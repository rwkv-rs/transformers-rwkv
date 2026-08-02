# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Audit a complete RWKV-7 Hugging Face artifact and render its side-effect-free upload dry run."""

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

from .convert_rwkv7_checkpoint_to_hf import validate_rwkv7_artifact_in_subprocess


_MANIFEST_NAME = "rwkv7_hf_upload_manifest.json"
_TOKENIZER_PAYLOADS = ("tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read publication metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Publication metadata must be a JSON object: {path}")
    return payload


def _copy_publication_file(source: str | os.PathLike | None, destination: Path) -> None:
    if source is None:
        return
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Publication source file does not exist: {source_path}")
    if source_path.resolve() != destination.resolve():
        shutil.copyfile(source_path, destination)


def _weight_files(artifact_dir: Path) -> list[Path]:
    index_path = artifact_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json requires a non-empty weight_map.")
        filenames = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and name.endswith(".safetensors") for name in filenames):
            raise ValueError("The shard index must reference only .safetensors files.")
        missing = [name for name in filenames if not (artifact_dir / name).is_file()]
        if missing:
            raise ValueError(f"The shard index references missing weight files: {missing}.")
        return [index_path, *(artifact_dir / name for name in filenames)]
    weight_path = artifact_dir / "model.safetensors"
    if not weight_path.is_file():
        raise ValueError("Publication requires model.safetensors or model.safetensors.index.json with all shards.")
    return [weight_path]


def _validate_artifact(artifact_dir: Path) -> dict[str, Any]:
    required = (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "README.md",
        "LICENSE",
        "rwkv7_conversion.json",
        "rwkv7_validation.json",
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    tokenizer_payloads = [name for name in _TOKENIZER_PAYLOADS if (artifact_dir / name).is_file()]
    if not tokenizer_payloads:
        missing.append("one tokenizer payload: " + ", ".join(_TOKENIZER_PAYLOADS))
    if missing:
        raise ValueError(f"RWKV-7 publication artifact is incomplete; missing {missing}.")

    config = _read_json(artifact_dir / "config.json")
    conversion = _read_json(artifact_dir / "rwkv7_conversion.json")
    validation = _read_json(artifact_dir / "rwkv7_validation.json")
    if config.get("model_type") != "rwkv7" or "Rwkv7ForCausalLM" not in config.get("architectures", []):
        raise ValueError("config.json must describe the native Rwkv7ForCausalLM architecture.")
    source_revision = conversion.get("source_revision")
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", source_revision) is None
    ):
        raise ValueError("rwkv7_conversion.json requires source_revision as a full 40- or 64-digit hexadecimal OID.")
    checkpoint_sha256 = conversion.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", checkpoint_sha256) is None:
        raise ValueError("rwkv7_conversion.json requires a SHA-256 checkpoint identity.")
    if validation.get("strict_load") is not True or not validation.get("generated_ids"):
        raise ValueError("rwkv7_validation.json must record strict load and deterministic generated_ids.")
    if validation.get("observed_wkv_backends") != ["flash_rwkv"]:
        raise ValueError("rwkv7_validation.json must record the fail-closed FlashRWKV provider.")
    for name in ("README.md", "LICENSE"):
        if not (artifact_dir / name).read_text(encoding="utf-8").strip():
            raise ValueError(f"Publication file must not be empty: {name}")

    return {
        "checkpoint_sha256": checkpoint_sha256,
        "source_revision": source_revision,
        "tokenizer_payloads": tokenizer_payloads,
        "validation": {
            "architecture": validation.get("architecture"),
            "observed_wkv_backends": validation.get("observed_wkv_backends"),
            "strict_load": True,
        },
        "weight_files": [path.name for path in _weight_files(artifact_dir)],
    }


def prepare_rwkv7_hf_upload(
    artifact_dir: str | os.PathLike,
    *,
    model_card_path: str | os.PathLike | None = None,
    license_path: str | os.PathLike | None = None,
    hub_repo_id: str | None = None,
    private: bool = False,
    create_pr: bool = False,
) -> dict[str, Any]:
    """Copy explicit publication metadata, audit the artifact, and write a no-network upload manifest."""
    artifact_path = Path(artifact_dir).expanduser()
    if not artifact_path.is_dir():
        raise FileNotFoundError(f"RWKV-7 artifact directory does not exist: {artifact_path}")
    _copy_publication_file(model_card_path, artifact_path / "README.md")
    _copy_publication_file(license_path, artifact_path / "LICENSE")
    audit = _validate_artifact(artifact_path)
    recorded_validation = _read_json(artifact_path / "rwkv7_validation.json")
    recorded_input_ids = recorded_validation.get("input_ids")
    if (
        not isinstance(recorded_input_ids, list)
        or len(recorded_input_ids) != 1
        or not isinstance(recorded_input_ids[0], list)
    ):
        raise ValueError("rwkv7_validation.json requires one validation input_ids row.")
    fresh_validation = validate_rwkv7_artifact_in_subprocess(
        artifact_path,
        input_ids=recorded_input_ids[0],
        max_new_tokens=recorded_validation.get("max_new_tokens", 4),
        device=recorded_validation.get("device", "cpu"),
        dtype=recorded_validation.get("dtype", "auto"),
    )
    for name in ("architecture", "generated_ids", "input_ids", "observed_wkv_backends", "strict_load"):
        if fresh_validation.get(name) != recorded_validation.get(name):
            raise ValueError(
                f"Fresh publication validation changed `{name}`: "
                f"recorded={recorded_validation.get(name)!r}, fresh={fresh_validation.get(name)!r}."
            )

    blockers = ["authentication_not_checked"]
    if hub_repo_id is None:
        blockers.append("hub_repo_id_missing")
        rendered_repo_id = "<namespace>/<repo>"
    elif re.fullmatch(r"[^/\s]+/[^/\s]+", hub_repo_id) is None:
        raise ValueError("Hub repository ID must have the form namespace/repo.")
    else:
        rendered_repo_id = hub_repo_id
    command = [
        "hf",
        "upload",
        rendered_repo_id,
        str(artifact_path.resolve()),
        ".",
        "--repo-type",
        "model",
        "--commit-message",
        f"Upload RWKV-7 artifact from {audit['source_revision']}",
    ]
    if private:
        command.append("--private")
    if create_pr:
        command.append("--create-pr")

    files = [
        {
            "name": path.relative_to(artifact_path).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(artifact_path.rglob("*"))
        if path.is_file() and path.name != _MANIFEST_NAME
    ]
    manifest = {
        "schema_version": 1,
        "artifact": {
            "directory": str(artifact_path.resolve()),
            "files": files,
            "fresh_validation": fresh_validation,
            **audit,
        },
        "dry_run": {
            "authentication_checked": False,
            "blockers": blockers,
            "command": command,
            "command_shell": shlex.join(command),
            "network_called": False,
            "artifact_ready": True,
            "command_ready": "hub_repo_id_missing" not in blockers,
        },
    }
    (artifact_path / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--model_card_path")
    parser.add_argument("--license_path")
    parser.add_argument("--hub_repo_id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--create_pr", action="store_true")
    args = parser.parse_args(argv)
    manifest = prepare_rwkv7_hf_upload(
        args.artifact_dir,
        model_card_path=args.model_card_path,
        license_path=args.license_path,
        hub_repo_id=args.hub_repo_id,
        private=args.private,
        create_pr=args.create_pr,
    )
    print(f"RWKV7_HF_UPLOAD_DRY_RUN={json.dumps(manifest, sort_keys=True)}")


if __name__ == "__main__":
    main()
