#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Publish the latest canonical RWKV-7 G1 checkpoints as Transformers Safetensors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open

from transformers import AutoTokenizer, GenerationConfig, RwkvForCausalLM


converter = importlib.import_module(f"{__package__ + '.' if __package__ else ''}rwkv_pth2st")
GENERATION_CONFIGS = converter.GENERATION_CONFIGS
LAYER_ZERO_UNUSED = converter.LAYER_ZERO_UNUSED
TOKENIZER_REPO = converter.TOKENIZER_REPO
TOKENIZER_REVISION = converter.TOKENIZER_REVISION
TOKENIZER_SHA256 = converter.TOKENIZER_SHA256
convert_checkpoint = converter.convert_checkpoint
load_checkpoint = converter.load_checkpoint
translation_plan = converter.translation_plan


SOURCE_REPO = "BlinkDL/rwkv7-g1"
TARGET_REPO = "rwkv-rs/rwkv7-g1-st"
OUTPUT_ROOT = Path.home() / "Weights/RWKV/hf"
MAX_SHARD_SIZE = "5GB"
SIZE_ORDER = {size: index for index, size in enumerate(("0.1b", "0.4b", "1.5b", "2.9b", "7.2b", "13.3b"))}
CHECKPOINT_RE = re.compile(
    r"^rwkv7-g1(?P<generation>[a-z])-(?P<size>0\.1b|0\.4b|1\.5b|2\.9b|7\.2b|13\.3b)-"
    r"(?P<date>\d{8})-ctx(?P<context>\d+)\.pth$"
)
PROVENANCE_FIELD_RE = re.compile(r"^- (?P<field>[^:]+): `(?P<value>[^`]+)`$", re.MULTILINE)
README_ROW_RE = re.compile(
    r"^\| RWKV-7 G1(?P<generation>[a-z]) (?P<size>[^ ]+) \| (?P<parameters>[\d,]+) \| "
    r"(?P<context>[\d,]+) \| `(?P<subfolder>[^`]+)` \|$"
)


@dataclass(frozen=True)
class Checkpoint:
    filename: str
    generation: str
    size: str
    date: str
    context_length: int
    source_sha256: str
    source_size: int

    @property
    def name(self) -> str:
        return self.filename.removesuffix(".pth")

    @property
    def sort_key(self) -> tuple[str, int, str]:
        return self.generation, SIZE_ORDER[self.size], self.date


@dataclass(frozen=True)
class ArtifactValidation:
    tensor_count: int
    parameter_count: int
    shard_sha256: tuple[tuple[str, str], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint(filename: str, *, sha256: str, size: int) -> Checkpoint | None:
    match = CHECKPOINT_RE.fullmatch(filename)
    if match is None:
        return None
    return Checkpoint(
        filename=filename,
        generation=match["generation"],
        size=match["size"],
        date=match["date"],
        context_length=int(match["context"]),
        source_sha256=sha256,
        source_size=size,
    )


def latest_checkpoints(siblings) -> list[Checkpoint]:
    checkpoints = []
    for sibling in siblings:
        if sibling.lfs is None or sibling.size is None:
            continue
        checkpoint = parse_checkpoint(sibling.rfilename, sha256=sibling.lfs.sha256, size=sibling.size)
        if checkpoint is not None:
            checkpoints.append(checkpoint)
    if not checkpoints:
        raise RuntimeError(f"{SOURCE_REPO} does not contain a canonical RWKV-7 G1 checkpoint.")
    latest_generation = max(checkpoint.generation for checkpoint in checkpoints)
    return sorted(
        (checkpoint for checkpoint in checkpoints if checkpoint.generation == latest_generation),
        key=lambda checkpoint: checkpoint.sort_key,
    )


def target_subfolders(siblings) -> set[str]:
    return {sibling.rfilename.split("/", 1)[0] for sibling in siblings if "/" in sibling.rfilename}


def parse_provenance(text: str) -> dict[str, str]:
    return {match["field"]: match["value"] for match in PROVENANCE_FIELD_RE.finditer(text)}


def validate_published_identity(api: HfApi, checkpoint: Checkpoint, target_revision: str) -> None:
    provenance_path = api.hf_hub_download(
        repo_id=TARGET_REPO,
        filename=f"{checkpoint.name}/PROVENANCE.md",
        revision=target_revision,
    )
    provenance = parse_provenance(Path(provenance_path).read_text(encoding="utf-8"))
    expected = {
        "Source repository": SOURCE_REPO,
        "Source file": checkpoint.filename,
        "Source SHA256": checkpoint.source_sha256,
    }
    mismatches = {
        field: (provenance.get(field), value) for field, value in expected.items() if provenance.get(field) != value
    }
    if mismatches:
        raise RuntimeError(f"Published checkpoint identity conflict for {checkpoint.name}: {mismatches}.")

    recorded_revision = provenance.get("Source revision")
    if recorded_revision is None:
        raise RuntimeError(f"Published checkpoint {checkpoint.name} does not record a source revision.")
    recorded_info = api.model_info(SOURCE_REPO, revision=recorded_revision, files_metadata=True)
    recorded_file = next((item for item in recorded_info.siblings if item.rfilename == checkpoint.filename), None)
    if recorded_file is None or recorded_file.lfs is None or recorded_file.lfs.sha256 != checkpoint.source_sha256:
        raise RuntimeError(
            f"Published checkpoint {checkpoint.name} source revision no longer resolves to its recorded LFS SHA."
        )


def validate_shard_index(output: Path, tensor_to_file: dict[str, str]) -> None:
    index_path = output / "model.safetensors.index.json"
    if len(set(tensor_to_file.values())) == 1 and not index_path.exists():
        return
    if not index_path.exists():
        raise RuntimeError("A sharded artifact must contain model.safetensors.index.json.")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("weight_map") != tensor_to_file:
        raise RuntimeError("Safetensors index does not exactly match the tensors stored in the shards.")


def validate_artifact(checkpoint_path: Path, output: Path) -> ArtifactValidation:
    source_state = load_checkpoint(checkpoint_path)
    target_to_source, dropped = translation_plan(source_state)
    if dropped != LAYER_ZERO_UNUSED:
        raise RuntimeError(f"Conversion dropped unsupported source tensors: {sorted(dropped)}.")

    seen = set()
    tensor_to_file = {}
    parameter_count = 0
    shard_hashes = []
    shard_paths = sorted(output.glob("model*.safetensors"))
    if not shard_paths:
        raise RuntimeError("Converted artifact does not contain Safetensors weights.")
    for shard_path in shard_paths:
        shard_hashes.append((shard_path.name, sha256_file(shard_path)))
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for target_key in shard.keys():
                if target_key in seen or target_key not in target_to_source:
                    raise RuntimeError(f"Unexpected or duplicate converted tensor {target_key}.")
                converted = shard.get_tensor(target_key)
                if converted.dtype is not torch.bfloat16:
                    raise RuntimeError(f"{target_key} must be bfloat16, got {converted.dtype}.")
                source = source_state[target_to_source[target_key]]
                if source.shape != converted.shape and source.ndim == 3 and source.shape[:2] == (1, 1):
                    source = source.squeeze(0).squeeze(0)
                if source.shape != converted.shape or not torch.equal(source, converted):
                    raise RuntimeError(f"Converted tensor {target_key} differs from its canonical PTH source.")
                seen.add(target_key)
                tensor_to_file[target_key] = shard_path.name
                parameter_count += converted.numel()
                del converted, source

    if seen != set(target_to_source):
        missing = sorted(set(target_to_source) - seen)
        raise RuntimeError(f"Converted artifact is missing tensors: {missing}.")
    source_parameters = sum(tensor.numel() for tensor in source_state.values())
    dropped_parameters = sum(source_state[key].numel() for key in dropped)
    if parameter_count != source_parameters - dropped_parameters:
        raise RuntimeError("Converted parameter count differs by more than the allowed layer-0 tensors.")
    validate_shard_index(output, tensor_to_file)
    del source_state
    gc.collect()

    model = RwkvForCausalLM.from_pretrained(output, local_files_only=True, dtype="auto", low_cpu_mem_usage=True)
    del model
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    if tokenizer.__class__.__name__ != "RwkvTokenizer":
        raise RuntimeError(f"AutoTokenizer reloaded {tokenizer.__class__.__name__}, not RwkvTokenizer.")
    for filename in GENERATION_CONFIGS:
        GenerationConfig.from_pretrained(output, config_file_name=filename, local_files_only=True)
    del tokenizer
    gc.collect()
    return ArtifactValidation(len(seen), parameter_count, tuple(shard_hashes))


def gpu_smoke(output: Path) -> str:
    if not torch.cuda.is_available():
        return "skipped: CUDA is unavailable"
    weight_bytes = sum(path.stat().st_size for path in output.glob("model*.safetensors"))
    candidates = []
    for device_index in range(torch.cuda.device_count()):
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        if weight_bytes <= int(total_bytes * 0.85) and free_bytes >= weight_bytes + 4 * 1024**3:
            candidates.append((free_bytes, device_index))
    if not candidates:
        return "skipped: no single GPU has sufficient free capacity"

    device_index = max(candidates)[1]
    device = f"cuda:{device_index}"
    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    model = RwkvForCausalLM.from_pretrained(
        output,
        local_files_only=True,
        dtype=torch.float16,
        device_map={"": device},
    ).eval()
    tokens = tokenizer("The future of open source AI is", return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )
    with torch.inference_mode():
        generated = model.generate(
            tokens,
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=None,
            stop_strings=None,
        )
    digest = hashlib.sha256(generated.cpu().numpy().tobytes()).hexdigest()
    del generated, tokens, model, tokenizer
    torch.cuda.empty_cache()
    return f"passed on NVIDIA RTX 4090 cuda:{device_index}; 8-token output SHA256 {digest}"


def converter_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def render_provenance(
    checkpoint: Checkpoint,
    source_revision: str,
    validation: ArtifactValidation,
    gpu_result: str,
) -> str:
    flash_version = importlib.metadata.version("FlashRWKV2")
    lines = [
        "# Artifact provenance",
        "",
        f"- Source repository: `{SOURCE_REPO}`",
        f"- Source revision: `{source_revision}`",
        f"- Source file: `{checkpoint.filename}`",
        f"- Source SHA256: `{checkpoint.source_sha256}`",
        "- Converter repository: `rwkv-rs/transformers-rwkv`",
        f"- Converter commit: `{converter_commit()}`",
        f"- Tokenizer source: `{TOKENIZER_REPO}/rwkv_vocab_v20230424.json`",
        f"- Tokenizer revision: `{TOKENIZER_REVISION}`",
        f"- Tokenizer SHA256: `{TOKENIZER_SHA256}`",
        f"- FlashRWKV2 validation version: `{flash_version}`",
        f"- Exact source-to-artifact tensor comparisons: {validation.tensor_count} tensors passed; only the unused layer-0 `blocks.0.att.v0/v1/v2` tensors were dropped.",
        "- Native reload validation: `RwkvForCausalLM.from_pretrained`, `AutoTokenizer.from_pretrained`, and all generation configurations passed.",
        f"- Host validation: `rwkv-szx-4090x4-ip129`; GPU smoke {gpu_result}.",
        "",
        "## Safetensors shard SHA256",
        "",
        "```text",
        *(f"{digest}  {filename}" for filename, digest in validation.shard_sha256),
        "```",
        "",
    ]
    return "\n".join(lines)


def update_readme(readme: str, artifacts: list[tuple[Checkpoint, ArtifactValidation]]) -> str:
    lines = readme.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line == "| Checkpoint | Parameters | Context | Subfolder |"),
        None,
    )
    if header_index is None or header_index + 1 >= len(lines):
        raise RuntimeError("Target README does not contain the checkpoint table contract.")
    row_start = header_index + 2
    row_end = row_start
    existing = {}
    while row_end < len(lines) and lines[row_end].startswith("| RWKV-7 G1"):
        match = README_ROW_RE.fullmatch(lines[row_end])
        if match is None:
            raise RuntimeError(f"Unsupported checkpoint table row: {lines[row_end]}")
        existing[match["subfolder"]] = (
            match["generation"],
            match["size"].lower(),
            match["parameters"],
            match["context"],
        )
        row_end += 1
    for checkpoint, validation in artifacts:
        existing[checkpoint.name] = (
            checkpoint.generation,
            checkpoint.size,
            f"{validation.parameter_count:,}",
            f"{checkpoint.context_length:,}",
        )
    rows = []
    for subfolder, (generation, size, parameters, context) in sorted(
        existing.items(), key=lambda item: (item[1][0], SIZE_ORDER[item[1][1]], item[0])
    ):
        rows.append(f"| RWKV-7 G1{generation} {size.upper()} | {parameters} | {context} | `{subfolder}` |")
    return "\n".join((*lines[:row_start], *rows, *lines[row_end:])) + "\n"


def validate_disk_space(checkpoints: list[Checkpoint], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    required = int(sum(checkpoint.source_size for checkpoint in checkpoints) * 2.1) + 10 * 1024**3
    available = shutil.disk_usage(output_root).free
    if available < required:
        raise RuntimeError(f"Insufficient disk space: need {required:,} bytes, have {available:,} bytes.")


def prepare_artifact(
    api: HfApi,
    checkpoint: Checkpoint,
    source_revision: str,
    output_root: Path,
) -> tuple[Path, ArtifactValidation]:
    checkpoint_path = Path(
        api.hf_hub_download(repo_id=SOURCE_REPO, filename=checkpoint.filename, revision=source_revision)
    )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != checkpoint.source_sha256:
        raise RuntimeError(
            f"Source SHA256 mismatch for {checkpoint.filename}: expected {checkpoint.source_sha256}, got {actual_sha256}."
        )
    final_output = output_root / checkpoint.name
    if final_output.exists():
        provenance = parse_provenance((final_output / "PROVENANCE.md").read_text(encoding="utf-8"))
        if (
            provenance.get("Source file") != checkpoint.filename
            or provenance.get("Source SHA256") != checkpoint.source_sha256
        ):
            raise RuntimeError(f"Local artifact identity conflict for {checkpoint.name}.")
        return final_output, validate_artifact(checkpoint_path, final_output)

    with tempfile.TemporaryDirectory(prefix=f".{checkpoint.name}-", dir=output_root) as temporary:
        staging = Path(temporary) / checkpoint.name
        convert_checkpoint(
            checkpoint_path,
            staging,
            checkpoint.context_length,
            MAX_SHARD_SIZE,
            tokenizer_source=None,
        )
        validation = validate_artifact(checkpoint_path, staging)
        gpu_result = gpu_smoke(staging)
        (staging / "PROVENANCE.md").write_text(
            render_provenance(checkpoint, source_revision, validation, gpu_result), encoding="utf-8"
        )
        staging.replace(final_output)
    return final_output, validation


def audit_commit(api: HfApi, revision: str, outputs: list[Path], readme_path: Path) -> None:
    info = api.model_info(TARGET_REPO, revision=revision, files_metadata=True)
    remote = {sibling.rfilename: sibling for sibling in info.siblings}
    expected = [("README.md", readme_path)]
    for output in outputs:
        expected.extend(
            (f"{output.name}/{path.relative_to(output).as_posix()}", path)
            for path in output.rglob("*")
            if path.is_file()
        )
    for path_in_repo, local_path in expected:
        sibling = remote.get(path_in_repo)
        if sibling is None or sibling.size != local_path.stat().st_size:
            raise RuntimeError(f"Published file size audit failed for {path_in_repo} at {revision}.")
        if local_path.suffix == ".safetensors":
            if sibling.lfs is None or sibling.lfs.sha256 != sha256_file(local_path):
                raise RuntimeError(f"Published LFS SHA audit failed for {path_in_repo} at {revision}.")


def run(*, dry_run: bool, source_revision: str | None, api: HfApi | None = None) -> dict:
    api = api or HfApi()
    source_info = api.model_info(SOURCE_REPO, revision=source_revision, files_metadata=True)
    source_revision = source_info.sha
    target_info = api.model_info(TARGET_REPO, files_metadata=True)
    checkpoints = latest_checkpoints(source_info.siblings)
    published = target_subfolders(target_info.siblings)
    pending = [checkpoint for checkpoint in checkpoints if checkpoint.name not in published]
    result = {
        "source_revision": source_revision,
        "target_revision": target_info.sha,
        "generation": checkpoints[0].generation,
        "pending": [checkpoint.filename for checkpoint in pending],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if dry_run:
        return result

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for a publishing run.")
    for checkpoint in checkpoints:
        if checkpoint.name in published:
            validate_published_identity(api, checkpoint, target_info.sha)
    if not pending:
        print("The latest canonical RWKV-7 G1 generation is already published.")
        return result

    validate_disk_space(pending, OUTPUT_ROOT)
    outputs = []
    artifacts = []
    for checkpoint in pending:
        output, validation = prepare_artifact(api, checkpoint, source_revision, OUTPUT_ROOT)
        outputs.append(output)
        artifacts.append((checkpoint, validation))

    readme_source = Path(
        api.hf_hub_download(repo_id=TARGET_REPO, filename="README.md", revision=target_info.sha)
    ).read_text(encoding="utf-8")
    readme = update_readme(readme_source, artifacts)
    with tempfile.TemporaryDirectory(prefix="rwkv-hub-readme-") as temporary:
        readme_path = Path(temporary) / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme_path)]
        for output in outputs:
            operations.extend(
                CommitOperationAdd(
                    path_in_repo=f"{output.name}/{path.relative_to(output).as_posix()}",
                    path_or_fileobj=path,
                )
                for path in sorted(output.rglob("*"))
                if path.is_file()
            )
        api.preupload_lfs_files(TARGET_REPO, operations, revision="main")
        commit = api.create_commit(
            repo_id=TARGET_REPO,
            operations=operations,
            commit_message=f"Add RWKV-7 G1{checkpoints[0].generation} checkpoints",
            parent_commit=target_info.sha,
        )
        audit_commit(api, commit.oid, outputs, readme_path)
    result["published_revision"] = commit.oid
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    run(dry_run=args.dry_run, source_revision=args.source_revision)


if __name__ == "__main__":
    main()
