# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import json

import pytest

import transformers.models.rwkv7.prepare_rwkv7_hf_upload as publication_rwkv7


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(path) -> None:
    path.mkdir()
    _write_json(path / "config.json", {"architectures": ["Rwkv7ForCausalLM"], "model_type": "rwkv7"})
    _write_json(path / "generation_config.json", {"do_sample": False})
    _write_json(path / "tokenizer_config.json", {"model_max_length": 4096})
    _write_json(path / "tokenizer.json", {"version": "1.0"})
    _write_json(
        path / "rwkv7_conversion.json",
        {"checkpoint_sha256": "b" * 64, "source_revision": "a" * 40},
    )
    _write_json(
        path / "rwkv7_validation.json",
        {
            "architecture": "Rwkv7ForCausalLM",
            "device": "cpu",
            "dtype": "float32",
            "generated_ids": [[1, 2, 3, 4]],
            "input_ids": [[1, 2, 3]],
            "max_new_tokens": 1,
            "observed_wkv_backends": ["flash_rwkv"],
            "strict_load": True,
        },
    )
    (path / "model.safetensors").write_bytes(b"safetensors-for-test")


def test_rwkv7_publication_audits_files_and_renders_side_effect_free_upload(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    _artifact(artifact)
    model_card = tmp_path / "MODEL_CARD.md"
    license_file = tmp_path / "SOURCE_LICENSE"
    model_card.write_text("# RWKV-7\n", encoding="utf-8")
    license_file.write_text("Apache License 2.0\n", encoding="utf-8")
    monkeypatch.setattr(
        publication_rwkv7,
        "validate_rwkv7_artifact_in_subprocess",
        lambda *args, **kwargs: {
            "architecture": "Rwkv7ForCausalLM",
            "device": "cpu",
            "dtype": "float32",
            "generated_ids": [[1, 2, 3, 4]],
            "input_ids": [[1, 2, 3]],
            "max_new_tokens": 1,
            "observed_wkv_backends": ["flash_rwkv"],
            "strict_load": True,
        },
    )

    blocked = publication_rwkv7.prepare_rwkv7_hf_upload(
        artifact,
        model_card_path=model_card,
        license_path=license_file,
    )

    assert blocked["dry_run"]["blockers"] == ["authentication_not_checked", "hub_repo_id_missing"]
    assert blocked["dry_run"]["command"][:3] == ["hf", "upload", "<namespace>/<repo>"]
    assert blocked["dry_run"]["network_called"] is False
    assert blocked["dry_run"]["artifact_ready"] is True
    assert blocked["dry_run"]["command_ready"] is False
    assert blocked["artifact"]["source_revision"] == "a" * 40
    assert blocked["artifact"]["checkpoint_sha256"] == "b" * 64
    assert blocked["artifact"]["weight_files"] == ["model.safetensors"]
    assert (artifact / "README.md").read_text(encoding="utf-8") == "# RWKV-7\n"
    assert (artifact / "LICENSE").read_text(encoding="utf-8") == "Apache License 2.0\n"
    assert (artifact / "rwkv7_hf_upload_manifest.json").is_file()

    ready = publication_rwkv7.prepare_rwkv7_hf_upload(artifact, hub_repo_id="rwkv-test/native-rwkv7")

    assert ready["dry_run"]["blockers"] == ["authentication_not_checked"]
    assert ready["dry_run"]["artifact_ready"] is True
    assert ready["dry_run"]["command_ready"] is True
    assert ready["dry_run"]["command"][:3] == ["hf", "upload", "rwkv-test/native-rwkv7"]
    assert "--token" not in ready["dry_run"]["command"]
    hashed_names = {file["name"] for file in ready["artifact"]["files"]}
    assert {
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "rwkv7_conversion.json",
        "rwkv7_validation.json",
        "tokenizer.json",
        "tokenizer_config.json",
    } == hashed_names


def test_rwkv7_publication_rejects_incomplete_shards_and_non_oid_revision(tmp_path, monkeypatch) -> None:
    artifact = tmp_path / "artifact"
    _artifact(artifact)
    (artifact / "README.md").write_text("# Card\n", encoding="utf-8")
    (artifact / "LICENSE").write_text("license\n", encoding="utf-8")
    (artifact / "model.safetensors").unlink()
    _write_json(
        artifact / "model.safetensors.index.json",
        {"weight_map": {"head.weight": "model-00001-of-00002.safetensors"}},
    )
    monkeypatch.setattr(publication_rwkv7, "validate_rwkv7_artifact_in_subprocess", lambda *args, **kwargs: {})

    with pytest.raises(ValueError, match="missing weight files"):
        publication_rwkv7.prepare_rwkv7_hf_upload(artifact)

    (artifact / "model-00001-of-00002.safetensors").write_bytes(b"shard")
    conversion = json.loads((artifact / "rwkv7_conversion.json").read_text(encoding="utf-8"))
    conversion["source_revision"] = "branch-name"
    _write_json(artifact / "rwkv7_conversion.json", conversion)
    with pytest.raises(ValueError, match="full 40- or 64-digit hexadecimal OID"):
        publication_rwkv7.prepare_rwkv7_hf_upload(artifact)
