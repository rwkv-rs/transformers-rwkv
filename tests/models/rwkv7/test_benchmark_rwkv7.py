# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from transformers import AutoModelForCausalLM
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config


_BENCHMARK_PATH = Path(__file__).parents[3] / "benchmark" / "rwkv7" / "standalone.py"
_SPEC = importlib.util.spec_from_file_location("rwkv7_standalone_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_rwkv7 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_rwkv7)


def _tiny_model():
    config = Rwkv7Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        head_size=4,
        context_length=16,
    )
    return AutoModelForCausalLM.from_config(config).eval()


def test_rwkv7_benchmark_requires_explicit_model() -> None:
    with pytest.raises(SystemExit):
        benchmark_rwkv7._parse_args(["--output", "result.json"])


def test_rwkv7_benchmark_correctness_gate_carries_recurrent_state(synthetic_fla_public_contract) -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    result, prefix_state = benchmark_rwkv7._correctness_gate(
        model,
        input_ids,
        prompt_tokens=3,
        dtype=torch.float32,
    )

    assert result["passed"] is True
    assert result["compared_tokens"] == 2
    assert result["observed_backends"] == ["flash_rwkv"]
    assert result["backend_observations"] == {
        "one_shot": ["flash_rwkv"],
        "prefix_prefill": ["flash_rwkv"],
        "staged_decode": ["flash_rwkv"],
    }
    assert result["state_components"] == 3
    assert result["input_state_preserved"] is True
    assert len(prefix_state) == 3


def test_rwkv7_benchmark_explicit_backend_observation_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="failed closed"):
        benchmark_rwkv7._require_observed_backend({"unexpected_provider"})


def test_rwkv7_benchmark_rejects_corrupt_final_recurrent_state(synthetic_fla_public_contract) -> None:
    class CorruptFinalState(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped
            self.decode_calls = 0

        def forward(self, *args, **kwargs):
            output = self.wrapped(*args, **kwargs)
            if kwargs.get("state") is not None:
                self.decode_calls += 1
                if self.decode_calls == 2:
                    output.state = (output.state[0] + 1, *output.state[1:])
            return output

    model = CorruptFinalState(_tiny_model())
    with pytest.raises(AssertionError):
        benchmark_rwkv7._correctness_gate(
            model,
            torch.tensor([[1, 2, 3, 4, 5]]),
            prompt_tokens=3,
            dtype=torch.float32,
        )


def test_rwkv7_benchmark_rejects_input_state_mutation(synthetic_fla_public_contract) -> None:
    class MutateInputState(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, *args, **kwargs):
            output = self.wrapped(*args, **kwargs)
            state = kwargs.get("state")
            if state is not None:
                with torch.no_grad():
                    state[0].add_(1)
            return output

    model = MutateInputState(_tiny_model())
    with pytest.raises(AssertionError):
        benchmark_rwkv7._correctness_gate(
            model,
            torch.tensor([[1, 2, 3, 4, 5]]),
            prompt_tokens=3,
            dtype=torch.float32,
        )


def test_rwkv7_benchmark_rejects_cross_stage_backend_changes() -> None:
    with pytest.raises(RuntimeError, match="stages selected inconsistent"):
        benchmark_rwkv7._require_consistent_backend_observations(
            {"prefill": {"unexpected_provider"}, "decode": {"flash_rwkv"}}
        )


def test_rwkv7_benchmark_validates_flash_runtime_provenance(monkeypatch) -> None:
    calls = []

    def validate():
        calls.append(True)
        return {"repository": "https://github.com/rwkv-rs/fla-rwkv.git", "revision": "a" * 40}

    monkeypatch.setattr(benchmark_rwkv7, "validate_rwkv7_runtime_provenance", validate)
    with pytest.raises(RuntimeError, match="failed closed"):
        benchmark_rwkv7._validated_operator_provenance({"unexpected_provider"})
    assert benchmark_rwkv7._validated_operator_provenance({"flash_rwkv"}) == {
        "repository": "https://github.com/rwkv-rs/fla-rwkv.git",
        "revision": "a" * 40,
    }
    assert calls == [True]


def _write_sharded_artifact(artifact_dir: Path) -> None:
    (artifact_dir / "config.json").write_text('{"model_type":"rwkv7"}\n')
    (artifact_dir / "tokenizer.json").write_text('{"version":"1.0"}\n')
    (artifact_dir / "model-00001-of-00002.safetensors").write_bytes(b"first shard")
    (artifact_dir / "model-00002-of-00002.safetensors").write_bytes(b"second shard")
    (artifact_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embeddings.weight": "model-00001-of-00002.safetensors",
                    "head.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )


def test_rwkv7_benchmark_hashes_local_artifact_files(tmp_path) -> None:
    _write_sharded_artifact(tmp_path)
    provenance = benchmark_rwkv7._artifact_provenance(str(tmp_path), SimpleNamespace(_commit_hash=None))

    assert provenance["identity"] == {"method": "direct_sha256"}
    assert provenance["weight_files"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    records = {record["name"]: record for record in provenance["files"]}
    assert records.keys() >= {
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert all(len(record["sha256"]) == 64 for record in records.values())


def test_rwkv7_benchmark_verifies_canonical_artifact_manifest(tmp_path) -> None:
    _write_sharded_artifact(tmp_path)
    tracked_paths, _ = benchmark_rwkv7._local_artifact_files(tmp_path)
    files = [benchmark_rwkv7._file_record(path, tmp_path) for path in tracked_paths]
    (tmp_path / "rwkv7_hf_upload_manifest.json").write_text(
        json.dumps({"schema_version": 1, "artifact": {"files": files}})
    )

    provenance = benchmark_rwkv7._artifact_provenance(str(tmp_path), SimpleNamespace(_commit_hash=None))
    assert provenance["identity"]["method"] == "rwkv7_hf_upload_manifest"

    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"changed shard")
    with pytest.raises(ValueError, match="digest changed"):
        benchmark_rwkv7._artifact_provenance(str(tmp_path), SimpleNamespace(_commit_hash=None))


def test_rwkv7_benchmark_rejects_local_artifact_without_weights(tmp_path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"rwkv7"}\n')
    with pytest.raises(ValueError, match="requires model.safetensors"):
        benchmark_rwkv7._artifact_provenance(str(tmp_path), SimpleNamespace(_commit_hash=None))


def test_rwkv7_benchmark_memory_excludes_setup_and_warms_after_empty_cache(monkeypatch) -> None:
    order = []
    allocator = {"allocated": 100, "peak": 100}

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing is True

        def record(self):
            order.append("event_record")

        def elapsed_time(self, other):
            assert isinstance(other, Event)
            return 2.0

    class BackendLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_wkv_backend = "flash_rwkv"

    model = torch.nn.Sequential(BackendLayer())

    def empty_cache():
        order.append("empty_cache")

    def setup():
        order.append("setup")
        allocator["allocated"] = 140
        allocator["peak"] = max(allocator["peak"], allocator["allocated"])
        return None

    def operation(_):
        order.append("operation")
        allocator["peak"] = 180
        return object()

    def reset_peak_memory_stats(_):
        order.append("reset_peak")
        allocator["peak"] = allocator["allocated"]

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: order.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak_memory_stats)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: allocator["allocated"])
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _: allocator["peak"])

    result = benchmark_rwkv7._measure_cuda(
        operation,
        setup,
        model=model,
        stage="decode",
        warmup=1,
        iterations=1,
        tokens_per_iteration=1,
        device=torch.device("cuda:0"),
    )

    assert order.index("empty_cache") < order.index("setup")
    assert order.index("reset_peak") > [index for index, value in enumerate(order) if value == "setup"][1]
    assert result["observed_backends"] == ["flash_rwkv"]
    assert result["torch_allocator_memory"] == {
        "scope": "PyTorch allocator only; excludes state setup and may not include native external allocations",
        "resident_before_operation_bytes_samples": [140],
        "peak_allocated_bytes_samples": [180],
        "operation_peak_increment_bytes_samples": [40],
        "max_operation_peak_increment_bytes": 40,
    }
