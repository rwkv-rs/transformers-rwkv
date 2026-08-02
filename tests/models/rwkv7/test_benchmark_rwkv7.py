# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
from pathlib import Path

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
        wkv_backend="reference",
    )
    return AutoModelForCausalLM.from_config(config).eval()


def test_rwkv7_benchmark_requires_explicit_model() -> None:
    with pytest.raises(SystemExit):
        benchmark_rwkv7._parse_args(["--output", "result.json"])


def test_rwkv7_benchmark_correctness_gate_carries_recurrent_state() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    result, prefix_state = benchmark_rwkv7._correctness_gate(
        model,
        input_ids,
        prompt_tokens=3,
        requested_backend="reference",
        dtype=torch.float32,
    )

    assert result["passed"] is True
    assert result["compared_tokens"] == 2
    assert result["observed_backends"] == ["reference"]
    assert len(prefix_state) == 3


def test_rwkv7_benchmark_explicit_backend_observation_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="failed closed"):
        benchmark_rwkv7._require_observed_backend("flash_rwkv", {"reference"})
