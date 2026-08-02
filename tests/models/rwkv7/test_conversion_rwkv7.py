# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from transformers import AutoModelForCausalLM
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import (
    convert_rwkv7_checkpoint_to_hf_format,
    convert_state_dict,
    infer_rwkv7_config,
)
from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM


def _config() -> Rwkv7Config:
    return Rwkv7Config(
        vocab_size=31,
        context_length=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        head_size=4,
    )


def _legacy_state_dict(model: Rwkv7ForCausalLM) -> dict[str, torch.Tensor]:
    raw = {}
    for name, tensor in model.state_dict().items():
        if name == "model.embeddings.weight":
            raw_name = "emb.weight"
        elif name.startswith("model."):
            raw_name = name.removeprefix("model.")
        else:
            raw_name = name
        raw[raw_name] = tensor.detach().clone()
    return raw


def test_convert_raw_checkpoint_strict_auto_load_round_trip(tmp_path) -> None:
    torch.manual_seed(0)
    source = Rwkv7ForCausalLM(_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected_logits = source(input_ids).logits
    checkpoint = tmp_path / "rwkv7-g1i-ctx32.pth"
    output_dir = tmp_path / "artifact"
    torch.save(_legacy_state_dict(source), checkpoint)

    convert_rwkv7_checkpoint_to_hf_format(str(checkpoint), str(output_dir))
    converted = AutoModelForCausalLM.from_pretrained(output_dir).eval()

    assert isinstance(converted, Rwkv7ForCausalLM)
    assert converted.config.context_length == 32
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(converted.state_dict()[name], tensor)
    torch.testing.assert_close(converted(input_ids).logits, expected_logits)


def test_converter_can_fuse_embedding_layer_norm(tmp_path) -> None:
    torch.manual_seed(0)
    source = Rwkv7ForCausalLM(_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected_logits = source(input_ids).logits
    checkpoint = tmp_path / "rwkv7-g1i-ctx32.pth"
    output_dir = tmp_path / "fused-artifact"
    torch.save(_legacy_state_dict(source), checkpoint)

    convert_rwkv7_checkpoint_to_hf_format(
        str(checkpoint),
        str(output_dir),
        fuse_embedding_layer_norm=True,
    )
    converted = AutoModelForCausalLM.from_pretrained(output_dir).eval()

    assert converted.config.embedding_layer_norm_fused
    torch.testing.assert_close(converted(input_ids).logits, expected_logits)


@pytest.mark.parametrize(
    "invalid_checkpoint, message",
    [
        ({"state_dict": {}}, "directly map string parameter names to tensors"),
        ({"emb.weight": torch.empty(31, 16)}, "required for config inference"),
    ],
)
def test_converter_rejects_non_raw_or_incomplete_state_dict(invalid_checkpoint, message) -> None:
    with pytest.raises(ValueError, match=message):
        infer_rwkv7_config(invalid_checkpoint)


def test_converter_rejects_unknown_keys_and_wrong_shapes() -> None:
    model = Rwkv7ForCausalLM(_config())
    raw = _legacy_state_dict(model)
    raw["blocks.0.att.w1"] = torch.empty(16, 31)
    with pytest.raises(ValueError, match="low-rank tensor"):
        infer_rwkv7_config(raw)

    raw = _legacy_state_dict(model)
    raw["optimizer.step"] = torch.tensor(1)
    with pytest.raises(ValueError, match="unsupported tensors"):
        convert_state_dict(raw, model.config)
