# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from transformers.models.rwkv7 import modeling_rwkv7


_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples/pytorch/any_to_rwkv/qwen3_5_conversion.py"
_SPEC = importlib.util.spec_from_file_location("any_to_rwkv_qwen3_5_conversion", _EXAMPLE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
qwen3_5_conversion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qwen3_5_conversion)


def test_any_to_rwkv_converter_remains_example_owned() -> None:
    assert _EXAMPLE_PATH.is_file()
    assert importlib.util.find_spec("transformers.models.rwkv7.convert_qwen3_5_to_any_to_rwkv") is None


@pytest.fixture
def public_recurrent_contract(monkeypatch):
    calls = []

    def recurrent_rwkv7(
        r,
        w,
        k,
        v,
        a,
        b,
        *,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        calls.append((r.shape, initial_state.shape, output_final_state, cu_seqlens, state_indices, mode))
        output = (r + w + k + v + a + b) / 6
        final_state = initial_state + torch.einsum("bthk,bthv->bhkv", k.float(), v.float())
        return output, final_state

    class PublicRwkv7:
        get_last_rwkv7_provider = staticmethod(lambda: "flash_rwkv")

    PublicRwkv7.recurrent_rwkv7 = staticmethod(recurrent_rwkv7)
    monkeypatch.setattr(qwen3_5_conversion, "validate_rwkv7_runtime_provenance", lambda: {})
    monkeypatch.setattr(qwen3_5_conversion.importlib, "import_module", lambda _name: PublicRwkv7)
    return calls


def test_any_to_rwkv_example_composes_and_round_trips_source_components(tmp_path, public_recurrent_contract) -> None:
    torch.manual_seed(0)
    source = qwen3_5_conversion.tiny_qwen_source()
    converted = qwen3_5_conversion.AnyToRwkvConvertedForCausalLM.from_source(source)

    assert type(converted.config) is qwen3_5_conversion.AnyToRwkvConvertedConfig
    assert type(converted) is qwen3_5_conversion.AnyToRwkvConvertedForCausalLM
    assert converted.config.model_type == "any_to_rwkv_converted"
    assert "qwen" not in converted.config.model_type and converted.config.model_type != "rwkv7"
    assert converted.config.source_architecture == source.config.model_type
    assert converted.config.source_config["model_type"] == "qwen3_5_moe_text"
    assert converted.embed_tokens is source.model.embed_tokens
    assert converted.lm_head is source.lm_head
    assert [layer.recurrent_mixer.head_size for layer in converted.layers] == [128, 256]
    assert [layer.recurrent_mixer.num_heads for layer in converted.layers] == [2, 1]
    assert isinstance(converted.norm, nn.LayerNorm)

    for target_layer, source_layer in zip(converted.layers, source.model.layers):
        assert isinstance(target_layer.input_layernorm, nn.LayerNorm)
        assert isinstance(target_layer.post_mixer_layernorm, nn.LayerNorm)
        assert target_layer.moe is source_layer.mlp
        assert target_layer.moe.gate.weight is source_layer.mlp.gate.weight
        assert target_layer.moe.experts.gate_up_proj is source_layer.mlp.experts.gate_up_proj
        assert target_layer.moe.experts.down_proj is source_layer.mlp.experts.down_proj

    input_ids = torch.tensor([[1, 5, 8, 13]])
    output = converted(input_ids, labels=input_ids)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert converted.embed_tokens.weight.grad is not None
    assert converted.lm_head.weight.grad is not None
    assert converted.layers[0].moe.gate.weight.grad is not None
    assert converted.layers[0].recurrent_mixer.r.weight.grad is not None
    assert [call[0][-1] for call in public_recurrent_contract] == [128, 256]
    assert all(call[2:] == (True, None, None, "fp32io16") for call in public_recurrent_contract)

    converted.eval()
    expected_logits = converted(input_ids).logits.detach()
    converted.save_pretrained(tmp_path, safe_serialization=True)
    reloaded, loading_info = qwen3_5_conversion.AnyToRwkvConvertedForCausalLM.from_pretrained(
        tmp_path, output_loading_info=True
    )
    reloaded.eval()
    assert type(reloaded.config) is qwen3_5_conversion.AnyToRwkvConvertedConfig
    assert type(reloaded) is qwen3_5_conversion.AnyToRwkvConvertedForCausalLM
    assert set(loading_info) == {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    assert all(not diagnostics for diagnostics in loading_info.values())
    assert "auto_map" not in reloaded.config.to_dict()
    assert (tmp_path / "model.safetensors").is_file()
    torch.testing.assert_close(reloaded(input_ids).logits, expected_logits)


@pytest.mark.parametrize(
    ("scenario", "distribution_name", "repository", "error_match"),
    [
        pytest.param(
            "wrong-package",
            "official-flash-linear-attention",
            modeling_rwkv7.RWKV7_FLA_REPOSITORY,
            "distribution identity does not match",
            id="wrong-package",
        ),
        pytest.param(
            "fork",
            modeling_rwkv7.RWKV7_FLA_DISTRIBUTION,
            "https://github.com/attacker/fla-rwkv.git",
            "repository provenance mismatch",
            id="fork",
        ),
    ],
)
def test_any_to_rwkv_example_validates_runtime_provenance_before_public_api_import(
    monkeypatch,
    tmp_path,
    scenario,
    distribution_name,
    repository,
    error_match,
) -> None:
    module_origin = tmp_path / "fla" / "__init__.py"

    class ProvenanceDistribution:
        metadata = {"Name": distribution_name}
        version = "0.5.2"

        @staticmethod
        def read_text(_filename):
            return json.dumps(
                {
                    "url": repository,
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": modeling_rwkv7.RWKV7_FLA_REVISION,
                        "commit_id": modeling_rwkv7.RWKV7_FLA_REVISION,
                    },
                }
            )

        @staticmethod
        def locate_file(_filename):
            return module_origin

    imported_modules = []

    def unexpected_import(module_name):
        imported_modules.append(module_name)
        raise AssertionError(f"{scenario} provenance reached public API import")

    monkeypatch.setattr(
        modeling_rwkv7.importlib_metadata,
        "distribution",
        lambda _name: ProvenanceDistribution(),
    )
    monkeypatch.setattr(
        modeling_rwkv7.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(module_origin)),
    )
    monkeypatch.setattr(qwen3_5_conversion.importlib, "import_module", unexpected_import)

    with pytest.raises(RuntimeError, match=error_match):
        qwen3_5_conversion._load_public_recurrent_rwkv7()

    assert imported_modules == []
