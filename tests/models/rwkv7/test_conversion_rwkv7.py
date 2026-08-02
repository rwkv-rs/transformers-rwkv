# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from transformers import PreTrainedTokenizerFast
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
        wkv_backend="reference",
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

    result = convert_rwkv7_checkpoint_to_hf_format(
        str(checkpoint),
        str(output_dir),
        source_revision="source-revision-for-test",
        validation_input_ids=[1, 2, 3],
        validation_max_new_tokens=2,
    )
    converted = Rwkv7ForCausalLM.from_pretrained(output_dir).eval()

    assert isinstance(converted, Rwkv7ForCausalLM)
    assert converted.config.context_length == 32
    assert converted.config.wkv_backend == "auto"
    converted.config.wkv_backend = "reference"
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(converted.state_dict()[name], tensor)
    torch.testing.assert_close(converted(input_ids).logits, expected_logits)
    assert result["validation"]["architecture"] == "Rwkv7ForCausalLM"
    assert result["validation"]["device"] == "cpu"
    assert result["validation"]["requested_wkv_backend"] == "auto"
    assert result["validation"]["strict_load"]
    assert result["validation"]["generated_ids"] == [[1, 2, 3, 2, 2]]
    assert "rwkv7_validation.json" in result["validation"]["artifact_files"]
    assert result["conversion"]["source_revision"] == "source-revision-for-test"
    assert len(result["conversion"]["checkpoint_sha256"]) == 64
    assert {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "rwkv7_conversion.json",
        "rwkv7_validation.json",
    }.issubset(path.name for path in output_dir.iterdir())
    assert json.loads((output_dir / "rwkv7_validation.json").read_text())["strict_load"]


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
        wkv_backend="reference",
        validation_max_new_tokens=2,
    )
    converted = Rwkv7ForCausalLM.from_pretrained(output_dir).eval()

    assert converted.config.embedding_layer_norm_fused
    assert converted.config.wkv_backend == "reference"
    torch.testing.assert_close(converted(input_ids).logits, expected_logits)


def test_converter_builds_publication_ready_hf_artifact_and_upload_dry_run(tmp_path) -> None:
    source = Rwkv7ForCausalLM(_config()).eval()
    checkpoint = tmp_path / "rwkv7-g1i-ctx32.pth"
    tokenizer_dir = tmp_path / "tokenizer"
    output_dir = tmp_path / "publication"
    model_card = tmp_path / "MODEL_CARD.md"
    license_file = tmp_path / "SOURCE_LICENSE"
    torch.save(_legacy_state_dict(source), checkpoint)
    vocabulary = {"[UNK]": 0, **{f"token-{index}": index for index in range(1, 31)}}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    tokenizer.save_pretrained(tokenizer_dir)
    model_card.write_text("# Native RWKV-7\n", encoding="utf-8")
    license_file.write_text("Apache License 2.0\n", encoding="utf-8")

    result = convert_rwkv7_checkpoint_to_hf_format(
        str(checkpoint),
        str(output_dir),
        tokenizer_name_or_path=str(tokenizer_dir),
        source_revision="a" * 40,
        wkv_backend="reference",
        publication_ready=True,
        model_card_path=str(model_card),
        license_path=str(license_file),
        validation_max_new_tokens=2,
    )

    publication = result["publication"]
    assert publication["artifact"]["fresh_validation"]["strict_load"]
    assert publication["artifact"]["fresh_validation"]["observed_wkv_backends"] == ["reference"]
    assert publication["dry_run"]["blockers"] == ["authentication_not_checked", "hub_repo_id_missing"]
    assert publication["dry_run"]["network_called"] is False
    assert {
        "LICENSE",
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "rwkv7_conversion.json",
        "rwkv7_hf_upload_manifest.json",
        "rwkv7_validation.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }.issubset(path.name for path in output_dir.iterdir())


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
