#!/usr/bin/env python
"""Compose selected Qwen3.5 GDN/GQA token mixers with RWKV-7."""

import tempfile
from collections.abc import Callable

import torch

from transformers import AutoModelForCausalLM, Qwen3_5ForCausalLM, Qwen3_5TextConfig


LayerSelector = Callable[[int, str], bool]


def compose_qwen3_5_rwkv7(
    config: Qwen3_5TextConfig,
    selector: LayerSelector,
) -> Qwen3_5ForCausalLM:
    """Replace selected Qwen3.5 token mixers while retaining the rest of each layer."""
    config_values = config.to_dict()
    config_values["layer_types"] = [
        "rwkv7" if selector(index, layer_type) else layer_type for index, layer_type in enumerate(config.layer_types)
    ]
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig.from_dict(config_values))


def tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=32,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        rwkv7_head_size=32,
        rwkv7_backend="reference",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def main() -> None:
    model = compose_qwen3_5_rwkv7(tiny_config(), lambda index, _kind: index in {0, 2})
    input_ids = torch.tensor([[1, 5, 8, 13]])
    loss = model(input_ids, labels=input_ids, use_cache=False).loss
    loss.backward()
    generated = model.generate(input_ids[:, :2], max_new_tokens=2)
    with tempfile.TemporaryDirectory() as checkpoint:
        model.save_pretrained(checkpoint)
        restored = AutoModelForCausalLM.from_pretrained(checkpoint)
    print({"loss": loss.item(), "generated_shape": tuple(generated.shape), "restored": type(restored).__name__})


if __name__ == "__main__":
    main()
