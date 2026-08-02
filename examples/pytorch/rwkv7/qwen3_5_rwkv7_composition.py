#!/usr/bin/env python
"""Compose Qwen3.5-MoE components with RWKV-7 token mixers and LayerNorm."""

import tempfile

import torch

from transformers import AutoModelForCausalLM, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig


RWKV7_HEAD_SIZE_BY_LAYER_TYPE = {"linear_attention": 128, "full_attention": 256}


def compose_qwen3_5_rwkv7(
    config: Qwen3_5MoeTextConfig,
) -> Qwen3_5MoeForCausalLM:
    """Replace Qwen3.5-MoE token mixers while retaining its embedding, MoE, and LM head."""
    config_values = config.to_dict()
    config_values["layer_types"] = ["rwkv7"] * config.num_hidden_layers
    config_values["rwkv7_head_sizes"] = [RWKV7_HEAD_SIZE_BY_LAYER_TYPE[kind] for kind in config.layer_types]
    config_values["use_rwkv7_layer_norm"] = True
    return Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig.from_dict(config_values))


def tiny_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
        layer_types=["linear_attention", "full_attention", "linear_attention"],
        rwkv7_backend="reference",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def main() -> None:
    model = compose_qwen3_5_rwkv7(tiny_config())
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
