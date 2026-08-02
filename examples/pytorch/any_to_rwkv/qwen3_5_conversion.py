#!/usr/bin/env python
"""Compose an independent Any-to-RWKV model from Qwen3.5-MoE components."""

from tempfile import TemporaryDirectory

import torch

from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.models.rwkv7.convert_qwen3_5_to_any_to_rwkv import (
    AnyToRwkvConvertedForCausalLM,
    convert_qwen3_5_to_any_to_rwkv,
)


def tiny_qwen_source():
    return Qwen3_5MoeForCausalLM(
        Qwen3_5MoeTextConfig(
            vocab_size=64,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts=4,
            num_experts_per_tok=2,
            layer_types=["linear_attention", "full_attention"],
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )


def main():
    source = tiny_qwen_source()
    converted = convert_qwen3_5_to_any_to_rwkv(source)

    assert converted.config.model_type == "any_to_rwkv_converted"
    assert converted.embed_tokens is source.model.embed_tokens
    assert converted.lm_head is source.lm_head
    assert all(target.moe is original.mlp for target, original in zip(converted.layers, source.model.layers))
    assert [layer.recurrent_mixer.head_size for layer in converted.layers] == [128, 256]  # GDN, then GQA
    assert isinstance(converted.norm, torch.nn.LayerNorm)

    input_ids = torch.tensor([[1, 5, 8, 13]])
    loss = converted(input_ids, labels=input_ids).loss
    loss.backward()
    converted.eval()
    generated = converted.generate(input_ids, max_new_tokens=1, do_sample=False)

    with TemporaryDirectory() as output_dir:
        converted.save_pretrained(output_dir, safe_serialization=True)
        reloaded = AnyToRwkvConvertedForCausalLM.from_pretrained(output_dir)
        torch.testing.assert_close(reloaded(input_ids).logits, converted(input_ids).logits)

    print({"architecture": converted.config.model_type, "loss": loss.item(), "generated": generated.tolist()})


if __name__ == "__main__":
    main()
