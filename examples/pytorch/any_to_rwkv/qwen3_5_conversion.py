#!/usr/bin/env python
"""Sketch an independent Any-to-RWKV model assembled from Qwen3.5-MoE components.

This example deliberately does not register an AutoClass or write an ``auto_map``.
Qwen3.5-MoE is the source of the embedding, MoE, and LM-head parameters; RWKV-7
is the lineage of the recurrent token mixer. The converted model has its own
configuration and model identity.

The public recurrent FLA entry point used below is part of the pending
``fla-rwkv`` runtime contract. Until that exact dependency revision is pinned,
the example fails closed instead of falling back to a chunk or reference kernel.
"""

import importlib

import torch
from torch import nn
from torch.nn import functional as F

from transformers import PretrainedConfig, PreTrainedModel, Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig
from transformers.modeling_outputs import CausalLMOutput


RWKV7_HEAD_SIZE_BY_SOURCE_LAYER = {"linear_attention": 128, "full_attention": 256}


def _load_public_recurrent_rwkv7():
    rwkv7 = importlib.import_module("fla.ops.rwkv7")
    recurrent_rwkv7 = getattr(rwkv7, "recurrent_rwkv7", None)
    get_last_provider = getattr(rwkv7, "get_last_rwkv7_provider", None)
    if not callable(recurrent_rwkv7) or not callable(get_last_provider):
        raise RuntimeError(
            "Any-to-RWKV requires the pinned fla-rwkv public recurrent_rwkv7 contract; "
            "chunk and reference fallbacks are disabled."
        )
    return recurrent_rwkv7, get_last_provider


class AnyToRwkvConvertedConfig(PretrainedConfig):
    """Independent converted-model config; source layer types only select mixer geometry."""

    model_type = "any_to_rwkv_converted"

    def __init__(
        self,
        vocab_size=64,
        hidden_size=256,
        source_architecture="qwen3_5_moe_text",
        source_layer_types=("linear_attention", "full_attention"),
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        unknown = sorted(set(source_layer_types) - RWKV7_HEAD_SIZE_BY_SOURCE_LAYER.keys())
        if unknown:
            raise ValueError(f"Unsupported Qwen3.5 source layer types: {unknown}.")
        head_sizes = [RWKV7_HEAD_SIZE_BY_SOURCE_LAYER[layer_type] for layer_type in source_layer_types]
        if any(hidden_size % head_size for head_size in head_sizes):
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by every recurrent head size: {head_sizes}."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.source_architecture = source_architecture
        self.source_layer_types = list(source_layer_types)
        self.head_sizes = head_sizes
        self.num_hidden_layers = len(source_layer_types)
        self.layer_norm_eps = layer_norm_eps


class AnyToRwkvRecurrentMixer(nn.Module):
    def __init__(self, hidden_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        recurrent_rwkv7, get_last_provider = _load_public_recurrent_rwkv7()
        batch_size, sequence_length, hidden_size = hidden_states.shape
        inputs = [
            projection(hidden_states).view(batch_size, sequence_length, self.num_heads, self.head_size)
            for projection in (self.r, self.w, self.k, self.v, self.a, self.b)
        ]
        initial_state = hidden_states.new_zeros(
            batch_size,
            self.num_heads,
            self.head_size,
            self.head_size,
            dtype=torch.float32,
        )
        result = recurrent_rwkv7(
            *inputs,
            initial_state=initial_state,
            output_final_state=True,
            mode="fp32io16",
        )
        if get_last_provider() != "flash_rwkv":
            raise RuntimeError("The public recurrent RWKV-7 call did not select FlashRWKV.")
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError("The public recurrent RWKV-7 call must return (output, final_state).")
        output, _ = result
        return self.output(output.reshape(batch_size, sequence_length, hidden_size))


class AnyToRwkvDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx, source_moe):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.recurrent_mixer = AnyToRwkvRecurrentMixer(config.hidden_size, config.head_sizes[layer_idx])
        self.post_mixer_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.moe = source_moe

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.recurrent_mixer(self.input_layernorm(hidden_states))
        return hidden_states + self.moe(self.post_mixer_layernorm(hidden_states))


class AnyToRwkvConvertedForCausalLM(PreTrainedModel):
    """Independent model that retains source Qwen parameter objects by identity."""

    config_class = AnyToRwkvConvertedConfig
    base_model_prefix = "any_to_rwkv"

    def __init__(self, config, source_model):
        super().__init__(config)
        source_layers = source_model.model.layers
        if len(source_layers) != config.num_hidden_layers:
            raise ValueError("The source Qwen layer count must match the independent converted config.")
        if (
            source_model.config.hidden_size != config.hidden_size
            or source_model.config.vocab_size != config.vocab_size
        ):
            raise ValueError("The source Qwen embedding/hidden dimensions must match the converted config.")

        self.embed_tokens = source_model.model.embed_tokens
        self.layers = nn.ModuleList(
            [AnyToRwkvDecoderLayer(config, index, layer.mlp) for index, layer in enumerate(source_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = source_model.lm_head

    def forward(self, input_ids, labels=None):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
        return CausalLMOutput(loss=loss, logits=logits)


def tiny_qwen_source():
    config = Qwen3_5MoeTextConfig(
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
    return Qwen3_5MoeForCausalLM(config)


def main():
    source = tiny_qwen_source()
    config = AnyToRwkvConvertedConfig(
        vocab_size=source.config.vocab_size,
        hidden_size=source.config.hidden_size,
        source_architecture=source.config.model_type,
        source_layer_types=source.config.layer_types,
        bos_token_id=source.config.bos_token_id,
        eos_token_id=source.config.eos_token_id,
        pad_token_id=source.config.pad_token_id,
    )
    converted = AnyToRwkvConvertedForCausalLM(config, source)

    assert "qwen" not in converted.config.model_type
    assert converted.config.model_type != "rwkv7"
    assert converted.embed_tokens is source.model.embed_tokens
    assert converted.lm_head is source.lm_head
    assert all(target.moe is original.mlp for target, original in zip(converted.layers, source.model.layers))

    input_ids = torch.tensor([[1, 5, 8, 13]])
    loss = converted(input_ids, labels=input_ids).loss
    loss.backward()
    print({"architecture": converted.config.model_type, "loss": loss.item()})


if __name__ == "__main__":
    main()
