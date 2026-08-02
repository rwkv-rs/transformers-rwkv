# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public Qwen3.5-to-Any-to-RWKV composition and direct reload utilities."""

import torch
from torch import nn
from torch.nn import functional as F

from ...configuration_utils import PretrainedConfig
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ..qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from ..qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM
from .modeling_rwkv7 import _rwkv7_flash


ANY_TO_RWKV_HEAD_SIZE_BY_QWEN3_5_LAYER = {"linear_attention": 128, "full_attention": 256}


class AnyToRwkvConvertedConfig(PretrainedConfig):
    """Configuration for an independent model composed by the Any-to-RWKV conversion tool."""

    model_type = "any_to_rwkv_converted"

    def __init__(
        self,
        vocab_size=64,
        hidden_size=256,
        source_architecture="qwen3_5_moe_text",
        source_config=None,
        source_layer_types=("linear_attention", "full_attention"),
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        unknown_layer_types = sorted(set(source_layer_types).difference(ANY_TO_RWKV_HEAD_SIZE_BY_QWEN3_5_LAYER))
        if unknown_layer_types:
            raise ValueError(f"Unsupported Qwen3.5 source layer types: {unknown_layer_types}.")
        head_sizes = [ANY_TO_RWKV_HEAD_SIZE_BY_QWEN3_5_LAYER[layer_type] for layer_type in source_layer_types]
        if any(hidden_size % head_size for head_size in head_sizes):
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by every recurrent head size: {head_sizes}."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.source_architecture = source_architecture
        self.source_config = source_config
        self.source_layer_types = list(source_layer_types)
        self.head_sizes = head_sizes
        self.num_hidden_layers = len(source_layer_types)
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = False


class _AnyToRwkvRecurrentMixer(nn.Module):
    def __init__(self, hidden_size, head_size):
        super().__init__()
        self.head_size = head_size
        self.num_heads = hidden_size // head_size
        self.receptance = nn.Linear(hidden_size, hidden_size, bias=False)
        self.raw_decay = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        initial_state = hidden_states.new_zeros(
            batch_size,
            self.num_heads,
            self.head_size,
            self.head_size,
            dtype=torch.float32,
        )
        output, _ = _rwkv7_flash(
            self.receptance(hidden_states),
            self.raw_decay(hidden_states),
            self.key(hidden_states),
            self.value(hidden_states),
            self.a(hidden_states),
            self.b(hidden_states),
            initial_state,
            self.head_size,
        )
        return self.output(output)


class _AnyToRwkvDecoderLayer(nn.Module):
    def __init__(self, config, layer_index, source_moe):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.recurrent_mixer = _AnyToRwkvRecurrentMixer(config.hidden_size, config.head_sizes[layer_index])
        self.post_mixer_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.moe = source_moe

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.recurrent_mixer(self.input_layernorm(hidden_states))
        return hidden_states + self.moe(self.post_mixer_layernorm(hidden_states))


class AnyToRwkvConvertedForCausalLM(PreTrainedModel, GenerationMixin):
    """Independent Any-to-RWKV model that owns reused Qwen3.5 embedding, LM-head, and MoE parameters."""

    config_class = AnyToRwkvConvertedConfig
    base_model_prefix = "any_to_rwkv"
    _no_split_modules = ["_AnyToRwkvDecoderLayer"]

    def __init__(self, config, source_model=None):
        super().__init__(config)
        is_reload = source_model is None
        if is_reload:
            if config.source_config is None:
                raise ValueError("Reloading the converted model requires its saved Qwen3.5 source component config.")
            source_config = Qwen3_5MoeTextConfig.from_dict(config.source_config)
            source_model = Qwen3_5MoeForCausalLM(source_config)
        _validate_qwen3_5_source(source_model, config)

        self.embed_tokens = source_model.model.embed_tokens
        self.layers = nn.ModuleList(
            [
                _AnyToRwkvDecoderLayer(config, layer_index, source_layer.mlp)
                for layer_index, source_layer in enumerate(source_model.model.layers)
            ]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = source_model.lm_head
        if is_reload:
            self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, labels=None, attention_mask=None, return_dict=None, **kwargs):
        if input_ids is None:
            raise ValueError("AnyToRwkvConvertedForCausalLM requires input_ids.")
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("Any-to-RWKV conversion currently requires unpadded input sequences.")
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100)
        if return_dict is False:
            return tuple(value for value in (loss, logits) if value is not None)
        return CausalLMOutput(loss=loss, logits=logits)


def convert_qwen3_5_to_any_to_rwkv(source_model):
    """Compose an independent Any-to-RWKV model while reusing Qwen3.5 embedding, LM-head, and MoE parameters."""
    _validate_qwen3_5_source_identity(source_model)
    source_config = source_model.config
    converted_config = AnyToRwkvConvertedConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        source_architecture=source_config.model_type,
        source_config=source_config.to_dict(),
        source_layer_types=source_config.layer_types,
        bos_token_id=source_config.bos_token_id,
        eos_token_id=source_config.eos_token_id,
        pad_token_id=source_config.pad_token_id,
        tie_word_embeddings=source_config.tie_word_embeddings,
    )
    return AnyToRwkvConvertedForCausalLM(converted_config, source_model)


def _validate_qwen3_5_source(source_model, converted_config):
    _validate_qwen3_5_source_identity(source_model)
    if converted_config.source_architecture != source_model.config.model_type:
        raise ValueError("The source architecture must match the independent converted config.")
    if len(source_model.model.layers) != converted_config.num_hidden_layers:
        raise ValueError("The source Qwen3.5 layer count must match the independent converted config.")
    if (
        source_model.config.hidden_size != converted_config.hidden_size
        or source_model.config.vocab_size != converted_config.vocab_size
    ):
        raise ValueError("The source Qwen3.5 embedding and hidden dimensions must match the converted config.")
    if list(source_model.config.layer_types) != converted_config.source_layer_types:
        raise ValueError("The source Qwen3.5 layer types must match the converted recurrent mixer geometry.")


def _validate_qwen3_5_source_identity(source_model):
    if not isinstance(source_model, Qwen3_5MoeForCausalLM):
        raise TypeError("Any-to-RWKV Qwen3.5 conversion requires Qwen3_5MoeForCausalLM as its source model.")
    if source_model.config.model_type != "qwen3_5_moe_text":
        raise ValueError("Any-to-RWKV Qwen3.5 conversion requires a qwen3_5_moe_text source config.")


__all__ = [
    "ANY_TO_RWKV_HEAD_SIZE_BY_QWEN3_5_LAYER",
    "AnyToRwkvConvertedConfig",
    "AnyToRwkvConvertedForCausalLM",
    "convert_qwen3_5_to_any_to_rwkv",
]
