# Copyright 2026 The HuggingFace Inc. team.
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
"""RWKV-7 configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BlinkDL/rwkv7-g1")
@strict
class Rwkv7Config(PreTrainedConfig):
    r"""
    context_length (`int`, *optional*, defaults to 4096):
        Maximum sequence length for a single forward pass. Recurrent decoding can continue beyond this length.
    head_size (`int`, *optional*, defaults to 64):
        Width of each RWKV-7 WKV head.
    group_norm_epsilon (`float`, *optional*, defaults to 0.00064):
        Epsilon used by the group normalization in the time-mix module.
    embedding_layer_norm_fused (`bool`, *optional*, defaults to `False`):
        Whether the block-0 layer normalization has already been fused into the embedding table.
    wkv_state_dtype (`str`, *optional*, defaults to `"float32"`):
        Dtype used to store the recurrent WKV matrix.
    decay_low_rank_dim (`int`, *optional*):
        Rank of the time-decay ``w1``/``w2`` projections.
    a_low_rank_dim (`int`, *optional*):
        Rank of the in-context learning-rate ``a1``/``a2`` projections.
    v_low_rank_dim (`int`, *optional*):
        Rank of the value-mixing ``v1``/``v2`` projections.
    gate_low_rank_dim (`int`, *optional*):
        Rank of the output-gate ``g1``/``g2`` projections.
    """

    model_type = "rwkv7"
    attribute_map = {"max_position_embeddings": "context_length"}

    vocab_size: int = 65536
    context_length: int = 4096
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    intermediate_size: int | None = None
    head_size: int = 64
    num_attention_heads: int | None = None
    layer_norm_epsilon: float = 1e-5
    group_norm_epsilon: float = 64e-5
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    embedding_layer_norm_fused: bool = False
    wkv_state_dtype: str = "float32"
    decay_low_rank_dim: int | None = None
    a_low_rank_dim: int | None = None
    v_low_rank_dim: int | None = None
    gate_low_rank_dim: int | None = None

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.num_attention_heads is None and self.head_size > 0:
            self.num_attention_heads = self.hidden_size // self.head_size
        default_decay_rank = max(32, round(2.5 * self.hidden_size**0.5 / 32) * 32)
        default_value_rank = max(32, round(1.7 * self.hidden_size**0.5 / 32) * 32)
        default_gate_rank = max(32, round(5.0 * self.hidden_size**0.5 / 32) * 32)
        if self.decay_low_rank_dim is None:
            self.decay_low_rank_dim = default_decay_rank
        if self.a_low_rank_dim is None:
            self.a_low_rank_dim = default_decay_rank
        if self.v_low_rank_dim is None:
            self.v_low_rank_dim = default_value_rank
        if self.gate_low_rank_dim is None:
            self.gate_low_rank_dim = default_gate_rank

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        """Part of `@strict`-powered validation. Validates the architecture of the config."""
        if self.hidden_size <= 0:
            raise ValueError(f"`hidden_size` must be positive, got {self.hidden_size}.")
        if self.num_hidden_layers <= 0:
            raise ValueError(f"`num_hidden_layers` must be positive, got {self.num_hidden_layers}.")
        if self.intermediate_size <= 0:
            raise ValueError(f"`intermediate_size` must be positive, got {self.intermediate_size}.")
        if self.head_size <= 0 or self.hidden_size % self.head_size != 0:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by a positive `head_size` ({self.head_size})."
            )

        inferred_num_heads = self.hidden_size // self.head_size
        if self.num_attention_heads != inferred_num_heads:
            raise ValueError(
                f"`num_attention_heads` ({self.num_attention_heads}) must equal hidden_size // head_size "
                f"({inferred_num_heads})."
            )
        if self.context_length <= 0:
            raise ValueError(f"`context_length` must be positive, got {self.context_length}.")
        if self.layer_norm_epsilon <= 0 or self.group_norm_epsilon <= 0:
            raise ValueError("Layer and group normalization epsilon values must be positive.")
        if self.wkv_state_dtype != "float32":
            raise ValueError("RWKV-7 requires `wkv_state_dtype='float32'`.")
        for name in ("decay_low_rank_dim", "a_low_rank_dim", "v_low_rank_dim", "gate_low_rank_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be positive, got {getattr(self, name)}.")


__all__ = ["Rwkv7Config"]
