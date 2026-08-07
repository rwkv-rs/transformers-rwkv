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
class RwkvConfig(PreTrainedConfig):
    r"""
    architecture_version (`str`, *optional*, defaults to `"rwkv7"`):
        Architecture generation stored in converted checkpoints. Only RWKV-7 is accepted.
    context_length (`int`, *optional*, defaults to 4096):
        Maximum sequence length used during pretraining. Recurrent inference can continue beyond this length.
    head_size (`int`, *optional*, defaults to 64):
        Width of each RWKV-7 recurrent head. The canonical FlashRWKV2 training and inference paths use 64.
    group_norm_epsilon (`float`, *optional*, defaults to 0.00064):
        Epsilon used by the head-wise normalization in TimeMix.
    embedding_layer_norm_fused (`bool`, *optional*, defaults to `False`):
        Whether block 0's `ln0` has already been folded into the embedding table.
    wkv_state_dtype (`str`, *optional*, defaults to `"float32"`):
        Dtype used for the recurrent WKV matrix.
    decay_low_rank_dim (`int`, *optional*):
        Rank of the time-decay projection. Inferred from `hidden_size` when omitted.
    a_low_rank_dim (`int`, *optional*):
        Rank of the in-context learning-rate projection. Inferred when omitted.
    v_low_rank_dim (`int`, *optional*):
        Rank of the value-residual projection. Inferred when omitted.
    gate_low_rank_dim (`int`, *optional*):
        Rank of the output-gate projection. Inferred when omitted.
    number_of_conv_states (`int`, *optional*, defaults to 2):
        Number of token-shift states stored in each cache layer.
    """

    model_type = "rwkv"
    attribute_map = {"max_position_embeddings": "context_length"}

    architecture_version: str = "rwkv7"
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
    pad_token_id: int | None = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    embedding_layer_norm_fused: bool = False
    wkv_state_dtype: str = "float32"
    decay_low_rank_dim: int | None = None
    a_low_rank_dim: int | None = None
    v_low_rank_dim: int | None = None
    gate_low_rank_dim: int | None = None
    number_of_conv_states: int = 2

    def __post_init__(self, **kwargs):
        legacy = sorted({"attention_hidden_size", "rescale_every"}.intersection(kwargs))
        if legacy:
            names = ", ".join(f"`{name}`" for name in legacy)
            raise ValueError(
                f"Legacy RWKV-4 configuration fields are no longer supported: {names}. "
                "This `rwkv` implementation requires an RWKV-7 checkpoint."
            )
        if self.intermediate_size is None:
            self.intermediate_size = int((3.5 * self.hidden_size) // 32 * 32)
        if self.num_attention_heads is None and self.head_size > 0:
            self.num_attention_heads = self.hidden_size // self.head_size

        # Keep the exact train_temp formulas, including Python's round-to-even behavior. These defaults are only
        # used for a new architecture. Converted and pretrained configs serialize their checkpoint-derived ranks.
        default_decay_rank = max(32, int(round((2.5 * self.hidden_size**0.5) / 32) * 32))
        default_value_rank = max(32, int(round((1.7 * self.hidden_size**0.5) / 32) * 32))
        default_gate_rank = max(32, int(round((5.0 * self.hidden_size**0.5) / 32) * 32))
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
        """Validate the canonical RWKV-7/FlashRWKV2 architecture contract."""
        if self.architecture_version != "rwkv7":
            raise ValueError(f"`architecture_version` must be 'rwkv7', got {self.architecture_version!r}.")
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0 or self.intermediate_size <= 0:
            raise ValueError("RWKV-7 hidden, layer, and intermediate dimensions must be positive.")
        if self.head_size != 64:
            raise ValueError(
                f"This integration requires the canonical FlashRWKV2 `head_size=64`, got {self.head_size}."
            )
        if self.hidden_size % self.head_size:
            raise ValueError(
                f"`hidden_size` ({self.hidden_size}) must be divisible by `head_size` ({self.head_size})."
            )
        inferred_heads = self.hidden_size // self.head_size
        if self.num_attention_heads != inferred_heads:
            raise ValueError(
                f"`num_attention_heads` ({self.num_attention_heads}) must equal hidden_size // head_size "
                f"({inferred_heads})."
            )
        if self.context_length <= 0:
            raise ValueError(f"`context_length` must be positive, got {self.context_length}.")
        if self.layer_norm_epsilon <= 0 or self.group_norm_epsilon <= 0:
            raise ValueError("LayerNorm and GroupNorm epsilon values must be positive.")
        if self.wkv_state_dtype != "float32":
            raise ValueError("RWKV-7 requires `wkv_state_dtype='float32'`.")
        if self.number_of_conv_states != 2:
            raise ValueError("RWKV-7 requires two shift-state cache entries per layer.")
        for name in ("decay_low_rank_dim", "a_low_rank_dim", "v_low_rank_dim", "gate_low_rank_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"`{name}` must be positive, got {getattr(self, name)}.")


__all__ = ["RwkvConfig"]
