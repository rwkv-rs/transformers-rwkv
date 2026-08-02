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
    rescale_every (`int`, *optional*, defaults to 0):
        If positive, divide residual activations and corresponding output weights by two at this layer interval during
        inference. Original RWKV-7 checkpoints use zero because their weights do not contain this rescaling.
    embedding_layer_norm_fused (`bool`, *optional*, defaults to `False`):
        Whether the block-0 layer normalization has already been fused into the embedding table.
    wkv_backend (`str`, *optional*, defaults to `"auto"`):
        WKV execution backend selected by the model implementation. Supported values are `"auto"`, `"reference"`,
        and `"flash_rwkv"`. `"auto"` falls back to the reference implementation when unsupported; an explicit
        `"flash_rwkv"` request fails closed.
    wkv_state_dtype (`str`, *optional*, defaults to `"float32"`):
        Dtype used to store the recurrent WKV matrix.
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
    rescale_every: int = 0
    tie_word_embeddings: bool = False
    use_cache: bool = True
    embedding_layer_norm_fused: bool = False
    wkv_backend: str = "auto"
    wkv_state_dtype: str = "float32"

    def __post_init__(self, **kwargs):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.num_attention_heads is None and self.head_size > 0:
            self.num_attention_heads = self.hidden_size // self.head_size

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
        if self.rescale_every < 0:
            raise ValueError(f"`rescale_every` must be non-negative, got {self.rescale_every}.")
        if self.wkv_backend not in {"auto", "reference", "flash_rwkv"}:
            raise ValueError("`wkv_backend` must be one of 'auto', 'reference', or 'flash_rwkv'.")
        if self.wkv_state_dtype != "float32":
            raise ValueError("RWKV-7 requires `wkv_state_dtype='float32'`.")


__all__ = ["Rwkv7Config"]
