# Copyright 2026 The RWKV-7 and HuggingFace Inc. teams.
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
    head_dim (`int`, *optional*, defaults to 64):
        Width of one RWKV recurrent head.
    attention_hidden_size (`int`, *optional*):
        Width of TimeMix. Defaults to `hidden_size`.
    intermediate_size (`int`, *optional*):
        Width of ChannelMix. Defaults to `4 * hidden_size`.
    decay_low_rank_dim (`int`, *optional*, defaults to 64):
        Rank of the TimeMix decay projection.
    a_low_rank_dim (`int`, *optional*, defaults to 64):
        Rank of the TimeMix in-context learning-rate projection.
    gate_low_rank_dim (`int`, *optional*, defaults to 128):
        Rank of the TimeMix output gate projection.
    value_low_rank_dim (`int`, *optional*, defaults to 32):
        Rank of the residual value projection used after the first layer.
    deep_embedding_size (`int`, *optional*, defaults to 0):
        DeepEmbedding matrix side. Set to `0` for ordinary ChannelMix and to
        the checkpoint value (normally `32`) for DeepEmbedding checkpoints.
    wkv_mode (`str`, *optional*, defaults to `"fp32io16"`):
        Recurrent precision. `"fp32io16"` keeps the recurrent state and WKV
        accumulation in float32 while preserving the model activation dtype;
        `"fp16"` keeps the recurrent state in float16.
    """

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 65536
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    head_dim: int = 64
    attention_hidden_size: int | None = None
    intermediate_size: int | None = None
    decay_low_rank_dim: int = 64
    a_low_rank_dim: int = 64
    gate_low_rank_dim: int = 128
    value_low_rank_dim: int = 32
    deep_embedding_size: int = 0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    wkv_mode: str = "fp32io16"
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 0
    pad_token_id: int | None = 0
    tie_word_embeddings: bool = False
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        self.attention_hidden_size = self.attention_hidden_size or self.hidden_size
        self.intermediate_size = self.intermediate_size or 4 * self.hidden_size
        if self.attention_hidden_size % self.head_dim != 0:
            raise ValueError("attention_hidden_size must be divisible by head_dim")
        if self.wkv_mode not in {"fp32io16", "fp16"}:
            raise ValueError("wkv_mode must be either 'fp32io16' or 'fp16'")
        if self.deep_embedding_size < 0:
            raise ValueError("deep_embedding_size must be non-negative")
        self.num_attention_heads = self.attention_hidden_size // self.head_dim
        super().__post_init__(**kwargs)


__all__ = ["Rwkv7Config"]
