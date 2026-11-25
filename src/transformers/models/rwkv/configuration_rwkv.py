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
    intermediate_size (`int`, *optional*):
        Width of the RWKV-7 ChannelMix activation. Defaults to four times `hidden_size`.
    head_size (`int`, *optional*, defaults to 64):
        Width of each RWKV-7 recurrent head.
    group_norm_epsilon (`float`, *optional*, defaults to 0.00064):
        Epsilon used by the head-wise normalization in TimeMix.
    wkv_mode (`str`, *optional*, defaults to `"fp32io16"`):
        Inference WKV state precision: `"fp32io16"` keeps FP32 state and matches training; `"fp16"` rounds
        state to FP16 after each token. Both use BF16 token IO and FP32 arithmetic. Training always uses FP32 state.
    decay_low_rank_dim (`int`, *optional*):
        Rank of the time-decay projection. Inferred from `hidden_size` when omitted.
    a_low_rank_dim (`int`, *optional*):
        Rank of the in-context learning-rate projection. Inferred when omitted.
    v_low_rank_dim (`int`, *optional*):
        Rank of the value-residual projection. Inferred when omitted.
    gate_low_rank_dim (`int`, *optional*):
        Rank of the output-gate projection. Inferred when omitted.
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
    wkv_mode: str = "fp32io16"
    decay_low_rank_dim: int | None = None
    a_low_rank_dim: int | None = None
    v_low_rank_dim: int | None = None
    gate_low_rank_dim: int | None = None

    def __post_init__(self, **kwargs):
        if self.architecture_version != "rwkv7" or {"attention_hidden_size", "rescale_every"}.intersection(kwargs):
            raise ValueError("This `rwkv` implementation requires an RWKV-7 checkpoint.")
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size
        if self.head_size not in (64, 128, 256) or self.hidden_size % self.head_size:
            raise ValueError("`head_size` must be 64, 128, or 256 and divide `hidden_size`.")
        self.num_attention_heads = self.hidden_size // self.head_size

        decay_rank = max(32, int(round((2.5 * self.hidden_size**0.5) / 32) * 32))
        value_rank = max(32, int(round((1.7 * self.hidden_size**0.5) / 32) * 32))
        gate_rank = max(32, int(round((5.0 * self.hidden_size**0.5) / 32) * 32))
        if self.decay_low_rank_dim is None:
            self.decay_low_rank_dim = decay_rank
        if self.a_low_rank_dim is None:
            self.a_low_rank_dim = decay_rank
        if self.v_low_rank_dim is None:
            self.v_low_rank_dim = value_rank
        if self.gate_low_rank_dim is None:
            self.gate_low_rank_dim = gate_rank

        if self.wkv_mode not in ("fp32io16", "fp16"):
            raise ValueError(f"`wkv_mode` must be 'fp32io16' or 'fp16', got {self.wkv_mode!r}.")
        super().__post_init__(**kwargs)


__all__ = ["RwkvConfig"]
