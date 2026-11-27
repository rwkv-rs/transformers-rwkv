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
"""RWKV-7S configuration."""

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BlinkDL/rwkv7-g1")
@strict
class Rwkv7sConfig(PreTrainedConfig):
    r"""
    architecture_version (`str`, *optional*, defaults to `"rwkv7s"`):
        Architecture generation stored in converted checkpoints. Only RWKV-7S is accepted.
    context_length (`int`, *optional*, defaults to 4096):
        Maximum sequence length used during pretraining. Recurrent inference can continue beyond this length.
    loop_layer_index_start (`int`, *optional*):
        Index of the first core layer. Defaults to 1 for models with at least four layers and 0 otherwise.
    loop_layer_index_end (`int`, *optional*):
        Index after the last core layer. Defaults to the largest index `<= num_hidden_layers - 1` yielding an even
        number of core layers from `loop_layer_index_start` (or 0 when `loop_layer_index_start == 0`).
    num_experts (`int`, *optional*, defaults to 8):
        Number of routed squared-ReLU MoE experts per layer.
    num_experts_per_tok (`int`, *optional*, defaults to 2):
        Number of routed experts activated per token.
    moe_intermediate_size (`int`, *optional*):
        Intermediate width of each routed squared-ReLU expert. Defaults to `hidden_size`.
    shared_expert_intermediate_size (`int`, *optional*):
        Intermediate width of the always-active squared-ReLU shared expert. Defaults to `hidden_size`.
    routed_scaling_factor (`float`, *optional*, defaults to 1.0):
        Scaling factor applied to the selected routed expert weights.
    norm_topk_prob (`bool`, *optional*, defaults to `True`):
        Whether to normalize the top-k sigmoid router weights so they sum to 1 before scaling.
    head_size (`int`, *optional*, defaults to 64):
        Width of each RWKV-7S recurrent head.
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
    train_autodeco (`bool`, *optional*, defaults to `False`):
        Whether `Rwkv7sForCausalLM.forward` computes the AutoDeco loss instead of the causal LM loss when `labels` are
        provided.
    deep_embed_dim (`int`, *optional*, defaults to 32):
        Low-rank projection dimension `r` for `rwkv7a` DeepEmbed (`s_emb` stores `r * r` values per token).
    rosa_dim (`int`, *optional*, defaults to 64):
        Low-rank embedding width for per-layer token-level ROSA (`rosa_emb` -> `rosa_proj`) injected in each FFN.
    mtp_depth (`int`, *optional*, defaults to 4):
        Number of looped LightMTP draft steps executed with the shared MTP block.
    mtp_intermediate_size (`int`, *optional*):
        Intermediate width of the LightMTP SwiGLU MLP. Defaults to `(hidden_size * 7) // 2`.
    mtp_loss_scaling_factor (`float`, *optional*, defaults to 0.0):
        Weight applied to the auxiliary MTP loss during joint LM training (`0.0` disables auxiliary MTP loss).
    train_mtp (`bool`, *optional*, defaults to `False`):
        Whether `Rwkv7sForCausalLM.forward` freezes the backbone and computes only the MTP loss when `labels` are
        provided.
    """

    model_type = "rwkv7s"
    attribute_map = {"max_position_embeddings": "context_length"}

    architecture_version: str = "rwkv7s"
    vocab_size: int = 65536
    context_length: int = 4096
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    loop_layer_index_start: int | None = None
    loop_layer_index_end: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
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
    train_autodeco: bool = False
    deep_embed_dim: int = 32
    rosa_dim: int = 64
    mtp_depth: int = 4
    mtp_intermediate_size: int | None = None
    mtp_loss_scaling_factor: float = 0.0
    train_mtp: bool = False

    def __post_init__(self, **kwargs):
        if self.architecture_version != "rwkv7s" or {"attention_hidden_size", "rescale_every"}.intersection(kwargs):
            raise ValueError("This `rwkv7s` implementation requires an RWKV-7S checkpoint.")
        if self.loop_layer_index_start is None:
            self.loop_layer_index_start = 1 if self.num_hidden_layers >= 4 else 0
        if self.loop_layer_index_end is None:
            if self.loop_layer_index_start == 0:
                self.loop_layer_index_end = 0
            else:
                end = self.num_hidden_layers - 1
                self.loop_layer_index_end = end - (end - self.loop_layer_index_start) % 2
        start, end = self.loop_layer_index_start, self.loop_layer_index_end
        if (start, end) != (0, 0) and (not 1 <= start < end <= self.num_hidden_layers - 1 or (end - start) % 2):
            raise ValueError(
                "`loop_layer_index_start` and `loop_layer_index_end` must both be 0 or define a positive, even "
                "number of core layers with at least one layer before and after them."
            )
        if self.num_experts < 1 or not 1 <= self.num_experts_per_tok <= self.num_experts:
            raise ValueError("`num_experts_per_tok` must satisfy 1 <= num_experts_per_tok <= num_experts.")
        if self.deep_embed_dim <= 0:
            raise ValueError("`deep_embed_dim` must be positive.")
        if self.rosa_dim <= 0:
            raise ValueError("`rosa_dim` must be positive.")
        if self.mtp_depth < 1:
            raise ValueError("`mtp_depth` must be >= 1.")
        if self.mtp_intermediate_size is None:
            self.mtp_intermediate_size = (self.hidden_size * 7) // 2
        if self.mtp_intermediate_size <= 0:
            raise ValueError("`mtp_intermediate_size` must be positive.")
        if self.mtp_loss_scaling_factor < 0:
            raise ValueError("`mtp_loss_scaling_factor` must be non-negative.")
        if self.train_autodeco and self.train_mtp:
            raise ValueError("`train_autodeco` and `train_mtp` cannot both be True.")
        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = self.hidden_size
        if self.shared_expert_intermediate_size is None:
            self.shared_expert_intermediate_size = self.hidden_size
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


__all__ = ["Rwkv7sConfig"]
