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
"""PyTorch RWKV-7 model backed exclusively by FlashRWKV2 operators."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...cache_utils import Cache, CacheLayerMixin, LinearAttentionLayer
from ...generation import GenerationMixin
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, logging
from .configuration_rwkv import RwkvConfig


logger = logging.get_logger(__name__)


_TRAINING_OPERATORS = (
    "pretrain_tmix_mix6_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_recurrent_bf16",
    "pretrain_tmix_lnx_rkvres_xg_bf16",
    "pretrain_cmix_bf16",
)

_STATEFUL_TRAINING_OPERATORS = (
    "statetune_tmix_mix6_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "statetune_recurrent_fp32io16",
    "pretrain_tmix_lnx_rkvres_xg_bf16",
    "statetune_cmix_bf16",
)

_INFERENCE_OPERATORS = (
    "infer_embedding_ln0_forward_varlen",
    "infer_tmix_mix6_forward_varlen",
    "infer_tmix_mix6_add_layer_norm_forward_varlen",
    "infer_tmix_linear_attention_c2c_forward_varlen",
    "infer_tmix_linear_ffn_key_forward_varlen",
    "infer_tmix_lowrank_in_forward_varlen",
    "infer_tmix_lowrank_wagv_in_forward_varlen",
    "infer_tmix_lowrank_out_forward_varlen",
    "infer_tmix_lowrank_vres_forward_varlen",
    "infer_tmix_kk_a_gate_forward_varlen",
    "infer_recurrent_fp32io16_forward_varlen",
    "infer_tmix_lnx_rkvres_xg_forward_varlen",
    "infer_cmix_mix_forward_varlen",
    "infer_cmix_add_layer_norm_mix_forward_varlen",
    "infer_cmix_sparse_down_relu_forward_varlen",
    "infer_cmix_sparse_forward_varlen",
    "infer_cmix_relu_square_forward_varlen",
    "infer_cmix_linear_ffn_down_forward_varlen",
    "infer_tmix_layer_norm_forward_varlen",
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_last_forward_varlen",
    "prepare_recurrent_metadata",
)


def _load_flash_rwkv2(mode: str, tensor: torch.Tensor | None = None):
    """Load the public FlashRWKV2 surface lazily and fail closed on contract drift."""
    required = {
        "training": _TRAINING_OPERATORS,
        "stateful training": _STATEFUL_TRAINING_OPERATORS,
        "inference": _INFERENCE_OPERATORS,
    }.get(mode)
    if required is None:
        raise ValueError(f"Unsupported FlashRWKV2 mode: {mode!r}.")
    try:
        module = importlib.import_module("flashrwkv2")
    except ImportError as error:
        raise RuntimeError(
            f"RWKV-7 {mode} requires `FlashRWKV2==0.1.0a6` and its public `flashrwkv2` root API; "
            f"import failed: {error}"
        ) from error
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        version = getattr(module, "__version__", "unknown")
        source = getattr(module, "__file__", "unknown")
        raise RuntimeError(
            f"RWKV-7 {mode} requires FlashRWKV2 public operators {missing}; "
            f"installed version={version}, source={source}."
        )
    if tensor is not None and (not tensor.is_cuda or tensor.device.type != "cuda"):
        raise RuntimeError(
            f"RWKV-7 {mode} has no product fallback and requires CUDA tensors; got "
            f"device={tensor.device}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}."
        )
    return module


def _infer_tmix_attention_linear(flashrwkv2, x: torch.Tensor, projection: nn.Module) -> torch.Tensor:
    """Run a TimeMix C2C projection while preserving active vanilla LoRA adapters."""

    get_base_layer = getattr(projection, "get_base_layer", None)
    base_layer = get_base_layer() if callable(get_base_layer) else projection
    if not isinstance(base_layer, nn.Linear):
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference requires TimeMix projections backed by torch.nn.Linear; "
            f"got {type(base_layer).__name__}."
        )
    if base_layer.bias is not None:
        raise RuntimeError("RWKV-7 FlashRWKV2 TimeMix C2C projections do not support a base bias.")

    base = flashrwkv2.infer_tmix_linear_attention_c2c_forward_varlen
    if base_layer is projection:
        return base(x, base_layer.weight.contiguous())

    required = (
        "lora_A",
        "lora_B",
        "lora_dropout",
        "scaling",
        "active_adapters",
        "disable_adapters",
        "merged",
    )
    missing = [name for name in required if not hasattr(projection, name)]
    if missing:
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference only supports vanilla PEFT LoRA wrappers around TimeMix projections; "
            f"{type(projection).__name__} is missing {missing}."
        )
    if getattr(projection, "fan_in_fan_out", False):
        raise RuntimeError("RWKV-7 FlashRWKV2 LoRA inference requires fan_in_fan_out=False.")

    if projection.disable_adapters:
        if projection.merged:
            unmerge = getattr(projection, "unmerge", None)
            if not callable(unmerge):
                raise RuntimeError("Disabled merged LoRA projection cannot be unmerged.")
            unmerge()
            base_layer = projection.get_base_layer()
        return base(x, base_layer.weight.contiguous())
    if projection.merged:
        return base(x, base_layer.weight.contiguous())

    active_adapters = projection.active_adapters
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]
    variants = getattr(projection, "lora_variant", {})
    use_dora = getattr(projection, "use_dora", {})
    active = []
    for adapter_name in active_adapters:
        if adapter_name not in projection.lora_A:
            continue
        if adapter_name in variants or use_dora.get(adapter_name, False):
            raise RuntimeError(
                "RWKV-7 FlashRWKV2 inference supports vanilla LoRA only; "
                f"adapter {adapter_name!r} uses a LoRA variant. Merge it before inference."
            )
        adapter_a = projection.lora_A[adapter_name]
        adapter_b = projection.lora_B[adapter_name]
        if adapter_b.bias is not None:
            raise RuntimeError(
                "RWKV-7 FlashRWKV2 inference does not support lora_bias; "
                f"adapter {adapter_name!r} must be merged before inference."
            )
        dropout = projection.lora_dropout[adapter_name]
        if projection.training and getattr(dropout, "p", 0.0) != 0.0:
            raise RuntimeError("Unmerged LoRA dropout is only supported in eval mode.")
        active.append((adapter_a, adapter_b, float(projection.scaling[adapter_name])))

    if not active:
        return base(x, base_layer.weight.contiguous())
    if len(active) != 1:
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference supports exactly one active vanilla LoRA adapter; "
            "merge multiple adapters before inference."
        )
    adapter_a, adapter_b, scale = active[0]
    return base(
        x,
        base_layer.weight.contiguous(),
        lora_a=adapter_a.weight.contiguous(),
        lora_b=adapter_b.weight.contiguous(),
        lora_scale=scale,
    )


@dataclass
class RwkvTrainingState:
    """Batch-local recurrent state used by chunked RWKV-7 training.

    Shift states use BF16, matching FlashRWKV2's stateful mixing operators.
    WKV accumulation is always FP32.
    """

    time_mix_shift: torch.Tensor
    wkv: torch.Tensor
    channel_mix_shift: torch.Tensor

    @classmethod
    def zeros(
        cls,
        config: RwkvConfig,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RwkvTrainingState:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if dtype != torch.bfloat16:
            raise TypeError(f"RWKV-7 training shift state must be bfloat16, got {dtype}.")
        shift_shape = (config.num_hidden_layers, batch_size, config.hidden_size)
        wkv_shape = (
            config.num_hidden_layers,
            batch_size,
            config.num_attention_heads,
            config.head_size,
            config.head_size,
        )
        return cls(
            time_mix_shift=torch.zeros(shift_shape, device=device, dtype=dtype),
            wkv=torch.zeros(wkv_shape, device=device, dtype=torch.float32),
            channel_mix_shift=torch.zeros(shift_shape, device=device, dtype=dtype),
        )

    def validate(
        self,
        config: RwkvConfig,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> RwkvTrainingState:
        expected_device = torch.device(device)
        expected_shift_shape = (config.num_hidden_layers, batch_size, config.hidden_size)
        expected_wkv_shape = (
            config.num_hidden_layers,
            batch_size,
            config.num_attention_heads,
            config.head_size,
            config.head_size,
        )
        fields = (
            ("time_mix_shift", self.time_mix_shift, expected_shift_shape, dtype),
            ("wkv", self.wkv, expected_wkv_shape, torch.float32),
            ("channel_mix_shift", self.channel_mix_shift, expected_shift_shape, dtype),
        )
        for name, value, shape, expected_dtype in fields:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"training_state.{name} must be a torch.Tensor.")
            if tuple(value.shape) != shape:
                raise ValueError(f"training_state.{name} must have shape {shape}, got {tuple(value.shape)}.")
            if value.device != expected_device:
                raise ValueError(f"training_state.{name} must be on {expected_device}, got {value.device}.")
            if value.dtype != expected_dtype:
                raise TypeError(f"training_state.{name} must have dtype {expected_dtype}, got {value.dtype}.")
        return self

    def clone(self) -> RwkvTrainingState:
        return type(self)(*(value.clone() for value in self.tensors()))

    def detach(self) -> RwkvTrainingState:
        return type(self)(*(value.detach() for value in self.tensors()))

    def clone_detach(self) -> RwkvTrainingState:
        return type(self)(*(value.clone().detach() for value in self.tensors()))

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.time_mix_shift, self.wkv, self.channel_mix_shift

    def reset_(
        self,
        batch_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        *,
        time_mix: bool = True,
        wkv: bool = True,
        channel_mix: bool = True,
    ) -> RwkvTrainingState:
        """Zero selected batch rows in place, or all rows when indices are omitted."""
        if not any((time_mix, wkv, channel_mix)):
            return self
        selected = (time_mix, wkv, channel_mix)
        with torch.no_grad():
            for enabled, value in zip(selected, self.tensors(), strict=True):
                if not enabled:
                    continue
                if batch_indices is None:
                    value.zero_()
                else:
                    indices = torch.as_tensor(batch_indices, device=value.device, dtype=torch.long)
                    value.index_fill_(1, indices, 0)
        return self

    def reset(
        self,
        batch_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        *,
        time_mix: bool = True,
        wkv: bool = True,
        channel_mix: bool = True,
    ) -> RwkvTrainingState:
        """Return a clone with selected batch rows reset to zero."""
        return self.clone().reset_(
            batch_indices,
            time_mix=time_mix,
            wkv=wkv,
            channel_mix=channel_mix,
        )


@dataclass
class RwkvModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor | None = None
    training_state: RwkvTrainingState | None = None
    past_key_values: RwkvCache | None = None


@dataclass
class RwkvCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    training_state: RwkvTrainingState | None = None
    past_key_values: RwkvCache | None = None


class RwkvDynamicCacheLayer(LinearAttentionLayer, CacheLayerMixin):
    """Linear-attention cache layer with RWKV's two shift states and sequence offset."""

    is_sliding = False

    def __init__(self, number_of_states: int = 2):
        CacheLayerMixin.__init__(self)
        LinearAttentionLayer.__init__(self, number_of_states=number_of_states)
        self.cumulative_length = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        raise RuntimeError("RWKV-7 updates shift and recurrent states through their dedicated cache methods.")

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    @property
    def batch_size(self) -> int:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                return self.conv_states[state_idx].shape[0]
            if self.is_recurrent_states_initialized[state_idx]:
                return self.recurrent_states[state_idx].shape[0]
        return -1

    def mark_updated(self, sequence_length: int) -> None:
        self.cumulative_length += sequence_length
        self.has_previous_state[0] = True
        self.has_previous_state[1] = True

    def batch_repeat_interleave(self, repeats: int) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                self.conv_states[state_idx] = self.conv_states[state_idx].repeat_interleave(repeats, dim=0)
            if self.is_recurrent_states_initialized[state_idx]:
                self.recurrent_states[state_idx] = self.recurrent_states[state_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for state_idx in range(self.number_of_states):
            if self.is_conv_states_initialized[state_idx]:
                self.conv_states[state_idx] = self.conv_states[state_idx].index_select(0, indices.to(self.device))
            if self.is_recurrent_states_initialized[state_idx]:
                self.recurrent_states[state_idx] = self.recurrent_states[state_idx].index_select(
                    0, indices.to(self.device)
                )

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0


class RwkvCache(Cache):
    """Standard Transformers cache containing RWKV-7 shift and FP32 WKV states."""

    def __init__(self, config: RwkvConfig):
        super().__init__(
            layers=[RwkvDynamicCacheLayer(config.number_of_conv_states) for _ in range(config.num_hidden_layers)]
        )
        self._rwkv_config = config
        self._rwkv_metadata_key = None
        self._rwkv_metadata = None

    def recurrent_metadata(self, flashrwkv2, batch_size: int, sequence_length: int, device: torch.device):
        key = (batch_size, sequence_length, device.type, device.index)
        if self._rwkv_metadata_key != key:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * sequence_length,
                sequence_length,
                dtype=torch.int32,
                device=device,
            )
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            ticket = flashrwkv2.prepare_recurrent_metadata(
                cu_seqlens,
                state_indices,
                total_tokens=batch_size * sequence_length,
                state_pool_size=batch_size,
                max_seqlen=sequence_length,
            )
            self._rwkv_metadata_key = key
            self._rwkv_metadata = (cu_seqlens, state_indices, ticket)
        return self._rwkv_metadata

    def reset(self) -> None:
        """Reset recurrent states and discard the stream-bound FlashRWKV2 metadata ticket."""
        super().reset()
        self._rwkv_metadata_key = None
        self._rwkv_metadata = None


def _cache_states(
    cache: RwkvCache,
    layer_idx: int,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, _, hidden_size = hidden_states.shape
    layer = cache.layers[layer_idx]
    if not isinstance(layer, RwkvDynamicCacheLayer):
        raise TypeError(f"RWKV-7 expected RwkvDynamicCacheLayer at index {layer_idx}, got {type(layer).__name__}.")
    for state_idx in (0, 1):
        if not layer.is_conv_states_initialized[state_idx]:
            seed = hidden_states.new_zeros(batch_size, hidden_size, 1)
            layer.lazy_initialization(conv_states=seed, state_idx=state_idx, conv_kernel_size=1)
    if not layer.is_recurrent_states_initialized[0]:
        config = cache._rwkv_config
        state = torch.zeros(
            batch_size,
            config.num_attention_heads,
            config.head_size,
            config.head_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        layer.lazy_initialization(recurrent_states=state, state_idx=0)
    return (
        layer.conv_states[0].squeeze(-1),
        layer.recurrent_states[0],
        layer.conv_states[1].squeeze(-1),
    )


class RwkvEmbedding(nn.Embedding):
    """Embedding carrying train_temp's SmallInitEmb contract."""

    def reset_parameters(self) -> None:
        init.uniform_(self.weight, -1e-4, 1e-4)


class RwkvLMHead(nn.Linear):
    """Untied LM head carrying train_temp's vocabulary-dependent orthogonal initialization."""

    def __init__(self, config: RwkvConfig):
        self.rwkv_hidden_size = config.hidden_size
        self.rwkv_vocab_size = config.vocab_size
        super().__init__(config.hidden_size, config.vocab_size, bias=False)

    def reset_parameters(self) -> None:
        gain = 0.5 * math.sqrt(self.rwkv_vocab_size / self.rwkv_hidden_size)
        if self.rwkv_vocab_size <= self.rwkv_hidden_size:
            gain = 0.5
        init.orthogonal_(self.weight, gain=gain)


class RwkvTimeMix(nn.Module):
    """Canonical RWKV-7 TimeMix component using FlashRWKV2's public training and inference APIs."""

    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        channels = config.hidden_size
        heads = config.num_attention_heads

        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, name, nn.Parameter(torch.empty(1, 1, channels)))
        self.w0 = nn.Parameter(torch.empty(1, 1, channels))
        self.w1 = nn.Parameter(torch.empty(channels, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.empty(config.decay_low_rank_dim, channels))
        self.a0 = nn.Parameter(torch.empty(1, 1, channels))
        self.a1 = nn.Parameter(torch.empty(channels, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.empty(config.a_low_rank_dim, channels))
        # Canonical checkpoints retain the value-residual parameters in layer 0
        # even though its forward path establishes `v_first` instead of using them.
        self.v0 = nn.Parameter(torch.empty(1, 1, channels))
        self.v1 = nn.Parameter(torch.empty(channels, config.v_low_rank_dim))
        self.v2 = nn.Parameter(torch.empty(config.v_low_rank_dim, channels))
        self.g1 = nn.Parameter(torch.empty(channels, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.empty(config.gate_low_rank_dim, channels))
        self.k_k = nn.Parameter(torch.empty(1, 1, channels))
        self.k_a = nn.Parameter(torch.empty(1, 1, channels))
        self.r_k = nn.Parameter(torch.empty(heads, config.head_size))
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.ln_x = nn.GroupNorm(heads, channels, eps=config.group_norm_epsilon)
        self.register_buffer("_zero_residual", None, persistent=False)
        for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"):
            self.register_buffer(f"_{name}_original", None, persistent=False)

    def prepare_for_inference(self) -> None:
        """Prepare the original low-rank layouts used by Albatross's shape-specific dispatch table."""
        for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"):
            parameter = getattr(self, name)
            if parameter.dtype != torch.float16 or not parameter.is_cuda:
                raise RuntimeError(
                    "RWKV-7 Albatross low-rank layouts require CUDA float16 parameters; "
                    f"got {name} with dtype={parameter.dtype}, device={parameter.device}."
                )
            setattr(self, f"_{name}_original", parameter.T.contiguous())
        self._zero_residual = torch.zeros(1, self.config.hidden_size, dtype=torch.float16, device=self.w1.device)

    def _inference_low_rank_layouts(self) -> tuple[torch.Tensor, ...]:
        names = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
        layouts = tuple(getattr(self, f"_{name}_original") for name in names)
        if any(layout is None for layout in layouts):
            raise RuntimeError(
                "RWKV-7 Albatross low-rank layouts are not prepared; call `model.prepare_for_inference()` "
                "after loading or modifying weights."
            )
        return layouts

    def reset_parameters(self) -> None:
        """Initialize parameters with the canonical train_temp layer schedule.

        This is the final effective state produced by the constructor and ``generate_init_weight()`` in
        RWKV-LM commit e6f74b63a06e08606d130043599d218209628bad.
        """
        channels = self.config.hidden_size
        ratio_0_to_1 = self.layer_idx / max(self.config.num_hidden_layers - 1, 1)
        ratio_1_to_almost0 = 1.0 - self.layer_idx / self.config.num_hidden_layers
        ddd = torch.arange(channels, dtype=torch.float32).view(1, 1, -1) / channels
        exponents = {"r": 0.2, "w": 0.9, "k": 0.7, "v": 0.7, "a": 0.9, "g": 0.2}
        with torch.no_grad():
            for name, exponent in exponents.items():
                init.copy_(getattr(self, f"x_{name}"), 1.0 - ddd.pow(exponent * ratio_1_to_almost0))
            linear = torch.arange(channels, dtype=torch.float32) / max(channels - 1, 1) - 0.5
            head_index = torch.arange(channels, dtype=torch.float32) % self.config.head_size
            zigzag = (head_index - (self.config.head_size - 1) / 2) / ((self.config.head_size - 1) / 2)
            zigzag = zigzag * zigzag.abs()
            decay = -6 + 6 * (torch.arange(channels, dtype=torch.float32) / max(channels - 1, 1)).pow(
                1 + ratio_0_to_1**0.3
            )
            init.copy_(self.w0, (decay + 0.5 + zigzag * 2.5).view(1, 1, -1))
            init.copy_(self.a0, (-0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, -1))
            init.copy_(self.v0, (0.73 - linear * 0.4).view(1, 1, -1))
            init.copy_(self.k_k, (0.71 - linear * 0.1).view(1, 1, -1))
            init.constant_(self.k_a, 1.02)
            init.constant_(self.r_k, -0.04)
            for name in ("w1", "a1", "v1", "g1"):
                init.zeros_(getattr(self, name))
            for name in ("w2", "a2", "v2", "g2"):
                parameter = getattr(self, name)
                gain = (
                    math.sqrt(parameter.shape[0] / parameter.shape[1])
                    if parameter.shape[0] > parameter.shape[1]
                    else 1
                )
                init.orthogonal_(parameter, gain=gain * 0.1)
            init.orthogonal_(self.receptance.weight, gain=1.0)
            init.orthogonal_(self.key.weight, gain=0.1)
            init.orthogonal_(self.value.weight, gain=1.0)
            init.zeros_(self.output.weight)
            layer_scale = (self.layer_idx + 1) / self.config.num_hidden_layers
            init.constant_(self.ln_x.weight, layer_scale**0.7)
            init.zeros_(self.ln_x.bias)

    def _training_forward(self, hidden_states: torch.Tensor, v_first: torch.Tensor | None):
        flash = _load_flash_rwkv2("training", hidden_states)
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                f"RWKV-7 training requires contiguous CUDA bfloat16 [B,T,C], got dtype={hidden_states.dtype}, "
                f"shape={tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[1] % 16:
            raise RuntimeError(
                f"RWKV-7 train_temp requires sequence length divisible by 16, got T={hidden_states.shape[1]}."
            )
        x = hidden_states.contiguous()
        xr, xw, xk, xv, xa, xg = flash.pretrain_tmix_mix6_bf16(
            x,
            self.x_r.reshape(-1).contiguous(),
            self.x_w.reshape(-1).contiguous(),
            self.x_k.reshape(-1).contiguous(),
            self.x_v.reshape(-1).contiguous(),
            self.x_a.reshape(-1).contiguous(),
            self.x_g.reshape(-1).contiguous(),
        )
        receptance = self.receptance(xr)
        decay_delta = torch.tanh(xw @ self.w1) @ self.w2
        decay_logits = self.w0 + decay_delta
        key = self.key(xk)
        value = self.value(xv)
        if self.layer_idx == 0:
            v_first = value
        else:
            if v_first is None:
                raise ValueError("`v_first` must be supplied to RWKV-7 TimeMix layers after layer 0.")
            value_delta = (xv @ self.v1) @ self.v2
            value = flash.pretrain_tmix_vres_gate_bf16(
                value.contiguous(),
                v_first.contiguous(),
                self.v0.reshape(-1).contiguous(),
                value_delta.contiguous(),
            )
        learning_rate_delta = (xa @ self.a1) @ self.a2
        learning_rate = flash.pretrain_tmix_a_gate_bf16(
            self.a0.reshape(-1).contiguous(), learning_rate_delta.contiguous()
        )
        gate = (torch.sigmoid(xg @ self.g1) @ self.g2).contiguous()
        key, recurrent_a, recurrent_b = flash.pretrain_tmix_kk_pre_bf16(
            key.contiguous(),
            self.k_k.reshape(-1).contiguous(),
            learning_rate.contiguous(),
            self.k_a.reshape(-1).contiguous(),
        )
        output = flash.pretrain_recurrent_bf16(
            receptance.contiguous(),
            decay_logits.contiguous(),
            key,
            value.contiguous(),
            recurrent_a,
            recurrent_b,
        )
        output = flash.pretrain_tmix_lnx_rkvres_xg_bf16(
            output,
            receptance.contiguous(),
            key,
            value.contiguous(),
            self.r_k.contiguous(),
            self.ln_x.weight.contiguous(),
            self.ln_x.bias.contiguous(),
            gate,
        )
        return self.output(output), v_first

    def _stateful_training_forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        shift_state: torch.Tensor,
        wkv_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flash = _load_flash_rwkv2("stateful training", hidden_states)
        if hidden_states.dtype != torch.bfloat16:
            raise RuntimeError(
                f"RWKV-7 stateful training requires CUDA bfloat16 activations; got dtype={hidden_states.dtype}."
            )
        batch_size, sequence_length, channels = hidden_states.shape
        if sequence_length <= 0:
            raise ValueError("RWKV-7 stateful training requires at least one token per chunk.")
        x = hidden_states.contiguous()
        xr, xw, xk, xv, xa, xg, next_shift_state = flash.statetune_tmix_mix6_bf16(
            x,
            shift_state.contiguous(),
            self.x_r.reshape(-1).contiguous(),
            self.x_w.reshape(-1).contiguous(),
            self.x_k.reshape(-1).contiguous(),
            self.x_v.reshape(-1).contiguous(),
            self.x_a.reshape(-1).contiguous(),
            self.x_g.reshape(-1).contiguous(),
        )
        receptance = self.receptance(xr)
        decay_logits = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        key = self.key(xk)
        value = self.value(xv)
        if self.layer_idx == 0:
            v_first = value
        else:
            if v_first is None:
                raise ValueError("`v_first` must be supplied to RWKV-7 TimeMix layers after layer 0.")
            value_delta = (xv @ self.v1) @ self.v2
            value = flash.pretrain_tmix_vres_gate_bf16(
                value.contiguous(),
                v_first.contiguous(),
                self.v0.reshape(-1).contiguous(),
                value_delta.contiguous(),
            )
        learning_rate_delta = (xa @ self.a1) @ self.a2
        learning_rate = flash.pretrain_tmix_a_gate_bf16(
            self.a0.reshape(-1).contiguous(), learning_rate_delta.contiguous()
        )
        gate = (torch.sigmoid(xg @ self.g1) @ self.g2).contiguous()
        heads = self.config.num_attention_heads
        head_size = self.config.head_size
        key, recurrent_a, recurrent_b = flash.pretrain_tmix_kk_pre_bf16(
            key.contiguous(),
            self.k_k.reshape(-1).contiguous(),
            learning_rate.contiguous(),
            self.k_a.reshape(-1).contiguous(),
        )

        packed_shape = (batch_size * sequence_length, heads, head_size)
        starts = torch.arange(
            0,
            batch_size * sequence_length,
            sequence_length,
            device=x.device,
            dtype=torch.int32,
        )
        ends = starts + sequence_length
        sequence_chunk_offsets = torch.arange(batch_size + 1, device=x.device, dtype=torch.int32)
        recurrent_output, next_wkv_state, _, _ = flash.statetune_recurrent_fp32io16(
            wkv_state.contiguous(),
            sequence_chunk_offsets,
            starts,
            ends,
            receptance.view(packed_shape).contiguous(),
            decay_logits.view(packed_shape).contiguous(),
            key.view(packed_shape).contiguous(),
            value.view(packed_shape).contiguous(),
            recurrent_a.view(packed_shape).contiguous(),
            recurrent_b.view(packed_shape).contiguous(),
        )
        output = flash.pretrain_tmix_lnx_rkvres_xg_bf16(
            recurrent_output.view(batch_size, sequence_length, channels).contiguous(),
            receptance.contiguous(),
            key,
            value.contiguous(),
            self.r_k.contiguous(),
            self.ln_x.weight.contiguous(),
            self.ln_x.bias.contiguous(),
            gate,
        )
        return self.output(output), v_first, next_shift_state, next_wkv_state

    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache,
        layer_norm: nn.LayerNorm | None = None,
        residual: torch.Tensor | None = None,
    ):
        flash = _load_flash_rwkv2("inference", hidden_states)
        if hidden_states.dtype != torch.float16:
            raise RuntimeError(
                f"RWKV-7 Albatross inference requires float16 hidden states; got dtype={hidden_states.dtype}. "
                "Call `model.prepare_for_inference()` after loading weights."
            )
        batch_size, sequence_length, channels = hidden_states.shape
        att_shift, wkv_state, _ = _cache_states(past_key_values, self.layer_idx, hidden_states)
        cu_seqlens, state_indices, ticket = past_key_values.recurrent_metadata(
            flash, batch_size, sequence_length, hidden_states.device
        )
        packed = hidden_states.reshape(-1, channels).contiguous()
        mix_parameters = (
            self.x_r.reshape(-1).contiguous(),
            self.x_w.reshape(-1).contiguous(),
            self.x_k.reshape(-1).contiguous(),
            self.x_v.reshape(-1).contiguous(),
            self.x_a.reshape(-1).contiguous(),
            self.x_g.reshape(-1).contiguous(),
        )
        if layer_norm is None:
            xr, xw, xk, xv, xa, xg = flash.infer_tmix_mix6_forward_varlen(
                packed,
                *mix_parameters,
                shift_state_pool=att_shift,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                max_seqlen=sequence_length,
                validated_metadata=ticket,
            )
        else:
            if self._zero_residual is None:
                raise RuntimeError("RWKV-7 Albatross fused TMix layout is not prepared.")
            residual = self._zero_residual if residual is None else residual.reshape(-1, channels).contiguous()
            summed, xr, xw, xk, xv, xa, xg = flash.infer_tmix_mix6_add_layer_norm_forward_varlen(
                packed,
                residual,
                layer_norm.weight.contiguous(),
                layer_norm.bias.contiguous(),
                *mix_parameters,
                shift_state_pool=att_shift,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                max_seqlen=sequence_length,
                eps=layer_norm.eps,
                validated_metadata=ticket,
            )
        receptance = _infer_tmix_attention_linear(flash, xr, self.receptance)
        key = _infer_tmix_attention_linear(flash, xk, self.key)
        value = _infer_tmix_attention_linear(flash, xv, self.value)
        w1, w2, a1, a2, v1, v2, g1, g2 = self._inference_low_rank_layouts()
        if self.layer_idx == 0:
            v_first = value
            wr, ar, gr = flash.infer_tmix_lowrank_in_forward_varlen(
                xw,
                xa,
                xg,
                w1,
                a1,
                g1,
                w1_runtime=self.w1.contiguous(),
                a1_runtime=self.a1.contiguous(),
                g1_runtime=self.g1.contiguous(),
            )
            decay_delta, learning_rate_delta, gate = flash.infer_tmix_lowrank_out_forward_varlen(
                wr,
                ar,
                gr,
                w2,
                a2,
                g2,
                w2_runtime=self.w2.contiguous(),
                a2_runtime=self.a2.contiguous(),
                g2_runtime=self.g2.contiguous(),
            )
        else:
            if v_first is None:
                raise ValueError("`v_first` must be supplied to RWKV-7 TimeMix layers after layer 0.")
            wr, ar, gr, vr = flash.infer_tmix_lowrank_wagv_in_forward_varlen(
                xw,
                xa,
                xg,
                xv,
                w1,
                a1,
                g1,
                v1,
                w1_runtime=self.w1.contiguous(),
                a1_runtime=self.a1.contiguous(),
                g1_runtime=self.g1.contiguous(),
                v1_runtime=self.v1.contiguous(),
            )
            decay_delta, learning_rate_delta, gate, value = flash.infer_tmix_lowrank_vres_forward_varlen(
                wr,
                ar,
                gr,
                vr,
                w2,
                a2,
                g2,
                v2,
                value,
                v_first,
                self.v0.reshape(-1).contiguous(),
                w2_runtime=self.w2.contiguous(),
                a2_runtime=self.a2.contiguous(),
                g2_runtime=self.g2.contiguous(),
                v2_runtime=self.v2.contiguous(),
            )
        key, recurrent_a, recurrent_b = flash.infer_tmix_kk_a_gate_forward_varlen(
            key,
            self.k_k.reshape(-1).contiguous(),
            self.a0.reshape(-1).contiguous(),
            learning_rate_delta,
            self.k_a.reshape(-1).contiguous(),
            batch_size=batch_size,
            max_seqlen=sequence_length,
        )
        heads = self.config.num_attention_heads
        head_size = self.config.head_size
        output = flash.infer_recurrent_fp32io16_forward_varlen(
            receptance.view(-1, heads, head_size).contiguous(),
            decay_delta.view(-1, heads, head_size).contiguous(),
            key.view(-1, heads, head_size).contiguous(),
            value.view(-1, heads, head_size).contiguous(),
            recurrent_a.view(-1, heads, head_size).contiguous(),
            recurrent_b.view(-1, heads, head_size).contiguous(),
            state_pool=wkv_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            decay_bias=self.w0.view(heads, head_size).contiguous(),
            max_seqlen=sequence_length,
            validated_metadata=ticket,
        ).view(-1, channels)
        output = flash.infer_tmix_lnx_rkvres_xg_forward_varlen(
            output,
            receptance,
            key,
            value,
            self.r_k.reshape(-1).contiguous(),
            self.ln_x.weight.contiguous(),
            self.ln_x.bias.contiguous(),
            gate,
            batch_size=batch_size,
            max_seqlen=sequence_length,
        )
        output = _infer_tmix_attention_linear(flash, output, self.output).view(batch_size, sequence_length, channels)
        if layer_norm is not None:
            return summed.view_as(hidden_states), output, v_first
        return output, v_first

    def inference_forward_with_layer_norm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        layer_norm: nn.LayerNorm,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.shape[:2] != (1, 1) or hidden_states.shape[2] != 4096:
            raise ValueError("The fused Albatross TMix LayerNorm path requires shape [1,1,4096].")
        return self._inference_forward(
            hidden_states, v_first, past_key_values, layer_norm=layer_norm, residual=residual
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None = None,
        past_key_values: RwkvCache | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        training_shift_state: torch.Tensor | None = None,
        training_wkv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("RWKV-7 training and the initial equal-length inference path require an all-ones mask.")
        if self.training:
            if past_key_values is not None:
                raise ValueError("Canonical train_temp pretraining does not accept recurrent inference state.")
            if (training_shift_state is None) != (training_wkv_state is None):
                raise ValueError("Stateful RWKV-7 training requires both shift and WKV state.")
            if training_shift_state is not None:
                return self._stateful_training_forward(
                    hidden_states, v_first, training_shift_state, training_wkv_state
                )
            return self._training_forward(hidden_states, v_first)
        if past_key_values is None:
            raise ValueError(
                "RWKV-7 inference requires an RwkvCache, including when the caller discards the final cache."
            )
        return self._inference_forward(hidden_states, v_first, past_key_values)


class RwkvChannelMix(nn.Module):
    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.register_buffer("_value_runtime", None, persistent=False)

    def prepare_for_inference(self) -> None:
        if self._value_runtime is not None and not self.value.weight.is_cuda:
            return
        if self.value.weight.dtype != torch.float16 or not self.value.weight.is_cuda:
            raise RuntimeError(
                "RWKV-7 Albatross ChannelMix layout requires a CUDA float16 value weight; "
                f"got dtype={self.value.weight.dtype}, device={self.value.weight.device}."
            )
        self._value_runtime = self.value.weight.T.contiguous()
        # Albatross replaces the canonical FFN-down layout during inference. Keep the serializable parameter on CPU
        # instead of retaining a second 4 GiB GPU copy for a 7.2B model; the non-persistent runtime layout is the only
        # one consumed after this explicit inference preparation step.
        self.value.weight.data = self.value.weight.data.cpu()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            channels = self.config.hidden_size
            ratio_1_to_almost0 = 1.0 - self.layer_idx / self.config.num_hidden_layers
            ddd = torch.arange(channels, dtype=torch.float32, device=self.x_k.device).view(1, 1, -1) / channels
            init.copy_(self.x_k, 1.0 - ddd.pow(ratio_1_to_almost0**4))
            init.orthogonal_(self.key.weight, gain=1.0)
            init.zeros_(self.value.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: RwkvCache | None = None,
        attention_mask: torch.Tensor | None = None,
        training_shift_state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("RWKV-7 ChannelMix currently requires an all-ones mask.")
        if self.training:
            if training_shift_state is not None:
                if hidden_states.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"RWKV-7 stateful ChannelMix requires bfloat16 activations; got {hidden_states.dtype}."
                    )
                flash = _load_flash_rwkv2("stateful training", hidden_states)
                return flash.statetune_cmix_bf16(
                    hidden_states.contiguous(),
                    training_shift_state.contiguous(),
                    self.x_k.reshape(-1).contiguous(),
                    self.key.weight.contiguous(),
                    self.value.weight.contiguous(),
                )
            flash = _load_flash_rwkv2("training", hidden_states)
            return flash.pretrain_cmix_bf16(
                hidden_states.contiguous(),
                self.x_k.reshape(-1).contiguous(),
                self.key.weight.contiguous(),
                self.value.weight.contiguous(),
            )
        if past_key_values is None:
            raise ValueError("RWKV-7 inference requires an RwkvCache.")
        flash = _load_flash_rwkv2("inference", hidden_states)
        if self._value_runtime is None:
            raise RuntimeError(
                "RWKV-7 Albatross ChannelMix layout is not prepared; call `model.prepare_for_inference()` "
                "after loading or modifying weights."
            )
        batch_size, sequence_length, channels = hidden_states.shape
        _, _, ffn_shift = _cache_states(past_key_values, self.layer_idx, hidden_states)
        cu_seqlens, state_indices, ticket = past_key_values.recurrent_metadata(
            flash, batch_size, sequence_length, hidden_states.device
        )
        packed = hidden_states.reshape(-1, channels).contiguous()
        if channels == 4096 and packed.shape[0] <= 19:
            return flash.infer_cmix_sparse_forward_varlen(
                packed,
                self.x_k.reshape(-1).contiguous(),
                self.key.weight.contiguous(),
                self._value_runtime,
                shift_state_pool=ffn_shift,
                cu_seqlens=cu_seqlens,
                state_indices=state_indices,
                max_seqlen=sequence_length,
                validated_metadata=ticket,
                deterministic=torch.are_deterministic_algorithms_enabled(),
            ).view(batch_size, sequence_length, channels)
        mixed = flash.infer_cmix_mix_forward_varlen(
            packed,
            self.x_k.reshape(-1).contiguous(),
            shift_state_pool=ffn_shift,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=sequence_length,
            validated_metadata=ticket,
        )
        key = flash.infer_tmix_linear_ffn_key_forward_varlen(mixed, self.key.weight.contiguous())
        key = flash.infer_cmix_relu_square_forward_varlen(key)
        output = flash.infer_cmix_linear_ffn_down_forward_varlen(key, self._value_runtime)
        return output.view(batch_size, sequence_length, channels)

    def inference_forward_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        past_key_values: RwkvCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Albatross's fused B1T1 residual/LayerNorm/ChannelMix path."""
        flash = _load_flash_rwkv2("inference", hidden_states)
        if self._value_runtime is None:
            raise RuntimeError(
                "RWKV-7 Albatross ChannelMix layout is not prepared; call `model.prepare_for_inference()` "
                "after loading or modifying weights."
            )
        batch_size, sequence_length, channels = hidden_states.shape
        if sequence_length != 1 or channels != 4096:
            raise ValueError("The fused Albatross ChannelMix residual path requires sequence_length=1 and C=4096.")
        _, _, ffn_shift = _cache_states(past_key_values, self.layer_idx, hidden_states)
        cu_seqlens, state_indices, ticket = past_key_values.recurrent_metadata(
            flash, batch_size, sequence_length, hidden_states.device
        )
        summed, mixed = flash.infer_cmix_add_layer_norm_mix_forward_varlen(
            hidden_states.reshape(-1, channels).contiguous(),
            residual.reshape(-1, channels).contiguous(),
            layer_norm.weight.contiguous(),
            layer_norm.bias.contiguous(),
            self.x_k.reshape(-1).contiguous(),
            shift_state_pool=ffn_shift,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=sequence_length,
            eps=layer_norm.eps,
            validated_metadata=ticket,
        )
        key = flash.infer_tmix_linear_ffn_key_forward_varlen(mixed, self.key.weight.contiguous())
        output = flash.infer_cmix_sparse_down_relu_forward_varlen(
            key,
            self._value_runtime,
            batch_size=batch_size,
            max_seqlen=sequence_length,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )
        return summed.view_as(hidden_states), output.view_as(hidden_states)


class RwkvBlock(nn.Module):
    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.att = RwkvTimeMix(config, layer_idx)
        self.ffn = RwkvChannelMix(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache | None,
        attention_mask: torch.Tensor | None,
        use_cache: bool,
        training_state: RwkvTrainingState | None = None,
    ) -> tuple[torch.Tensor, ...]:
        att_result = self.att(
            self.ln1(hidden_states),
            v_first=v_first,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            training_shift_state=None if training_state is None else training_state.time_mix_shift[self.layer_idx],
            training_wkv_state=None if training_state is None else training_state.wkv[self.layer_idx],
        )
        if training_state is None:
            output, v_first = att_result
        else:
            output, v_first, next_att_shift, next_wkv = att_result
        hidden_states = hidden_states + output
        ffn_result = self.ffn(
            self.ln2(hidden_states),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            training_shift_state=None if training_state is None else training_state.channel_mix_shift[self.layer_idx],
        )
        if training_state is None:
            hidden_states = hidden_states + ffn_result
        else:
            ffn_output, next_ffn_shift = ffn_result
            hidden_states = hidden_states + ffn_output
        if past_key_values is not None:
            layer = past_key_values.layers[self.layer_idx]
            if isinstance(layer, RwkvDynamicCacheLayer):
                layer.mark_updated(hidden_states.shape[1])
        if training_state is None:
            return hidden_states, v_first
        return hidden_states, v_first, next_att_shift, next_wkv, next_ffn_shift

    def inference_forward_with_residual(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one Albatross B1T1 block while carrying the ChannelMix residual into the next block."""
        hidden_states, output, v_first = self.att.inference_forward_with_layer_norm(
            hidden_states, residual, self.ln1, v_first, past_key_values
        )
        hidden_states, residual = self.ffn.inference_forward_with_residual(
            hidden_states, output, self.ln2, past_key_values
        )
        layer = past_key_values.layers[self.layer_idx]
        if isinstance(layer, RwkvDynamicCacheLayer):
            layer.mark_updated(hidden_states.shape[1])
        return hidden_states, residual, v_first


@auto_docstring
class RwkvPreTrainedModel(PreTrainedModel):
    config_class = RwkvConfig
    base_model_prefix = "model"
    _no_split_modules = ["RwkvBlock"]
    _is_stateful = True
    supports_gradient_checkpointing = False

    # trf-ignore: TRF018
    def _init_weights(self, module):
        # These owning modules preserve Transformers' per-module loading markers, so from_pretrained never
        # reinitializes a complete model after loading a checkpoint.
        if isinstance(module, RwkvEmbedding | RwkvLMHead | RwkvTimeMix | RwkvChannelMix):
            module.reset_parameters()
        elif isinstance(module, nn.LayerNorm):
            module.reset_parameters()


@auto_docstring
class RwkvModel(RwkvPreTrainedModel):
    def __init__(self, config: RwkvConfig):
        if config.architecture_version != "rwkv7":
            raise ValueError(
                f"RwkvModel only supports `architecture_version='rwkv7'`, got {config.architecture_version!r}."
            )
        super().__init__(config)
        self.emb = RwkvEmbedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([RwkvBlock(config, index) for index in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def reset_parameters(self) -> None:
        """Apply the final canonical train_temp initialization in model order."""
        with torch.no_grad():
            init.uniform_(self.emb.weight, -1e-4, 1e-4)
            for block in self.blocks:
                for layer_norm in (getattr(block, "ln0", None), block.ln1, block.ln2):
                    if layer_norm is not None:
                        init.ones_(layer_norm.weight)
                        init.zeros_(layer_norm.bias)
                block.att.reset_parameters()
                block.ffn.reset_parameters()
            init.ones_(self.ln_out.weight)
            init.zeros_(self.ln_out.bias)

    def get_input_embeddings(self):
        return self.emb

    def set_input_embeddings(self, value):
        self.emb = value

    def _new_cache(self) -> RwkvCache:
        cache = RwkvCache(self.config)
        cache._rwkv_config = self.config
        return cache

    def prepare_for_inference(self):
        """Convert weights to Albatross's mixed BF16-embedding/FP16-runtime layout."""
        self.to(dtype=torch.float16)
        self.emb.to(dtype=torch.bfloat16)
        if not self.config.embedding_layer_norm_fused:
            self.blocks[0].ln0.to(dtype=torch.bfloat16)
        for block in self.blocks:
            block.att.prepare_for_inference()
            block.ffn.prepare_for_inference()
        return self

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: RwkvCache | None = None,
        training_state: RwkvTrainingState | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> RwkvModelOutput | BaseModelOutputWithPast | tuple:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if past_key_values is not None and not isinstance(past_key_values, RwkvCache):
            raise TypeError(f"RWKV-7 requires `RwkvCache`, got {type(past_key_values).__name__}.")
        use_cache = self.config.use_cache if use_cache is None else use_cache
        return_dict = self.config.return_dict if return_dict is None else return_dict

        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)
        if self.training:
            if past_key_values is not None:
                raise ValueError("Canonical RWKV-7 pretraining does not accept recurrent inference state.")
            if training_state is not None:
                training_state.validate(
                    self.config,
                    batch_size=inputs_embeds.shape[0],
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
            hidden_states = inputs_embeds
            if not self.config.embedding_layer_norm_fused:
                hidden_states = self.blocks[0].ln0(hidden_states)
            cache = None
        else:
            if training_state is not None:
                raise ValueError("`training_state` is only valid while the RWKV-7 model is in training mode.")
            cache = past_key_values if past_key_values is not None else self._new_cache()
            if inputs_embeds.dtype != torch.bfloat16:
                raise RuntimeError(
                    f"RWKV-7 Albatross embedding path requires bfloat16 embeddings, got {inputs_embeds.dtype}; "
                    "call `model.prepare_for_inference()`."
                )
            if self.config.embedding_layer_norm_fused:
                hidden_states = inputs_embeds.to(torch.float16)
            else:
                flash = _load_flash_rwkv2("inference", inputs_embeds)
                hidden_states = flash.infer_embedding_ln0_forward_varlen(
                    inputs_embeds.reshape(-1, self.config.hidden_size).contiguous(),
                    self.blocks[0].ln0.weight.contiguous(),
                    self.blocks[0].ln0.bias.contiguous(),
                    eps=self.config.layer_norm_epsilon,
                ).view_as(inputs_embeds)

        v_first = None
        next_att_shifts: list[torch.Tensor] = []
        next_wkv_states: list[torch.Tensor] = []
        next_ffn_shifts: list[torch.Tensor] = []
        fused_decode = not self.training and hidden_states.shape == (1, 1, 4096) and cache is not None
        if fused_decode:
            residual = None
            for block in self.blocks:
                hidden_states, residual, v_first = block.inference_forward_with_residual(
                    hidden_states, residual, v_first, cache
                )
            hidden_states = hidden_states + residual
        else:
            for block in self.blocks:
                block_result = block(
                    hidden_states,
                    v_first,
                    cache,
                    attention_mask,
                    use_cache,
                    training_state,
                )
                if training_state is None:
                    hidden_states, v_first = block_result
                else:
                    hidden_states, v_first, next_att_shift, next_wkv, next_ffn_shift = block_result
                    next_att_shifts.append(next_att_shift)
                    next_wkv_states.append(next_wkv)
                    next_ffn_shifts.append(next_ffn_shift)

        if self.training:
            hidden_states = self.ln_out(hidden_states)
        else:
            flash = _load_flash_rwkv2("inference", hidden_states)
            batch_size, sequence_length, channels = hidden_states.shape
            hidden_states = flash.infer_tmix_layer_norm_forward_varlen(
                hidden_states.reshape(-1, channels).contiguous(),
                self.ln_out.weight.contiguous(),
                self.ln_out.bias.contiguous(),
                eps=self.config.layer_norm_epsilon,
            ).view(batch_size, sequence_length, channels)

        final_cache = cache if use_cache else None
        next_training_state = None
        if training_state is not None:
            next_training_state = RwkvTrainingState(
                time_mix_shift=torch.stack(next_att_shifts),
                wkv=torch.stack(next_wkv_states),
                channel_mix_shift=torch.stack(next_ffn_shifts),
            )
        if not return_dict:
            if next_training_state is not None:
                return hidden_states, next_training_state
            return hidden_states, final_cache
        if next_training_state is not None:
            return RwkvModelOutput(last_hidden_state=hidden_states, training_state=next_training_state)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=final_cache)


@auto_docstring
class RwkvForCausalLM(RwkvPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}

    def __init__(self, config: RwkvConfig):
        super().__init__(config)
        self.model = RwkvModel(config)
        self.head = RwkvLMHead(config)
        self.post_init()

    def reset_head_parameters(self) -> None:
        """Apply train_temp's vocabulary-dependent LM-head initialization."""
        self.head.reset_parameters()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, value):
        self.head = value

    def prepare_for_inference(self):
        self.model.prepare_for_inference()
        self.head.to(dtype=torch.float16)
        return self

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        **kwargs,
    ):
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": self.config.use_cache if use_cache is None else use_cache,
            "logits_to_keep": kwargs.get("logits_to_keep", 1),
        }

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: RwkvCache | None = None,
        training_state: RwkvTrainingState | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> RwkvCausalLMOutput | CausalLMOutputWithPast | tuple:
        return_dict = self.config.return_dict if return_dict is None else return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            training_state=training_state,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if labels is not None and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            raise ValueError("RWKV-7 training with `labels` requires `logits_to_keep=0` for the shifted loss.")
        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError(f"`logits_to_keep` must be non-negative, got {logits_to_keep}.")
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        selected_hidden_states = hidden_states[:, slice_indices, :]
        if self.training:
            logits = self.head(selected_hidden_states)
        else:
            flash = _load_flash_rwkv2("inference", hidden_states)
            batch_size, sequence_length, channels = hidden_states.shape
            if isinstance(logits_to_keep, int) and logits_to_keep == 1:
                logits = flash.infer_head_linear_last_forward_varlen(
                    selected_hidden_states.reshape(batch_size, channels).contiguous(),
                    self.head.weight.contiguous(),
                    tokens_count=sequence_length,
                ).view(batch_size, 1, self.config.vocab_size)
            else:
                selected_length = selected_hidden_states.shape[1]
                logits = flash.infer_head_linear_all_forward_varlen(
                    selected_hidden_states.reshape(-1, channels).contiguous(), self.head.weight.contiguous()
                ).view(batch_size, selected_length, self.config.vocab_size)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if not return_dict:
            return tuple(
                value
                for value in (loss, logits, getattr(outputs, "training_state", None), outputs.past_key_values)
                if value is not None
            )
        if getattr(outputs, "training_state", None) is not None:
            return RwkvCausalLMOutput(
                loss=loss,
                logits=logits,
                training_state=outputs.training_state,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


__all__ = [
    "RwkvCache",
    "RwkvCausalLMOutput",
    "RwkvForCausalLM",
    "RwkvModel",
    "RwkvModelOutput",
    "RwkvPreTrainedModel",
    "RwkvTimeMix",
    "RwkvTrainingState",
]
