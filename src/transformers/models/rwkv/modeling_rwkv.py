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

import math
from dataclasses import dataclass

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, LinearAttentionLayer
from ...generation import GenerationMixin
from ...integrations.flash_rwkv2 import load_flash_rwkv2 as _load_flash_rwkv2
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from .configuration_rwkv import RwkvConfig


_STATEFUL_BOUNDARY_CHUNK_LEN = 16


def _validate_rwkv_attention_mask(attention_mask: torch.Tensor | None, hidden_states: torch.Tensor) -> None:
    """Reject padding and ragged batches before they reach a FlashRWKV2 provider."""
    if attention_mask is None:
        return
    if attention_mask.ndim != 2:
        raise ValueError(
            f"RWKV-7 attention masks must be two-dimensional [batch, sequence], got {attention_mask.ndim} dimensions."
        )
    if attention_mask.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            "RWKV-7 attention-mask batch size must match the input batch size, got "
            f"{attention_mask.shape[0]} and {hidden_states.shape[0]}."
        )
    if attention_mask.shape[1] < hidden_states.shape[1]:
        raise ValueError(
            "RWKV-7 attention masks cannot be shorter than the current input, got "
            f"{attention_mask.shape[1]} and {hidden_states.shape[1]}."
        )
    if not torch.all(attention_mask == 1):
        raise ValueError(
            "RWKV-7 does not support padding or ragged batches; use an all-ones mask and bucket inputs by length."
        )


def _stateful_training_metadata(
    batch_size: int, sequence_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe canonical 16-token replay boundaries inside each logical sequence chunk."""

    chunks_per_sequence = (sequence_length + _STATEFUL_BOUNDARY_CHUNK_LEN - 1) // _STATEFUL_BOUNDARY_CHUNK_LEN
    sequence_chunk_offsets = torch.arange(
        0,
        (batch_size + 1) * chunks_per_sequence,
        chunks_per_sequence,
        device=device,
        dtype=torch.int32,
    )
    sequence_starts = (torch.arange(batch_size, device=device, dtype=torch.int32) * sequence_length).unsqueeze(1)
    within_sequence_starts = torch.arange(
        0,
        sequence_length,
        _STATEFUL_BOUNDARY_CHUNK_LEN,
        device=device,
        dtype=torch.int32,
    ).unsqueeze(0)
    chunk_token_starts = (sequence_starts + within_sequence_starts).flatten()
    sequence_ends = sequence_starts + sequence_length
    chunk_token_ends = torch.minimum(
        chunk_token_starts.view(batch_size, chunks_per_sequence) + _STATEFUL_BOUNDARY_CHUNK_LEN,
        sequence_ends,
    ).flatten()
    return sequence_chunk_offsets, chunk_token_starts, chunk_token_ends


def _infer_linear_attention_projection_spec(
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, float]:
    """Resolve one bias-free Linear and an optional active vanilla LoRA adapter."""

    get_base_layer = getattr(projection, "get_base_layer", None)
    base_layer = get_base_layer() if callable(get_base_layer) else projection
    if not isinstance(base_layer, nn.Linear):
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference requires projections backed by torch.nn.Linear; "
            f"got {type(base_layer).__name__}."
        )
    if base_layer.bias is not None:
        raise RuntimeError("RWKV-7 FlashRWKV2 inference projections do not support a base bias.")

    if base_layer is projection:
        return base_layer.weight.contiguous(), None, None, 1.0

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
            "RWKV-7 FlashRWKV2 inference only supports vanilla PEFT LoRA wrappers around linear-attention "
            "projections; "
            f"{type(projection).__name__} is missing {missing}."
        )
    if getattr(projection, "fan_in_fan_out", False):
        raise RuntimeError("RWKV-7 FlashRWKV2 LoRA inference requires fan_in_fan_out=False.")

    if projection.disable_adapters:
        if projection.merged:
            raise RuntimeError(
                "RWKV-7 inference does not mutate PEFT adapter state in forward. Unmerge the LoRA projection "
                "before disabling adapters."
            )
        return base_layer.weight.contiguous(), None, None, 1.0
    if projection.merged:
        return base_layer.weight.contiguous(), None, None, 1.0

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
        return base_layer.weight.contiguous(), None, None, 1.0
    if len(active) != 1:
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference supports exactly one active vanilla LoRA adapter; "
            "merge multiple adapters before inference."
        )
    adapter_a, adapter_b, scale = active[0]
    return (
        base_layer.weight.contiguous(),
        adapter_a.weight.contiguous(),
        adapter_b.weight.contiguous(),
        scale,
    )


@dataclass
class RwkvTrainingState:
    """Batch-local recurrent state used by chunked RWKV-7 training.

    Shift states use BF16, matching FlashRWKV2's stateful mixing operators.
    WKV accumulation is always FP32.
    """

    attention_shift: torch.Tensor
    recurrent_state: torch.Tensor
    mlp_shift: torch.Tensor

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
            attention_shift=torch.zeros(shift_shape, device=device, dtype=dtype),
            recurrent_state=torch.zeros(wkv_shape, device=device, dtype=torch.float32),
            mlp_shift=torch.zeros(shift_shape, device=device, dtype=dtype),
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
            ("attention_shift", self.attention_shift, expected_shift_shape, dtype),
            ("recurrent_state", self.recurrent_state, expected_wkv_shape, torch.float32),
            ("mlp_shift", self.mlp_shift, expected_shift_shape, dtype),
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
        return self.attention_shift, self.recurrent_state, self.mlp_shift

    def reset_(
        self,
        batch_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        *,
        attention: bool = True,
        recurrent: bool = True,
        mlp: bool = True,
    ) -> RwkvTrainingState:
        """Zero selected batch rows in place, or all rows when indices are omitted."""
        if not any((attention, recurrent, mlp)):
            return self
        selected = (attention, recurrent, mlp)
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
        attention: bool = True,
        recurrent: bool = True,
        mlp: bool = True,
    ) -> RwkvTrainingState:
        """Return a clone with selected batch rows reset to zero."""
        return self.clone().reset_(
            batch_indices,
            attention=attention,
            recurrent=recurrent,
            mlp=mlp,
        )


@auto_docstring
@dataclass
class RwkvModelOutput(ModelOutput):
    r"""
    training_state (`RwkvTrainingState`, *optional*, returned when `training_state` is provided):
        Recurrent state at the end of the input chunk. It can be passed to the next training forward to continue
        stateful training.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: RwkvCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    training_state: RwkvTrainingState | None = None


@auto_docstring
@dataclass
class RwkvCausalLMOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss for next-token prediction.
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head before SoftMax.
    training_state (`RwkvTrainingState`, *optional*, returned when `training_state` is provided):
        Recurrent state at the end of the input chunk. It can be passed to the next training forward to continue
        stateful training.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: RwkvCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    training_state: RwkvTrainingState | None = None


class RwkvDynamicCacheLayer(LinearAttentionLayer):
    """Linear-attention cache layer with RWKV's two shift states and sequence offset."""

    is_compileable = False
    is_sliding = False

    def __init__(self, number_of_states: int = 2):
        super().__init__(number_of_states=number_of_states)
        self.cumulative_length = 0

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return query_length, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def mark_updated(self, sequence_length: int) -> None:
        self.cumulative_length += sequence_length
        self.has_previous_state[0] = True
        self.has_previous_state[1] = True

    def reset(self) -> None:
        super().reset()
        self.cumulative_length = 0


class RwkvCache(Cache):
    """Standard Transformers cache containing RWKV-7 shift and FP32 WKV states."""

    def __init__(self, config):
        super().__init__(
            layers=[RwkvDynamicCacheLayer(config.number_of_conv_states) for _ in range(config.num_hidden_layers)]
        )
        self._rwkv_metadata_key = None
        self._rwkv_metadata = None

    def _clear_recurrent_metadata(self) -> None:
        self._rwkv_metadata_key = None
        self._rwkv_metadata = None

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the logical recurrent offset tracked by an RWKV layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        """Return RWKV's query-only mask size; recurrent history has no key/value axis."""
        if layer_idx >= len(self.layers):
            return query_length, 0
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def recurrent_metadata(self, flashrwkv2, batch_size: int, sequence_length: int, device: torch.device):
        stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
        key = (batch_size, sequence_length, device.type, device.index, stream)
        if self._rwkv_metadata_key != key:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * sequence_length,
                sequence_length,
                dtype=torch.int32,
                device=device,
            )
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            ticket = flashrwkv2.prepare_tmix_wkv7_recurrent_metadata(
                cu_seqlens,
                state_indices,
                total_tokens=batch_size * sequence_length,
                state_pool_size=batch_size,
                max_seqlen=sequence_length,
            )
            self._rwkv_metadata_key = key
            self._rwkv_metadata = (cu_seqlens, state_indices, ticket)
        return self._rwkv_metadata

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        super().reorder_cache(beam_idx)
        self._clear_recurrent_metadata()

    def batch_repeat_interleave(self, repeats: int) -> None:
        super().batch_repeat_interleave(repeats)
        self._clear_recurrent_metadata()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        super().batch_select_indices(indices)
        self._clear_recurrent_metadata()

    def reset(self) -> None:
        """Reset recurrent states and discard the stream-bound FlashRWKV2 metadata ticket."""
        super().reset()
        self._clear_recurrent_metadata()


def _cache_states(
    cache: RwkvCache,
    layer_idx: int,
    hidden_states: torch.Tensor,
    config,
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


class RwkvLinearAttention(nn.Module):
    """RWKV-7 linear attention using FlashRWKV2's public training and inference APIs."""

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
        self.r_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.o_proj = nn.Linear(channels, channels, bias=False)
        self.g_norm = nn.GroupNorm(heads, channels, eps=config.group_norm_epsilon)
        for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"):
            self.register_buffer(f"_{name}_original", None, persistent=False)
        self.register_load_state_dict_post_hook(self._clear_inference_layouts)

    def _clear_inference_layouts(self, *args) -> None:
        for name in ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"):
            setattr(self, f"_{name}_original", None)

    def _apply(self, fn, recurse=True):
        self._clear_inference_layouts()
        return super()._apply(fn, recurse=recurse)

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
            init.orthogonal_(self.r_proj.weight, gain=1.0)
            init.orthogonal_(self.k_proj.weight, gain=0.1)
            init.orthogonal_(self.v_proj.weight, gain=1.0)
            init.zeros_(self.o_proj.weight)
            layer_scale = (self.layer_idx + 1) / self.config.num_hidden_layers
            init.constant_(self.g_norm.weight, layer_scale**0.7)
            init.zeros_(self.g_norm.bias)

    def _training_projections(self, flash, mixed: tuple[torch.Tensor, ...], v_first: torch.Tensor | None):
        xr, xw, xk, xv, xa, xg = mixed
        receptance = self.r_proj(xr)
        decay_logits = self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        key = self.k_proj(xk)
        value = self.v_proj(xv)
        if self.layer_idx == 0:
            v_first = value
        else:
            if v_first is None:
                raise ValueError("`v_first` must be supplied to RWKV-7 linear-attention layers after layer 0.")
            value = flash.pretrain_tmix_vres_gate_bf16(
                value.contiguous(),
                v_first.contiguous(),
                self.v0.reshape(-1).contiguous(),
                ((xv @ self.v1) @ self.v2).contiguous(),
            )
        learning_rate = flash.pretrain_tmix_a_gate_bf16(
            self.a0.reshape(-1).contiguous(), ((xa @ self.a1) @ self.a2).contiguous()
        )
        gate = (torch.sigmoid(xg @ self.g1) @ self.g2).contiguous()
        key, recurrent_a, recurrent_b = flash.pretrain_tmix_kk_pre_bf16(
            key.contiguous(),
            self.k_k.reshape(-1).contiguous(),
            learning_rate.contiguous(),
            self.k_a.reshape(-1).contiguous(),
        )
        return receptance, decay_logits, key, value, v_first, gate, recurrent_a, recurrent_b

    def _finish_training_output(
        self,
        flash,
        recurrent_output: torch.Tensor,
        receptance: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        output = flash.pretrain_tmix_readout_bf16(
            recurrent_output,
            receptance.contiguous(),
            key,
            value.contiguous(),
            self.r_k.contiguous(),
            self.g_norm.weight.contiguous(),
            self.g_norm.bias.contiguous(),
            gate,
        )
        return self.o_proj(output)

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
        mixed = flash.pretrain_tmix_tokenshift_bf16(
            x,
            self.x_r.reshape(-1).contiguous(),
            self.x_w.reshape(-1).contiguous(),
            self.x_k.reshape(-1).contiguous(),
            self.x_v.reshape(-1).contiguous(),
            self.x_a.reshape(-1).contiguous(),
            self.x_g.reshape(-1).contiguous(),
        )
        receptance, decay_logits, key, value, v_first, gate, recurrent_a, recurrent_b = self._training_projections(
            flash, mixed, v_first
        )
        output = flash.pretrain_tmix_wkv7_recurrent_bf16(
            receptance.contiguous(),
            decay_logits.contiguous(),
            key,
            value.contiguous(),
            recurrent_a,
            recurrent_b,
        )
        return self._finish_training_output(flash, output, receptance, key, value, gate), v_first

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
        *mixed, next_shift_state = flash.statetune_tmix_tokenshift_bf16(
            x,
            shift_state.contiguous(),
            self.x_r.reshape(-1).contiguous(),
            self.x_w.reshape(-1).contiguous(),
            self.x_k.reshape(-1).contiguous(),
            self.x_v.reshape(-1).contiguous(),
            self.x_a.reshape(-1).contiguous(),
            self.x_g.reshape(-1).contiguous(),
        )
        receptance, decay_logits, key, value, v_first, gate, recurrent_a, recurrent_b = self._training_projections(
            flash, mixed, v_first
        )
        heads = self.config.num_attention_heads
        head_size = self.config.head_size

        packed_shape = (batch_size * sequence_length, heads, head_size)
        sequence_chunk_offsets, starts, ends = _stateful_training_metadata(batch_size, sequence_length, x.device)
        recurrent_output, next_wkv_state, _, _ = flash.statetune_tmix_wkv7_recurrent_fp32io16(
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
        output = self._finish_training_output(
            flash,
            recurrent_output.view(batch_size, sequence_length, channels).contiguous(),
            receptance,
            key,
            value,
            gate,
        )
        return output, v_first, next_shift_state, next_wkv_state

    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache,
    ):
        flash = _load_flash_rwkv2("inference", hidden_states)
        if hidden_states.dtype != torch.float16:
            raise RuntimeError(
                f"RWKV-7 Albatross inference requires float16 hidden states; got dtype={hidden_states.dtype}. "
                "Call `model.prepare_for_inference()` after loading weights."
            )
        batch_size, sequence_length, channels = hidden_states.shape
        att_shift, wkv_state, _ = _cache_states(past_key_values, self.layer_idx, hidden_states, self.config)
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
        residual = residual.reshape(-1, channels).contiguous()
        summed, xr, xw, xk, xv, xa, xg = flash.infer_tmix_postnorm_tokenshift_forward_varlen(
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
        receptance_weight, receptance_lora_a, receptance_lora_b, receptance_lora_scale = (
            _infer_linear_attention_projection_spec(self.r_proj)
        )
        key_weight, key_lora_a, key_lora_b, key_lora_scale = _infer_linear_attention_projection_spec(self.k_proj)
        value_weight, value_lora_a, value_lora_b, value_lora_scale = _infer_linear_attention_projection_spec(
            self.v_proj
        )
        w1, w2, a1, a2, v1, v2, g1, g2 = self._inference_low_rank_layouts()
        if self.layer_idx != 0 and v_first is None:
            raise ValueError("`v_first` must be supplied to RWKV-7 linear-attention layers after layer 0.")
        (
            receptance,
            decay_delta,
            key,
            value,
            recurrent_a,
            recurrent_b,
            gate,
            v_first,
        ) = flash.infer_tmix_wkv_prepare_forward_varlen(
            xr,
            xw,
            xk,
            xv,
            xa,
            xg,
            receptance_weight,
            key_weight,
            value_weight,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            self.v0.reshape(-1).contiguous(),
            self.k_k.reshape(-1).contiguous(),
            self.a0.reshape(-1).contiguous(),
            self.k_a.reshape(-1).contiguous(),
            v_first=None if self.layer_idx == 0 else v_first,
            w1_runtime=self.w1.contiguous(),
            a1_runtime=self.a1.contiguous(),
            g1_runtime=self.g1.contiguous(),
            v1_runtime=self.v1.contiguous(),
            w2_runtime=self.w2.contiguous(),
            a2_runtime=self.a2.contiguous(),
            g2_runtime=self.g2.contiguous(),
            v2_runtime=self.v2.contiguous(),
            receptance_lora_a=receptance_lora_a,
            receptance_lora_b=receptance_lora_b,
            receptance_lora_scale=receptance_lora_scale,
            key_lora_a=key_lora_a,
            key_lora_b=key_lora_b,
            key_lora_scale=key_lora_scale,
            value_lora_a=value_lora_a,
            value_lora_b=value_lora_b,
            value_lora_scale=value_lora_scale,
            head_size=self.config.head_size,
            batch_size=batch_size,
            max_seqlen=sequence_length,
        )
        heads = self.config.num_attention_heads
        head_size = self.config.head_size
        output = flash.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
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
        output_weight, output_lora_a, output_lora_b, output_lora_scale = _infer_linear_attention_projection_spec(
            self.o_proj
        )
        output = flash.infer_tmix_readout_forward_varlen(
            output,
            receptance,
            key,
            value,
            self.r_k.reshape(-1).contiguous(),
            self.g_norm.weight.contiguous(),
            self.g_norm.bias.contiguous(),
            gate,
            output_weight,
            output_lora_a=output_lora_a,
            output_lora_b=output_lora_b,
            output_lora_scale=output_lora_scale,
            head_size=self.config.head_size,
            batch_size=batch_size,
            max_seqlen=sequence_length,
        )
        output = output.view(batch_size, sequence_length, channels)
        return summed.view_as(hidden_states), output, v_first

    def inference_forward_with_postnorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._inference_forward(hidden_states, residual, layer_norm, v_first, past_key_values)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None = None,
        past_key_values: RwkvCache | None = None,
        attention_mask: torch.Tensor | None = None,
        training_shift_state: torch.Tensor | None = None,
        training_wkv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        _validate_rwkv_attention_mask(attention_mask, hidden_states)
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
        raise RuntimeError(
            "RWKV-7 inference must run linear attention through its owning decoder layer so FlashRWKV2 can fuse "
            "residual, "
            "LayerNorm and TokenShift."
        )


class RwkvMLP(nn.Module):
    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.x_k = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.register_buffer("_down_proj_runtime", None, persistent=False)
        self.register_load_state_dict_post_hook(self._clear_inference_layout)

    def _clear_inference_layout(self, *args) -> None:
        self._down_proj_runtime = None

    def _apply(self, fn, recurse=True):
        self._clear_inference_layout()
        return super()._apply(fn, recurse=recurse)

    def prepare_for_inference(self) -> None:
        runtime_device = self.up_proj.weight.device
        if runtime_device.type != "cuda" or self.up_proj.weight.dtype != torch.float16:
            raise RuntimeError(
                "RWKV-7 Albatross MLP layout requires CUDA float16 runtime weights; "
                f"got dtype={self.up_proj.weight.dtype}, device={runtime_device}."
            )
        self._down_proj_runtime = self.down_proj.weight.to(device=runtime_device, dtype=torch.float16).T.contiguous()
        # Albatross replaces the canonical FFN-down layout during inference. Keep the serializable parameter on CPU
        # instead of retaining a second 4 GiB GPU copy for a 7.2B model; the non-persistent runtime layout is the only
        # one consumed after this explicit inference preparation step.
        self.down_proj.weight.data = self.down_proj.weight.data.cpu()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            channels = self.config.hidden_size
            ratio_1_to_almost0 = 1.0 - self.layer_idx / self.config.num_hidden_layers
            ddd = torch.arange(channels, dtype=torch.float32, device=self.x_k.device).view(1, 1, -1) / channels
            init.copy_(self.x_k, 1.0 - ddd.pow(ratio_1_to_almost0**4))
            init.orthogonal_(self.up_proj.weight, gain=1.0)
            init.zeros_(self.down_proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: RwkvCache | None = None,
        attention_mask: torch.Tensor | None = None,
        training_shift_state: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _validate_rwkv_attention_mask(attention_mask, hidden_states)
        if self.training:
            if training_shift_state is not None:
                if hidden_states.dtype != torch.bfloat16:
                    raise RuntimeError(
                        f"RWKV-7 stateful MLP requires bfloat16 activations; got {hidden_states.dtype}."
                    )
                flash = _load_flash_rwkv2("stateful training", hidden_states)
                return flash.statetune_cmix_bf16(
                    hidden_states.contiguous(),
                    training_shift_state.contiguous(),
                    self.x_k.reshape(-1).contiguous(),
                    self.up_proj.weight.contiguous(),
                    self.down_proj.weight.contiguous(),
                )
            flash = _load_flash_rwkv2("training", hidden_states)
            return flash.pretrain_cmix_bf16(
                hidden_states.contiguous(),
                self.x_k.reshape(-1).contiguous(),
                self.up_proj.weight.contiguous(),
                self.down_proj.weight.contiguous(),
            )
        if past_key_values is None:
            raise ValueError("RWKV-7 inference requires an RwkvCache.")
        raise RuntimeError(
            "RWKV-7 inference must run the MLP through its owning decoder layer so FlashRWKV2 can fuse residual, "
            "LayerNorm, TokenShift and the complete FFN."
        )

    def inference_forward_with_postnorm(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        past_key_values: RwkvCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the complete FlashRWKV2 MLP inference island."""
        flash = _load_flash_rwkv2("inference", hidden_states)
        if self._down_proj_runtime is None:
            raise RuntimeError(
                "RWKV-7 Albatross MLP layout is not prepared; call `model.prepare_for_inference()` "
                "after loading or modifying weights."
            )
        batch_size, sequence_length, channels = hidden_states.shape
        _, _, ffn_shift = _cache_states(past_key_values, self.layer_idx, hidden_states, self.config)
        cu_seqlens, state_indices, ticket = past_key_values.recurrent_metadata(
            flash, batch_size, sequence_length, hidden_states.device
        )
        packed = hidden_states.reshape(-1, channels).contiguous()
        summed, output = flash.infer_cmix_forward_varlen(
            packed,
            residual.reshape(-1, channels).contiguous(),
            layer_norm.weight.contiguous(),
            layer_norm.bias.contiguous(),
            self.x_k.reshape(-1).contiguous(),
            self.up_proj.weight.contiguous(),
            self._down_proj_runtime,
            shift_state_pool=ffn_shift,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=sequence_length,
            eps=layer_norm.eps,
            validated_metadata=ticket,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )
        return summed.view_as(hidden_states), output.view_as(hidden_states)


class RwkvDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_state_boundary = nn.Identity()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.linear_attn = RwkvLinearAttention(config, layer_idx)
        self.mlp = RwkvMLP(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        training_state: RwkvTrainingState | None = None,
        past_key_values: RwkvCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.hidden_state_boundary(hidden_states)
        if not self.training:
            if past_key_values is None:
                raise ValueError("RWKV-7 inference requires an RwkvCache.")
            raise RuntimeError("RWKV-7 inference decoder layers require an explicit residual tensor.")
        att_result = self.linear_attn(
            self.input_layernorm(hidden_states),
            v_first=v_first,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            training_shift_state=None if training_state is None else training_state.attention_shift[self.layer_idx],
            training_wkv_state=None if training_state is None else training_state.recurrent_state[self.layer_idx],
        )
        if training_state is None:
            output, v_first = att_result
        else:
            output, v_first, next_att_shift, next_wkv = att_result
        hidden_states = hidden_states + output
        ffn_result = self.mlp(
            self.post_attention_layernorm(hidden_states),
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            training_shift_state=None if training_state is None else training_state.mlp_shift[self.layer_idx],
        )
        if training_state is None:
            hidden_states = hidden_states + ffn_result
        else:
            ffn_output, next_ffn_shift = ffn_result
            hidden_states = hidden_states + ffn_output
        if training_state is None:
            return hidden_states, v_first
        return hidden_states, v_first, next_att_shift, next_wkv, next_ffn_shift

    def inference_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        v_first: torch.Tensor | None,
        past_key_values: RwkvCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one decoder layer while carrying FlashRWKV2's residual between fusion islands."""
        layer_input, output, v_first = self.linear_attn.inference_forward_with_postnorm(
            hidden_states, residual, self.input_layernorm, v_first, past_key_values
        )
        layer_input = self.hidden_state_boundary(layer_input)
        hidden_states, residual = self.mlp.inference_forward_with_postnorm(
            layer_input, output, self.post_attention_layernorm, past_key_values
        )
        layer = past_key_values.layers[self.layer_idx]
        if isinstance(layer, RwkvDynamicCacheLayer):
            layer.mark_updated(hidden_states.shape[1])
        return hidden_states, residual, v_first, layer_input


@auto_docstring
class RwkvPreTrainedModel(PreTrainedModel):
    config_class = RwkvConfig
    base_model_prefix = "model"
    _no_split_modules = ["RwkvDecoderLayer"]
    _is_stateful = True
    _can_compile_fullgraph = False
    _can_record_outputs = {
        "hidden_states": OutputRecorder(
            nn.Identity,
            layer_name="hidden_state_boundary",
            capture_initial_hidden_state=False,
        )
    }
    supports_gradient_checkpointing = True

    # trf-ignore: TRF018
    def _init_weights(self, module):
        # These owning modules preserve Transformers' per-module loading markers, so from_pretrained never
        # reinitializes a complete model after loading a checkpoint.
        if isinstance(module, RwkvEmbedding | RwkvLMHead | RwkvLinearAttention | RwkvMLP):
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
        self.embed_tokens = RwkvEmbedding(config.vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layers = nn.ModuleList([RwkvDecoderLayer(config, index) for index in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.hidden_state_boundary = nn.Identity()
        self.post_init()

    def reset_parameters(self) -> None:
        """Apply the final canonical train_temp initialization in model order."""
        with torch.no_grad():
            init.uniform_(self.embed_tokens.weight, -1e-4, 1e-4)
            init.ones_(self.embedding_norm.weight)
            init.zeros_(self.embedding_norm.bias)
            for layer in self.layers:
                for layer_norm in (layer.input_layernorm, layer.post_attention_layernorm):
                    init.ones_(layer_norm.weight)
                    init.zeros_(layer_norm.bias)
                layer.linear_attn.reset_parameters()
                layer.mlp.reset_parameters()
            init.ones_(self.norm.weight)
            init.zeros_(self.norm.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _new_cache(self) -> RwkvCache:
        return RwkvCache(self.config)

    def prepare_for_inference(self):
        """Convert weights in-place to Albatross's mixed BF16-embedding/FP16-runtime layout."""
        self.to(dtype=torch.float16)
        self.embed_tokens.to(dtype=torch.bfloat16)
        if not self.config.embedding_layer_norm_fused:
            self.embedding_norm.to(dtype=torch.bfloat16)
        for layer in self.layers:
            layer.linear_attn.prepare_for_inference()
            layer.mlp.prepare_for_inference()
        return self

    @merge_with_config_defaults
    @capture_outputs(tie_last_hidden_states=False)
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: RwkvCache | None = None,
        training_state: RwkvTrainingState | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RwkvModelOutput:
        r"""
        training_state (`RwkvTrainingState`, *optional*):
            Recurrent state returned by a previous training forward, used to continue stateful training across input
            chunks. This argument is not valid during inference.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if past_key_values is not None and not isinstance(past_key_values, RwkvCache):
            raise TypeError(f"RWKV-7 requires `RwkvCache`, got {type(past_key_values).__name__}.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        _validate_rwkv_attention_mask(attention_mask, inputs_embeds)
        if self.training:
            if any(layer.mlp._down_proj_runtime is not None for layer in self.layers):
                raise RuntimeError(
                    "RWKV-7 is still using the in-place Albatross inference layout. To resume training, move the "
                    "complete model to CUDA bfloat16 with `model.to(device='cuda', dtype=torch.bfloat16)`, then call "
                    "`model.train()` before the next forward."
                )
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
                hidden_states = self.embedding_norm(hidden_states)
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
                    self.embedding_norm.weight.contiguous(),
                    self.embedding_norm.bias.contiguous(),
                    eps=self.config.layer_norm_epsilon,
                ).view_as(inputs_embeds)

        v_first = None
        next_att_shifts: list[torch.Tensor] = []
        next_wkv_states: list[torch.Tensor] = []
        next_ffn_shifts: list[torch.Tensor] = []
        if self.training:
            for layer in self.layers:
                layer_result = layer(
                    hidden_states,
                    v_first,
                    training_state,
                    past_key_values=cache,
                    attention_mask=attention_mask,
                )
                if training_state is None:
                    hidden_states, v_first = layer_result
                else:
                    hidden_states, v_first, next_att_shift, next_wkv, next_ffn_shift = layer_result
                    next_att_shifts.append(next_att_shift)
                    next_wkv_states.append(next_wkv)
                    next_ffn_shifts.append(next_ffn_shift)
            hidden_states = self.norm(hidden_states)
        else:
            flash = _load_flash_rwkv2("inference", hidden_states)
            if cache is None:
                raise RuntimeError("RWKV-7 inference cache initialization failed.")
            residual = torch.zeros_like(hidden_states)
            for layer in self.layers:
                hidden_states, residual, v_first, _ = layer.inference_forward(hidden_states, residual, v_first, cache)
            batch_size, sequence_length, channels = hidden_states.shape
            hidden_states = flash.infer_post_norm_output_forward_varlen(
                hidden_states.reshape(-1, channels).contiguous(),
                residual.reshape(-1, channels).contiguous(),
                self.norm.weight.contiguous(),
                self.norm.bias.contiguous(),
                eps=self.config.layer_norm_epsilon,
            ).view(batch_size, sequence_length, channels)
        hidden_states = self.hidden_state_boundary(hidden_states)

        final_cache = cache if use_cache else None
        next_training_state = None
        if training_state is not None:
            next_training_state = RwkvTrainingState(
                attention_shift=torch.stack(next_att_shifts),
                recurrent_state=torch.stack(next_wkv_states),
                mlp_shift=torch.stack(next_ffn_shifts),
            )
        return RwkvModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=final_cache,
            training_state=next_training_state,
        )


@auto_docstring
class RwkvForCausalLM(RwkvPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {}

    def __init__(self, config: RwkvConfig):
        super().__init__(config)
        self.model = RwkvModel(config)
        self.lm_head = RwkvLMHead(config)
        self.post_init()

    def reset_lm_head_parameters(self) -> None:
        """Apply train_temp's vocabulary-dependent LM-head initialization."""
        self.lm_head.reset_parameters()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_for_inference(self):
        self.model.prepare_for_inference()
        self.lm_head.to(dtype=torch.float16)
        return self

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length: int | None = None,
        past_key_values: RwkvCache | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        use_cache: bool | None = None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            is_first_iteration=is_first_iteration,
            use_cache=use_cache,
            **kwargs,
        )
        if model_inputs.get("use_cache") and not is_first_iteration:
            model_inputs.pop("attention_mask", None)
        return model_inputs

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: RwkvCache | None = None,
        training_state: RwkvTrainingState | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> RwkvCausalLMOutput:
        r"""
        training_state (`RwkvTrainingState`, *optional*):
            Recurrent state returned by a previous training forward, used to continue stateful training across input
            chunks. This argument is not valid during inference.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            training_state=training_state,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
            logits = self.lm_head(selected_hidden_states)
        else:
            flash = _load_flash_rwkv2("inference", hidden_states)
            batch_size, sequence_length, channels = hidden_states.shape
            if isinstance(logits_to_keep, int) and logits_to_keep == 1:
                logits = flash.infer_head_linear_last_forward_varlen(
                    selected_hidden_states.reshape(batch_size, channels).contiguous(),
                    self.lm_head.weight.contiguous(),
                    tokens_count=sequence_length,
                ).view(batch_size, 1, self.config.vocab_size)
            else:
                selected_length = selected_hidden_states.shape[1]
                logits = flash.infer_head_linear_all_forward_varlen(
                    selected_hidden_states.reshape(-1, channels).contiguous(), self.lm_head.weight.contiguous()
                ).view(batch_size, selected_length, self.config.vocab_size)
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )
        return RwkvCausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            training_state=outputs.training_state,
        )


__all__ = [
    "RwkvCache",
    "RwkvForCausalLM",
    "RwkvModel",
    "RwkvPreTrainedModel",
    "RwkvLinearAttention",
    "RwkvTrainingState",
]
