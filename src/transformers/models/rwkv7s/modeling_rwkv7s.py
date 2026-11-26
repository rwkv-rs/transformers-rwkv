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
"""PyTorch RWKV-7S model backed exclusively by FlashRWKV2."""

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ... import initialization as init
from ...activations import ACT2FN
from ...cache_utils import Cache, LinearAttentionLayer
from ...integrations import use_experts_implementation
from ...integrations.flash_rwkv2 import flash_rwkv2_linear_spec, load_flash_rwkv2
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel, slice_logits_to_keep
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ...utils.deprecation import deprecate_kwarg
from .configuration_rwkv7s import Rwkv7sConfig
from .generation_rwkv7s import (
    Rwkv7sGenerationMixin,
    _materialize_recurrent_state,
    _recurrent_forward,
    _recurrent_metadata,
    _Rwkv7sGraphState,
    _Rwkv7sRosa,
    compute_rwkv7s_rosa_ids,
)


class _Rwkv7sCacheLayer(LinearAttentionLayer):
    def __init__(self, inference_recurrent_states):
        self._inference_recurrent_states = inference_recurrent_states
        super().__init__(number_of_states=2)

    def _release_recurrent_state(self):
        if self._inference_recurrent_states:
            pool = self._recurrent_states[0]
            if pool is not None:
                key = pool.data_ptr()
                handle = self._inference_recurrent_states.get(key)
                if handle is not None:
                    _materialize_recurrent_state(handle, pool)
                    del self._inference_recurrent_states[key]

    @property
    def recurrent_states(self):
        # The inherited cache API exposes a mutable dictionary of raw tensors.
        self._release_recurrent_state()
        return self._recurrent_states

    @recurrent_states.setter
    def recurrent_states(self, states):
        self._release_recurrent_state()
        self._recurrent_states = states


class Rwkv7sCache(Cache):
    """RWKV-7S recurrent cache with two token shifts and one WKV state per layer, FP32 by default."""

    def __init__(self, config: Rwkv7sConfig):
        self._inference_recurrent_states: dict[int, object] = {}
        super().__init__(
            layers=[_Rwkv7sCacheLayer(self._inference_recurrent_states) for _ in range(config.num_hidden_layers)]
        )
        self.config = config
        self._seen_tokens = 0
        self._stream_lengths: torch.Tensor | None = None
        self._inference_metadata: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, object]] = {}
        self._rosa_states: list[_Rwkv7sRosa] | None = None

    def step_rosa(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._rosa_states is None:
            self._rosa_states = [_Rwkv7sRosa() for _ in range(input_ids.shape[0])]
        rows = input_ids.detach().cpu().tolist()
        matched = [state.extend(row) for state, row in zip(self._rosa_states, rows, strict=True)]
        return torch.tensor(matched, dtype=torch.long, device=input_ids.device)

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def batch_size(self) -> int:
        if self._stream_lengths is not None:
            return self._stream_lengths.numel()
        for layer in self.layers:
            for state in (*layer.conv_states.values(), *layer._recurrent_states.values()):
                if state is not None:
                    return state.shape[0]
        return -1

    @property
    def stream_lengths(self) -> torch.Tensor | None:
        return None if self._stream_lengths is None else self._stream_lengths.clone()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        return query_length, self._seen_tokens

    def layer_states(
        self, layer_idx: int, hidden_states: torch.Tensor, *, training: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._layer_states(layer_idx, hidden_states, training=training, materialize=True)

    def _layer_states(
        self, layer_idx: int, hidden_states: torch.Tensor, *, training: bool, materialize: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        batch_size = hidden_states.shape[0]
        wkv_shape = (
            batch_size,
            self.config.num_attention_heads,
            self.config.head_size,
            self.config.head_size,
        )
        state_dtype = torch.float16 if not training and self.config.wkv_mode == "fp16" else torch.float32

        if not layer.is_conv_states_initialized[0]:
            template = hidden_states[:, -1].unsqueeze(-1)
            layer.lazy_initialization(conv_states=template, state_idx=0, conv_kernel_size=1)
        if not layer.is_recurrent_states_initialized[0]:
            template = torch.empty(wkv_shape, dtype=state_dtype, device=hidden_states.device)
            layer.lazy_initialization(recurrent_states=template, state_idx=0)
        if not layer.is_conv_states_initialized[1]:
            template = hidden_states[:, -1].unsqueeze(-1)
            layer.lazy_initialization(conv_states=template, state_idx=1, conv_kernel_size=1)

        attention_shift = layer.conv_states[0]
        wkv_state = (layer.recurrent_states if materialize else layer._recurrent_states)[0]
        feed_forward_shift = layer.conv_states[1]
        return attention_shift.squeeze(-1), wkv_state, feed_forward_shift.squeeze(-1)

    def update_layer(
        self,
        layer_idx: int,
        attention_shift: torch.Tensor | None = None,
        wkv_state: torch.Tensor | None = None,
        feed_forward_shift: torch.Tensor | None = None,
        *,
        training: bool,
    ) -> None:
        layer = self.layers[layer_idx]
        updates = ((0, attention_shift), (1, feed_forward_shift))
        for state_idx, state in updates:
            if state is None:
                continue
            state = state.unsqueeze(-1)
            if training:
                layer.conv_states[state_idx] = state
            else:
                layer.conv_states[state_idx].copy_(state)
            layer.is_conv_states_initialized[state_idx] = True
            layer.has_previous_state[state_idx] = True
        if wkv_state is not None:
            if training:
                layer.recurrent_states[0] = wkv_state
            else:
                layer.recurrent_states[0].copy_(wkv_state)
            layer.is_recurrent_states_initialized[0] = True
            layer.has_previous_state[0] = True

    def mark_layer_updated(self, layer_idx: int) -> None:
        layer = self.layers[layer_idx]
        layer.has_previous_state[0] = True
        layer.has_previous_state[1] = True

    def advance(self, num_tokens: int, batch_size: int, device: torch.device) -> None:
        if self._stream_lengths is None:
            self._stream_lengths = torch.full((batch_size,), self._seen_tokens, dtype=torch.long, device=device)
        self._stream_lengths = self._stream_lengths + num_tokens
        self._seen_tokens += num_tokens

    def recurrent_metadata(
        self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, object]:
        key = (device, batch_size, sequence_length, torch.cuda.current_stream(device).cuda_stream)
        if key not in self._inference_metadata:
            self._inference_metadata[key] = _recurrent_metadata(flash_rwkv2, batch_size, sequence_length, device)
        return self._inference_metadata[key]

    def recurrent_forward(
        self,
        flash_rwkv2,
        receptance: torch.Tensor,
        decay_logits: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_a: torch.Tensor,
        recurrent_b: torch.Tensor,
        wkv_state: torch.Tensor,
        decay_bias: torch.Tensor,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object],
    ) -> torch.Tensor:
        recurrent_state = self._inference_recurrent_states.get(wkv_state.data_ptr())
        if recurrent_state is None:
            prepare_state = getattr(
                flash_rwkv2, f"prepare_tmix_wkv7_recurrent_{self.config.wkv_mode}_state_from_tensor"
            )
            recurrent_state = prepare_state(wkv_state)
            self._inference_recurrent_states[wkv_state.data_ptr()] = recurrent_state
        return _recurrent_forward(
            flash_rwkv2,
            self.config.wkv_mode,
            recurrent_state,
            receptance,
            decay_logits,
            key,
            value,
            recurrent_a,
            recurrent_b,
            decay_bias,
            inference_metadata,
        )

    def _copy(self, *, detach: bool) -> "Rwkv7sCache":
        copied = type(self)(self.config)
        copied._seen_tokens = self._seen_tokens
        if self._stream_lengths is not None:
            stream_lengths = self._stream_lengths.detach() if detach else self._stream_lengths
            copied._stream_lengths = stream_lengths.clone()
        if self._rosa_states is not None:
            copied._rosa_states = [state.clone() for state in self._rosa_states]
        for source, target in zip(self.layers, copied.layers, strict=True):
            target.device = source.device
            target.dtype = source.dtype
            target.record_past = source.record_past
            for state_idx in range(source.number_of_states):
                for states_name in ("conv_states", "recurrent_states"):
                    state = getattr(source, states_name)[state_idx]
                    if state is not None:
                        state = state.detach() if detach else state
                        getattr(target, states_name)[state_idx] = state.clone()
                target.is_conv_states_initialized[state_idx] = source.is_conv_states_initialized[state_idx]
                target.is_recurrent_states_initialized[state_idx] = source.is_recurrent_states_initialized[state_idx]
                target.has_previous_state[state_idx] = source.has_previous_state[state_idx]
                target.conv_kernel_size[state_idx] = source.conv_kernel_size[state_idx]
        return copied

    def clone(self) -> "Rwkv7sCache":
        return self._copy(detach=False)

    def detach(self) -> "Rwkv7sCache":
        return self._copy(detach=True)

    def reset(self, batch_indices: torch.LongTensor | None = None) -> None:
        if batch_indices is None:
            for layer in self.layers:
                for state_idx in range(layer.number_of_states):
                    for states_name in ("conv_states", "recurrent_states"):
                        state = getattr(layer, states_name)[state_idx]
                        if state is not None:
                            if torch.is_grad_enabled() and state.requires_grad:
                                getattr(layer, states_name)[state_idx] = torch.zeros_like(state)
                            else:
                                state.zero_()
                    layer.has_previous_state[state_idx] = False
            if self._stream_lengths is not None:
                self._stream_lengths.zero_()
            if self._rosa_states is not None:
                for state in self._rosa_states:
                    state.reset()
            self._seen_tokens = 0
        else:
            for layer in self.layers:
                for state_idx in range(layer.number_of_states):
                    for states_name in ("conv_states", "recurrent_states"):
                        state = getattr(layer, states_name)[state_idx]
                        if state is None:
                            continue
                        indices = batch_indices.to(state.device)
                        if torch.is_grad_enabled() and state.requires_grad:
                            getattr(layer, states_name)[state_idx] = state.index_fill(0, indices, 0)
                        else:
                            state.index_fill_(0, indices, 0)
            if self._stream_lengths is not None:
                self._stream_lengths.index_fill_(0, batch_indices.to(self._stream_lengths.device), 0)
                self._seen_tokens = int(self._stream_lengths.max().item())
            if self._rosa_states is not None:
                for idx in batch_indices.tolist():
                    self._rosa_states[idx].reset()
        self._inference_metadata.clear()

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            for state_idx in range(layer.number_of_states):
                for states_name in ("conv_states", "recurrent_states"):
                    state = getattr(layer, states_name)[state_idx]
                    if state is not None:
                        getattr(layer, states_name)[state_idx] = state.repeat_interleave(repeats, dim=0)
        if self._stream_lengths is not None:
            self._stream_lengths = self._stream_lengths.repeat_interleave(repeats, dim=0)
        if self._rosa_states is not None:
            self._rosa_states = [state.clone() for state in self._rosa_states for _ in range(repeats)]
        self._inference_metadata.clear()

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for layer in self.layers:
            for state_idx in range(layer.number_of_states):
                for states_name in ("conv_states", "recurrent_states"):
                    state = getattr(layer, states_name)[state_idx]
                    if state is not None:
                        getattr(layer, states_name)[state_idx] = state.index_select(0, indices.to(state.device))
        if self._stream_lengths is not None:
            self._stream_lengths = self._stream_lengths.index_select(0, indices.to(self._stream_lengths.device))
            self._seen_tokens = int(self._stream_lengths.max().item())
        if self._rosa_states is not None:
            self._rosa_states = [self._rosa_states[idx].clone() for idx in indices.tolist()]
        self._inference_metadata.clear()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class Rwkv7sAttention(nn.Module):
    """TimeMix, called `RWKV_Tmix_x070` in RWKV-LM and `RWKV7Attention` in FLA and hf_rwkv."""

    def __init__(self, config: Rwkv7sConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_size = config.head_size
        self.num_heads = config.num_attention_heads

        self.x_r = nn.Parameter(torch.empty(config.hidden_size))
        self.x_w = nn.Parameter(torch.empty(config.hidden_size))
        self.x_k = nn.Parameter(torch.empty(config.hidden_size))
        self.x_v = nn.Parameter(torch.empty(config.hidden_size))
        self.x_a = nn.Parameter(torch.empty(config.hidden_size))
        self.x_g = nn.Parameter(torch.empty(config.hidden_size))

        self.w0 = nn.Parameter(torch.empty(config.hidden_size))
        self.w1 = nn.Parameter(torch.empty(config.hidden_size, config.decay_low_rank_dim))
        self.w2 = nn.Parameter(torch.empty(config.decay_low_rank_dim, config.hidden_size))
        self.a0 = nn.Parameter(torch.empty(config.hidden_size))
        self.a1 = nn.Parameter(torch.empty(config.hidden_size, config.a_low_rank_dim))
        self.a2 = nn.Parameter(torch.empty(config.a_low_rank_dim, config.hidden_size))
        if layer_idx != 0:
            self.v0 = nn.Parameter(torch.empty(config.hidden_size))
            self.v1 = nn.Parameter(torch.empty(config.hidden_size, config.v_low_rank_dim))
            self.v2 = nn.Parameter(torch.empty(config.v_low_rank_dim, config.hidden_size))
        self.g1 = nn.Parameter(torch.empty(config.hidden_size, config.gate_low_rank_dim))
        self.g2 = nn.Parameter(torch.empty(config.gate_low_rank_dim, config.hidden_size))

        self.k_k = nn.Parameter(torch.empty(config.hidden_size))
        self.k_a = nn.Parameter(torch.empty(config.hidden_size))
        self.r_k = nn.Parameter(torch.empty(config.num_attention_heads, config.head_size))

        self.r_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.g_norm = nn.GroupNorm(
            config.num_attention_heads,
            config.hidden_size,
            eps=config.group_norm_epsilon,
            affine=True,
        )

    def _training_forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor | None,
        wkv_state: torch.Tensor | None,
        use_unfused_projections: bool,
        cu_seqlens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        flash_rwkv2 = load_flash_rwkv2()
        hidden_states = hidden_states.contiguous()
        shifts = flash_rwkv2.pretrain_tmix_tokenshift_bf16(
            hidden_states.squeeze(0) if cu_seqlens is not None else hidden_states,
            self.x_r.to(hidden_states.dtype),
            self.x_w.to(hidden_states.dtype),
            self.x_k.to(hidden_states.dtype),
            self.x_v.to(hidden_states.dtype),
            self.x_a.to(hidden_states.dtype),
            self.x_g.to(hidden_states.dtype),
            initial_shift=attention_shift.contiguous() if attention_shift is not None else None,
            cu_seqlens=cu_seqlens,
            return_final_state=attention_shift is not None,
        )
        if cu_seqlens is not None:
            shifts = tuple(shift.unsqueeze(0) for shift in shifts)
        xr, xw, xk, xv, xa, xg = shifts[:6]
        next_attention_shift = shifts[6] if attention_shift is not None else None

        if use_unfused_projections:
            dtype = hidden_states.dtype
            projection_dtype = self.w0.dtype
            receptance = self.r_proj(xr.to(projection_dtype)).to(dtype)
            key = self.k_proj(xk.to(projection_dtype)).to(dtype)
            value = self.v_proj(xv.to(projection_dtype)).to(dtype)
            decay = ((xw @ self.w1.to(dtype)).tanh() @ self.w2.to(dtype)).float()
            # The training recurrence applies the decay activation to biased logits.
            decay_logits = (decay + self.w0.float()).to(dtype)
            rate = (self.a0.float() + (xa @ self.a1.to(dtype) @ self.a2.to(dtype)).float()).sigmoid().to(dtype)
            gate = (xg @ self.g1.to(dtype)).sigmoid() @ self.g2.to(dtype)
            if self.layer_idx == 0:
                v_first = value
            else:
                value_gate = (self.v0.float() + (xv @ self.v1.to(dtype) @ self.v2.to(dtype)).float()).sigmoid()
                value = (value.float() + (v_first.float() - value.float()) * value_gate).to(dtype)
            direction = F.normalize(
                (key.float() * self.k_k.float()).unflatten(-1, (self.num_heads, self.head_size)), dim=-1
            ).flatten(-2)
            negative_direction = (-direction).to(dtype)
            scaled_direction = (direction * rate.float()).to(dtype)
            key = (key.float() * (1.0 + (rate.float() - 1.0) * self.k_a.float())).to(dtype)
            o_weight, o_a, o_b, o_scale, o_nora = None, None, None, 1.0, False
        else:
            r_weight, r_a, r_b, r_scale, r_nora = flash_rwkv2_linear_spec(self.r_proj)
            k_weight, k_a, k_b, k_scale, k_nora = flash_rwkv2_linear_spec(self.k_proj)
            v_weight, v_a, v_b, v_scale, v_nora = flash_rwkv2_linear_spec(self.v_proj)
            o_weight, o_a, o_b, o_scale, o_nora = flash_rwkv2_linear_spec(self.o_proj)
            receptance, decay_logits, key, value, negative_direction, scaled_direction, gate, v_first = (
                flash_rwkv2.pretrain_tmix_wkv_prepare_bf16(
                    xr,
                    xw,
                    xk,
                    xv,
                    xa,
                    xg,
                    r_weight,
                    k_weight,
                    v_weight,
                    self.w0,
                    self.w1,
                    self.w2,
                    self.a0,
                    self.a1,
                    self.a2,
                    self.g1,
                    self.g2,
                    self.k_k,
                    self.k_a,
                    v_first=v_first if self.layer_idx != 0 else None,
                    v0=self.v0 if self.layer_idx != 0 else None,
                    v1=self.v1 if self.layer_idx != 0 else None,
                    v2=self.v2 if self.layer_idx != 0 else None,
                    receptance_lora_a=r_a,
                    receptance_lora_b=r_b,
                    receptance_lora_scale=r_scale,
                    key_lora_a=k_a,
                    key_lora_b=k_b,
                    key_lora_scale=k_scale,
                    value_lora_a=v_a,
                    value_lora_b=v_b,
                    value_lora_scale=v_scale,
                    receptance_use_nora=r_nora,
                    key_use_nora=k_nora,
                    value_use_nora=v_nora,
                    head_size=self.head_size,
                )
            )

        recurrent_inputs = (receptance, decay_logits, key, value, negative_direction, scaled_direction)
        if cu_seqlens is not None:
            recurrent_inputs = tuple(t.view(-1, self.num_heads, self.head_size) for t in recurrent_inputs)
        recurrent_output = flash_rwkv2.pretrain_tmix_wkv7_recurrent_bf16(
            *recurrent_inputs,
            head_size=self.head_size,
            initial_state=wkv_state,
            cu_seqlens=cu_seqlens,
            return_final_state=wkv_state is not None,
        )
        next_wkv_state = None
        if wkv_state is not None:
            recurrent_output, next_wkv_state = recurrent_output
        if cu_seqlens is not None:
            recurrent_output = recurrent_output.view_as(hidden_states)

        output = flash_rwkv2.pretrain_tmix_readout_bf16(
            recurrent_output.contiguous(),
            receptance,
            key,
            value,
            self.r_k.to(hidden_states.dtype),
            self.g_norm.weight.to(hidden_states.dtype),
            self.g_norm.bias.to(hidden_states.dtype),
            gate,
            head_size=self.head_size,
            output_weight=o_weight,
            output_lora_a=o_a,
            output_lora_b=o_b,
            output_lora_scale=o_scale,
            output_use_nora=o_nora,
        )
        if use_unfused_projections:
            output = self.o_proj(output.to(self.w0.dtype)).to(hidden_states.dtype)
        return output, v_first, next_attention_shift, next_wkv_state

    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor,
        wkv_state: object,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object],
        cache: Rwkv7sCache | _Rwkv7sGraphState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flash_rwkv2 = load_flash_rwkv2()
        cu_seqlens, state_indices, max_seqlen, ticket = inference_metadata
        saved_tmix = getattr(cache, "saved_tmix_norm", None)
        residual, xr, xw, xk, xv, xa, xg = flash_rwkv2.infer_tmix_postnorm_tokenshift_forward_varlen(
            hidden_states.contiguous(),
            residual.contiguous(),
            layer_norm.weight.contiguous(),
            layer_norm.bias.contiguous(),
            self.x_r,
            self.x_w,
            self.x_k,
            self.x_v,
            self.x_a,
            self.x_g,
            shift_state_pool=attention_shift,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=max_seqlen,
            eps=self.config.layer_norm_epsilon,
            validated_metadata=ticket,
            save_normalized=None if saved_tmix is None else saved_tmix[self.layer_idx],
        )

        r_weight, r_lora_a, r_lora_b, r_lora_scale, r_nora = flash_rwkv2_linear_spec(self.r_proj)
        k_weight, k_lora_a, k_lora_b, k_lora_scale, k_nora = flash_rwkv2_linear_spec(self.k_proj)
        v_weight, v_lora_a, v_lora_b, v_lora_scale, v_nora = flash_rwkv2_linear_spec(self.v_proj)
        receptance, decay_logits, key, value, recurrent_a, recurrent_b, gate, v_first = (
            flash_rwkv2.infer_tmix_wkv_prepare_forward_varlen(
                xr,
                xw,
                xk,
                xv,
                xa,
                xg,
                r_weight,
                k_weight,
                v_weight,
                self.w1.t(),
                self.a1.t(),
                self.g1.t(),
                self.v1.t() if self.layer_idx != 0 else None,
                self.w2.t(),
                self.a2.t(),
                self.g2.t(),
                self.v2.t() if self.layer_idx != 0 else None,
                self.v0 if self.layer_idx != 0 else None,
                self.k_k,
                self.a0,
                self.k_a,
                v_first=v_first,
                receptance_lora_a=r_lora_a,
                receptance_lora_b=r_lora_b,
                receptance_lora_scale=r_lora_scale,
                receptance_use_nora=r_nora,
                key_lora_a=k_lora_a,
                key_lora_b=k_lora_b,
                key_lora_scale=k_lora_scale,
                key_use_nora=k_nora,
                value_lora_a=v_lora_a,
                value_lora_b=v_lora_b,
                value_lora_scale=v_lora_scale,
                value_use_nora=v_nora,
                head_size=self.head_size,
                batch_size=state_indices.numel(),
                max_seqlen=max_seqlen,
            )
        )
        recurrent_output = cache.recurrent_forward(
            flash_rwkv2,
            receptance.view(-1, self.num_heads, self.head_size),
            decay_logits.view(-1, self.num_heads, self.head_size),
            key.view(-1, self.num_heads, self.head_size),
            value.view(-1, self.num_heads, self.head_size),
            recurrent_a.view(-1, self.num_heads, self.head_size),
            recurrent_b.view(-1, self.num_heads, self.head_size),
            wkv_state,
            self.w0,
            inference_metadata,
        )
        o_weight, o_lora_a, o_lora_b, o_lora_scale, o_nora = flash_rwkv2_linear_spec(self.o_proj)
        output = flash_rwkv2.infer_tmix_readout_forward_varlen(
            recurrent_output.view(-1, self.hidden_size),
            receptance,
            key,
            value,
            self.r_k.flatten().contiguous(),
            self.g_norm.weight,
            self.g_norm.bias,
            gate,
            o_weight,
            output_lora_a=o_lora_a,
            output_lora_b=o_lora_b,
            output_lora_scale=o_lora_scale,
            output_use_nora=o_nora,
            head_size=self.head_size,
            batch_size=state_indices.numel(),
            max_seqlen=max_seqlen,
        )
        return output, residual, v_first

    @deprecate_kwarg("cache_params", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None = None,
        attention_shift: torch.Tensor | None = None,
        wkv_state: object | None = None,
        residual: torch.Tensor | None = None,
        layer_norm: nn.LayerNorm | None = None,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object] | None = None,
        cache: Rwkv7sCache | _Rwkv7sGraphState | None = None,
        use_unfused_projections: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if hidden_states.ndim == 3:
            return self._training_forward(
                hidden_states,
                v_first,
                attention_shift,
                wkv_state,
                use_unfused_projections,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
            )
        return self._inference_forward(
            hidden_states,
            residual,
            layer_norm,
            v_first,
            attention_shift,
            wkv_state,
            inference_metadata,
            cache,
        )


class Rwkv7sMLP(nn.Module):
    """Squared-ReLU shared expert MLP (`NemotronHMLP`)."""

    def __init__(self, config: Rwkv7sConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.shared_expert_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.shared_expert_intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN["relu2"]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


class Rwkv7sTopkRouter(nn.Module):
    """Aux-loss-free sigmoid top-k router (`DeepseekV3TopkRouter` / `NemotronHTopkRouter`)."""

    def __init__(self, config: Rwkv7sConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, dtype=torch.float32))
        self.e_score_correction_bias = nn.Buffer(torch.zeros(self.num_experts, dtype=torch.float32))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.weight.float())
        scores = router_logits.sigmoid()
        topk_indices = torch.topk(scores + self.e_score_correction_bias.float(), k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_indices


@use_experts_implementation(has_gate=False)
class Rwkv7sExperts(nn.Module):
    """Non-gated squared-ReLU (`relu2`) MoE experts stored as 3D tensors (`NemotronHExperts`)."""

    def __init__(self, config: Rwkv7sConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN["relu2"]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states, dtype=top_k_weights.dtype)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = F.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(current_hidden_states)
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states.to(hidden_states.dtype)


class Rwkv7sFeedForward(nn.Module):
    """Token-shifted squared-ReLU MoE ChannelMix with `rwkv7a` low-rank DeepEmbed channelwise scaling."""

    def __init__(self, config: Rwkv7sConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.x_k = nn.Parameter(torch.empty(config.hidden_size))
        self.gate = Rwkv7sTopkRouter(config)
        self.experts = Rwkv7sExperts(config)
        self.shared_experts = Rwkv7sMLP(config)
        rank = config.deep_embed_dim
        self.s_emb = nn.Embedding(config.vocab_size, rank * rank)
        self.s1 = nn.Parameter(torch.empty(config.hidden_size, rank))
        self.s2 = nn.Parameter(torch.empty(rank, config.hidden_size))
        self.s0 = nn.Parameter(torch.ones(config.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        feed_forward_shift: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        layer_norm: nn.LayerNorm | None = None,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object] | None = None,
        input_ids: torch.LongTensor | None = None,
        saved_cmix_norm: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if hidden_states.ndim == 3:
            cu_seqlens = kwargs.pop("cu_seq_lens_q", None)
            normed = hidden_states
            previous = (
                F.pad(normed[:, :-1], (0, 0, 1, 0))
                if feed_forward_shift is None
                else torch.cat((feed_forward_shift[:, None], normed[:, :-1]), dim=1)
            )
            if cu_seqlens is not None:
                previous = previous.index_fill(1, cu_seqlens[:-1].long(), 0)
            state_out = normed[:, -1] if feed_forward_shift is not None else None
        else:
            _, state_indices, max_seqlen, _ = inference_metadata
            state_out = hidden_states + residual
            normed = (
                load_flash_rwkv2()
                .infer_embedding_ln0_forward_varlen(
                    state_out.contiguous(),
                    layer_norm.weight,
                    layer_norm.bias,
                    eps=self.config.layer_norm_epsilon,
                )
                .view(state_indices.numel(), max_seqlen, -1)
            )
            if saved_cmix_norm is not None:
                saved_cmix_norm.copy_(normed)
            previous = (
                feed_forward_shift[:, None, :]
                if max_seqlen == 1
                else torch.cat((feed_forward_shift[:, None, :], normed[:, :-1, :]), dim=1)
            )

        x = normed.float()
        flat = (x + (previous.float() - x) * self.x_k.float()).to(normed.dtype).reshape(-1, normed.shape[-1])
        if hidden_states.ndim == 2:
            feed_forward_shift.copy_(normed[:, -1, :])
        _, topk_weights, topk_indices = self.gate(flat)
        output = (self.experts(flat, topk_indices, topk_weights) + self.shared_experts(flat.to(self.x_k.dtype))).to(
            normed.dtype
        )
        if input_ids is not None:
            rank = self.config.deep_embed_dim
            x_flat = normed.reshape(-1, normed.shape[-1])
            emb = self.s_emb(input_ids.reshape(-1)).view(-1, rank, rank).to(x_flat.dtype)
            ss = torch.bmm((x_flat @ self.s1.to(x_flat.dtype)).unsqueeze(1), emb).squeeze(1)
            output = output * ((ss @ self.s2.to(x_flat.dtype)) + self.s0.to(x_flat.dtype))
        return output.view_as(hidden_states), state_out


class Rwkv7sDecoderLayer(GradientCheckpointingLayer):
    """RWKV-7S block, called `Block` in RWKV-LM and `RWKV7Block` in FLA and hf_rwkv."""

    def __init__(self, config: Rwkv7sConfig, layer_idx: int):
        super().__init__()
        self.linear_attn = Rwkv7sAttention(config, layer_idx)
        self.mlp = Rwkv7sFeedForward(config, layer_idx)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    @deprecate_kwarg("cache_params", "5.40.0", new_name="cache")
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor | None,
        wkv_state: object | None,
        feed_forward_shift: torch.Tensor | None,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object] | None,
        cache: Rwkv7sCache | _Rwkv7sGraphState | None = None,
        use_unfused_projections: bool = False,
        input_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if hidden_states.ndim == 3:
            flash_rwkv2 = load_flash_rwkv2()
            attention_output, v_first, next_attention_shift, next_wkv_state = self.linear_attn(
                flash_rwkv2.infer_embedding_ln0_forward_varlen(
                    hidden_states,
                    self.input_layernorm.weight.to(hidden_states.dtype),
                    self.input_layernorm.bias.to(hidden_states.dtype),
                    eps=self.input_layernorm.eps,
                ),
                v_first,
                attention_shift,
                wkv_state,
                use_unfused_projections=use_unfused_projections,
                **kwargs,
            )
            hidden_states = hidden_states + attention_output
            feed_forward_output, next_feed_forward_shift = self.mlp(
                flash_rwkv2.infer_embedding_ln0_forward_varlen(
                    hidden_states,
                    self.post_attention_layernorm.weight.to(hidden_states.dtype),
                    self.post_attention_layernorm.bias.to(hidden_states.dtype),
                    eps=self.post_attention_layernorm.eps,
                ),
                feed_forward_shift,
                input_ids=input_ids,
                **kwargs,
            )
            hidden_states = hidden_states + feed_forward_output
            return hidden_states, v_first, next_attention_shift, next_wkv_state, next_feed_forward_shift

        attention_output, layer_input, v_first = self.linear_attn(
            hidden_states,
            v_first,
            attention_shift,
            wkv_state,
            residual=residual,
            layer_norm=self.input_layernorm,
            inference_metadata=inference_metadata,
            cache=cache,
        )
        saved_cmix = getattr(cache, "saved_cmix_norm", None)
        hidden_states, residual = self.mlp(
            layer_input,
            feed_forward_shift,
            residual=attention_output,
            layer_norm=self.post_attention_layernorm,
            inference_metadata=inference_metadata,
            input_ids=input_ids,
            saved_cmix_norm=None if saved_cmix is None else saved_cmix[self.mlp.layer_idx],
        )
        return hidden_states, residual, v_first, None, None


@auto_docstring
@dataclass
class Rwkv7sModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    autodeco_hidden_state (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
        Normalized hidden states from the second-to-last decoder layer (or final layer if fewer than two layers).
    """

    autodeco_hidden_state: torch.Tensor | None = None


@auto_docstring
@dataclass
class Rwkv7sCausalLMOutputWithPast(CausalLMOutputWithPast):
    r"""
    temp_logits (`torch.Tensor` of shape `(batch_size, sequence_length, 1)`, *optional*):
        Per-token AutoDeco temperature predictions in `(0, 2)`.
    top_p_logits (`torch.Tensor` of shape `(batch_size, sequence_length, 1)`, *optional*):
        Per-token AutoDeco top-p predictions in `(0, 1)`.
    """

    temp_logits: torch.Tensor | None = None
    top_p_logits: torch.Tensor | None = None


class Rwkv7sAutoDeco(nn.Module):
    """AutoDeco dynamic temperature and top-p heads (`TempHead` + `TopPHead` in RWKV-AUTODECO)."""

    def __init__(self, hidden_size: int, intermediate_size: int = 64):
        super().__init__()
        self.temp_fc1 = nn.Linear(hidden_size, intermediate_size)
        self.temp_fc2 = nn.Linear(intermediate_size, 1)
        self.top_p_fc1 = nn.Linear(hidden_size + 5, intermediate_size)
        self.top_p_fc2 = nn.Linear(intermediate_size, 1)

    def forward(self, hidden_states: torch.Tensor, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden_states.detach()
        temp_in = hidden.to(self.temp_fc1.weight.dtype)
        temp = torch.sigmoid(self.temp_fc2(F.relu(self.temp_fc1(temp_in)))) * 2

        probs = torch.softmax(logits.detach().float() / (temp.detach().float() + 1e-8), dim=-1)
        top_p_dtype = self.top_p_fc1.weight.dtype
        stats = torch.cat(
            (
                probs.max(dim=-1, keepdim=True).values,
                -(probs * torch.log(probs + 1e-8)).sum(dim=-1, keepdim=True),
                probs.var(dim=-1, keepdim=True),
                probs.topk(min(5, probs.size(-1)), dim=-1).values.sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        ).to(top_p_dtype)
        top_p_in = torch.cat((hidden.to(top_p_dtype), temp.detach().to(top_p_dtype), stats), dim=-1)
        top_p = torch.sigmoid(self.top_p_fc2(F.relu(self.top_p_fc1(top_p_in))))
        return temp, top_p


class Rwkv7sMTP(nn.Module):
    """Shared LightMTP head (`RwkvMTPBlock` in irwkv), looped `mtp_depth` times with shared `lm_head`."""

    def __init__(self, config: Rwkv7sConfig):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.mtp_intermediate_size
        eps = config.layer_norm_epsilon
        self.norm_h = nn.LayerNorm(hidden_size, eps=eps)
        self.norm_e = nn.LayerNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.norm_mlp = nn.LayerNorm(hidden_size, eps=eps)
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.norm_out = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        dtype = self.proj.weight.dtype
        hidden = hidden_states.to(dtype)
        fused = self.proj(torch.cat((self.norm_e(token_embeds.to(dtype)), self.norm_h(hidden)), dim=-1))
        hidden = hidden + fused
        mlp_in = self.norm_mlp(hidden)
        return hidden + self.down(F.silu(self.gate(mlp_in)) * self.up(mlp_in))


@auto_docstring
class Rwkv7sPreTrainedModel(PreTrainedModel):
    config: Rwkv7sConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Rwkv7sDecoderLayer"]
    _is_stateful = True
    _can_compile_fullgraph = False
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "mlp.gate.weight"]

    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Embedding):
            init.uniform_(module.weight, -1e-4, 1e-4)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])
        elif hasattr(self, "lm_head") and module is self.lm_head:
            gain = (
                0.5 * math.sqrt(self.config.vocab_size / self.config.hidden_size)
                if self.config.vocab_size > self.config.hidden_size
                else 0.5
            )
            init.orthogonal_(module.weight, gain=gain)
        elif isinstance(module, Rwkv7sAttention):
            channels = module.hidden_size
            position = torch.arange(channels, device=module.x_r.device, dtype=torch.float32)
            ddd = position / channels
            linear = position / max(channels - 1, 1) - 0.5
            zigzag = (position.remainder(module.head_size) - (module.head_size - 1) / 2) / ((module.head_size - 1) / 2)
            zigzag = zigzag * zigzag.abs()
            ratio_0_to_1 = module.layer_idx / max(self.config.num_hidden_layers - 1, 1)
            ratio_1_to_almost0 = 1.0 - module.layer_idx / self.config.num_hidden_layers
            decay = -6 + 6 * (position / max(channels - 1, 1)).pow(1 + ratio_0_to_1**0.3)

            for parameter, exponent in (
                (module.x_r, 0.2),
                (module.x_w, 0.9),
                (module.x_k, 0.7),
                (module.x_v, 0.7),
                (module.x_a, 0.9),
                (module.x_g, 0.2),
            ):
                init.copy_(parameter, (1 - ddd.pow(exponent * ratio_1_to_almost0)).to(parameter.dtype))
            init.copy_(module.w0, (decay + 0.5 + zigzag * 2.5).to(module.w0.dtype))
            init.copy_(module.a0, (-0.19 + zigzag * 0.3 + linear * 0.4).to(module.a0.dtype))
            if module.layer_idx != 0:
                init.copy_(module.v0, (0.73 - linear * 0.4).to(module.v0.dtype))
            init.copy_(module.k_k, (0.71 - linear * 0.1).to(module.k_k.dtype))
            init.constant_(module.k_a, 1.02)
            init.constant_(module.r_k, -0.04)

            for parameter in (module.w1, module.a1, module.g1):
                init.zeros_(parameter)
            if module.layer_idx != 0:
                init.zeros_(module.v1)
            for parameter in (module.w2, module.a2, module.g2):
                init.orthogonal_(parameter, gain=0.1)
            if module.layer_idx != 0:
                init.orthogonal_(module.v2, gain=0.1)
            init.orthogonal_(module.r_proj.weight)
            init.orthogonal_(module.k_proj.weight, gain=0.1)
            init.orthogonal_(module.v_proj.weight)
            init.zeros_(module.o_proj.weight)
            init.constant_(
                module.g_norm.weight,
                ((module.layer_idx + 1) / self.config.num_hidden_layers) ** 0.7,
            )
            init.zeros_(module.g_norm.bias)
        elif isinstance(module, Rwkv7sFeedForward):
            channels = self.config.hidden_size
            ddd = torch.arange(channels, device=module.x_k.device, dtype=torch.float32) / channels
            ratio = 1.0 - module.layer_idx / self.config.num_hidden_layers
            init.copy_(module.x_k, (1 - ddd.pow(ratio**4)).to(module.x_k.dtype))
            init.orthogonal_(module.shared_experts.up_proj.weight)
            init.zeros_(module.shared_experts.down_proj.weight)
            init.normal_(module.s_emb.weight, mean=0.0, std=0.02)
            init.normal_(module.s1, mean=0.0, std=0.02)
            init.zeros_(module.s2)
            init.ones_(module.s0)
        elif isinstance(module, Rwkv7sTopkRouter):
            init.normal_(module.weight, mean=0.0, std=0.02)
            init.zeros_(module.e_score_correction_bias)
        elif isinstance(module, Rwkv7sExperts):
            init.normal_(module.up_proj, mean=0.0, std=0.02)
            init.normal_(module.down_proj, mean=0.0, std=0.02)
        elif isinstance(module, Rwkv7sModel):
            init.zeros_(module.rosa_gate)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)


@auto_docstring
class Rwkv7sModel(Rwkv7sPreTrainedModel):
    def __init__(self, config: Rwkv7sConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.rosa_embed = nn.Embedding(config.vocab_size + 1, config.hidden_size, padding_idx=config.vocab_size)
        self.rosa_gate = nn.Parameter(torch.zeros(config.hidden_size))
        self.layers = nn.ModuleList(
            [Rwkv7sDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def _supports_fused_projections(self) -> bool:
        return all(
            flash_rwkv2_linear_spec(projection) is not None
            for layer in self.layers
            for projection in (
                layer.linear_attn.r_proj,
                layer.linear_attn.k_proj,
                layer.linear_attn.v_proj,
                layer.linear_attn.o_proj,
            )
        )

    def _forward_state_only(
        self,
        input_ids: torch.LongTensor | None,
        inputs_embeds: torch.FloatTensor | None,
        cache: Rwkv7sCache | _Rwkv7sGraphState,
        output_hidden_states: bool,
        return_autodeco_hidden: bool = False,
        rosa_ids: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...] | None, torch.Tensor | None]:
        if inputs_embeds is None:
            batch_size, sequence_length = input_ids.shape
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = inputs_embeds.reshape(-1, self.config.hidden_size).contiguous()
        flash_rwkv2 = load_flash_rwkv2()
        hidden_states = flash_rwkv2.infer_embedding_ln0_forward_varlen(
            inputs_embeds,
            self.embedding_norm.weight,
            self.embedding_norm.bias,
            eps=self.config.layer_norm_epsilon,
        )
        if rosa_ids is not None:
            rosa_tokens = torch.where(rosa_ids < 0, self.config.vocab_size, rosa_ids).reshape(-1)
            hidden_states = hidden_states + (self.rosa_embed(rosa_tokens) * self.rosa_gate).to(hidden_states.dtype)
        residual = torch.zeros_like(hidden_states)
        inference_metadata = cache.recurrent_metadata(flash_rwkv2, batch_size, sequence_length, hidden_states.device)

        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            all_hidden_states += ((hidden_states + residual).view(batch_size, sequence_length, -1),)

        second_last_idx = len(self.layers) - 2
        autodeco_hidden = None
        v_first = None
        for layer_idx, decoder_layer in enumerate(self.layers):
            state_reference = hidden_states.view(batch_size, sequence_length, -1)
            if isinstance(cache, Rwkv7sCache):
                attention_shift, wkv_state, feed_forward_shift = cache._layer_states(
                    layer_idx, state_reference, training=False, materialize=False
                )
            else:
                attention_shift, wkv_state, feed_forward_shift = cache.layer_states(layer_idx, state_reference)
            hidden_states, residual, v_first, _, _ = decoder_layer(
                hidden_states,
                residual,
                v_first,
                attention_shift,
                wkv_state,
                feed_forward_shift,
                inference_metadata,
                cache,
                input_ids=input_ids,
            )
            if isinstance(cache, Rwkv7sCache):
                cache.mark_layer_updated(layer_idx)
            if return_autodeco_hidden and layer_idx == second_last_idx:
                autodeco_hidden = flash_rwkv2.infer_post_norm_output_forward_varlen(
                    hidden_states,
                    residual,
                    self.norm.weight,
                    self.norm.bias,
                    eps=self.config.layer_norm_epsilon,
                )

            if output_hidden_states:
                all_hidden_states += ((hidden_states + residual).view(batch_size, sequence_length, -1),)

        if isinstance(cache, Rwkv7sCache):
            cache.advance(sequence_length, batch_size, hidden_states.device)
        return hidden_states, residual, all_hidden_states, autodeco_hidden

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache", raise_if_both_names=True)
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        rosa_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Rwkv7sModelOutputWithPast:
        r"""
        rosa_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Precomputed token-level ROSA successor IDs (`-1` when unmatched), e.g. produced asynchronously by CPU
            DataLoader workers via `compute_rwkv7s_rosa_ids`. Computed automatically from `input_ids` when omitted.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens_q is not None:
            if cache is not None:
                raise ValueError("RWKV-7S packing cannot be used with cache.")
            use_cache = False
        inputs = input_ids if inputs_embeds is None else inputs_embeds
        if not inputs.is_cuda or not self.norm.weight.is_cuda:
            raise RuntimeError("RWKV-7S requires CUDA inputs and model weights.")
        use_unfused_projections = kwargs.pop("_use_unfused_projections", None)
        if use_unfused_projections is None:
            use_unfused_projections = not self._supports_fused_projections()
        kbit_training = use_unfused_projections and getattr(self.config, "quantization_config", None) is not None
        if not kbit_training and (
            self.norm.weight.dtype != torch.bfloat16
            or (inputs_embeds is not None and inputs_embeds.dtype != torch.bfloat16)
        ):
            raise TypeError("RWKV-7S requires bfloat16 model weights and input embeddings.")
        training_path = self.training or torch.is_grad_enabled()
        if use_cache is None:
            use_cache = cache is not None or (not training_path and self.config.use_cache)
        training_path = training_path or use_unfused_projections or cu_seq_lens_q is not None
        output_hidden_states = kwargs.pop("output_hidden_states", None)
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        _ = position_ids

        batch_size, sequence_length = inputs.shape[:2]
        if (
            not training_path
            and attention_mask is not None
            and (
                attention_mask.ndim != 2
                or attention_mask.shape[0] != batch_size
                or attention_mask.shape[1] < sequence_length
                or not torch.all(attention_mask != 0)
            )
        ):
            raise ValueError("RWKV-7S inference requires an unpadded batch.")

        if cache is not None:
            if not isinstance(cache, Rwkv7sCache):
                raise TypeError(f"RWKV-7S accepts only Rwkv7sCache, got {type(cache).__name__}.")
            cache_batch_size = cache.batch_size
            if cache_batch_size not in (-1, batch_size):
                raise ValueError(f"RWKV-7S cache batch size is {cache_batch_size}, but the input has {batch_size}.")
            state_dtype = torch.float16 if not training_path and self.config.wkv_mode == "fp16" else torch.float32
            if any(
                (state := layer._recurrent_states[0]) is not None and state.dtype != state_dtype
                for layer in cache.layers
            ):
                raise ValueError(f"RWKV-7S requires a WKV cache with dtype {state_dtype} for this forward.")
        # FlashRWKV2 inference is recurrent even when the caller does not want the
        # state returned, so use a temporary cache for that call.
        if cache is None and (use_cache or not training_path):
            cache = Rwkv7sCache(self.config)

        if rosa_ids is not None:
            rosa_ids = rosa_ids.to(device=inputs.device)
        elif input_ids is not None:
            rosa_ids = (
                cache.step_rosa(input_ids)
                if cache is not None
                else compute_rwkv7s_rosa_ids(input_ids, cu_seqlens=cu_seq_lens_q)
            )

        flash_rwkv2 = load_flash_rwkv2()
        autodeco_hidden = None
        if training_path:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            hidden_states = flash_rwkv2.infer_embedding_ln0_forward_varlen(
                inputs_embeds.to(torch.bfloat16),
                self.embedding_norm.weight.to(torch.bfloat16),
                self.embedding_norm.bias.to(torch.bfloat16),
                eps=self.config.layer_norm_epsilon,
            )
            if rosa_ids is not None:
                rosa_tokens = torch.where(rosa_ids < 0, self.config.vocab_size, rosa_ids)
                hidden_states = hidden_states + (self.rosa_embed(rosa_tokens) * self.rosa_gate).to(hidden_states.dtype)
            all_hidden_states = () if output_hidden_states else None
            if output_hidden_states:
                all_hidden_states += (hidden_states.view(batch_size, sequence_length, -1),)

            second_last_idx = len(self.layers) - 2
            prev_hidden_states = None
            v_first = None
            for layer_idx, decoder_layer in enumerate(self.layers):
                attention_shift = wkv_state = feed_forward_shift = None
                if cache is not None:
                    attention_shift, wkv_state, feed_forward_shift = cache.layer_states(layer_idx, hidden_states)

                hidden_states, v_first, next_attention_shift, next_wkv_state, next_feed_forward_shift = decoder_layer(
                    hidden_states,
                    None,
                    v_first,
                    attention_shift,
                    wkv_state,
                    feed_forward_shift,
                    None,
                    use_unfused_projections=use_unfused_projections,
                    input_ids=input_ids,
                    cu_seq_lens_q=cu_seq_lens_q,
                )
                if cache is not None:
                    cache.update_layer(
                        layer_idx,
                        next_attention_shift,
                        next_wkv_state,
                        next_feed_forward_shift,
                        training=True,
                    )
                if layer_idx == second_last_idx:
                    prev_hidden_states = hidden_states

                if output_hidden_states:
                    all_hidden_states += (hidden_states.view(batch_size, sequence_length, -1),)

            if cache is not None:
                cache.advance(sequence_length, batch_size, hidden_states.device)
            hidden_states = flash_rwkv2.infer_embedding_ln0_forward_varlen(
                hidden_states,
                self.norm.weight.to(hidden_states.dtype),
                self.norm.bias.to(hidden_states.dtype),
                eps=self.config.layer_norm_epsilon,
            )
            if prev_hidden_states is not None:
                autodeco_hidden = flash_rwkv2.infer_embedding_ln0_forward_varlen(
                    prev_hidden_states,
                    self.norm.weight.to(prev_hidden_states.dtype),
                    self.norm.bias.to(prev_hidden_states.dtype),
                    eps=self.config.layer_norm_epsilon,
                )
            else:
                autodeco_hidden = hidden_states
        else:
            hidden_states, residual, all_hidden_states, autodeco_hidden = self._forward_state_only(
                input_ids,
                inputs_embeds,
                cache,
                output_hidden_states,
                return_autodeco_hidden=True,
                rosa_ids=rosa_ids,
            )
            hidden_states = flash_rwkv2.infer_post_norm_output_forward_varlen(
                hidden_states,
                residual,
                self.norm.weight,
                self.norm.bias,
                eps=self.config.layer_norm_epsilon,
            ).view(batch_size, sequence_length, -1)
            autodeco_hidden = (
                autodeco_hidden.view(batch_size, sequence_length, -1) if autodeco_hidden is not None else hidden_states
            )
        if output_hidden_states:
            all_hidden_states = all_hidden_states[:-1] + (hidden_states,)

        return Rwkv7sModelOutputWithPast(
            last_hidden_state=hidden_states,
            cache=cache if use_cache else None,
            hidden_states=all_hidden_states,
            autodeco_hidden_state=autodeco_hidden,
        )


@auto_docstring
class Rwkv7sForCausalLM(Rwkv7sPreTrainedModel, Rwkv7sGenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Rwkv7sConfig):
        super().__init__(config)
        self.generation_config.presence_penalty = 0.0
        self.generation_config.frequency_penalty = 0.0
        self.generation_config.penalty_decay = 0.996
        self.generation_config.use_autodeco = True
        self.model = Rwkv7sModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.autodeco = Rwkv7sAutoDeco(config.hidden_size)
        self.mtp = Rwkv7sMTP(config)
        self.post_init()

    def _project_lm_head(
        self, hidden_states: torch.Tensor, head_spec, use_unfused_projections: bool, *, detach_head: bool = False
    ) -> torch.Tensor:
        if use_unfused_projections:
            if detach_head:
                weight = self.lm_head.weight.detach()
                return F.linear(hidden_states.to(weight.dtype), weight)
            return self.lm_head(hidden_states.to(self.model.norm.weight.dtype))
        weight, lora_a, lora_b, scale, use_nora = head_spec
        if detach_head:
            weight = weight.detach()
            lora_a = None if lora_a is None else lora_a.detach()
            lora_b = None if lora_b is None else lora_b.detach()
        return load_flash_rwkv2().pretrain_head_linear_bf16(
            hidden_states.contiguous(), weight, lora_a=lora_a, lora_b=lora_b, lora_scale=scale, use_nora=use_nora
        )

    def _mtp_loss(
        self,
        hidden_states: torch.Tensor,
        token_embeds: torch.Tensor,
        labels: torch.Tensor,
        head_spec,
        use_unfused_projections: bool,
        *,
        detach_head: bool = False,
        shift_labels: torch.Tensor | None = None,
        cu_seq_lens_q: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = shift_labels if shift_labels is not None else F.pad(labels, (0, 1), value=-100)[..., 1:]
        targets = targets.to(hidden_states.device)
        seq_ids = None
        if cu_seq_lens_q is not None:
            lengths = (cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]).long()
            seq_ids = torch.repeat_interleave(
                torch.arange(lengths.numel(), device=hidden_states.device), lengths
            ).unsqueeze(0)

        h = hidden_states
        total_loss = None
        active_steps = 0
        seq_total = hidden_states.shape[1]
        for step in range(self.config.mtp_depth):
            span = seq_total - (step + 2)
            if span <= 0:
                break
            h = self.mtp(h[:, :span], token_embeds[:, step + 1 : step + 1 + span])
            step_targets = targets[:, step + 1 : step + 1 + span]
            if seq_ids is not None:
                same_seq = seq_ids[:, :span] == seq_ids[:, step + 2 : step + 2 + span]
                step_targets = step_targets.masked_fill(~same_seq, -100)
            if not (step_targets != -100).any():
                continue
            step_logits = self._project_lm_head(
                self.mtp.norm_out(h), head_spec, use_unfused_projections, detach_head=detach_head
            )
            step_loss = F.cross_entropy(
                step_logits.reshape(-1, self.vocab_size).float(), step_targets.reshape(-1), ignore_index=-100
            )
            total_loss = step_loss if total_loss is None else total_loss + step_loss
            active_steps += 1
        if total_loss is None:
            return h.sum().float() * 0.0
        return total_loss / active_steps

    def _autodeco_loss(
        self,
        logits: torch.Tensor,
        temp_logits: torch.Tensor,
        top_p_logits: torch.Tensor,
        labels: torch.Tensor,
        shift_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        targets = shift_labels if shift_labels is not None else F.pad(labels, (0, 1), value=-100)[..., 1:]
        targets = targets.to(logits.device)
        unscaled = logits.detach().float()
        temp = temp_logits.float().clamp_min(1e-2)
        top_p = top_p_logits.float()

        valid = targets != -100
        with torch.no_grad():
            base_probs = torch.softmax(unscaled, dim=-1)
            if self.training:
                correct = base_probs.argmax(dim=-1) == targets.clamp_min(0)
                valid = valid & ~((torch.rand_like(targets, dtype=torch.float32) < 0.6) & correct)
        if not valid.any():
            return temp.sum() * 0.0 + top_p.sum() * 0.0

        scaled = (unscaled / temp)[valid]
        top_p_v = top_p[valid]
        y = targets[valid]
        weights = base_probs[valid].gather(1, y.unsqueeze(-1)).squeeze(-1)

        temp_loss = (F.cross_entropy(scaled, y, reduction="none") * weights).mean()
        probs = torch.softmax(scaled, dim=-1)
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cum_probs = sorted_probs.cumsum(dim=-1)

        decay = torch.exp(-30.0 * F.relu(cum_probs - top_p_v))
        masked = probs * torch.zeros_like(probs).scatter_(-1, sorted_idx, decay)
        log_probs = torch.log(masked / (masked.sum(dim=-1, keepdim=True) + 1e-9) + 1e-9)
        top_p_soft = (F.nll_loss(log_probs, y, reduction="none") * weights).mean()

        rank = (sorted_idx == y.unsqueeze(-1)).float().argmax(dim=-1)
        err = top_p_v.squeeze(-1) - cum_probs.gather(-1, rank.unsqueeze(-1)).squeeze(-1).detach()
        top_p_mse = torch.where(err < 0, 2.0 * err.pow(2), err.pow(2)).mean()
        return temp_loss + top_p_soft + top_p_mse

    @deprecate_kwarg("past_key_values", "5.40.0", new_name="cache", raise_if_both_names=True)
    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        cache: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        train_autodeco: bool | None = None,
        train_mtp: bool | None = None,
        rosa_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Rwkv7sCausalLMOutputWithPast:
        r"""
        train_autodeco (`bool`, *optional*):
            When `True`, runs the backbone and LM head without gradients and computes the AutoDeco loss when `labels`
            are provided. Defaults to `config.train_autodeco`.
        train_mtp (`bool`, *optional*):
            When `True`, runs the backbone without gradients and computes only the LightMTP loss when `labels` are
            provided. Defaults to `config.train_mtp`.
        rosa_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Precomputed token-level ROSA successor IDs (`-1` when unmatched). Computed automatically when omitted.
        """
        if train_autodeco is None:
            train_autodeco = self.config.train_autodeco
        if train_mtp is None:
            train_mtp = self.config.train_mtp
        if train_autodeco and train_mtp:
            raise ValueError("`train_autodeco` and `train_mtp` cannot both be True.")

        head_spec = flash_rwkv2_linear_spec(self.lm_head)
        use_unfused_projections = head_spec is None or not self.model._supports_fused_projections()
        freeze_backbone = train_autodeco or train_mtp
        with torch.no_grad() if freeze_backbone else nullcontext():
            outputs: Rwkv7sModelOutputWithPast = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                rosa_ids=rosa_ids,
                _use_unfused_projections=use_unfused_projections,
                **kwargs,
            )
            selected = slice_logits_to_keep(outputs.last_hidden_state, logits_to_keep)
            unsqueeze_scalar = (
                selected.ndim == 2 and isinstance(logits_to_keep, torch.Tensor) and logits_to_keep.ndim == 0
            )
            if unsqueeze_scalar:
                selected = selected.unsqueeze(1)
            logits = self._project_lm_head(selected, head_spec, use_unfused_projections)

        autodeco_hidden = slice_logits_to_keep(outputs.autodeco_hidden_state, logits_to_keep)
        if unsqueeze_scalar:
            autodeco_hidden = autodeco_hidden.unsqueeze(1)
        temp_logits, top_p_logits = self.autodeco(autodeco_hidden, logits)

        loss = None
        if labels is not None:
            if train_autodeco:
                loss = self._autodeco_loss(logits, temp_logits, top_p_logits, labels, kwargs.get("shift_labels"))
            else:
                if not train_mtp:
                    loss = self.loss_function(
                        logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
                    )
                if train_mtp or self.config.mtp_loss_scaling_factor > 0.0:
                    with torch.no_grad() if train_mtp else nullcontext():
                        token_embeds = (
                            inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
                        )
                    mtp_loss = self._mtp_loss(
                        outputs.last_hidden_state.detach() if train_mtp else outputs.last_hidden_state,
                        token_embeds.detach() if train_mtp else token_embeds,
                        labels,
                        head_spec,
                        use_unfused_projections,
                        detach_head=train_mtp,
                        shift_labels=kwargs.get("shift_labels"),
                        cu_seq_lens_q=kwargs.get("cu_seq_lens_q"),
                    )
                    loss = mtp_loss if train_mtp else loss + self.config.mtp_loss_scaling_factor * mtp_loss

        return Rwkv7sCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            cache=outputs.cache,
            hidden_states=outputs.hidden_states,
            temp_logits=temp_logits,
            top_p_logits=top_p_logits,
        )


__all__ = [
    "Rwkv7sAttention",
    "Rwkv7sAutoDeco",
    "Rwkv7sCache",
    "Rwkv7sCausalLMOutputWithPast",
    "Rwkv7sDecoderLayer",
    "Rwkv7sExperts",
    "Rwkv7sFeedForward",
    "Rwkv7sForCausalLM",
    "Rwkv7sMLP",
    "Rwkv7sMTP",
    "Rwkv7sModel",
    "Rwkv7sModelOutputWithPast",
    "Rwkv7sPreTrainedModel",
    "Rwkv7sTopkRouter",
    "compute_rwkv7s_rosa_ids",
]
