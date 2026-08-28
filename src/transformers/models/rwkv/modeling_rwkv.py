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
"""PyTorch RWKV-7 model backed exclusively by FlashRWKV2."""

from __future__ import annotations

import math

import torch
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache, LinearAttentionLayer
from ...generation import GenerationMixin
from ...integrations.flash_rwkv2 import flash_rwkv2_linear_spec, load_flash_rwkv2
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling
from .configuration_rwkv import RwkvConfig


_TRAIN_ATTENTION_OPERATORS = (
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_readout_bf16",
    "pretrain_tmix_tokenshift_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "statetune_tmix_tokenshift_bf16",
    "statetune_tmix_wkv7_recurrent_fp32io16",
)
_INFER_ATTENTION_OPERATORS = (
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "infer_tmix_readout_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "infer_tmix_wkv_prepare_forward_varlen",
)
_TRAIN_FEED_FORWARD_OPERATORS = ("pretrain_cmix_bf16", "statetune_cmix_bf16")
_INFER_FEED_FORWARD_OPERATORS = ("infer_cmix_forward_varlen",)


def _validate_attention_mask(attention_mask: torch.Tensor | None, batch_size: int, sequence_length: int) -> None:
    if attention_mask is None:
        return
    if attention_mask.ndim != 2 or attention_mask.shape[0] != batch_size:
        raise ValueError(
            "RWKV-7 accepts only a two-dimensional all-ones attention mask with shape [batch_size, sequence_length]."
        )
    if attention_mask.shape[1] < sequence_length or not bool(torch.all(attention_mask != 0)):
        raise ValueError("RWKV-7 does not support padding or ragged batches; bucket inputs by sequence length.")


class RwkvCache(Cache):
    """RWKV-7 recurrent cache with two token shifts and one FP32 WKV state per layer."""

    _chunk_length = 16

    def __init__(self, config: RwkvConfig):
        config.validate_architecture()
        super().__init__(layers=[LinearAttentionLayer(number_of_states=2) for _ in range(config.num_hidden_layers)])
        self.config = config
        self._seen_tokens = 0
        self._stream_lengths: torch.Tensor | None = None
        self._inference_metadata: dict[tuple, tuple[torch.Tensor, torch.Tensor, int, object]] = {}
        self._training_metadata: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @property
    def is_compileable(self) -> bool:
        return False

    @property
    def batch_size(self) -> int:
        if self._stream_lengths is not None:
            return self._stream_lengths.numel()
        for layer in self.layers:
            for state in (*layer.conv_states.values(), *layer.recurrent_states.values()):
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

    def _invalidate_metadata(self) -> None:
        self._inference_metadata.clear()
        self._training_metadata.clear()

    def layer_states(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layer = self.layers[layer_idx]
        batch_size, _, hidden_size = hidden_states.shape
        expected_shift_shape = (batch_size, hidden_size, 1)
        expected_wkv_shape = (
            batch_size,
            self.config.num_attention_heads,
            self.config.head_size,
            self.config.head_size,
        )

        if not layer.is_conv_states_initialized[0]:
            template = hidden_states[:, -1].unsqueeze(-1)
            layer.lazy_initialization(conv_states=template, state_idx=0, conv_kernel_size=1)
        if not layer.is_recurrent_states_initialized[0]:
            template = torch.empty(expected_wkv_shape, dtype=torch.float32, device=hidden_states.device)
            layer.lazy_initialization(recurrent_states=template, state_idx=0)
        if not layer.is_conv_states_initialized[1]:
            template = hidden_states[:, -1].unsqueeze(-1)
            layer.lazy_initialization(conv_states=template, state_idx=1, conv_kernel_size=1)

        attention_shift = layer.conv_states[0]
        wkv_state = layer.recurrent_states[0]
        feed_forward_shift = layer.conv_states[1]
        if attention_shift.shape != expected_shift_shape or feed_forward_shift.shape != expected_shift_shape:
            raise ValueError(
                f"RWKV-7 cache batch/hidden shape does not match this input: expected {expected_shift_shape}."
            )
        if wkv_state.shape != expected_wkv_shape or wkv_state.dtype != torch.float32:
            raise ValueError(
                f"RWKV-7 WKV cache must have shape {expected_wkv_shape} and dtype float32; "
                f"got shape={tuple(wkv_state.shape)}, dtype={wkv_state.dtype}."
            )
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
        elif self._stream_lengths.numel() != batch_size:
            raise ValueError(
                f"RWKV-7 cache batch size is {self._stream_lengths.numel()}, but the input batch size is {batch_size}."
            )
        self._stream_lengths = self._stream_lengths + num_tokens
        self._seen_tokens += num_tokens

    def recurrent_metadata(
        self, flash_rwkv2, batch_size: int, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, int, object]:
        stream = torch.cuda.current_stream(device)
        key = (device, batch_size, sequence_length, stream.cuda_stream)
        if key not in self._inference_metadata:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * sequence_length,
                sequence_length,
                dtype=torch.int32,
                device=device,
            )
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
            ticket = flash_rwkv2.prepare_tmix_wkv7_recurrent_metadata(
                cu_seqlens,
                state_indices,
                total_tokens=batch_size * sequence_length,
                state_pool_size=batch_size,
                max_seqlen=sequence_length,
            )
            self._inference_metadata[key] = (cu_seqlens, state_indices, sequence_length, ticket)
        return self._inference_metadata[key]

    def training_metadata(
        self, batch_size: int, sequence_length: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (device, batch_size, sequence_length)
        if key not in self._training_metadata:
            chunks_per_sequence = math.ceil(sequence_length / self._chunk_length)
            sequence_chunk_offsets = torch.arange(
                0,
                (batch_size + 1) * chunks_per_sequence,
                chunks_per_sequence,
                dtype=torch.int32,
                device=device,
            )
            sequence_starts = torch.arange(batch_size, dtype=torch.int32, device=device) * sequence_length
            chunk_offsets = torch.arange(chunks_per_sequence, dtype=torch.int32, device=device) * self._chunk_length
            chunk_token_starts = (sequence_starts[:, None] + chunk_offsets[None, :]).flatten()
            sequence_ends = sequence_starts[:, None] + sequence_length
            chunk_token_ends = torch.minimum(
                chunk_token_starts.view(batch_size, -1) + self._chunk_length, sequence_ends
            )
            self._training_metadata[key] = (
                sequence_chunk_offsets.contiguous(),
                chunk_token_starts.contiguous(),
                chunk_token_ends.flatten().contiguous(),
            )
        return self._training_metadata[key]

    def _copy(self, *, detach: bool) -> RwkvCache:
        copied = type(self)(self.config)
        copied._seen_tokens = self._seen_tokens
        if self._stream_lengths is not None:
            stream_lengths = self._stream_lengths.detach() if detach else self._stream_lengths
            copied._stream_lengths = stream_lengths.clone()
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

    def clone(self) -> RwkvCache:
        return self._copy(detach=False)

    def detach(self) -> RwkvCache:
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
        self._invalidate_metadata()

    def batch_repeat_interleave(self, repeats: int) -> None:
        for layer in self.layers:
            for state_idx in range(layer.number_of_states):
                for states_name in ("conv_states", "recurrent_states"):
                    state = getattr(layer, states_name)[state_idx]
                    if state is not None:
                        getattr(layer, states_name)[state_idx] = state.repeat_interleave(repeats, dim=0)
        if self._stream_lengths is not None:
            self._stream_lengths = self._stream_lengths.repeat_interleave(repeats, dim=0)
        self._invalidate_metadata()

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
        self._invalidate_metadata()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self.batch_select_indices(beam_idx)


class RwkvAttention(nn.Module):
    def __init__(self, config: RwkvConfig, layer_idx: int):
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

        self._low_rank_canonical: tuple[torch.Tensor, ...] | None = None
        self._layer_zero_v: tuple[torch.Tensor, ...] | None = None
        self.register_load_state_dict_post_hook(self._clear_runtime_hook)

    def _clear_runtime_hook(self, module, incompatible_keys) -> None:
        self._clear_runtime()

    def _clear_runtime(self) -> None:
        self._low_rank_canonical = None
        self._layer_zero_v = None

    def _apply(self, fn, recurse=True):
        self._clear_runtime()
        return super()._apply(fn, recurse=recurse)

    def _inference_low_rank_weights(self) -> tuple[torch.Tensor, ...]:
        if self._low_rank_canonical is None:
            canonical = [
                self.w1.t().contiguous(),
                self.a1.t().contiguous(),
                self.g1.t().contiguous(),
            ]
            if self.layer_idx == 0:
                rank = self.config.v_low_rank_dim
                v1 = torch.zeros(rank, self.hidden_size, dtype=self.w1.dtype, device=self.w1.device)
                v2 = torch.zeros(self.hidden_size, rank, dtype=self.w1.dtype, device=self.w1.device)
                v0 = torch.zeros(self.hidden_size, dtype=self.w1.dtype, device=self.w1.device)
                v1_runtime = v2
                v2_runtime = v1
                self._layer_zero_v = (v1, v2, v0, v1_runtime, v2_runtime)
            else:
                v1 = self.v1.t().contiguous()
                v2 = self.v2.t().contiguous()
            canonical.extend(
                (
                    v1,
                    self.w2.t().contiguous(),
                    self.a2.t().contiguous(),
                    self.g2.t().contiguous(),
                    v2,
                )
            )
            self._low_rank_canonical = tuple(canonical)
        return self._low_rank_canonical

    def _training_forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor | None,
        wkv_state: torch.Tensor | None,
        training_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        flash_rwkv2 = load_flash_rwkv2(_TRAIN_ATTENTION_OPERATORS, hidden_states, "training")
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"RWKV-7 training requires bfloat16 hidden states, got {hidden_states.dtype}.")
        hidden_states = hidden_states.contiguous()
        if attention_shift is None:
            xr, xw, xk, xv, xa, xg = flash_rwkv2.pretrain_tmix_tokenshift_bf16(
                hidden_states, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g
            )
            next_attention_shift = None
        else:
            xr, xw, xk, xv, xa, xg, next_attention_shift = flash_rwkv2.statetune_tmix_tokenshift_bf16(
                hidden_states,
                attention_shift.contiguous(),
                self.x_r,
                self.x_w,
                self.x_k,
                self.x_v,
                self.x_a,
                self.x_g,
            )

        receptance = self.r_proj(xr).contiguous()
        decay_logits = (self.w0 + torch.tanh(xw @ self.w1) @ self.w2).contiguous()
        key = self.k_proj(xk).contiguous()
        value = self.v_proj(xv).contiguous()
        if self.layer_idx == 0:
            v_first = value
        else:
            if v_first is None:
                raise ValueError("RWKV-7 layers after layer 0 require the first layer's value residual.")
            v12 = ((xv @ self.v1) @ self.v2).contiguous()
            value = flash_rwkv2.pretrain_tmix_vres_gate_bf16(value, v_first, self.v0, v12)

        recurrent_gate = flash_rwkv2.pretrain_tmix_a_gate_bf16(self.a0, ((xa @ self.a1) @ self.a2).contiguous())
        gate = (torch.sigmoid(xg @ self.g1) @ self.g2).contiguous()
        key, negative_direction, scaled_direction = flash_rwkv2.pretrain_tmix_kk_pre_bf16(
            key, self.k_k, recurrent_gate, self.k_a, head_size=self.head_size
        )

        batch_size, sequence_length, _ = hidden_states.shape
        stateful = wkv_state is not None
        if wkv_state is None and sequence_length % RwkvCache._chunk_length == 0:
            recurrent_output = flash_rwkv2.pretrain_tmix_wkv7_recurrent_bf16(
                receptance,
                decay_logits,
                key,
                value,
                negative_direction,
                scaled_direction,
                head_size=self.head_size,
            )
            next_wkv_state = None
        else:
            if training_metadata is None:
                raise ValueError("RWKV-7 StateTune execution requires chunk metadata.")
            if wkv_state is None:
                wkv_state = torch.zeros(
                    batch_size,
                    self.num_heads,
                    self.head_size,
                    self.head_size,
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
            sequence_chunk_offsets, chunk_token_starts, chunk_token_ends = training_metadata
            recurrent_output, final_wkv_state, _, _ = flash_rwkv2.statetune_tmix_wkv7_recurrent_fp32io16(
                wkv_state,
                sequence_chunk_offsets,
                chunk_token_starts,
                chunk_token_ends,
                receptance.view(-1, self.num_heads, self.head_size),
                decay_logits.view(-1, self.num_heads, self.head_size),
                key.view(-1, self.num_heads, self.head_size),
                value.view(-1, self.num_heads, self.head_size),
                negative_direction.view(-1, self.num_heads, self.head_size),
                scaled_direction.view(-1, self.num_heads, self.head_size),
            )
            recurrent_output = recurrent_output.view(batch_size, sequence_length, self.hidden_size)
            next_wkv_state = final_wkv_state if stateful else None

        output = flash_rwkv2.pretrain_tmix_readout_bf16(
            recurrent_output.contiguous(),
            receptance,
            key,
            value,
            self.r_k,
            self.g_norm.weight,
            self.g_norm.bias,
            gate,
            head_size=self.head_size,
        )
        return self.o_proj(output), v_first, next_attention_shift, next_wkv_state

    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor,
        wkv_state: torch.Tensor,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flash_rwkv2 = load_flash_rwkv2(_INFER_ATTENTION_OPERATORS, hidden_states, "inference")
        if hidden_states.dtype != torch.float16 or residual.dtype != torch.float16:
            raise TypeError("RWKV-7 inference requires float16 hidden and residual tensors.")
        cu_seqlens, state_indices, max_seqlen, ticket = inference_metadata
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
        )

        r_weight, r_lora_a, r_lora_b, r_lora_scale = flash_rwkv2_linear_spec(self.r_proj)
        k_weight, k_lora_a, k_lora_b, k_lora_scale = flash_rwkv2_linear_spec(self.k_proj)
        v_weight, v_lora_a, v_lora_b, v_lora_scale = flash_rwkv2_linear_spec(self.v_proj)
        low_rank = self._inference_low_rank_weights()
        if self.layer_idx == 0:
            _, _, v0, v1_runtime, v2_runtime = self._layer_zero_v
        else:
            v0, v1_runtime, v2_runtime = self.v0, self.v1, self.v2
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
                *low_rank,
                v0,
                self.k_k,
                self.a0,
                self.k_a,
                v_first=v_first,
                w1_runtime=self.w1,
                a1_runtime=self.a1,
                g1_runtime=self.g1,
                v1_runtime=v1_runtime,
                w2_runtime=self.w2,
                a2_runtime=self.a2,
                g2_runtime=self.g2,
                v2_runtime=v2_runtime,
                receptance_lora_a=r_lora_a,
                receptance_lora_b=r_lora_b,
                receptance_lora_scale=r_lora_scale,
                key_lora_a=k_lora_a,
                key_lora_b=k_lora_b,
                key_lora_scale=k_lora_scale,
                value_lora_a=v_lora_a,
                value_lora_b=v_lora_b,
                value_lora_scale=v_lora_scale,
                head_size=self.head_size,
                batch_size=state_indices.numel(),
                max_seqlen=max_seqlen,
            )
        )
        recurrent_output = flash_rwkv2.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            receptance.view(-1, self.num_heads, self.head_size),
            decay_logits.view(-1, self.num_heads, self.head_size),
            key.view(-1, self.num_heads, self.head_size),
            value.view(-1, self.num_heads, self.head_size),
            recurrent_a.view(-1, self.num_heads, self.head_size),
            recurrent_b.view(-1, self.num_heads, self.head_size),
            state_pool=wkv_state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            decay_bias=self.w0,
            max_seqlen=max_seqlen,
            validated_metadata=ticket,
        )
        o_weight, o_lora_a, o_lora_b, o_lora_scale = flash_rwkv2_linear_spec(self.o_proj)
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
            head_size=self.head_size,
            batch_size=state_indices.numel(),
            max_seqlen=max_seqlen,
        )
        return output, residual, v_first

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor | None = None,
        attention_shift: torch.Tensor | None = None,
        wkv_state: torch.Tensor | None = None,
        training_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        residual: torch.Tensor | None = None,
        layer_norm: nn.LayerNorm | None = None,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object] | None = None,
    ):
        if hidden_states.ndim == 3:
            return self._training_forward(hidden_states, v_first, attention_shift, wkv_state, training_metadata)
        if residual is None or layer_norm is None or inference_metadata is None:
            raise ValueError("RWKV-7 inference attention requires residual, layer norm, cache states, and metadata.")
        return self._inference_forward(
            hidden_states,
            residual,
            layer_norm,
            v_first,
            attention_shift,
            wkv_state,
            inference_metadata,
        )


class RwkvFeedForward(nn.Module):
    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.x_k = nn.Parameter(torch.empty(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self._value_runtime: torch.Tensor | None = None
        self.register_load_state_dict_post_hook(self._clear_runtime_hook)

    def _clear_runtime_hook(self, module, incompatible_keys) -> None:
        base_layer = self.value.get_base_layer() if hasattr(self.value, "get_base_layer") else self.value
        if base_layer.weight.device != self.key.weight.device or base_layer.weight.dtype != self.key.weight.dtype:
            base_layer.weight.data = base_layer.weight.data.to(self.key.weight.device, dtype=self.key.weight.dtype)
        self._value_runtime = None

    def _apply(self, fn, recurse=True):
        self._value_runtime = None
        return super()._apply(fn, recurse=recurse)

    def _training_forward(
        self, hidden_states: torch.Tensor, feed_forward_shift: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        flash_rwkv2 = load_flash_rwkv2(_TRAIN_FEED_FORWARD_OPERATORS, hidden_states, "training")
        if hidden_states.dtype != torch.bfloat16:
            raise TypeError(f"RWKV-7 training requires bfloat16 hidden states, got {hidden_states.dtype}.")
        if self._value_runtime is not None:
            self._value_runtime = None
        if self.value.weight.device != hidden_states.device or self.value.weight.dtype != hidden_states.dtype:
            self.value.weight.data = self.value.weight.data.to(hidden_states.device, dtype=hidden_states.dtype)
        if feed_forward_shift is None:
            return (
                flash_rwkv2.pretrain_cmix_bf16(
                    hidden_states.contiguous(), self.x_k, self.key.weight, self.value.weight
                ),
                None,
            )
        return flash_rwkv2.statetune_cmix_bf16(
            hidden_states.contiguous(),
            feed_forward_shift.contiguous(),
            self.x_k,
            self.key.weight,
            self.value.weight,
        )

    def _inference_value_weight(self) -> torch.Tensor:
        if self._value_runtime is None:
            value_weight, value_lora_a, value_lora_b, _ = flash_rwkv2_linear_spec(self.value)
            if value_lora_a is not None or value_lora_b is not None:
                raise RuntimeError("Merge ChannelMix value LoRA adapters before RWKV-7 inference.")
            self._value_runtime = value_weight.t().contiguous()
            base_layer = self.value.get_base_layer() if hasattr(self.value, "get_base_layer") else self.value
            base_layer.weight.data = base_layer.weight.data.cpu()
        return self._value_runtime

    def _inference_forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        layer_norm: nn.LayerNorm,
        feed_forward_shift: torch.Tensor,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flash_rwkv2 = load_flash_rwkv2(_INFER_FEED_FORWARD_OPERATORS, hidden_states, "inference")
        key_weight, key_lora_a, key_lora_b, _ = flash_rwkv2_linear_spec(self.key)
        if key_lora_a is not None or key_lora_b is not None:
            raise RuntimeError("Merge ChannelMix key LoRA adapters before RWKV-7 inference.")
        cu_seqlens, state_indices, max_seqlen, ticket = inference_metadata
        hidden_states, residual = flash_rwkv2.infer_cmix_forward_varlen(
            hidden_states.contiguous(),
            residual.contiguous(),
            layer_norm.weight,
            layer_norm.bias,
            self.x_k,
            key_weight,
            self._inference_value_weight(),
            shift_state_pool=feed_forward_shift,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            max_seqlen=max_seqlen,
            eps=self.config.layer_norm_epsilon,
            validated_metadata=ticket,
            deterministic=torch.are_deterministic_algorithms_enabled(),
        )
        return hidden_states, residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        feed_forward_shift: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        layer_norm: nn.LayerNorm | None = None,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object] | None = None,
    ):
        if hidden_states.ndim == 3:
            return self._training_forward(hidden_states, feed_forward_shift)
        if residual is None or layer_norm is None or inference_metadata is None:
            raise ValueError("RWKV-7 inference ChannelMix requires residual, layer norm, cache state, and metadata.")
        return self._inference_forward(hidden_states, residual, layer_norm, feed_forward_shift, inference_metadata)


class RwkvDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: RwkvConfig, layer_idx: int):
        super().__init__()
        self.linear_attn = RwkvAttention(config, layer_idx)
        self.mlp = RwkvFeedForward(config, layer_idx)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        v_first: torch.Tensor | None,
        attention_shift: torch.Tensor | None,
        wkv_state: torch.Tensor | None,
        feed_forward_shift: torch.Tensor | None,
        training_metadata: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        inference_metadata: tuple[torch.Tensor, torch.Tensor, int, object] | None,
    ):
        if hidden_states.ndim == 3:
            attention_output, v_first, next_attention_shift, next_wkv_state = self.linear_attn(
                self.input_layernorm(hidden_states).contiguous(),
                v_first,
                attention_shift,
                wkv_state,
                training_metadata,
            )
            hidden_states = hidden_states + attention_output
            feed_forward_output, next_feed_forward_shift = self.mlp(
                self.post_attention_layernorm(hidden_states).contiguous(), feed_forward_shift
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
        )
        hidden_states, residual = self.mlp(
            layer_input,
            feed_forward_shift,
            residual=attention_output,
            layer_norm=self.post_attention_layernorm,
            inference_metadata=inference_metadata,
        )
        return hidden_states, residual, v_first, None, None


class RwkvPreTrainedModel(PreTrainedModel):
    config_class = RwkvConfig
    config: RwkvConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RwkvDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _is_stateful = True
    _can_compile_fullgraph = False

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Embedding):
            init.uniform_(module.weight, -1e-4, 1e-4)
        elif hasattr(self, "lm_head") and module is self.lm_head:
            gain = (
                0.5 * math.sqrt(self.config.vocab_size / self.config.hidden_size)
                if self.config.vocab_size > self.config.hidden_size
                else 0.5
            )
            init.orthogonal_(module.weight, gain=gain)
        elif isinstance(module, RwkvAttention):
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
        elif isinstance(module, RwkvFeedForward):
            channels = self.config.hidden_size
            ddd = torch.arange(channels, device=module.x_k.device, dtype=torch.float32) / channels
            ratio = 1.0 - module.layer_idx / self.config.num_hidden_layers
            init.copy_(module.x_k, (1 - ddd.pow(ratio**4)).to(module.x_k.dtype))
            init.orthogonal_(module.key.weight)
            init.zeros_(module.value.weight)
        elif isinstance(module, nn.LayerNorm):
            init.ones_(module.weight)
            init.zeros_(module.bias)


class RwkvModel(RwkvPreTrainedModel):
    def __init__(self, config: RwkvConfig):
        if config.architecture_version != "rwkv7":
            raise ValueError(f"RwkvModel requires architecture_version='rwkv7', got {config.architecture_version!r}.")
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layers = nn.ModuleList(
            [RwkvDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_inference_embedding(self) -> None:
        if self.embed_tokens.weight.dtype != torch.bfloat16:
            self.embed_tokens.weight.data = self.embed_tokens.weight.data.to(dtype=torch.bfloat16)
        if self.embedding_norm.weight.dtype != torch.bfloat16:
            self.embedding_norm.weight.data = self.embedding_norm.weight.data.to(dtype=torch.bfloat16)
            self.embedding_norm.bias.data = self.embedding_norm.bias.data.to(dtype=torch.bfloat16)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if is_torchdynamo_compiling():
            raise RuntimeError("RWKV-7 FlashRWKV2 execution does not support torch.compile or export.")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds.")
        if past_key_values is not None and type(past_key_values) is not RwkvCache:
            raise TypeError(f"RWKV-7 accepts only RwkvCache, got {type(past_key_values).__name__}.")
        training_path = self.norm.weight.dtype == torch.bfloat16
        if self.training and not training_path:
            raise TypeError(
                f"RWKV-7 training requires the model in bfloat16, got model dtype {self.norm.weight.dtype}."
            )
        if not training_path and self.norm.weight.dtype != torch.float16:
            raise TypeError(
                "RWKV-7 execution requires a bfloat16 model for training/evaluation or a float16 model for "
                f"inference, got model dtype {self.norm.weight.dtype}."
            )
        if use_cache is None:
            use_cache = past_key_values is not None or (not training_path and self.config.use_cache)
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        _ = position_ids

        if inputs_embeds is None:
            batch_size, sequence_length = input_ids.shape
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        if sequence_length == 0:
            raise ValueError("RWKV-7 requires at least one input token.")
        _validate_attention_mask(attention_mask, batch_size, sequence_length)

        cache_params = past_key_values
        # FlashRWKV2 inference is recurrent even when the caller does not want the
        # state returned, so use a temporary cache for that call.
        if cache_params is None and (use_cache or not training_path):
            cache_params = RwkvCache(self.config)

        if training_path:
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if inputs_embeds.dtype != torch.bfloat16:
                raise TypeError(f"RWKV-7 training requires bfloat16 embeddings, got {inputs_embeds.dtype}.")
            hidden_states = self.embedding_norm(inputs_embeds).contiguous()
            residual = None
            inference_metadata = None
        else:
            self._prepare_inference_embedding()
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16).reshape(-1, self.config.hidden_size).contiguous()
            flash_rwkv2 = load_flash_rwkv2(
                ("infer_embedding_ln0_forward_varlen", "prepare_tmix_wkv7_recurrent_metadata"),
                inputs_embeds,
                "inference",
            )
            hidden_states = flash_rwkv2.infer_embedding_ln0_forward_varlen(
                inputs_embeds,
                self.embedding_norm.weight,
                self.embedding_norm.bias,
                eps=self.config.layer_norm_epsilon,
            ).to(dtype=torch.float16)
            residual = torch.zeros_like(hidden_states)
            inference_metadata = cache_params.recurrent_metadata(
                flash_rwkv2, batch_size, sequence_length, hidden_states.device
            )

        all_hidden_states = () if output_hidden_states else None
        if output_hidden_states:
            initial_hidden = hidden_states if training_path else hidden_states + residual
            all_hidden_states += (initial_hidden.view(batch_size, sequence_length, -1),)

        v_first = None
        shared_training_metadata = None
        if training_path and (cache_params is not None or sequence_length % RwkvCache._chunk_length):
            metadata_owner = cache_params if cache_params is not None else RwkvCache(self.config)
            shared_training_metadata = metadata_owner.training_metadata(
                batch_size, sequence_length, hidden_states.device
            )
        for layer_idx, decoder_layer in enumerate(self.layers):
            attention_shift = wkv_state = feed_forward_shift = None
            if cache_params is not None:
                state_reference = hidden_states
                if not training_path:
                    state_reference = hidden_states.view(batch_size, sequence_length, -1)
                attention_shift, wkv_state, feed_forward_shift = cache_params.layer_states(layer_idx, state_reference)

            hidden_states, layer_value, next_attention_shift, next_wkv_state, next_feed_forward_shift = decoder_layer(
                hidden_states,
                residual,
                v_first,
                attention_shift,
                wkv_state,
                feed_forward_shift,
                shared_training_metadata,
                inference_metadata,
            )
            if training_path:
                v_first = layer_value
                if cache_params is not None:
                    cache_params.update_layer(
                        layer_idx,
                        next_attention_shift,
                        next_wkv_state,
                        next_feed_forward_shift,
                        training=self.training,
                    )
            else:
                residual = layer_value
                v_first = next_attention_shift
                cache_params.mark_layer_updated(layer_idx)

            if output_hidden_states:
                layer_hidden = hidden_states if training_path else hidden_states + residual
                all_hidden_states += (layer_hidden.view(batch_size, sequence_length, -1),)

        if cache_params is not None:
            cache_params.advance(sequence_length, batch_size, hidden_states.device)

        if training_path:
            hidden_states = self.norm(hidden_states)
        else:
            flash_rwkv2 = load_flash_rwkv2("infer_post_norm_output_forward_varlen", hidden_states, "inference")
            hidden_states = flash_rwkv2.infer_post_norm_output_forward_varlen(
                hidden_states,
                residual,
                self.norm.weight,
                self.norm.bias,
                eps=self.config.layer_norm_epsilon,
            ).view(batch_size, sequence_length, -1)
        if output_hidden_states:
            all_hidden_states = all_hidden_states[:-1] + (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=cache_params if use_cache else None,
            hidden_states=all_hidden_states,
        )


@auto_docstring
class RwkvForCausalLM(RwkvPreTrainedModel, GenerationMixin):
    def __init__(self, config: RwkvConfig):
        super().__init__(config)
        self.model = RwkvModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        if labels is not None and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
            raise ValueError("Set logits_to_keep=0 when labels are provided.")
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        if self.training:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])
        else:
            if isinstance(logits_to_keep, int):
                if logits_to_keep < 0:
                    raise ValueError("logits_to_keep must be non-negative.")
                selected = hidden_states if logits_to_keep == 0 else hidden_states[:, -logits_to_keep:, :]
            else:
                selected = hidden_states[:, logits_to_keep, :]
                if selected.ndim == 2:
                    selected = selected.unsqueeze(1)
            batch_size, selected_length, hidden_size = selected.shape
            selected = selected.reshape(-1, hidden_size).contiguous()
            operators = (
                "infer_head_linear_last_forward_varlen"
                if isinstance(logits_to_keep, int) and logits_to_keep == 1
                else "infer_head_linear_all_forward_varlen"
            )
            flash_rwkv2 = load_flash_rwkv2(operators, selected, "inference")
            if operators == "infer_head_linear_last_forward_varlen":
                logits = flash_rwkv2.infer_head_linear_last_forward_varlen(
                    selected, self.lm_head.weight, tokens_count=batch_size
                )
            else:
                logits = flash_rwkv2.infer_head_linear_all_forward_varlen(selected, self.lm_head.weight)
            logits = logits.view(batch_size, selected_length, self.vocab_size)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )


__all__ = [
    "RwkvAttention",
    "RwkvCache",
    "RwkvDecoderLayer",
    "RwkvFeedForward",
    "RwkvForCausalLM",
    "RwkvModel",
    "RwkvPreTrainedModel",
]
