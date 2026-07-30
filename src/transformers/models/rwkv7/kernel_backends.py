# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Python adapters around existing RWKV-7 kernels."""

from collections.abc import Callable

import torch
from torch.nn import functional as F

from .kernel_loader import load_rwkv7_inference_kernel, load_rwkv7_training_kernel


RWKV7_WKV_BACKENDS: dict[str, Callable] = {}


def register_rwkv7_wkv_backend(name: str, backend: Callable):
    """Register a WKV backend without changing the RWKV-7 model implementation."""

    if not isinstance(name, str) or not name:
        raise ValueError("An RWKV-7 WKV backend name must be a non-empty string.")
    if not callable(backend):
        raise TypeError("An RWKV-7 WKV backend must be callable.")
    RWKV7_WKV_BACKENDS[name] = backend


def _validate_common_inputs(tensors: tuple[torch.Tensor, ...], state: torch.Tensor, head_size: int):
    receptance = tensors[0]
    if receptance.ndim != 3 or any(tensor.shape != receptance.shape for tensor in tensors[1:]):
        raise ValueError("RWKV-7 WKV inputs must share shape (batch, sequence, hidden_size).")
    if any(tensor.device != receptance.device for tensor in tensors[1:]):
        raise ValueError("RWKV-7 WKV inputs must be on the same device.")
    batch_size, _, hidden_size = receptance.shape
    if head_size != 64 or hidden_size % head_size != 0:
        raise ValueError("The bundled RWKV-7 kernels require `head_size=64`.")
    expected_state_shape = (batch_size, hidden_size // head_size, head_size, head_size)
    if state.shape != expected_state_shape:
        raise ValueError(f"RWKV-7 WKV state must have shape {expected_state_shape}, got {tuple(state.shape)}.")


def _pack_inference_inputs(
    tensors: tuple[torch.Tensor, ...],
    state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cu_seq_lens: torch.Tensor | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    receptance = tensors[0]
    batch_size, sequence_length, hidden_size = receptance.shape
    flat_tensors = tuple(tensor.reshape(-1, hidden_size) for tensor in tensors)

    if cu_seq_lens is not None:
        if batch_size != 1:
            raise ValueError("Packed RWKV-7 input must use a single batch row.")
        query_start_loc = cu_seq_lens.to(device=receptance.device, dtype=torch.int32).contiguous()
        request_count = query_start_loc.numel() - 1
        slot_indices = torch.arange(request_count, device=receptance.device, dtype=torch.int32)
        kernel_state = state.new_zeros(request_count, *state.shape[1:])
        return flat_tensors, query_start_loc, slot_indices, kernel_state, None

    if attention_mask is None:
        query_start_loc = torch.arange(
            0,
            (batch_size + 1) * sequence_length,
            sequence_length,
            device=receptance.device,
            dtype=torch.int32,
        )
        slot_indices = torch.arange(batch_size, device=receptance.device, dtype=torch.int32)
        return flat_tensors, query_start_loc, slot_indices, state, None

    valid = attention_mask.to(device=receptance.device, dtype=torch.bool)
    lengths = valid.sum(dim=1, dtype=torch.int32)
    active_rows = torch.nonzero(lengths, as_tuple=False).flatten()
    if active_rows.numel() == 0:
        return tuple(tensor[:0] for tensor in flat_tensors), lengths.new_zeros(1), active_rows.int(), state, valid
    packed_tensors = tuple(tensor[valid.reshape(-1)] for tensor in flat_tensors)
    query_start_loc = torch.cat((lengths.new_zeros(1), lengths[active_rows].cumsum(0))).contiguous()
    return packed_tensors, query_start_loc, active_rows.to(torch.int32), state, valid


def rwkv7_cuda_inference(
    receptance: torch.Tensor,
    raw_decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    negative_key: torch.Tensor,
    scaled_key: torch.Tensor,
    state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    head_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run vllm-rwkv's canonical packed-varlen inference kernel."""

    tensors = (receptance, raw_decay, key, value, negative_key, scaled_key)
    _validate_common_inputs(tensors, state, head_size)
    if receptance.device.type != "cuda" or any(tensor.dtype != torch.float16 for tensor in tensors):
        raise ValueError("The bundled RWKV-7 inference kernel requires CUDA FP16 inputs.")
    if state.dtype != torch.float32:
        raise ValueError("The bundled RWKV-7 inference kernel requires FP32 recurrent state.")
    if cu_seq_lens is not None and attention_mask is not None:
        raise ValueError("`cu_seq_lens` and `attention_mask` are mutually exclusive.")

    packed, query_start_loc, slot_indices, kernel_state, valid = _pack_inference_inputs(
        tensors, state, attention_mask, cu_seq_lens
    )
    if packed[0].shape[0] == 0:
        return torch.zeros_like(receptance), state

    load_rwkv7_inference_kernel()
    packed = tuple(tensor.contiguous() for tensor in packed)
    output = torch.empty_like(packed[0])
    torch.ops.rwkv7_wkv_fp32_v2.wkv(
        query_start_loc,
        slot_indices,
        kernel_state,
        *packed,
        output,
    )

    if cu_seq_lens is not None:
        return output.view_as(receptance), kernel_state[-1:]
    if valid is None:
        return output.view_as(receptance), kernel_state
    unpacked_output = torch.zeros_like(receptance).reshape(-1, receptance.shape[-1])
    unpacked_output[valid.reshape(-1)] = output
    return unpacked_output.view_as(receptance), kernel_state


class _WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_decay, receptance, key, value, negative_key, scaled_key):
        load_rwkv7_training_kernel()
        batch_size, sequence_length, num_heads, head_size = log_decay.shape
        output = torch.empty_like(value)
        saved_state = torch.empty(
            batch_size,
            num_heads,
            sequence_length // 16,
            head_size,
            head_size,
            dtype=torch.float32,
            device=log_decay.device,
        )
        state_dot_negative_key = torch.empty(
            batch_size,
            sequence_length,
            num_heads,
            head_size,
            dtype=torch.float32,
            device=log_decay.device,
        )
        torch.ops.wind_backstepping.forward(
            log_decay,
            receptance,
            key,
            value,
            negative_key,
            scaled_key,
            output,
            saved_state,
            state_dot_negative_key,
        )
        ctx.save_for_backward(
            log_decay,
            receptance,
            key,
            value,
            negative_key,
            scaled_key,
            saved_state,
            state_dot_negative_key,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        inputs = saved[:6]
        gradients = tuple(torch.empty_like(tensor) for tensor in inputs)
        torch.ops.wind_backstepping.backward(
            *inputs,
            grad_output.contiguous(),
            *saved[6:],
            *gradients,
        )
        return gradients


def rwkv7_cuda_training(
    receptance: torch.Tensor,
    raw_decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    negative_key: torch.Tensor,
    scaled_key: torch.Tensor,
    state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    cu_seq_lens: torch.Tensor | None = None,
    head_size: int = 64,
) -> tuple[torch.Tensor, None]:
    """Run RWKV-v7's official BF16 wind-backstepping training kernel."""

    tensors = (receptance, raw_decay, key, value, negative_key, scaled_key)
    _validate_common_inputs(tensors, state, head_size)
    if receptance.device.type != "cuda" or any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise ValueError("The bundled RWKV-7 training kernel requires CUDA BF16 inputs.")
    if attention_mask is not None or cu_seq_lens is not None:
        raise ValueError("The bundled RWKV-7 training kernel does not support padded or packed batches.")
    if torch.count_nonzero(state).item() != 0:
        raise ValueError("The bundled RWKV-7 training kernel only supports a zero initial recurrent state.")

    batch_size, sequence_length, hidden_size = receptance.shape
    num_heads = hidden_size // head_size
    padding = (-sequence_length) % 16
    log_decay = -F.softplus(-raw_decay) - 0.5
    kernel_inputs = (log_decay, receptance, key, value, negative_key, scaled_key)
    if padding:
        kernel_inputs = tuple(F.pad(tensor, (0, 0, 0, padding)) for tensor in kernel_inputs)
    kernel_inputs = tuple(
        tensor.view(batch_size, sequence_length + padding, num_heads, head_size).contiguous()
        for tensor in kernel_inputs
    )
    output = _WindBackstepping.apply(*kernel_inputs)
    return output[:, :sequence_length].reshape(batch_size, sequence_length, hidden_size), None


register_rwkv7_wkv_backend("cuda", rwkv7_cuda_inference)
register_rwkv7_wkv_backend("cuda_training", rwkv7_cuda_training)


def run_rwkv7_wkv(backend: str, training: bool, *args, **kwargs):
    selected_backend = "cuda_training" if backend == "auto" and training else "cuda" if backend == "auto" else backend
    if selected_backend not in RWKV7_WKV_BACKENDS:
        raise ValueError(
            f"Unknown RWKV-7 WKV backend `{selected_backend}`. Available backends: {sorted(RWKV7_WKV_BACKENDS)}."
        )
    return RWKV7_WKV_BACKENDS[selected_backend](*args, **kwargs)


__all__ = ["RWKV7_WKV_BACKENDS", "register_rwkv7_wkv_backend", "run_rwkv7_wkv"]
