# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""Triton sparse channel-mix value projection for RWKV-7 decode.

Kept out of `modeling_rwkv7.py` so the model file stays importable without Triton;
`sparse_channel_mix_value` there dispatches here and falls back to a dense matmul.

The weight is indexed `[inter, hidden]` so one selected input channel reads one
contiguous row. Cross-tile partials land in an fp32 accumulator through atomics,
and the finalize pass re-zeros that accumulator as it casts, so no separate clear
is ever launched.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_value_kernel(
    act_ptr,
    w_ptr,
    acc_ptr,
    inter,
    hidden,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_i, pid_h = tl.program_id(0), tl.program_id(1)
    base_i = pid_i * BLOCK_I
    offs_i = base_i + tl.arange(0, BLOCK_I)
    act = tl.load(act_ptr + offs_i, mask=offs_i < inter, other=0.0).to(tl.float32)
    # one vector load and one reduction decide the whole tile: an all-zero tile
    # returns before touching any weight at all
    if tl.sum(tl.abs(act)) == 0.0:
        return
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for j in tl.static_range(BLOCK_I):
        a = tl.sum(tl.where(tl.arange(0, BLOCK_I) == j, act, 0.0))
        if a != 0.0:
            w = tl.load(w_ptr + (base_i + j) * hidden + offs_h, mask=mask_h, other=0.0).to(tl.float32)
            acc += a * w
    tl.atomic_add(acc_ptr + offs_h, acc, mask=mask_h)


@triton.jit
def _sparse_finalize_kernel(acc_ptr, out_ptr, hidden, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden
    value = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, value.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(acc_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)


def triton_sparse_value(activation, weight_t, accumulator):
    inter, hidden = weight_t.shape
    out = torch.empty(hidden, device=activation.device, dtype=activation.dtype)
    block_h, block_i = 1024, 16
    _sparse_value_kernel[(triton.cdiv(inter, block_i), triton.cdiv(hidden, block_h))](
        activation,
        weight_t,
        accumulator,
        inter,
        hidden,
        BLOCK_H=block_h,
        BLOCK_I=block_i,
        num_warps=8,
    )
    _sparse_finalize_kernel[(triton.cdiv(hidden, 256),)](accumulator, out, hidden, BLOCK=256)
    return out
