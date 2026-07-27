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

Registered as a custom op rather than called directly: a raw Triton launch is on
dynamo's skip list, so calling it from the model breaks the graph at every layer
(21 breaks in a 7.2B forward, measured) and costs about 8% end to end. As an
opaque op it stays inside the single captured graph.

The weight is indexed `[inter, hidden]` so one selected input channel reads one
contiguous row.

The launch geometry below is the best of five structures measured in situ, and
the two that looked most promising on paper are the two that lost:

  tile-scan, (inter/16, hidden/1024) grid   826 us/step   <- shipped
  compacted indices, one extra launch       415 us/step   but +1.3 ms of bubbles
  scalar re-read instead of a reduction     837 us/step
  input axis moved off the grid (512 prog)  1043 us/step
  coarse output tiles, few programs         far worse

Cutting the atomic traffic by using fewer, fatter programs makes it *slower*: the
serial work per program grows faster than the atomics shrink, and a few hundred
programs cannot cover memory latency on this card. Each program re-derives which
of its own input channels are nonzero. Compacting
the surviving indices into a shared list first is the obvious improvement and was
tried: it halves this kernel's GPU time in situ (826 -> 415 us/step on a 7.2B
decode). It is still slower end to end, by 10%. Decoding is a strict dependency
chain -- layer N+1's activation comes from layer N's output -- so the extra
launch, and the barrier its data-dependent trip count forces, costs ~1.3 ms/step
of pipeline bubbles against the 0.43 ms of work it saves. Measure any variant of
this end to end, not as a standalone kernel; three different standalone harnesses
each favoured a configuration that lost in the model, by allowing L2 reuse, by
letting independent iterations overlap, and by using random weights whose
`relu(x)**2` is ~50% dense instead of ~7%. Cross-tile partials land in an fp32 accumulator through atomics,
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


@torch.library.custom_op("rwkv7::sparse_channel_mix_value", mutates_args={"accumulator"})
def triton_sparse_value(activation: torch.Tensor, weight_t: torch.Tensor, accumulator: torch.Tensor) -> torch.Tensor:
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


@triton_sparse_value.register_fake
def _(activation, weight_t, accumulator):
    return torch.empty(weight_t.shape[1], device=activation.device, dtype=activation.dtype)
