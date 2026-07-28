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
(21 breaks in a 7.2B forward, measured) and costs about 8% end to end. As an opaque
op it stays inside the single captured graph.

The weight is indexed `[inter, hidden]` so one selected input channel reads one
contiguous row.

STRUCTURE. The surviving indices are compacted into a shared list first, and the
projection then walks that list with scalar loads and a runtime trip count. Five
structures were measured in situ on a 7.2B decode -- end to end, batch 1, under
`max-autotune` with CUDA graphs engaged:

  compacted indices, SPLIT=128              133.9 tok/s   <- shipped
  tile-scan, (inter/16, hidden/1024) grid   127.8
  scalar re-read instead of a reduction     slower than tile-scan
  input axis moved off the grid, 512 prog   slower than tile-scan
  coarse output tiles, few programs         far worse

The compacted structure was tried once before and rejected, and why is worth
keeping, because the reason was true and has stopped being true. Measured with one
launch per stage and no CUDA graph, its extra launch and the barrier its
data-dependent trip count forces cost ~1.3 ms/step of pipeline bubbles against the
0.43 ms of kernel time it saved. Once the decode compiled into a single graph with
cudagraphs engaged, GPU-busy time equals wall time and there are no bubbles left to
pay -- so the arithmetic that killed it no longer applied to anything.

The first re-test still lost, 122.3 against 127.8, because it inherited SPLIT=8:
128 programs against the tile-scan's 4096. The split decides how much of the card
the walk occupies and had never been swept. It is the whole difference:

  SPLIT   8 -> 122.3      SPLIT  64 -> 133.3
  SPLIT  16 -> 131.1      SPLIT 128 -> 133.9   <- optimum, bracketed both ways
  SPLIT  32 -> 133.3      SPLIT 256 -> 132.9
                          SPLIT 512 -> 132.4

The durable lesson is about the record rather than the kernel: a rejection is only
as good as the conditions it was measured under, and one that does not name them
gets re-read as a fact about the algorithm. 128 is measured at hidden=4096; other
widths have not been swept.

Measure any variant of this end to end, not as a standalone kernel. Three separate
standalone harnesses each favoured a configuration that lost in the model -- by
allowing L2 reuse, by letting independent iterations overlap, and by using random
weights whose `relu(x)**2` is ~50% dense instead of the ~10% a real checkpoint
gives at any decode step.

Cross-tile partials land in an fp32 accumulator through atomics, and the finalize
pass re-zeros both that accumulator and the counter as it casts, so no separate
clear is ever launched.
"""

import torch
import triton
import triton.language as tl


# How many programs the walk is split into. Measured optimum at hidden=4096; it sets
# how much of the card the matvec occupies, so it is the first thing to sweep on a
# model of a different width.
_SPLIT = 128
_BLOCK_H = 256
_BLOCK_C = 256


@triton.jit
def _compact_kernel(act_ptr, idx_ptr, val_ptr, cnt_ptr, inter, BLOCK: tl.constexpr):
    """Append this tile's surviving (index, value) pairs to one shared list."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    act = tl.load(act_ptr + offs, mask=offs < inter, other=0.0).to(tl.float32)
    nonzero = act != 0.0
    count = tl.sum(nonzero.to(tl.int32))
    if count > 0:
        # reserve a contiguous range, then place each survivor at its rank within it
        base = tl.atomic_add(cnt_ptr, count)
        rank = tl.cumsum(nonzero.to(tl.int32)) - nonzero.to(tl.int32)
        tl.store(idx_ptr + base + rank, offs, mask=nonzero)
        tl.store(val_ptr + base + rank, act, mask=nonzero)


@triton.jit
def _sparse_value_kernel(
    idx_ptr,
    val_ptr,
    cnt_ptr,
    w_ptr,
    acc_ptr,
    hidden,
    SPLIT: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Walk a slice of the compacted list, accumulating into one output tile."""
    pid_h, pid_s = tl.program_id(0), tl.program_id(1)
    total = tl.load(cnt_ptr)
    per = (total + SPLIT - 1) // SPLIT
    start = pid_s * per
    stop = tl.minimum(start + per, total)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < hidden
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for i in range(start, stop):
        idx = tl.load(idx_ptr + i)
        a = tl.load(val_ptr + i)
        acc += a * tl.load(w_ptr + idx * hidden + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    if stop > start:
        tl.atomic_add(acc_ptr + offs_h, acc, mask=mask_h)


@triton.jit
def _sparse_finalize_kernel(acc_ptr, out_ptr, cnt_ptr, hidden, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < hidden
    value = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, value.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(acc_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=mask)
    if tl.program_id(0) == 0:
        tl.store(cnt_ptr, 0)


@torch.library.custom_op("rwkv7::sparse_channel_mix_value", mutates_args={"accumulator", "index", "value", "counter"})
def triton_sparse_value(
    activation: torch.Tensor,
    weight_t: torch.Tensor,
    accumulator: torch.Tensor,
    index: torch.Tensor,
    value: torch.Tensor,
    counter: torch.Tensor,
) -> torch.Tensor:
    inter, hidden = weight_t.shape
    out = torch.empty(hidden, device=activation.device, dtype=activation.dtype)
    _compact_kernel[(triton.cdiv(inter, _BLOCK_C),)](activation, index, value, counter, inter, BLOCK=_BLOCK_C)
    _sparse_value_kernel[(triton.cdiv(hidden, _BLOCK_H), _SPLIT)](
        index,
        value,
        counter,
        weight_t,
        accumulator,
        hidden,
        SPLIT=_SPLIT,
        BLOCK_H=_BLOCK_H,
        num_warps=4,
    )
    _sparse_finalize_kernel[(triton.cdiv(hidden, 256),)](accumulator, out, counter, hidden, BLOCK=256)
    return out


@triton_sparse_value.register_fake
def _(activation, weight_t, accumulator, index, value, counter):
    return activation.new_empty(weight_t.shape[1])
