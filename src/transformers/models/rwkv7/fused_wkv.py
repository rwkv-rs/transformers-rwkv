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
"""Single-token WKV in one Triton kernel: the state is read once and written once.

The portable recurrence in `modeling_rwkv7.rwkv7_recurrent` touches the state four
times for one decoded token -- once to form `sa = (-kk) @ S`, once to update `S`,
once to read it back for `r @ S`, once to store it. That is the natural way to write
it and it is free at batch 1, where the state is a megabyte a layer and invisible
next to the weights being streamed.

It stops being free with batch. At batch 256 and 7.2B's shape the state is
`[256, 64, 64, 64]` -- 268 MB a layer, 8.6 GB across 32 layers -- and profiling put
77% of a decode step inside the time-mix while its own projections were 5.5% of it.
The cost was the state, not the weights, and four passes over it rather than two.

This does the whole update with the `[head_dim, head_dim]` tile resident. Measured on
one RTX 5090 at 7.2B: 5.7x on the recurrence at batch 256, and the whole model's
batched decode goes from 55% of the albatross reference to 85%.

Called as a plain function rather than wrapped in a `custom_op`. Dynamo traces a
`@triton.jit` launch natively, and the wrapper is opaque to it -- with the state
declared as mutated, inductor stops issuing CUDA graphs for the region. That is
invisible at batch 256, where the kernel wins anyway, and costs everything at batch
1: 138.9 tok/s wrapped becomes 57.9, which is eager speed. `sparse_channel_mix` is a
`custom_op` because its three kernels and its compaction buffers are genuinely
opaque; one traced kernel is better off without the wrapper.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _wkv_one_kernel(
    state_ptr,
    r_ptr,
    w_ptr,
    k_ptr,
    v_ptr,
    kk_ptr,
    a_ptr,
    out_ptr,
    stride_sb,
    stride_sh,
    stride_si,
    stride_sj,
    stride_vb,
    stride_vh,
    N: tl.constexpr,
):
    """One program per (batch, head).

    `N` is `head_dim` and is a compile-time constant, so the `[N, N]` tile lives in
    registers for the whole body: 64x64 fp32 is 16 KB, which is what makes reading
    the state once sufficient.
    """
    batch = tl.program_id(0)
    head = tl.program_id(1)

    i = tl.arange(0, N)
    j = tl.arange(0, N)
    vec = batch * stride_vb + head * stride_vh
    r = tl.load(r_ptr + vec + i).to(tl.float32)
    w = tl.load(w_ptr + vec + i).to(tl.float32)
    k = tl.load(k_ptr + vec + i).to(tl.float32)
    v = tl.load(v_ptr + vec + j).to(tl.float32)
    kk = tl.load(kk_ptr + vec + i).to(tl.float32)
    a = tl.load(a_ptr + vec + i).to(tl.float32)

    # Every axis of the state is strided, the last one included. Hardcoding unit
    # stride there reads the tile transposed when the caller hands over anything
    # that is not contiguous, and the result is wrong by 80-98% with nothing
    # raised: the shape is right, so neither the kernel nor Triton notices.
    off = batch * stride_sb + head * stride_sh + i[:, None] * stride_si + j[None, :] * stride_sj
    state = tl.load(state_ptr + off).to(tl.float32)

    # Same three lines as the reference, in the same order and in fp32: `sa` uses the
    # PRE-update state and the output uses the POST-update one.
    sa = tl.sum((-kk)[:, None] * state, axis=0)
    state = tl.exp(w)[:, None] * state + (kk * a)[:, None] * sa[None, :] + k[:, None] * v[None, :]
    out = tl.sum(r[:, None] * state, axis=0)

    tl.store(state_ptr + off, state.to(state_ptr.dtype.element_ty))
    tl.store(out_ptr + vec + j, out.to(out_ptr.dtype.element_ty))


def fused_wkv_one(
    r: torch.Tensor,
    w_log: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    batch, _, heads, head_dim = r.shape
    # The grid comes from `r`, so a state that disagrees with it is read at offsets
    # belonging to another row and returns a plausible answer for the wrong sequence.
    # Checked rather than assumed: the kernel cannot tell, and neither can the caller.
    if tuple(state.shape) != (batch, heads, head_dim, head_dim):
        raise ValueError(
            f"state has shape {tuple(state.shape)}, but the vectors imply {(batch, heads, head_dim, head_dim)}"
        )
    out = torch.empty(batch, heads, head_dim, device=r.device, dtype=r.dtype)
    flat = [t.reshape(batch, heads, head_dim).contiguous() for t in (r, w_log, k, v, kk, a)]
    _wkv_one_kernel[(batch, heads)](
        state,
        *flat,
        out,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        state.stride(3),
        out.stride(0),
        out.stride(1),
        N=head_dim,
    )
    return out.view(batch, 1, heads, head_dim)
