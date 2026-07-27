# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2024 BlinkDL and contributors.
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
"""PyTorch RWKV-7 model."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    auto_docstring,
    logging,
)
from .configuration_rwkv7 import Rwkv7Config


logger = logging.get_logger(__name__)


# ===========================================================================================
# WKV7 Computation Kernels
# ===========================================================================================
# The WKV7 operation is the core recurrent computation in RWKV-v7.
# It computes: state_t = state_{t-1} * w_t + state_{t-1} @ a_t @ b_t + v_t @ k_t^T
#              out_t = state_t @ r_t
#
# Multiple backend implementations are supported:
#   - "pytorch": Pure PyTorch (works everywhere, slower)
#   - "cuda":    Custom CUDA kernel (requires compilation, fast for GPT-mode prefill)
#   - "varlen":  Variable-length CUDA kernel (from vllm-rwkv, for continuous batching)
# ===========================================================================================

try:
    from torch.utils.cpp_extension import load as _cpp_load
    _HAS_CPP_EXTENSION = True
except ImportError:
    _HAS_CPP_EXTENSION = False


_wkv7_cuda_kernel = None
_wkv7_head_size = None


def _try_load_cuda_kernel(head_size: int):
    """Try to load the WKV7 CUDA kernel."""
    global _wkv7_cuda_kernel, _wkv7_head_size
    if _wkv7_cuda_kernel is not None and _wkv7_head_size == head_size:
        return True
    if not _HAS_CPP_EXTENSION:
        return False
    try:
        _cpp_load(
            name="wkv7",
            sources=["cuda/wkv7_op.cpp", "cuda/wkv7.cu"],
            is_python_module=False,
            verbose=False,
            extra_cuda_cflags=[
                "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3",
                "--extra-device-vectorization", f"-D_N_={head_size}"
            ],
        )
        _wkv7_cuda_kernel = torch.ops.wkv7
        _wkv7_head_size = head_size
        return True
    except Exception:
        return False


def _wkv7_pytorch(
    r: torch.Tensor,  # (B, T, C) or (B, C) for single token
    w: torch.Tensor,  # (B, T, C) or (B, C)
    k: torch.Tensor,  # (B, T, C) or (B, C)
    v: torch.Tensor,  # (B, T, C) or (B, C)
    a: torch.Tensor,  # (B, T, C) or (B, C) -- -kk (negative normalized key)
    b: torch.Tensor,  # (B, T, C) or (B, C) -- kk * icl_rate
    state: Optional[torch.Tensor] = None,  # (B, H, N, N)
    head_size: int = 64,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pure PyTorch implementation of WKV7.

    Args:
        r: receptance vector
        w: decay (will be transformed: exp(-exp(w)) internally)
        k: key vector (already modulated with ICL rate)
        v: value vector
        a: negative normalized key (-kk), shape (B, T, C) or (B, C)
        b: kk * a (normalized key times ICL rate), shape (B, T, C) or (B, C)
        state: previous WKV state of shape (B, H, N, N), optional
        head_size: size of each head (N)
        output_dtype: dtype for output

    Returns:
        output: (B, T, C) or (B, C)
        state: (B, H, N, N)
    """
    orig_shape = r.shape
    if r.dim() == 2:
        # Single token: (B, C) -> (B, 1, C)
        r = r.unsqueeze(1)
        w = w.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        a = a.unsqueeze(1)
        b = b.unsqueeze(1)

    B, T, C = r.shape
    H = C // head_size
    N = head_size

    # Reshape to (B, T, H, N)
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))

    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)

    if state is None:
        state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    else:
        state = state.float()

    for t in range(T):
        kk = k[:, t].view(B, H, 1, N)
        rr = r[:, t].view(B, H, N, 1)
        vv = v[:, t].view(B, H, N, 1)
        aa = a[:, t].view(B, H, N, 1)
        bb = b[:, t].view(B, H, 1, N)
        # state = state * w + state @ a @ b + v @ k^T
        state = state * w[:, t, None, :] + state @ aa @ bb + vv @ kk
        out[:, t] = (state @ rr).view(B, H, N)

    out = out.view(orig_shape)
    if output_dtype is not None:
        out = out.to(output_dtype)

    return out, state


def _wkv7_cuda(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    head_size: int = 64,
    output_dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CUDA kernel implementation of WKV7 (GPT-mode, processes full sequence).

    Uses the wkv7 CUDA kernel for fast parallel computation.
    State is returned for continuation in RNN mode.
    """
    B, T, C = r.shape
    H = C // head_size
    N = head_size

    assert r.dtype == w.dtype == k.dtype == v.dtype == a.dtype == b.dtype
    assert all(x.is_contiguous() for x in [r, w, k, v, a, b])

    y = torch.empty((B, T, C), device=k.device, dtype=r.dtype, memory_format=torch.contiguous_format)

    if state is not None:
        # RNN-mode: uses wkv7s kernel with state
        state = state.to(dtype=k.dtype).contiguous()
        _wkv7_cuda_kernel.forward(1, T, C, H, state, r, w, k, v, a, b, y)
    else:
        # GPT-mode: full sequence parallel
        _wkv7_cuda_kernel.forward(B, T, C, H, r, w, k, v, a, b, y)

    if output_dtype is not None and y.dtype != output_dtype:
        y = y.to(output_dtype)

    return y, state


class Rwkv7WKVFunction(torch.autograd.Function):
    """
    Custom autograd Function for WKV7 computation.

    This provides:
    - Forward: Dispatch to CUDA kernel or PyTorch fallback
    - Backward: Placeholder (training uses WindBackstepping kernel via separate path)

    During inference, this is used in torch.no_grad() context.
    """

    @staticmethod
    def forward(
        ctx,
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        state: Optional[torch.Tensor],
        head_size: int,
        use_cuda: bool,
    ):
        ctx.head_size = head_size
        ctx.use_cuda = use_cuda

        if use_cuda and _wkv7_cuda_kernel is not None:
            y, new_state = _wkv7_cuda(r, w, k, v, a, b, state, head_size, output_dtype=r.dtype)
        else:
            y, new_state = _wkv7_pytorch(r, w, k, v, a, b, state, head_size, output_dtype=r.dtype)

        ctx.save_for_backward(r, w, k, v, a, b)
        return y, new_state

    @staticmethod
    def backward(ctx, grad_output, grad_state=None):
        # For inference-only, backward is not needed
        raise NotImplementedError(
            "WKV7 backward pass is not implemented in the HF port. "
            "Use the training-optimized WindBackstepping kernel for training."
        )


def wkv7_attention(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: Optional[torch.Tensor] = None,
    head_size: int = 64,
    use_cuda: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    WKV7 attention computation. Dispatches to the best available backend.

    Args:
        r: receptance (B, T, C) or (B, C)
        w: decay (B, T, C) or (B, C)
        k: key (B, T, C) or (B, C)
        v: value (B, T, C) or (B, C)
        a: -kk, negative normalized key (B, T, C) or (B, C)
        b: kk * icl_rate (B, T, C) or (B, C)
        state: previous WKV state (B, H, N, N), optional
        head_size: size of each attention head
        use_cuda: whether to try CUDA kernel

    Returns:
        output: (B, T, C) or (B, C)
        state: updated state (B, H, N, N), or None
    """
    return Rwkv7WKVFunction.apply(r, w, k, v, a, b, state, head_size, use_cuda)


# ===========================================================================================
# RWKV-7 TimeMix (Attention-like) Module
# ===========================================================================================

class Rwkv7TimeMix(nn.Module):
    """
    RWKV-7 Time Mixing block (attention-like mechanism).

    This block implements the "time mixing" component of RWKV-v7, which includes:
    - Token-shift mixing of x with previous timestep
    - Computation of r (receptance), w (decay), k (key), v (value), a (ICL rate), g (gate)
    - The WKV7 linear attention operator
    - GroupNorm on output
    - Local attention bonus term

    This module is designed to be independently replaceable.
    """

    def __init__(self, config: Rwkv7Config, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        hidden_size = config.hidden_size
        head_size = config.head_size
        self.head_size = head_size
        self.n_head = hidden_size // head_size
        assert hidden_size % self.n_head == 0, f"hidden_size ({hidden_size}) must be divisible by n_head ({self.n_head})"

        H = self.n_head
        N = head_size
        C = hidden_size

        # Time-mix parameters (token-shift): x_{t} mixes with x_{t-1}
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # Decay (w): low-rank decomposition w0 + tanh(xw @ w1) @ w2
        D_DECAY_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))

        # ICL learning rate (a): a0 + (xa @ a1) @ a2
        D_AAA_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))

        # Value residual (v): v0 + (xv @ v1) @ v2
        D_MV_LORA = max(32, int(round((1.7 * (C ** 0.5)) / 32) * 32))
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))

        # Output gate (g): sigmoid(xg @ g1) @ g2
        D_GATE_LORA = max(32, int(round((5 * (C ** 0.5)) / 32) * 32))
        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))

        # Key modulation
        self.k_k = nn.Parameter(torch.empty(1, 1, C))  # key normalization factor
        self.k_a = nn.Parameter(torch.empty(1, 1, C))  # key ICL modulation factor

        # Local attention bonus factor
        self.r_k = nn.Parameter(torch.empty(H, N))

        # Token shift
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        # Linear projections
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)

        # Group normalization on WKV output
        self.ln_x = nn.GroupNorm(H, C, eps=config.group_norm_epsilon)

    def _token_shift(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        Token shift: concatenate last step's x with current x, then compute shifts.

        For training (T > 1): use time_shift padding
        For inference (T == 1) with state: use the state value as the previous step
        """
        B, T, C = x.shape
        if T == 1 and state is not None:
            shifted = state
        else:
            shifted = self.time_shift(x)
            if state is not None and T > 1:
                shifted[:, 0] = state

        xx = shifted - x
        return xx, x[:, -1]  # Return the difference and the last x (for next state)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        att_x_prev: Optional[torch.Tensor] = None,
        att_state: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the TimeMix block.

        Args:
            hidden_states: (B, T, C) input tensor
            v_first: (B, T, C) or (B, C) — the value from the first layer (for residual)
            att_x_prev: (B, C) — previous hidden state for token shift
            att_state: (B, H, N, N) — WKV state from previous step

        Returns:
            output: (B, T, C)
            v_first: updated v_first (pass-through from layer 0, or same)
            att_x_prev_new: (B, C) — new x for next step
            att_state_new: (B, H, N, N) — new WKV state
        """
        B, T, C = hidden_states.shape
        H = self.n_head
        N = self.head_size

        # Token shift
        xx, x_last = self._token_shift(hidden_states, att_x_prev)

        # Apply time-mix
        xr = hidden_states + xx * self.x_r
        xw = hidden_states + xx * self.x_w
        xk = hidden_states + xx * self.x_k
        xv = hidden_states + xx * self.x_v
        xa = hidden_states + xx * self.x_a
        xg = hidden_states + xx * self.x_g

        # Linear projections
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)

        # Decay: soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5

        # Value residual from first layer (skip if layer 0)
        if self.layer_id == 0:
            v_first = v
        else:
            if v_first is not None:
                v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # ICL learning rate
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)

        # Output gate
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # Key normalization and modulation
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # WKV7 operation
        use_cuda = (
            hidden_states.device.type == "cuda"
            and _wkv7_cuda_kernel is not None
            and _wkv7_head_size == self.head_size
        )
        if use_cuda:
            # w is already transformed by the CUDA kernel (softplus + offset)
            wkv_out, att_state_new = _wkv7_cuda(
                r, w, k, v, -kk, kk * a,
                state=att_state, head_size=N,
            )
        else:
            wkv_out, att_state_new = _wkv7_pytorch(
                r, w, k, v, -kk, kk * a,
                state=att_state, head_size=N,
                output_dtype=hidden_states.dtype,
            )

        # Ensure wkv_out has the right batch/time dimensions
        if wkv_out.dim() == 2:
            wkv_out = wkv_out.unsqueeze(1)

        # GroupNorm on output
        wkv_out = self.ln_x(wkv_out.view(B * T, C)).view(B, T, C)

        # Local attention bonus term
        local_bonus = (
            (r.view(B, T, H, N) * k.view(B, T, H, N) * self.r_k).sum(dim=-1, keepdim=True)
            * v.view(B, T, H, N)
        ).view(B, T, C)
        wkv_out = wkv_out + local_bonus

        # Output projection with gate
        output = self.output(wkv_out * g)

        return output, v_first, x_last, att_state_new


# ===========================================================================================
# RWKV-7 ChannelMix (FFN) Module
# ===========================================================================================

class Rwkv7ChannelMix(nn.Module):
    """
    RWKV-7 Channel Mixing block (feed-forward network).

    This block implements the "channel mixing" component of RWKV-v7:
    - Token-shift mixing of x with previous timestep
    - Squared ReLU activation on the key projection
    - Value projection

    This module is designed to be independently replaceable.
    """

    def __init__(self, config: Rwkv7Config, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, hidden_size))

        self.key = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.value = nn.Linear(intermediate_size, hidden_size, bias=False)

    def _token_shift(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Token shift for the FFN input."""
        B, T, C = x.shape
        if T == 1 and state is not None:
            shifted = state
        else:
            shifted = self.time_shift(x)
            if state is not None and T > 1:
                shifted[:, 0] = state

        xx = shifted - x
        return xx, x[:, -1]

    def forward(
        self,
        hidden_states: torch.Tensor,
        ffn_x_prev: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass for the ChannelMix block.

        Args:
            hidden_states: (B, T, C) input tensor
            ffn_x_prev: (B, C) — previous hidden state for token shift

        Returns:
            output: (B, T, C)
            ffn_x_prev_new: (B, C) — new x for next step
        """
        # Token shift
        xx, x_last = self._token_shift(hidden_states, ffn_x_prev)

        # Apply time-mix
        k = hidden_states + xx * self.x_k

        # Squared ReLU activation
        k = torch.relu(self.key(k)) ** 2

        # Value projection
        output = self.value(k)

        return output, x_last


# ===========================================================================================
# RWKV-7 Block (combining TimeMix + ChannelMix)
# ===========================================================================================

class Rwkv7Block(GradientCheckpointingLayer):
    """
    A single RWKV-7 block, combining TimeMix and ChannelMix with residual connections
    and layer normalization.

    Block structure:
        x = x + TimeMix(ln1(x))
        x = x + ChannelMix(ln2(x))
    With an optional pre-ln (ln0) at block 0.
    """

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        # Pre-ln is only used at block 0 (for deep embedding / fused embedding)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.att = Rwkv7TimeMix(config, layer_id)
        self.ffn = Rwkv7ChannelMix(config, layer_id)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: Optional[torch.Tensor] = None,
        att_x_prev: Optional[torch.Tensor] = None,
        att_state: Optional[torch.Tensor] = None,
        ffn_x_prev: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        """
        Forward pass for one RWKV-7 block.

        Returns:
            hidden_states: output tensor
            v_first: value residual from first block (pass-through)
            att_x_prev_new: (B, C) — for state cache
            att_state_new: (B, H, N, N) — for state cache
            ffn_x_prev_new: (B, C) — for state cache
            attention: output of TimeMix (for output_attentions)
        """
        # Pre-ln at block 0
        if self.layer_id == 0:
            hidden_states = self.ln0(hidden_states)

        # TimeMix
        residual = hidden_states
        attn_out, v_first, att_x_prev_new, att_state_new = self.att(
            self.ln1(hidden_states),
            v_first=v_first,
            att_x_prev=att_x_prev,
            att_state=att_state,
        )
        hidden_states = residual + attn_out

        # ChannelMix
        residual = hidden_states
        ffn_out, ffn_x_prev_new = self.ffn(
            self.ln2(hidden_states),
            ffn_x_prev=ffn_x_prev,
        )
        hidden_states = residual + ffn_out

        outputs = (hidden_states, v_first)
        if use_cache:
            outputs += ((att_x_prev_new, att_state_new, ffn_x_prev_new),)
        else:
            outputs += (None,)

        if output_attentions:
            outputs += (attn_out,)
        else:
            outputs += (None,)

        return outputs


# ===========================================================================================
# RWKV-7 Pretrained Model (base class)
# ===========================================================================================

@auto_docstring
class Rwkv7PreTrainedModel(PreTrainedModel):
    """Base class for RWKV-7 models."""

    config_class = Rwkv7Config
    base_model_prefix = "rwkv7"
    _no_split_modules = ["Rwkv7Block"]
    _keep_in_fp32_modules = ["w0"]
    supports_gradient_checkpointing = True
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module: nn.Module):
        """Initialize the weights following the original RWKV-7 initialization scheme."""
        super()._init_weights(module)

        if isinstance(module, Rwkv7TimeMix):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            head_size = module.config.head_size
            H = module.n_head
            N = head_size
            C = hidden_size

            ratio_0_to_1 = layer_id / (num_hidden_layers - 1) if num_hidden_layers > 1 else 0
            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers) if num_hidden_layers > 1 else 0

            # Time-mix parameters: x_*
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            init_val_map = {
                "x_r": 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),
                "x_w": 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),
                "x_k": 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0),
                "x_v": 1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0),
                "x_a": 1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0),
                "x_g": 1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0),
            }

            for name, init_val in init_val_map.items():
                nn.init.constant_(getattr(module, name), 0)  # placeholder
                with torch.no_grad():
                    getattr(module, name).copy_(init_val)

            # Decay parameters: w0, w1, w2
            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            for n in range(C):
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + ratio_0_to_1 ** 0.3)

            nn.init.zeros_(module.w1)
            self._ortho_init(module.w2, 0.1)
            with torch.no_grad():
                module.w0.copy_((www.reshape(1, 1, C) + 0.5 + zigzag * 2.5))

            # ICL rate parameters: a0, a1, a2
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
            nn.init.zeros_(module.a1)
            self._ortho_init(module.a2, 0.1)
            with torch.no_grad():
                module.a0.copy_((torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4))

            # Value residual parameters: v0, v1, v2
            nn.init.zeros_(module.v1)
            self._ortho_init(module.v2, 0.1)
            with torch.no_grad():
                module.v0.copy_((torch.zeros(1, 1, C) + 0.73 - linear * 0.4))

            # Gate parameters: g1, g2
            nn.init.zeros_(module.g1)
            self._ortho_init(module.g2, 0.1)

            # Key modulation
            with torch.no_grad():
                module.k_k.copy_(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
                module.k_a.copy_(torch.zeros(1, 1, C) + 1.02)
                module.r_k.copy_(torch.zeros(H, N) - 0.04)

            # Linear projections: uniform initialization
            module.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            module.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            module.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            module.output.weight.data.zero_()

        elif isinstance(module, Rwkv7ChannelMix):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size

            ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers) if num_hidden_layers > 1 else 0

            ddd = torch.ones(1, 1, hidden_size)
            for i in range(hidden_size):
                ddd[0, 0, i] = i / hidden_size

            with torch.no_grad():
                module.x_k.copy_(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))

            module.key.weight.data.uniform_(-0.5 / (hidden_size ** 0.5), 0.5 / (hidden_size ** 0.5))
            module.value.weight.data.zero_()

        elif isinstance(module, nn.Linear):
            shape = module.weight.shape
            gain = 1.0
            scale = 1.0
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if shape[0] > shape[1]:
                gain = math.sqrt(shape[0] / shape[1])
            if shape[0] == self.config.vocab_size and shape[1] == self.config.hidden_size:
                # Final projection (LM head)
                scale = 0.5
            gain *= scale
            nn.init.orthogonal_(module.weight, gain=gain)

        elif isinstance(module, nn.Embedding):
            shape = module.weight.shape
            gain = 1e-4 * math.sqrt(max(shape[0], shape[1]))
            nn.init.orthogonal_(module.weight, gain=gain)

        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _ortho_init(tensor: nn.Parameter, scale: float):
        """Orthogonal initialization for weight tensors."""
        shape = tensor.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            nn.init.orthogonal_(tensor, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                nn.init.orthogonal_(tensor[i], gain=gain * scale)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, Rwkv7Model):
            module.gradient_checkpointing = value


# ===========================================================================================
# RWKV-7 Model Outputs
# ===========================================================================================

@dataclass
class Rwkv7Output(ModelOutput):
    """
    Base class for RWKV-7 model outputs.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state (`List[torch.FloatTensor]`, *optional*):
            State of the model at the last time step. Can be used in a forward method with the next `input_ids`.
            State is a list of `num_hidden_layers` tuples, each containing:
                (att_x_prev, att_state, ffn_x_prev)
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer).
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for each layer) of the attention output.
    """

    last_hidden_state: torch.FloatTensor | None = None
    state: list | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Rwkv7CausalLMOutput(ModelOutput):
    """
    Base class for RWKV-7 causal LM outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head.
        state (`List`, *optional*):
            State of the model at the last time step.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Hidden-states of the model at the output of each layer.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention outputs of each layer.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    state: list | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


# ===========================================================================================
# RWKV-7 Model
# ===========================================================================================

@auto_docstring
class Rwkv7Model(Rwkv7PreTrainedModel):
    """
    The bare RWKV-7 model outputting raw hidden-states without any specific head on top.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods
    the library implements for all its models.
    """

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [Rwkv7Block(config, layer_id=i) for i in range(config.num_hidden_layers)]
        )
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.layers_are_rescaled = False
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings = new_embeddings

    def _apply_deep_embedding(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Apply deep embedding: fuse embedding output with ln0 of block 0.

        This is done at load time for efficiency. If the model weights were saved
        with deep embedding already applied, this is a no-op.
        """
        if self.config.deep_embedding and hasattr(self.blocks[0], 'ln0'):
            # Deep embedding: embed + pre-ln is fused by normalizing the embedding
            # weights with ln0 parameters. This is applied at load time.
            return inputs_embeds
        return inputs_embeds

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, Rwkv7Output]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        state (`List`, *optional*):
            The state of the model from the previous forward pass. Contains `num_hidden_layers` entries,
            each being a tuple of `(att_x_prev, att_state, ffn_x_prev)`.
        use_cache (`bool`, *optional*):
            If set to `True`, the model returns the updated state.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if self.training == self.layers_are_rescaled:
            self._rescale_layers()

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Initialize state if needed
        batch_size = inputs_embeds.size(0)
        hidden_size = self.config.hidden_size
        head_size = self.config.head_size
        n_head = hidden_size // head_size

        if use_cache and state is None:
            state = []
            for _ in range(self.config.num_hidden_layers):
                state.append((
                    torch.zeros(batch_size, hidden_size, dtype=inputs_embeds.dtype, device=inputs_embeds.device),
                    torch.zeros(batch_size, n_head, head_size, head_size, dtype=torch.float32, device=inputs_embeds.device),
                    torch.zeros(batch_size, hidden_size, dtype=inputs_embeds.dtype, device=inputs_embeds.device),
                ))

        hidden_states = inputs_embeds
        v_first = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        new_state = [] if use_cache else None

        for idx, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # Extract per-layer state
            att_x_prev = state[idx][0] if (use_cache and state is not None) else None
            att_kv_state = state[idx][1] if (use_cache and state is not None) else None
            ffn_x_prev = state[idx][2] if (use_cache and state is not None) else None

            # Gradient checkpointing is handled automatically by GradientCheckpointingLayer.__call__
            hidden_states, v_first, layer_state, attentions = block(
                hidden_states,
                v_first=v_first,
                att_x_prev=att_x_prev,
                att_state=att_kv_state,
                ffn_x_prev=ffn_x_prev,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            if use_cache:
                new_state.append(layer_state)

            # Rescale at inference
            if (
                self.layers_are_rescaled
                and self.config.rescale_every > 0
                and (idx + 1) % self.config.rescale_every == 0
            ):
                hidden_states = hidden_states / 2

            if output_attentions:
                all_self_attentions = all_self_attentions + (attentions,)

        hidden_states = self.ln_out(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                x for x in [hidden_states, new_state, all_hidden_states, all_self_attentions] if x is not None
            )

        return Rwkv7Output(
            last_hidden_state=hidden_states,
            state=new_state,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def _rescale_layers(self):
        """Rescale layer weights for inference stability."""
        if self.layers_are_rescaled == (not self.training):
            return
        if self.config.rescale_every > 0:
            with torch.no_grad():
                for block_id, block in enumerate(self.blocks):
                    if self.training:
                        block.att.output.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                        block.ffn.value.weight.mul_(2 ** int(block_id // self.config.rescale_every))
                    else:
                        block.att.output.weight.div_(2 ** int(block_id // self.config.rescale_every))
                        block.ffn.value.weight.div_(2 ** int(block_id // self.config.rescale_every))
        self.layers_are_rescaled = not self.training


# ===========================================================================================
# RWKV-7 For Causal Language Modeling
# ===========================================================================================

@auto_docstring(
    custom_intro="""
    The RWKV-7 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    """
    RWKV-7 model with a language modeling head for causal LM tasks.

    This model supports:
    - Autoregressive text generation (via `GenerationMixin`)
    - Next-token prediction
    - KV-caching for fast inference (via the `state` mechanism)
    """

    _tied_weights_keys = {"head.weight": "rwkv7.embeddings.weight"}

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.rwkv7 = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new_embeddings):
        self.head = new_embeddings

    def get_input_embeddings(self):
        return self.rwkv7.embeddings

    def set_input_embeddings(self, new_embeddings):
        self.rwkv7.embeddings = new_embeddings

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        state: Optional[List] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, Rwkv7CausalLMOutput]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        state (`List`, *optional*):
            The state of the model from the previous forward pass.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Labels are shifted inside the model.
        use_cache (`bool`, *optional*):
            If set to `True`, the model returns the updated state for fast autoregressive generation.
        logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
            If > 0, only compute the last `logits_to_keep` logits to save memory.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        rwkv7_outputs = self.rwkv7(
            input_ids,
            inputs_embeds=inputs_embeds,
            state=state,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = rwkv7_outputs.last_hidden_state

        # Only compute necessary logits
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )

        if not return_dict:
            output = (logits,) + (rwkv7_outputs[1:])
            return ((loss,) + output) if loss is not None else output

        return Rwkv7CausalLMOutput(
            loss=loss,
            logits=logits,
            state=rwkv7_outputs.state,
            hidden_states=rwkv7_outputs.hidden_states,
            attentions=rwkv7_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        state: Optional[List] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        Prepare inputs for autoregressive generation.

        When `state` is provided, only the last token's input_ids are used.
        """
        if state is not None:
            # In RNN mode, we only need the last token
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "state": state,
            "inputs_embeds": inputs_embeds,
            "use_cache": True,
        }


# ===========================================================================================
# Backend configuration helpers
# ===========================================================================================

def configure_wkv7_backend(head_size: int, backend: str = "auto"):
    """
    Configure the WKV7 computation backend.

    Args:
        head_size: The head size (N) for the model.
        backend: One of "auto", "pytorch", "cuda", or "varlen".
            - "auto": Try CUDA first, fall back to PyTorch.
            - "pytorch": Pure PyTorch implementation (works everywhere).
            - "cuda": Use custom CUDA kernel (requires compilation).
            - "varlen": Use vllm-rwkv variable-length kernel for continuous batching.
    """
    if backend == "auto":
        if torch.cuda.is_available() and _HAS_CPP_EXTENSION:
            _try_load_cuda_kernel(head_size)

    elif backend == "cuda":
        if not _try_load_cuda_kernel(head_size):
            raise RuntimeError(f"Failed to load CUDA kernel for head_size={head_size}")

    elif backend == "varlen":
        # vllm-rwkv integration — defer to vllm for kernel loading
        logger.info("varlen backend selected; ensure vllm-rwkv CUDA kernels are compiled.")

    elif backend == "pytorch":
        pass  # PyTorch fallback is always available

    else:
        raise ValueError(f"Unknown backend: {backend}. Choose from: auto, pytorch, cuda, varlen.")


__all__ = [
    "Rwkv7PreTrainedModel",
    "Rwkv7Model",
    "Rwkv7ForCausalLM",
    "Rwkv7TimeMix",
    "Rwkv7ChannelMix",
    "Rwkv7Block",
    "Rwkv7Output",
    "Rwkv7CausalLMOutput",
    "configure_wkv7_backend",
]
