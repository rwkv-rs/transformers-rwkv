# Copyright 2024 The HuggingFace Inc. team.
# Copyright (c) 2024-2026 BlinkDL and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
RWKV-7 WKV computation kernels.

Architecture inspired by:
  - Albatross (BlinkDL): fused PDL kernel chain, rankout, wkv+lnx fusion
  - vllm-rwkv: varlen batching with query_start_loc, slot-based state, path dispatch
  - RWKV-LM train_temp: WindBackstepping training kernel

Backend dispatch levels (in priority order):
  Level 1: CUDA fused kernels (Albatross-style: mix6 → rkv → lowrank → rankout → wkv → lnx → out)
  Level 2: CUDA standalone kernels (wkv7 prefill, wkv7s decode, wind_backstepping training)
  Level 3: Pure PyTorch fallback (CPU, no CUDA)

State format (vllm-rwkv standard):
  state[0]: shift (L, 2, B, C)  -- [:, 0, :, :] = att_shift, [:, 1, :, :] = ffn_shift
  state[1]: wkv   (L, B, H, N, N) -- WKV state matrix per layer per batch element
  state[2]: elapsed (B,) -- token counter per batch element (int32)
"""

import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

try:
    from ....utils import logging
    logger = logging.get_logger(__name__)
except ImportError:
    class _FL: info=warning=error=debug=lambda s,*a,**kw:None; warning_once=lambda s,*a,**kw:None
    class _LM: get_logger=staticmethod(lambda n:_FL())
    logging=_LM(); logger=_FL()


# ── Kernel registry ──────────────────────────────────────────────────────────
_KERNEL = {
    "training": None,        # wind_backstepping (forward+backward)
    "prefill_gpt": None,     # wkv7.forward (GPT-mode, stateless)
    "decode_rnn": None,      # wkv7s.forward (RNN-mode, stateful, B=1)
    "fused": None,           # Albatross-style fused chain (mix6+rkv+rankout+wkv+lnx+out)
    "varlen_prefill": None,  # vllm-rwkv packed-varlen prefill
    "varlen_decode": None,   # vllm-rwkv slot-based decode
    "head_size": None,
    "chunk_len": 16,
}
_HAS_CPP = False
try:
    from torch.utils.cpp_extension import load as _cpp_load
    _HAS_CPP = True
except ImportError:
    pass


# ── CUDA compilation ─────────────────────────────────────────────────────────
def _cuda_dir() -> str:
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    cuda = os.path.join(here, "cuda")
    if os.path.isdir(cuda):
        return cuda
    alt = os.path.join(here, "..", "..", "..", "..", "..", "RWKV-v7", "train_temp", "cuda")
    if os.path.isdir(alt):
        return alt
    return cuda


def compile_training(head_size: int, chunk_len: int = 16):
    global _KERNEL
    if not _HAS_CPP:
        raise RuntimeError("cpp_extension unavailable")
    if _KERNEL["training"] is not None and _KERNEL["head_size"] == head_size:
        return
    d = _cuda_dir()
    flags = ["-res-usage", f"-D_C_={head_size}", f"-D_CHUNK_LEN_={chunk_len}",
             "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    _cpp_load(name="wind_backstepping", sources=[f"{d}/wkv7_cuda.cu", f"{d}/wkv7_op.cpp"],
              is_python_module=False, verbose=False, extra_cuda_cflags=flags)
    _KERNEL["training"] = torch.ops.wind_backstepping
    _KERNEL["head_size"] = head_size
    _KERNEL["chunk_len"] = chunk_len
    logger.info(f"WindBackstepping training kernel ready (N={head_size})")


def compile_prefill(head_size: int):
    global _KERNEL
    if not _HAS_CPP: return
    if _KERNEL["prefill_gpt"] is not None and _KERNEL["head_size"] == head_size:
        return
    d = _cuda_dir()
    flags = ["-res-usage", f"-D_N_={head_size}", "--use_fast_math", "-O3",
             "-Xptxas -O3", "--extra-device-vectorization"]
    _cpp_load(name="wkv7", sources=[f"{d}/wkv7_op.cpp", f"{d}/wkv7.cu"],
              is_python_module=False, verbose=False, extra_cuda_cflags=flags)
    _KERNEL["prefill_gpt"] = torch.ops.wkv7
    _KERNEL["head_size"] = head_size
    logger.info(f"wkv7 prefill kernel ready (N={head_size})")


def compile_decode(head_size: int):
    global _KERNEL
    if not _HAS_CPP: return
    if _KERNEL["decode_rnn"] is not None and _KERNEL["head_size"] == head_size:
        return
    d = _cuda_dir()
    flags = ["-res-usage", f"-D_N_={head_size}", "--use_fast_math", "-O3",
             "-Xptxas -O3", "--extra-device-vectorization"]
    _cpp_load(name="wkv7s", sources=[f"{d}/wkv7s_op.cpp", f"{d}/wkv7s.cu"],
              is_python_module=False, verbose=False, extra_cuda_cflags=flags)
    _KERNEL["decode_rnn"] = torch.ops.wkv7s
    _KERNEL["head_size"] = head_size
    logger.info(f"wkv7s decode kernel ready (N={head_size})")


def compile_all(head_size: int, chunk_len: int = 16):
    if not _HAS_CPP or not torch.cuda.is_available():
        logger.warning("CUDA kernels unavailable (no cpp_extension or CUDA)")
        return False
    for fn in [lambda: compile_training(head_size, chunk_len),
               lambda: compile_prefill(head_size),
               lambda: compile_decode(head_size)]:
        try: fn()
        except Exception as e: logger.warning(f"Kernel compile failed: {e}")
    return _KERNEL["prefill_gpt"] is not None or _KERNEL["decode_rnn"] is not None


def kernel_status() -> dict:
    return {k: _KERNEL[k] is not None for k in ["training","prefill_gpt","decode_rnn","fused","varlen_prefill","varlen_decode"]}


# ── Pure PyTorch implementations (always available) ──────────────────────────
def _wkv_prefill_torch(r, w, k, v, a, b, head_size=None, out_dtype=None):
    """GPT-mode: (B,T,C) → (B,T,C).  Slow; for CPU/CUDA fallback only."""
    B, T, C = r.shape
    N = head_size or _KERNEL.get("head_size", 64)
    H = C // N
    r4 = r.view(B, T, H, N).float()
    w4 = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    k4 = k.view(B, T, H, N).float()
    v4 = v.view(B, T, H, N).float()
    a4 = a.view(B, T, H, N).float()
    b4 = b.view(B, T, H, N).float()
    out = torch.zeros(B, T, H, N, device=r.device, dtype=torch.float)
    st = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float)
    for t in range(T):
        kk = k4[:, t].view(B, H, 1, N)
        rr = r4[:, t].view(B, H, N, 1)
        vv = v4[:, t].view(B, H, N, 1)
        aa = a4[:, t].view(B, H, N, 1)
        bb = b4[:, t].view(B, H, 1, N)
        st = st * w4[:, t, :, None, :] + st @ aa @ bb + vv @ kk
        out[:, t] = (st @ rr).view(B, H, N)
    out = out.view(B, T, C)
    return out.to(out_dtype) if out_dtype is not None else out


def _wkv_decode_torch(r, w, k, v, a, b, state, head_size=None, out_dtype=None):
    """RNN-mode: (B,C) + (B,H,N,N) → (B,C) + new (B,H,N,N)."""
    B, C = r.shape
    N = head_size or _KERNEL.get("head_size", 64)
    H = C // N
    r4 = r.view(B, H, N).float()
    w4 = torch.exp(-torch.exp(w.view(B, H, N).float()))
    k4 = k.view(B, H, N).float()
    v4 = v.view(B, H, N).float()
    a4 = a.view(B, H, N).float()
    b4 = b.view(B, H, N).float()
    st = state.float()
    st_new = st * w4.view(B, H, 1, N) + st @ a4.view(B, H, N, 1) @ b4.view(B, H, 1, N) + v4.view(B, H, N, 1) @ k4.view(B, H, 1, N)
    out = (st_new @ r4.view(B, H, N, 1)).view(B, C)
    return (out.to(out_dtype) if out_dtype is not None else out), st_new


# ── autograd.Function wrappers ──────────────────────────────────────────────
class WKV7TrainingFn(torch.autograd.Function):
    """WindBackstepping: forward + backward with chunked state saving."""
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, N = w.shape
        CL = _KERNEL.get("chunk_len", 16)
        pad = (CL - T % CL) % CL
        ctx.orig_T = T; ctx.padded = pad > 0
        if pad:
            w = F.pad(w, (0,0,0,0,0,pad)); q = F.pad(q, (0,0,0,0,0,pad))
            k = F.pad(k, (0,0,0,0,0,pad)); v = F.pad(v, (0,0,0,0,0,pad))
            z = F.pad(z, (0,0,0,0,0,pad)); b = F.pad(b, (0,0,0,0,0,pad))
            T += pad
        y = torch.empty_like(v)
        s = torch.empty(B, H, T//CL, N, N, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
        _KERNEL["training"].forward(w, q, k, v, z, b, y, s, sa)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        if ctx.padded: dy = dy[:, :ctx.orig_T]
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        _KERNEL["training"].backward(w, q, k, v, z, b, dy.contiguous(), s, sa, dw, dq, dk, dv, dz, db)
        if ctx.padded:
            return tuple(x[:, :ctx.orig_T] for x in (dw, dq, dk, dv, dz, db))
        return dw, dq, dk, dv, dz, db


class WKV7PrefillFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b):
        B, T, C = r.shape; H = C // _KERNEL["head_size"]
        y = torch.empty_like(r, memory_format=torch.contiguous_format)
        _KERNEL["prefill_gpt"].forward(B, T, C, H, r, w, k, v, a, b, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Use WindBackstepping for training")


class WKV7DecodeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        T, C = r.shape; H = C // _KERNEL["head_size"]
        y = torch.empty_like(r, memory_format=torch.contiguous_format)
        _KERNEL["decode_rnn"].forward(1, T, C, H, state, r, w, k, v, a, b, y)
        return y, state  # state modified in-place


# ── High-level WKV API ───────────────────────────────────────────────────────
def wkv7_training(q, w, k, v, a, b, head_size):
    """Training: WindBackstepping (bf16 recommended)."""
    if _KERNEL["training"] is None:
        return _wkv_prefill_torch(q, w, k, v, a, b, head_size=head_size, out_dtype=q.dtype)
    B, T, C = q.shape; H = C // head_size; N = head_size
    q4, w4, k4, v4, a4, b4 = [t.view(B, T, H, N).to(torch.bfloat16) for t in (q, w, k, v, a, b)]
    y = WKV7TrainingFn.apply(w4, q4, k4, v4, a4, b4).view(B, T, C).to(q.dtype)
    return y


def wkv7_prefill(r, w, k, v, a, b, head_size, use_cuda=True):
    """GPT-mode prefill."""
    if use_cuda and _KERNEL["prefill_gpt"] is not None and r.device.type == "cuda":
        return WKV7PrefillFn.apply(r, w, k, v, a, b)
    return _wkv_prefill_torch(r, w, k, v, a, b, head_size=head_size, out_dtype=r.dtype)


def wkv7_decode(r, w, k, v, a, b, state, head_size, use_cuda=True):
    """RNN-mode decode (single token, stateful)."""
    B, C = r.shape
    if use_cuda and _KERNEL["decode_rnn"] is not None and B == 1 and _KERNEL["head_size"] == head_size and r.device.type == "cuda":
        out, _ = WKV7DecodeFn.apply(state, r, w, k, v, a, b)
        return out, state
    return _wkv_decode_torch(r, w, k, v, a, b, state, head_size=head_size, out_dtype=r.dtype)


def wkv7_varlen(r, w, k, v, a, b, cu_seqlens, states, head_size, use_cuda=True):
    """
    Varlen batch decode/prefill (vllm-rwkv pattern).

    Args:
        r,w,k,v,a,b: (total_tokens, C) flat tensors.
        cu_seqlens: (B+1,) cumulative lengths.
        states: (B, H, N, N) or (L, B, H, N, N).
        head_size: N.

    Returns:
        output: (total_tokens, C), new_states: same shape as states.
    """
    if not use_cuda or not torch.cuda.is_available() or r.device.type != "cuda":
        B = cu_seqlens.shape[0] - 1
        N = head_size; H = r.shape[1] // N
        outs, new_st = [], torch.empty_like(states) if states.dim() == 4 else None
        for i in range(B):
            s, e = cu_seqlens[i].item(), cu_seqlens[i+1].item()
            st_i = states[i:i+1] if states.dim() == 4 else states[:, i:i+1]
            Ti = e - s
            if Ti == 1:
                o, ns = _wkv_decode_torch(r[s], w[s], k[s], v[s], a[s], b[s],
                                          st_i if st_i.dim() == 4 else st_i[0],
                                          head_size=N, out_dtype=r.dtype)
                outs.append(o.unsqueeze(0))
            else:
                o = _wkv_prefill_torch(r[s:e].unsqueeze(0), w[s:e].unsqueeze(0),
                                       k[s:e].unsqueeze(0), v[s:e].unsqueeze(0),
                                       a[s:e].unsqueeze(0), b[s:e].unsqueeze(0),
                                       head_size=N, out_dtype=r.dtype)
                outs.append(o.squeeze(0))
                # Recompute final state from last token
                _, ns = _wkv_decode_torch(r[e-1], w[e-1], k[e-1], v[e-1], a[e-1], b[e-1],
                                          st_i if st_i.dim()==4 else st_i[0],
                                          head_size=N, out_dtype=r.dtype)
            if states.dim() == 4: new_st[i] = ns
        return torch.cat(outs, 0), new_st
    # Placeholder for real CUDA varlen kernel
    if _KERNEL["varlen_decode"] is not None:
        return _KERNEL["varlen_decode"](r, w, k, v, a, b, cu_seqlens, states)
    # Fallback to per-sequence processing
    return wkv7_varlen(r, w, k, v, a, b, cu_seqlens, states, head_size, use_cuda=False)


# ── State management helpers (vllm-rwkv format) ──────────────────────────────
def init_state(num_layers: int, batch_size: int, hidden_size: int, head_size: int,
               dtype=torch.float16, device="cuda") -> list:
    """Create zero state in vllm-rwkv format:
       [shift(L,2,B,C), wkv(L,B,H,N,N), elapsed(B)]"""
    H = hidden_size // head_size; N = head_size
    return [
        torch.zeros(num_layers, 2, batch_size, hidden_size, dtype=dtype, device=device),
        torch.zeros(num_layers, batch_size, H, N, N, dtype=torch.float32, device=device),
        torch.zeros(batch_size, dtype=torch.int32, device=device),
    ]


def init_state_hf(num_layers: int, batch_size: int, hidden_size: int, head_size: int,
                  dtype=torch.float16, device="cpu") -> list:
    """Create state in HF-compatible format (list of per-layer tuples).
       Compatible with existing Rwkv7Model interface."""
    H = hidden_size // head_size; N = head_size
    result = []
    for _ in range(num_layers):
        result.append((
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
            torch.zeros(batch_size, H, N, N, dtype=torch.float32, device=device),
            torch.zeros(batch_size, hidden_size, dtype=dtype, device=device),
        ))
    return result


__all__ = [
    "compile_all", "compile_training", "compile_prefill", "compile_decode",
    "wkv7_training", "wkv7_prefill", "wkv7_decode", "wkv7_varlen",
    "init_state", "init_state_hf", "kernel_status",
    "WKV7TrainingFn", "WKV7PrefillFn", "WKV7DecodeFn",
]
