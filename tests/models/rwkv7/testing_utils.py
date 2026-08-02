# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch


_last_rwkv7_kernel = None


def _run_sequence(tensors, state):
    r, w, k, v, a, b = tensors
    outputs = []
    current_state = state.float()
    for token_index in range(r.shape[1]):
        decay = w[:, token_index].float().exp().unsqueeze(-1)
        projection = torch.einsum("bhk,bhkv->bhv", a[:, token_index].float(), current_state)
        current_state = (
            decay * current_state
            + b[:, token_index].float().unsqueeze(-1) * projection.unsqueeze(-2)
            + k[:, token_index].float().unsqueeze(-1) * v[:, token_index].float().unsqueeze(-2)
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", r[:, token_index].float(), current_state))
    return torch.stack(outputs, dim=1).to(v.dtype), current_state


def recurrent_rwkv7(
    r,
    w,
    k,
    v,
    a,
    b,
    *,
    initial_state,
    output_final_state,
    cu_seqlens,
    state_indices,
    mode,
):
    """CPU oracle for the public recurrent FLA call shape; this is not a FlashRWKV operator E2E helper."""
    global _last_rwkv7_kernel
    assert output_final_state is True
    assert mode == "fp32io16"
    tensors = (r, w, k, v, a, b)
    if state_indices is None:
        _last_rwkv7_kernel = (
            "pretrain_recurrent_fp32io16_forward"
            if any(tensor.requires_grad for tensor in (*tensors, initial_state))
            else "rwkv7"
        )
        return _run_sequence(tensors, initial_state)

    _last_rwkv7_kernel = "rwkv7_recurrent_stateful"
    outputs = []
    for sequence_index in range(state_indices.numel()):
        start = int(cu_seqlens[sequence_index])
        end = int(cu_seqlens[sequence_index + 1])
        state_index = int(state_indices[sequence_index])
        output, final_state = _run_sequence(
            tuple(tensor[:, start:end] for tensor in tensors),
            initial_state[state_index : state_index + 1],
        )
        initial_state[state_index].copy_(final_state[0])
        outputs.append(output)
    return torch.cat(outputs, dim=1), initial_state


# Explicit chunk/reference comparisons may keep using this test-only oracle name.
chunk_rwkv7 = recurrent_rwkv7


def get_last_rwkv7_provider():
    return "flash_rwkv"


def get_last_rwkv7_kernel():
    return _last_rwkv7_kernel


def _unavailable_fused_operator(*args, **kwargs):
    del args, kwargs
    raise RuntimeError("Synthetic CPU contract does not execute FlashRWKV fused operators.")


class _FlashRwkvPublicApi:
    infer_cmix_mix_fp16 = staticmethod(_unavailable_fused_operator)
    infer_tmix_kk_a_gate_fp16 = staticmethod(_unavailable_fused_operator)
    infer_tmix_lnx_rkvres_xg_fp16 = staticmethod(_unavailable_fused_operator)
    infer_tmix_mix6_fp16 = staticmethod(_unavailable_fused_operator)
    infer_tmix_vres_gate_fp16 = staticmethod(_unavailable_fused_operator)


flash_rwkv = _FlashRwkvPublicApi()
