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

from __future__ import annotations

import importlib
from types import ModuleType

import torch

from ..dependency_versions_table import deps


_TRAINING_OPERATORS = (
    "pretrain_tmix_tokenshift_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "pretrain_tmix_wkv7_recurrent_bf16",
    "pretrain_tmix_readout_bf16",
    "pretrain_cmix_bf16",
)

_STATEFUL_TRAINING_OPERATORS = (
    "statetune_tmix_tokenshift_bf16",
    "pretrain_tmix_a_gate_bf16",
    "pretrain_tmix_vres_gate_bf16",
    "pretrain_tmix_kk_pre_bf16",
    "statetune_tmix_wkv7_recurrent_fp32io16",
    "pretrain_tmix_readout_bf16",
    "statetune_cmix_bf16",
)

_INFERENCE_OPERATORS = (
    "infer_embedding_ln0_forward_varlen",
    "infer_tmix_postnorm_tokenshift_forward_varlen",
    "infer_tmix_wkv_prepare_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
    "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
    "infer_tmix_wkv7_chunk_bf16_forward_varlen",
    "infer_tmix_readout_forward_varlen",
    "infer_cmix_forward_varlen",
    "infer_post_norm_output_forward_varlen",
    "infer_head_linear_all_forward_varlen",
    "infer_head_linear_last_forward_varlen",
    "prepare_tmix_wkv7_recurrent_metadata",
)

_OPERATORS_BY_MODE = {
    "training": _TRAINING_OPERATORS,
    "stateful training": _STATEFUL_TRAINING_OPERATORS,
    "inference": _INFERENCE_OPERATORS,
}


def load_flash_rwkv2(mode: str, tensor: torch.Tensor | None = None) -> ModuleType:
    """Load the pinned public FlashRWKV2 API lazily and fail closed on contract drift."""
    required = _OPERATORS_BY_MODE.get(mode)
    if required is None:
        raise ValueError(f"Unsupported FlashRWKV2 mode: {mode!r}.")
    if tensor is not None and (not tensor.is_cuda or tensor.device.type != "cuda"):
        raise RuntimeError(
            f"RWKV-7 {mode} has no product fallback and requires CUDA tensors; got "
            f"device={tensor.device}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}."
        )
    try:
        module = importlib.import_module("flashrwkv2")
    except ImportError as error:
        raise RuntimeError(
            f"RWKV-7 {mode} requires the public `flashrwkv2` root API. Install this checkout with its "
            f"`rwkv` extra; import failed: {error}"
        ) from error

    requirement = deps["FlashRWKV2"]
    expected_version = requirement.split("==", maxsplit=1)[1]
    version = getattr(module, "__version__", "unknown")
    source = getattr(module, "__file__", "unknown")
    if version != expected_version:
        raise RuntimeError(f"RWKV-7 {mode} requires {requirement}; installed version={version}, source={source}.")

    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            f"RWKV-7 {mode} requires FlashRWKV2 public operators {missing}; "
            f"installed version={version}, source={source}."
        )
    return module


__all__ = ["load_flash_rwkv2"]
