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
from torch import nn

from ..dependency_versions_table import deps


def load_flash_rwkv2(
    operators: str | tuple[str, ...], tensor: torch.Tensor | None = None, mode: str = "execution"
) -> ModuleType:
    """Load the pinned FlashRWKV2 public API and validate only the operators used by this path."""
    if tensor is not None and not tensor.is_cuda:
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

    required = (operators,) if isinstance(operators, str) else operators
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise RuntimeError(
            f"RWKV-7 {mode} requires FlashRWKV2 public operators {missing}; "
            f"installed version={version}, source={source}."
        )
    return module


def flash_rwkv2_linear_spec(
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, float]:
    """Resolve a bias-free Linear and at most one active, unmerged vanilla PEFT LoRA adapter."""
    if isinstance(projection, nn.Linear):
        if projection.bias is not None:
            raise RuntimeError("RWKV-7 FlashRWKV2 inference projections do not support a base bias.")
        return projection.weight.contiguous(), None, None, 1.0

    try:
        from peft.tuners.lora import LoraLayer
    except ImportError as error:
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference only supports torch.nn.Linear or PEFT LoraLayer projections."
        ) from error
    if not isinstance(projection, LoraLayer):
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference only supports torch.nn.Linear or PEFT LoraLayer projections; "
            f"got {type(projection).__name__}."
        )

    base_layer = projection.get_base_layer()
    if not isinstance(base_layer, nn.Linear) or base_layer.bias is not None:
        raise RuntimeError("RWKV-7 FlashRWKV2 LoRA requires a bias-free torch.nn.Linear base layer.")
    if projection.fan_in_fan_out:
        raise RuntimeError("RWKV-7 FlashRWKV2 LoRA inference requires fan_in_fan_out=False.")
    if projection.disable_adapters:
        if projection.merged:
            raise RuntimeError("Unmerge the LoRA projection before disabling adapters for RWKV-7 inference.")
        return base_layer.weight.contiguous(), None, None, 1.0
    if projection.merged:
        return base_layer.weight.contiguous(), None, None, 1.0

    active_adapters = projection.active_adapters
    if isinstance(active_adapters, str):
        active_adapters = [active_adapters]
    variants = getattr(projection, "lora_variant", {})
    use_dora = getattr(projection, "use_dora", {})
    active = []
    for adapter_name in active_adapters:
        if adapter_name not in projection.lora_A:
            continue
        if adapter_name in variants or use_dora.get(adapter_name, False):
            raise RuntimeError(
                "RWKV-7 FlashRWKV2 inference supports vanilla LoRA only; "
                f"adapter {adapter_name!r} must be merged before inference."
            )
        adapter_a = projection.lora_A[adapter_name]
        adapter_b = projection.lora_B[adapter_name]
        if adapter_b.bias is not None:
            raise RuntimeError(
                "RWKV-7 FlashRWKV2 inference does not support lora_bias; "
                f"adapter {adapter_name!r} must be merged before inference."
            )
        dropout = projection.lora_dropout[adapter_name]
        if projection.training and getattr(dropout, "p", 0.0) != 0.0:
            raise RuntimeError("Unmerged LoRA dropout is only supported in eval mode.")
        active.append((adapter_a, adapter_b, float(projection.scaling[adapter_name])))

    if not active:
        return base_layer.weight.contiguous(), None, None, 1.0
    if len(active) != 1:
        raise RuntimeError(
            "RWKV-7 FlashRWKV2 inference supports exactly one active vanilla LoRA adapter; "
            "merge multiple adapters before inference."
        )
    adapter_a, adapter_b, scale = active[0]
    return base_layer.weight.contiguous(), adapter_a.weight.contiguous(), adapter_b.weight.contiguous(), scale


__all__ = ["flash_rwkv2_linear_spec", "load_flash_rwkv2"]
