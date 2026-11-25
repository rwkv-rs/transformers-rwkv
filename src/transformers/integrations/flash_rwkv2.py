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

from functools import cache
from types import ModuleType

import torch
from torch import nn

from ..dependency_versions_table import deps


@cache
def load_flash_rwkv2() -> ModuleType:
    """Load the pinned FlashRWKV2 public API on first use."""
    try:
        import flashrwkv2
    except ImportError as error:
        raise RuntimeError(
            "RWKV-7 requires the public `flashrwkv2` root API. Install this checkout with its "
            f"`rwkv` extra; import failed: {error}"
        ) from error

    requirement = deps["FlashRWKV2"]
    expected_version = requirement.split("==", maxsplit=1)[1]
    version = getattr(flashrwkv2, "__version__", "unknown")
    if version != expected_version:
        raise RuntimeError(
            f"RWKV-7 requires {requirement}; installed version={version}, source={flashrwkv2.__file__}."
        )
    return flashrwkv2


def flash_rwkv2_linear_spec(
    projection: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, float, bool] | None:
    """Return the weights of a fusable projection, preserving raw NoRA parameters."""
    if type(projection) is nn.Linear:
        if projection.bias is None and projection.weight.dtype == torch.bfloat16:
            return projection.weight.contiguous(), None, None, 1.0, False
        return None

    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        return None
    if not isinstance(projection, LoraLayer):
        return None

    base_layer = projection.get_base_layer()
    if (
        type(base_layer) is not nn.Linear
        or base_layer.bias is not None
        or base_layer.weight.dtype != torch.bfloat16
        or projection.fan_in_fan_out
    ):
        return None
    if projection.disable_adapters:
        if projection.merged:
            return None
        return base_layer.weight.contiguous(), None, None, 1.0, False
    if projection.merged:
        return base_layer.weight.contiguous(), None, None, 1.0, False

    active_adapters = [name for name in projection.active_adapters if name in projection.lora_A]
    if len(active_adapters) > 1:
        return None
    if not active_adapters:
        return base_layer.weight.contiguous(), None, None, 1.0, False

    (adapter_name,) = active_adapters
    variants = getattr(projection, "lora_variant", {})
    use_dora = getattr(projection, "use_dora", {})
    adapter_a = projection.lora_A[adapter_name]
    adapter_b = projection.lora_B[adapter_name]
    dropout = projection.lora_dropout[adapter_name]
    if (
        adapter_name in variants
        or use_dora.get(adapter_name, False)
        or adapter_b.bias is not None
        or (projection.training and getattr(dropout, "p", 0.0) != 0.0)
    ):
        return None
    return (
        base_layer.weight.contiguous(),
        adapter_a.weight.contiguous(),
        adapter_b.weight.contiguous(),
        float(projection.scaling[adapter_name]),
        getattr(projection, "use_nora", {}).get(adapter_name) is True,
    )


__all__ = ["flash_rwkv2_linear_spec", "load_flash_rwkv2"]
