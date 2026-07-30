# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Lazy loaders for the unmodified upstream RWKV-7 CUDA kernels."""

import threading
from pathlib import Path

import torch

from ...utils import logging


logger = logging.get_logger(__name__)

_KERNEL_ROOT = Path(__file__).resolve().parent / "kernels"
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOADED = False
_TRAINING_LOADED = False


def _load_cuda_extension(name: str, sources: list[Path], extra_cflags: list[str], extra_cuda_cflags: list[str]):
    if not torch.cuda.is_available():
        raise RuntimeError(f"The RWKV-7 `{name}` backend requires CUDA.")

    try:
        from torch.utils.cpp_extension import CUDA_HOME, load
    except ImportError as error:
        raise ImportError("Building the RWKV-7 CUDA kernels requires PyTorch C++ extension support.") from error

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME must point to a CUDA toolkit before loading RWKV-7 kernels.")
    missing_sources = [str(source) for source in sources if not source.is_file()]
    if missing_sources:
        raise RuntimeError(f"RWKV-7 kernel sources are missing: {missing_sources}.")

    logger.info("Compiling the RWKV-7 `%s` CUDA extension. This is only done once per environment.", name)
    try:
        load(
            name=name,
            sources=[str(source) for source in sources],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            is_python_module=False,
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to build the RWKV-7 `{name}` CUDA extension. Install Ninja and a compiler supported by the "
            "installed CUDA toolkit, then retry."
        ) from error


def load_rwkv7_inference_kernel():
    """Load vllm-rwkv's canonical FP16-I/O, FP32-state packed-varlen WKV operator."""

    global _INFERENCE_LOADED
    if _INFERENCE_LOADED or hasattr(torch.ops.rwkv7_wkv_fp32_v2, "wkv"):
        _INFERENCE_LOADED = True
        return

    with _LOAD_LOCK:
        if _INFERENCE_LOADED or hasattr(torch.ops.rwkv7_wkv_fp32_v2, "wkv"):
            _INFERENCE_LOADED = True
            return
        source_root = _KERNEL_ROOT / "inference"
        _load_cuda_extension(
            "rwkv7_vllm_wkv_fp32",
            [source_root / "rwkv7_wkv_fp32_v2.cpp", source_root / "rwkv7_wkv_fp32_v2.cu"],
            extra_cflags=["-O3", "-D_IO_FP16_"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas", "-O3", "-D_IO_FP16_"],
        )
        _INFERENCE_LOADED = True


def load_rwkv7_training_kernel():
    """Load RWKV-v7's official BF16 wind-backstepping forward/backward operator."""

    global _TRAINING_LOADED
    if _TRAINING_LOADED or hasattr(torch.ops.wind_backstepping, "forward"):
        _TRAINING_LOADED = True
        return

    with _LOAD_LOCK:
        if _TRAINING_LOADED or hasattr(torch.ops.wind_backstepping, "forward"):
            _TRAINING_LOADED = True
            return
        source_root = _KERNEL_ROOT / "training"
        _load_cuda_extension(
            "rwkv7_wind_backstepping_64_16",
            [source_root / "wkv7_cuda.cu", source_root / "wkv7_op.cpp"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-Xptxas",
                "-O3",
                "--extra-device-vectorization",
                "-D_C_=64",
                "-D_CHUNK_LEN_=16",
            ],
        )
        _TRAINING_LOADED = True


__all__ = ["load_rwkv7_inference_kernel", "load_rwkv7_training_kernel"]
