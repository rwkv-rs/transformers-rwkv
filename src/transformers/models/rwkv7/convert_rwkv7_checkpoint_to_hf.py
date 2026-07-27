# Copyright 2026 The RWKV-7 and HuggingFace Inc. teams.
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
"""Convert a BlinkDL RWKV-7 checkpoint to the Hugging Face format."""

import argparse
import math
import os
import re
from collections.abc import Mapping

import torch
from huggingface_hub import save_torch_state_dict

from .configuration_rwkv7 import Rwkv7Config
from .modeling_rwkv7 import Rwkv7ForCausalLM


_LAYER_PATTERN = re.compile(r"^blocks\.(\d+)\.")
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _unwrap_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint must contain a mapping of parameter names to tensors")
    for container_key in ("state_dict", "model"):
        nested = checkpoint.get(container_key)
        if isinstance(nested, Mapping) and all(isinstance(value, torch.Tensor) for value in nested.values()):
            checkpoint = nested
            break
    state_dict = {str(key): value for key, value in checkpoint.items() if isinstance(value, torch.Tensor)}
    if not state_dict:
        raise ValueError("The checkpoint does not contain any tensors")
    return state_dict


def _required_shape(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Size:
    if key not in state_dict:
        raise ValueError(f"The checkpoint is missing required RWKV-7 tensor {key!r}")
    return state_dict[key].shape


def _infer_num_hidden_layers(state_dict: Mapping[str, torch.Tensor]) -> int:
    layer_ids = sorted(
        {int(match.group(1)) for key in state_dict for match in [_LAYER_PATTERN.match(key)] if match is not None}
    )
    if not layer_ids or layer_ids != list(range(layer_ids[-1] + 1)):
        raise ValueError(f"RWKV-7 block indices must be contiguous from zero, found {layer_ids}")
    return layer_ids[-1] + 1


def _infer_rank(state_dict: Mapping[str, torch.Tensor], key: str, hidden_size: int) -> int:
    shape = _required_shape(state_dict, key)
    if len(shape) != 2 or shape[0] != hidden_size:
        raise ValueError(f"Expected {key!r} to have shape ({hidden_size}, rank), found {tuple(shape)}")
    return shape[1]


def infer_rwkv7_config(
    state_dict: Mapping[str, torch.Tensor],
    *,
    head_dim: int | None = None,
    wkv_mode: str = "fp32io16",
) -> Rwkv7Config:
    """Infer an upstream `Rwkv7Config` from official Bo-style parameter names."""
    embedding_shape = _required_shape(state_dict, "emb.weight")
    if len(embedding_shape) != 2:
        raise ValueError(f"Expected 'emb.weight' to be two-dimensional, found {tuple(embedding_shape)}")
    vocab_size, hidden_size = embedding_shape
    attention_hidden_size = _required_shape(state_dict, "blocks.0.att.receptance.weight")[0]
    intermediate_size = _required_shape(state_dict, "blocks.0.ffn.key.weight")[0]

    r_k_shape = _required_shape(state_dict, "blocks.0.att.r_k")
    inferred_head_dim = r_k_shape[-1] if len(r_k_shape) >= 2 else 64
    head_dim = head_dim or inferred_head_dim
    if attention_hidden_size % head_dim != 0:
        raise ValueError(
            f"TimeMix width {attention_hidden_size} is not divisible by the requested head dimension {head_dim}"
        )
    if math.prod(r_k_shape) != attention_hidden_size:
        raise ValueError(
            f"Expected 'blocks.0.att.r_k' to contain {attention_hidden_size} values, found {math.prod(r_k_shape)}"
        )

    value_rank_key = next((key for key in ("blocks.1.att.v1", "blocks.0.att.v1") if key in state_dict), None)
    deep_embedding_size = 0
    if "blocks.0.ffn.s_emb.weight" in state_dict:
        deep_width = state_dict["blocks.0.ffn.s_emb.weight"].shape[-1]
        deep_embedding_size = math.isqrt(deep_width)
        if deep_embedding_size * deep_embedding_size != deep_width:
            raise ValueError(f"DeepEmbedding width {deep_width} is not a square number")

    floating_tensor = next((value for value in state_dict.values() if value.is_floating_point()), None)
    dtype = floating_tensor.dtype if floating_tensor is not None else torch.float32
    return Rwkv7Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=_infer_num_hidden_layers(state_dict),
        head_dim=head_dim,
        attention_hidden_size=attention_hidden_size,
        intermediate_size=intermediate_size,
        decay_low_rank_dim=_infer_rank(state_dict, "blocks.0.att.w1", hidden_size),
        a_low_rank_dim=_infer_rank(state_dict, "blocks.0.att.a1", hidden_size),
        gate_low_rank_dim=_infer_rank(state_dict, "blocks.0.att.g1", hidden_size),
        value_low_rank_dim=_infer_rank(state_dict, value_rank_key, hidden_size) if value_rank_key else 32,
        deep_embedding_size=deep_embedding_size,
        wkv_mode=wkv_mode,
        architectures=["Rwkv7ForCausalLM"],
        dtype=dtype,
    )


def convert_state_dict(
    state_dict: Mapping[str, torch.Tensor], expected_shapes: Mapping[str, torch.Size] | None = None
) -> dict[str, torch.Tensor]:
    """Apply only the uniform HF body prefix and shape-only singleton normalization.

    RWKV-7's modules deliberately retain BlinkDL's parameter names and matrix
    orientation. Consequently this conversion does not perform semantic key
    renaming or matrix transposition.
    """
    converted = {}
    for source_name, tensor in state_dict.items():
        if source_name in {"blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"}:
            continue
        target_name = (
            source_name if source_name == "head.weight" or source_name.startswith("model.") else f"model.{source_name}"
        )
        if target_name in converted:
            raise ValueError(f"Multiple source tensors map to {target_name!r}")
        expected_shape = expected_shapes.get(target_name) if expected_shapes is not None else None
        if expected_shape is not None and tensor.shape != expected_shape:
            if tensor.numel() != math.prod(expected_shape):
                raise ValueError(
                    f"Tensor {source_name!r} has shape {tuple(tensor.shape)}, expected {tuple(expected_shape)}"
                )
            tensor = tensor.reshape(expected_shape)
        converted[target_name] = tensor
    return converted


def convert_rwkv7_checkpoint_to_hf(
    checkpoint_file: str,
    output_dir: str,
    *,
    dtype: str | None = None,
    head_dim: int | None = None,
    wkv_mode: str = "fp32io16",
    max_shard_size: str = "5GB",
) -> Rwkv7Config:
    checkpoint = torch.load(checkpoint_file, map_location="cpu", mmap=True, weights_only=True)
    source_state_dict = _unwrap_state_dict(checkpoint)
    config = infer_rwkv7_config(source_state_dict, head_dim=head_dim, wkv_mode=wkv_mode)
    target_dtype = _DTYPES[dtype] if dtype is not None else config.dtype
    config.dtype = target_dtype

    with torch.device("meta"):
        model = Rwkv7ForCausalLM(config)
    expected_shapes = {name: parameter.shape for name, parameter in model.state_dict().items()}
    state_dict = convert_state_dict(source_state_dict, expected_shapes)
    missing = sorted(expected_shapes.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected_shapes.keys())
    if missing or unexpected:
        raise ValueError(f"Converted checkpoint key mismatch. Missing: {missing}; unexpected: {unexpected}")

    if dtype is not None:
        state_dict = {
            name: tensor.to(target_dtype) if tensor.is_floating_point() else tensor
            for name, tensor in state_dict.items()
        }
    os.makedirs(output_dir, exist_ok=True)
    config.save_pretrained(output_dir)
    save_torch_state_dict(
        state_dict,
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_file", required=True, help="Path to an official BlinkDL RWKV-7 .pth file")
    parser.add_argument("--output_dir", required=True, help="Directory in which to write the HF checkpoint")
    parser.add_argument("--dtype", choices=sorted(_DTYPES), help="Optionally cast all floating-point weights")
    parser.add_argument("--head_dim", type=int, help="Override head size when r_k is stored flattened")
    parser.add_argument("--wkv_mode", choices=("fp32io16", "fp16"), default="fp32io16")
    parser.add_argument("--max_shard_size", default="5GB")
    args = parser.parse_args()
    convert_rwkv7_checkpoint_to_hf(**vars(args))


if __name__ == "__main__":
    main()
