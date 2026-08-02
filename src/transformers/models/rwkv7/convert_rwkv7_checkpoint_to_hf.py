# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Convert a legacy RWKV-7 raw PyTorch state dict to a Transformers artifact."""

import argparse
import copy
import os
import re

import torch
from torch.nn import functional as F

from .configuration_rwkv7 import Rwkv7Config
from .modeling_rwkv7 import Rwkv7ForCausalLM


_SUPPORTED_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _projection_ranks(hidden_size: int) -> tuple[int, int, int]:
    return (
        max(32, round(2.5 * hidden_size**0.5 / 32) * 32),
        max(32, round(1.7 * hidden_size**0.5 / 32) * 32),
        max(32, round(5.0 * hidden_size**0.5 / 32) * 32),
    )


def _validate_tensor_state_dict(state_dict) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("RWKV-7 checkpoint must be a non-empty tensor state dictionary.")
    invalid = [
        name
        for name, tensor in state_dict.items()
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor)
    ]
    if invalid:
        raise ValueError(
            f"RWKV-7 checkpoint must directly map string parameter names to tensors; invalid entries: {invalid[:5]}."
        )
    return state_dict


def infer_rwkv7_config(
    state_dict: dict[str, torch.Tensor],
    checkpoint_name: str = "",
    embedding_layer_norm_fused: bool = False,
) -> Rwkv7Config:
    """Infer the public RWKV-7 configuration from validated raw tensor shapes."""
    state_dict = _validate_tensor_state_dict(state_dict)
    required = ("emb.weight", "blocks.0.att.r_k", "blocks.0.ffn.key.weight")
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError(f"Checkpoint is missing tensors required for config inference: {missing}.")

    embedding = state_dict["emb.weight"]
    if embedding.ndim != 2:
        raise ValueError(f"`emb.weight` must be two-dimensional, got {tuple(embedding.shape)}.")
    vocab_size, hidden_size = embedding.shape

    r_k = state_dict["blocks.0.att.r_k"]
    if r_k.ndim != 2:
        raise ValueError(f"`blocks.0.att.r_k` must be two-dimensional, got {tuple(r_k.shape)}.")
    num_attention_heads, head_size = r_k.shape
    if num_attention_heads * head_size != hidden_size:
        raise ValueError(
            f"Checkpoint head layout {num_attention_heads} * {head_size} does not equal hidden size {hidden_size}."
        )

    layer_ids = {int(match.group(1)) for name in state_dict if (match := re.fullmatch(r"blocks\.(\d+)\..+", name))}
    if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError(f"Checkpoint layer ids must be contiguous from zero, got {sorted(layer_ids)}.")
    if len(layer_ids) < 2:
        raise ValueError("RWKV-7 raw checkpoints must contain at least two layers.")

    ffn_key = state_dict["blocks.0.ffn.key.weight"]
    if ffn_key.ndim != 2 or ffn_key.shape[1] != hidden_size:
        raise ValueError(f"Invalid block-0 FFN key shape: {tuple(ffn_key.shape)}.")

    decay_rank, value_rank, gate_rank = _projection_ranks(hidden_size)
    rank_contract = {
        "blocks.0.att.w1": (hidden_size, decay_rank),
        "blocks.0.att.w2": (decay_rank, hidden_size),
        "blocks.0.att.a1": (hidden_size, decay_rank),
        "blocks.0.att.a2": (decay_rank, hidden_size),
        "blocks.1.att.v1": (hidden_size, value_rank),
        "blocks.1.att.v2": (value_rank, hidden_size),
        "blocks.0.att.g1": (hidden_size, gate_rank),
        "blocks.0.att.g2": (gate_rank, hidden_size),
    }
    for name, expected_shape in rank_contract.items():
        if name not in state_dict:
            raise ValueError(f"Checkpoint is missing low-rank tensor `{name}`.")
        if tuple(state_dict[name].shape) != expected_shape:
            raise ValueError(
                f"Checkpoint low-rank tensor `{name}` must have shape {expected_shape}, "
                f"got {tuple(state_dict[name].shape)}."
            )

    context_match = re.search(r"ctx(\d+)", os.path.basename(checkpoint_name), flags=re.IGNORECASE)
    return Rwkv7Config(
        vocab_size=vocab_size,
        context_length=int(context_match.group(1)) if context_match else 4096,
        hidden_size=hidden_size,
        intermediate_size=ffn_key.shape[0],
        num_hidden_layers=len(layer_ids),
        head_size=head_size,
        num_attention_heads=num_attention_heads,
        embedding_layer_norm_fused=embedding_layer_norm_fused,
    )


def _raw_name(model_name: str) -> str:
    if model_name == "model.embeddings.weight":
        return "emb.weight"
    if model_name.startswith("model."):
        return model_name.removeprefix("model.")
    return model_name


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: Rwkv7Config,
    fuse_embedding_layer_norm: bool = False,
) -> dict[str, torch.Tensor]:
    """Validate raw keys and shapes, then map them to the public Transformers model names."""
    source = dict(_validate_tensor_state_dict(state_dict))
    for name in ("v0", "v1", "v2"):
        source.pop(f"blocks.0.att.{name}", None)

    validation_config = copy.deepcopy(config)
    validation_config.embedding_layer_norm_fused = False
    expected_model = Rwkv7ForCausalLM(validation_config)
    expected_shapes = {_raw_name(name): tuple(tensor.shape) for name, tensor in expected_model.state_dict().items()}
    missing = sorted(expected_shapes.keys() - source.keys())
    unexpected = sorted(source.keys() - expected_shapes.keys())
    if missing:
        raise ValueError(f"Checkpoint is missing required tensors: {missing}.")
    if unexpected:
        raise ValueError(f"Checkpoint contains unsupported tensors: {unexpected}.")
    for name, expected_shape in expected_shapes.items():
        if tuple(source[name].shape) != expected_shape:
            raise ValueError(
                f"Checkpoint tensor `{name}` must have shape {expected_shape}, got {tuple(source[name].shape)}."
            )

    if fuse_embedding_layer_norm:
        embedding = source["emb.weight"]
        source["emb.weight"] = F.layer_norm(
            embedding.float(),
            (config.hidden_size,),
            source["blocks.0.ln0.weight"].float(),
            source["blocks.0.ln0.bias"].float(),
        ).to(embedding.dtype)
        del source["blocks.0.ln0.weight"]
        del source["blocks.0.ln0.bias"]

    converted = {}
    for name, tensor in source.items():
        if name == "emb.weight":
            target_name = "model.embeddings.weight"
        elif name == "head.weight":
            target_name = name
        else:
            target_name = f"model.{name}"
        converted[target_name] = tensor.detach().clone(memory_format=torch.contiguous_format)
    return converted


def convert_rwkv7_checkpoint_to_hf_format(
    checkpoint_path: str,
    output_dir: str,
    *,
    fuse_embedding_layer_norm: bool = False,
    dtype: str | None = None,
    safe_serialization: bool = True,
) -> None:
    """Convert one local legacy raw ``.pth`` checkpoint and save a strict-loadable artifact."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if dtype is not None and dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype `{dtype}`. Choose from {sorted(_SUPPORTED_DTYPES)} or preserve it.")

    raw_state_dict = _validate_tensor_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    config = infer_rwkv7_config(
        raw_state_dict,
        checkpoint_path,
        embedding_layer_norm_fused=fuse_embedding_layer_norm,
    )
    converted = convert_state_dict(raw_state_dict, config, fuse_embedding_layer_norm)
    model = Rwkv7ForCausalLM(config)
    try:
        model.load_state_dict(converted, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint does not satisfy the RWKV-7 model contract: {error}") from error
    if dtype is not None:
        model.to(_SUPPORTED_DTYPES[dtype])
    config.architectures = ["Rwkv7ForCausalLM"]
    model.save_pretrained(output_dir, safe_serialization=safe_serialization)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Local legacy RWKV-7 raw .pth state dict.")
    parser.add_argument("--output_dir", required=True, help="Destination for the Transformers artifact.")
    parser.add_argument(
        "--dtype", choices=sorted(_SUPPORTED_DTYPES), help="Optional output dtype; preserve by default."
    )
    parser.add_argument("--fuse_embedding_layer_norm", action="store_true")
    parser.add_argument("--no_safe_serialization", action="store_true")
    args = parser.parse_args(argv)
    convert_rwkv7_checkpoint_to_hf_format(
        args.checkpoint_path,
        args.output_dir,
        fuse_embedding_layer_norm=args.fuse_embedding_layer_norm,
        dtype=args.dtype,
        safe_serialization=not args.no_safe_serialization,
    )


if __name__ == "__main__":
    main()
