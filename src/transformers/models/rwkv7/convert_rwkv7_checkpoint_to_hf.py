# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert BlinkDL RWKV-7 checkpoints to the Hugging Face format."""

import argparse
import os
import re

import torch
from huggingface_hub import hf_hub_download, save_torch_state_dict
from torch.nn import functional as F

from ...utils import logging
from .configuration_rwkv7 import Rwkv7Config


logger = logging.get_logger(__name__)

_SUPPORTED_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _projection_ranks(hidden_size: int) -> tuple[int, int, int, int]:
    decay_rank = max(32, int(round(2.5 * hidden_size**0.5 / 32) * 32))
    in_context_rank = max(32, int(round(2.5 * hidden_size**0.5 / 32) * 32))
    value_rank = max(32, int(round(1.7 * hidden_size**0.5 / 32) * 32))
    gate_rank = max(32, int(round(5.0 * hidden_size**0.5 / 32) * 32))
    return decay_rank, in_context_rank, value_rank, gate_rank


def infer_rwkv7_config(
    state_dict: dict[str, torch.Tensor], checkpoint_name: str = "", embedding_layer_norm_fused: bool = True
) -> Rwkv7Config:
    """Infer an RWKV-7 configuration from checkpoint tensor shapes."""

    required_tensors = ("emb.weight", "blocks.0.att.r_k", "blocks.0.ffn.key.weight")
    missing = [name for name in required_tensors if name not in state_dict]
    if missing:
        raise ValueError(f"Checkpoint is missing tensors required for config inference: {missing}.")

    embedding = state_dict["emb.weight"]
    if embedding.ndim != 2:
        raise ValueError(f"`emb.weight` must be two-dimensional, got {tuple(embedding.shape)}.")
    vocab_size, hidden_size = embedding.shape

    local_attention = state_dict["blocks.0.att.r_k"]
    if local_attention.ndim != 2:
        raise ValueError(f"`blocks.0.att.r_k` must be two-dimensional, got {tuple(local_attention.shape)}.")
    num_attention_heads, head_size = local_attention.shape
    if num_attention_heads * head_size != hidden_size:
        raise ValueError(
            "Checkpoint head layout is inconsistent: "
            f"{num_attention_heads} * {head_size} does not equal hidden size {hidden_size}."
        )

    layer_ids = {
        int(match.group(1)) for name in state_dict if (match := re.match(r"blocks\.(\d+)\.", name)) is not None
    }
    if not layer_ids or layer_ids != set(range(max(layer_ids) + 1)):
        raise ValueError(f"Checkpoint layer ids must be contiguous from zero, got {sorted(layer_ids)}.")
    num_hidden_layers = len(layer_ids)

    ffn_weight = state_dict["blocks.0.ffn.key.weight"]
    if ffn_weight.ndim != 2 or ffn_weight.shape[1] != hidden_size:
        raise ValueError(f"Invalid block-0 FFN key shape: {tuple(ffn_weight.shape)}.")
    intermediate_size = ffn_weight.shape[0]

    expected_ranks = _projection_ranks(hidden_size)
    rank_tensors = (
        ("blocks.0.att.w1", "blocks.0.att.w2"),
        ("blocks.0.att.a1", "blocks.0.att.a2"),
        ("blocks.1.att.v1", "blocks.1.att.v2"),
        ("blocks.0.att.g1", "blocks.0.att.g2"),
    )
    if num_hidden_layers < 2:
        raise ValueError("RWKV-7 checkpoints must contain at least two layers to infer the value-residual rank.")
    for expected_rank, (first_name, second_name) in zip(expected_ranks, rank_tensors):
        if first_name not in state_dict or second_name not in state_dict:
            raise ValueError(f"Checkpoint is missing low-rank tensors `{first_name}` or `{second_name}`.")
        first_shape = tuple(state_dict[first_name].shape)
        second_shape = tuple(state_dict[second_name].shape)
        if first_shape != (hidden_size, expected_rank) or second_shape != (expected_rank, hidden_size):
            raise ValueError(
                f"Checkpoint low-rank tensors `{first_name}` and `{second_name}` have shapes "
                f"{first_shape} and {second_shape}; this implementation expects "
                f"{(hidden_size, expected_rank)} and {(expected_rank, hidden_size)}."
            )

    context_match = re.search(r"ctx(\d+)", os.path.basename(checkpoint_name), flags=re.IGNORECASE)
    context_length = int(context_match.group(1)) if context_match else 4096
    return Rwkv7Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        head_size=head_size,
        num_attention_heads=num_attention_heads,
        context_length=context_length,
        embedding_layer_norm_fused=embedding_layer_norm_fused,
        rescale_every=0,
    )


def _expected_shapes(config: Rwkv7Config) -> dict[str, tuple[int, ...]]:
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_heads = config.num_attention_heads
    head_size = config.head_size
    decay_rank, in_context_rank, value_rank, gate_rank = _projection_ranks(hidden_size)

    shapes = {
        "emb.weight": (config.vocab_size, hidden_size),
        "head.weight": (config.vocab_size, hidden_size),
        "ln_out.weight": (hidden_size,),
        "ln_out.bias": (hidden_size,),
    }
    vector_parameters = ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a")
    for layer_id in range(config.num_hidden_layers):
        block_prefix = f"blocks.{layer_id}"
        if layer_id == 0:
            shapes[f"{block_prefix}.ln0.weight"] = (hidden_size,)
            shapes[f"{block_prefix}.ln0.bias"] = (hidden_size,)
        for norm_name in ("ln1", "ln2"):
            shapes[f"{block_prefix}.{norm_name}.weight"] = (hidden_size,)
            shapes[f"{block_prefix}.{norm_name}.bias"] = (hidden_size,)

        attention_prefix = f"{block_prefix}.att"
        for name in vector_parameters:
            if layer_id == 0 and name == "v0":
                continue
            shapes[f"{attention_prefix}.{name}"] = (1, 1, hidden_size)
        shapes.update(
            {
                f"{attention_prefix}.w1": (hidden_size, decay_rank),
                f"{attention_prefix}.w2": (decay_rank, hidden_size),
                f"{attention_prefix}.a1": (hidden_size, in_context_rank),
                f"{attention_prefix}.a2": (in_context_rank, hidden_size),
                f"{attention_prefix}.g1": (hidden_size, gate_rank),
                f"{attention_prefix}.g2": (gate_rank, hidden_size),
                f"{attention_prefix}.r_k": (num_heads, head_size),
                f"{attention_prefix}.ln_x.weight": (hidden_size,),
                f"{attention_prefix}.ln_x.bias": (hidden_size,),
            }
        )
        if layer_id > 0:
            shapes[f"{attention_prefix}.v1"] = (hidden_size, value_rank)
            shapes[f"{attention_prefix}.v2"] = (value_rank, hidden_size)
        for projection_name in ("receptance", "key", "value", "output"):
            shapes[f"{attention_prefix}.{projection_name}.weight"] = (hidden_size, hidden_size)

        ffn_prefix = f"{block_prefix}.ffn"
        shapes[f"{ffn_prefix}.x_k"] = (1, 1, hidden_size)
        shapes[f"{ffn_prefix}.key.weight"] = (intermediate_size, hidden_size)
        shapes[f"{ffn_prefix}.value.weight"] = (hidden_size, intermediate_size)
    return shapes


def convert_state_dict(
    state_dict: dict[str, torch.Tensor], config: Rwkv7Config, fuse_embedding_layer_norm: bool = True
) -> dict[str, torch.Tensor]:
    """Validate and rename a raw RWKV-7 state dict."""

    expected_shapes = _expected_shapes(config)
    source_state_dict = dict(state_dict)
    redundant_block_zero_value_parameters = {
        f"blocks.0.att.{name}" for name in ("v0", "v1", "v2")
    } & source_state_dict.keys()
    for name in redundant_block_zero_value_parameters:
        del source_state_dict[name]
    unexpected_missing = set(expected_shapes) - source_state_dict.keys()
    unexpected = source_state_dict.keys() - expected_shapes.keys()
    if unexpected_missing:
        raise ValueError(f"Checkpoint is missing required tensors: {sorted(unexpected_missing)}.")
    if unexpected:
        raise ValueError(f"Checkpoint contains unsupported tensors: {sorted(unexpected)}.")
    for name, expected_shape in expected_shapes.items():
        tensor = source_state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Checkpoint entry `{name}` is not a tensor.")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Checkpoint tensor `{name}` must have shape {expected_shape}, got {tuple(tensor.shape)}."
            )

    if fuse_embedding_layer_norm:
        embedding = source_state_dict["emb.weight"]
        source_state_dict["emb.weight"] = F.layer_norm(
            embedding.float(),
            (config.hidden_size,),
            weight=source_state_dict["blocks.0.ln0.weight"].float(),
            bias=source_state_dict["blocks.0.ln0.bias"].float(),
        ).to(embedding.dtype)

    converted = {}
    for name, tensor in source_state_dict.items():
        if fuse_embedding_layer_norm and name.startswith("blocks.0.ln0."):
            continue
        if name == "head.weight":
            converted_name = name
        elif name == "emb.weight":
            converted_name = "rwkv7.embeddings.weight"
        else:
            converted_name = f"rwkv7.{name}"
        converted[converted_name] = tensor.detach().clone(memory_format=torch.contiguous_format)
    return converted


def convert_rwkv7_checkpoint_to_hf_format(
    output_dir: str,
    checkpoint_path: str | None = None,
    repo_id: str | None = None,
    checkpoint_file: str | None = None,
    tokenizer_name_or_path: str | None = None,
    push_to_hub: bool = False,
    model_name: str | None = None,
    fuse_embedding_layer_norm: bool = True,
    dtype: str = "float16",
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
):
    """Convert and save either a local checkpoint or a checkpoint hosted on the Hub."""

    if checkpoint_path is not None:
        if repo_id is not None or checkpoint_file is not None:
            raise ValueError("Use either `checkpoint_path` or `repo_id` with `checkpoint_file`, not both.")
        model_file = checkpoint_path
    else:
        if repo_id is None or checkpoint_file is None:
            raise ValueError("Provide `checkpoint_path` or both `repo_id` and `checkpoint_file`.")
        model_file = hf_hub_download(repo_id=repo_id, filename=checkpoint_file)
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Checkpoint does not exist: {model_file}")

    raw_state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    if not isinstance(raw_state_dict, dict):
        raise ValueError("RWKV-7 checkpoint must contain a tensor state dictionary.")
    config = infer_rwkv7_config(raw_state_dict, model_file, embedding_layer_norm_fused=fuse_embedding_layer_norm)
    converted_state_dict = convert_state_dict(raw_state_dict, config, fuse_embedding_layer_norm)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype `{dtype}`. Choose from {sorted(_SUPPORTED_DTYPES)}.")
    target_dtype = _SUPPORTED_DTYPES[dtype]
    converted_state_dict = {
        name: tensor.to(dtype=target_dtype, memory_format=torch.contiguous_format)
        for name, tensor in converted_state_dict.items()
    }
    config.architectures = ["Rwkv7ForCausalLM"]
    config.dtype = target_dtype

    os.makedirs(output_dir, exist_ok=True)
    config.save_pretrained(output_dir)
    save_torch_state_dict(
        converted_state_dict,
        output_dir,
        max_shard_size=max_shard_size,
        safe_serialization=safe_serialization,
    )

    tokenizer = None
    if tokenizer_name_or_path is not None:
        from ..auto.tokenization_auto import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if len(tokenizer) != config.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size {len(tokenizer)} does not match model vocabulary size {config.vocab_size}."
            )
        tokenizer.save_pretrained(output_dir)
    else:
        logger.warning("No tokenizer was saved. Provide an RWKV World tokenizer with %s entries.", config.vocab_size)

    if push_to_hub:
        if model_name is None:
            raise ValueError("`model_name` is required when `push_to_hub=True`.")
        from .modeling_rwkv7 import Rwkv7ForCausalLM

        model = Rwkv7ForCausalLM.from_pretrained(output_dir)
        model.push_to_hub(model_name, max_shard_size=max_shard_size)
        if tokenizer is not None:
            tokenizer.push_to_hub(model_name)

    logger.info("Converted RWKV-7 checkpoint saved to %s.", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a BlinkDL RWKV-7 checkpoint to Hugging Face format.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint_path", type=str, help="Path to a local .pth checkpoint.")
    source.add_argument("--repo_id", type=str, help="Hub repository containing the checkpoint.")
    parser.add_argument("--checkpoint_file", type=str, help="Checkpoint filename when --repo_id is used.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--max_shard_size", type=str, default="5GB")
    parser.add_argument("--dtype", choices=sorted(_SUPPORTED_DTYPES), default="float16")
    parser.add_argument("--no_safe_serialization", action="store_true")
    parser.add_argument("--no_embedding_layer_norm_fusion", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    if args.repo_id is not None and args.checkpoint_file is None:
        parser.error("--checkpoint_file is required with --repo_id")
    convert_rwkv7_checkpoint_to_hf_format(
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        checkpoint_file=args.checkpoint_file,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
        fuse_embedding_layer_norm=not args.no_embedding_layer_norm_fusion,
        dtype=args.dtype,
        max_shard_size=args.max_shard_size,
        safe_serialization=not args.no_safe_serialization,
    )
