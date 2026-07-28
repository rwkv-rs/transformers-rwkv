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
"""Convert a RWKV-7 checkpoint from BlinkDL format to the Hugging Face format."""

import argparse
import gc
import json
import os

import torch
from huggingface_hub import hf_hub_download, split_torch_state_dict_into_shards
from torch.nn import functional as F

from .configuration_rwkv7 import Rwkv7Config
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_utils import WEIGHTS_INDEX_NAME
from transformers.utils import logging


logger = logging.get_logger(__name__)


# ===========================================================================================
# Size presets for standard RWKV-7 models
# ===========================================================================================

NUM_HIDDEN_LAYERS_MAPPING = {
    "0.1B": 12,
    "0.4B": 24,
    "0.6B": 24,
    "1.5B": 24,
    "3B": 32,
    "7B": 32,
}

HIDDEN_SIZE_MAPPING = {
    "0.1B": 768,
    "0.4B": 1024,
    "0.6B": 1536,
    "1.5B": 2048,
    "3B": 2560,
    "7B": 4096,
}

VOCAB_SIZE = 65536  # Standard RWKV vocabulary size
HEAD_SIZE = 64  # Standard head size for RWKV-7


def convert_state_dict(
    state_dict: dict,
    config: Rwkv7Config,
    apply_deep_embedding: bool = True,
) -> dict:
    """
    Convert a raw RWKV-7 checkpoint state_dict to HuggingFace format.

    Key mappings (checkpoint → HF):
        emb.weight                          → rwkv7.embeddings.weight
        blocks.i.ln0.weight/bias            → rwkv7.blocks.i.ln0.weight/bias
        blocks.i.ln1.weight/bias            → rwkv7.blocks.i.ln1.weight/bias
        blocks.i.ln2.weight/bias            → rwkv7.blocks.i.ln2.weight/bias
        blocks.i.att.x_r/x_w/x_k/x_v/x_a/x_g → rwkv7.blocks.i.att.x_*
        blocks.i.att.w0/w1/w2               → rwkv7.blocks.i.att.w*
        blocks.i.att.a0/a1/a2               → rwkv7.blocks.i.att.a*
        blocks.i.att.v0/v1/v2               → rwkv7.blocks.i.att.v*
        blocks.i.att.g1/g2                  → rwkv7.blocks.i.att.g*
        blocks.i.att.k_k/k_a                → rwkv7.blocks.i.att.k_*
        blocks.i.att.r_k                    → rwkv7.blocks.i.att.r_k
        blocks.i.att.receptance.weight      → rwkv7.blocks.i.att.receptance.weight
        blocks.i.att.key.weight             → rwkv7.blocks.i.att.key.weight
        blocks.i.att.value.weight           → rwkv7.blocks.i.att.value.weight
        blocks.i.att.output.weight          → rwkv7.blocks.i.att.output.weight
        blocks.i.att.ln_x.weight/bias       → rwkv7.blocks.i.att.ln_x.weight/bias
        blocks.i.ffn.key.weight             → rwkv7.blocks.i.ffn.key.weight
        blocks.i.ffn.value.weight           → rwkv7.blocks.i.ffn.value.weight
        blocks.i.ffn.x_k                    → rwkv7.blocks.i.ffn.x_k
        ln_out.weight/bias                  → rwkv7.ln_out.weight/bias
        head.weight                         → head.weight

    Args:
        state_dict: Raw checkpoint state_dict.
        config: RWKV-7 model configuration.
        apply_deep_embedding: Whether to fuse emb.weight with blocks.0.ln0.

    Returns:
        Converted state_dict with HF-compatible keys.
    """
    new_state_dict = {}
    hidden_size = config.hidden_size
    head_size = config.head_size

    for name, param in state_dict.items():
        # Skip non-tensor entries
        if not isinstance(param, torch.Tensor):
            continue

        # Squeeze 1-dims (some original checkpoints have extra dims)
        param = param.squeeze()

        # ============================================================================
        # Global parameters
        # ============================================================================

        if name == "emb.weight":
            new_name = "rwkv7.embeddings.weight"

        elif name == "head.weight":
            new_name = "head.weight"

        elif name == "ln_out.weight":
            new_name = "rwkv7.ln_out.weight"

        elif name == "ln_out.bias":
            new_name = "rwkv7.ln_out.bias"

        # ============================================================================
        # Block-level parameters
        # ============================================================================

        elif name.startswith("blocks."):
            parts = name.split(".")
            block_idx = parts[1]

            if parts[2] == "ln0":
                # blocks.i.ln0.weight/bias → rwkv7.blocks.i.ln0.weight/bias
                suffix = parts[3]  # weight or bias
                new_name = f"rwkv7.blocks.{block_idx}.ln0.{suffix}"

            elif parts[2] == "ln1":
                suffix = parts[3]
                new_name = f"rwkv7.blocks.{block_idx}.ln1.{suffix}"

            elif parts[2] == "ln2":
                suffix = parts[3]
                new_name = f"rwkv7.blocks.{block_idx}.ln2.{suffix}"

            elif parts[2] == "att":
                att_parts = parts[3:]  # e.g., ['x_r'] or ['receptance', 'weight']
                att_name = ".".join(att_parts)

                # Direct parameter mappings (time-mix, LoRA, key modulation, local bonus)
                if att_name in {
                    "x_r", "x_w", "x_k", "x_v", "x_a", "x_g",
                    "w0", "w1", "w2",
                    "a0", "a1", "a2",
                    "v0", "v1", "v2",
                    "g1", "g2",
                    "k_k", "k_a",
                    "r_k",
                    "ln_x.weight", "ln_x.bias",
                }:
                    new_name = f"rwkv7.blocks.{block_idx}.att.{att_name}"

                # Linear layer weights
                elif att_name in {
                    "receptance.weight", "key.weight",
                    "value.weight", "output.weight",
                }:
                    new_name = f"rwkv7.blocks.{block_idx}.att.{att_name}"

                else:
                    logger.warning(f"Unknown att parameter: {name}, skipping.")
                    continue

            elif parts[2] == "ffn":
                ffn_parts = parts[3:]
                ffn_name = ".".join(ffn_parts)

                if ffn_name in {"key.weight", "value.weight", "x_k"}:
                    new_name = f"rwkv7.blocks.{block_idx}.ffn.{ffn_name}"
                else:
                    logger.warning(f"Unknown ffn parameter: {name}, skipping.")
                    continue

            else:
                logger.warning(f"Unknown block parameter: {name}, skipping.")
                continue

        else:
            logger.warning(f"Unknown parameter: {name}, skipping.")
            continue

        # ============================================================================
        # Shape adjustments
        # ============================================================================

        # r_k: originally (n_head, head_size), flatten if needed
        if new_name.endswith(".att.r_k"):
            n_head = hidden_size // head_size
            if param.numel() == n_head * head_size:
                param = param.reshape(n_head, head_size)

        # Fix shape for time-mix and 1d parameters (should be (1, 1, hidden_size))
        if any(new_name.endswith(s) for s in [
            ".att.x_r", ".att.x_w", ".att.x_k", ".att.x_v", ".att.x_a", ".att.x_g",
            ".att.w0", ".att.a0", ".att.v0",
            ".att.k_k", ".att.k_a",
            ".ffn.x_k",
        ]):
            if param.dim() == 1:
                param = param.unsqueeze(0).unsqueeze(0)
            elif param.dim() == 2:
                param = param.unsqueeze(0)

        new_state_dict[new_name] = param

    # ============================================================================
    # Deep Embedding: fuse emb.weight with blocks.0.ln0
    # ============================================================================
    if apply_deep_embedding:
        emb_key = "rwkv7.embeddings.weight"
        ln0_w_key = "rwkv7.blocks.0.ln0.weight"
        ln0_b_key = "rwkv7.blocks.0.ln0.bias"

        if all(k in new_state_dict for k in [emb_key, ln0_w_key, ln0_b_key]):
            emb = new_state_dict[emb_key]
            ln0_w = new_state_dict[ln0_w_key]
            ln0_b = new_state_dict[ln0_b_key]

            # Fuse: emb = LayerNorm(emb, weight=ln0_w, bias=ln0_b)
            # Block 0's ln0 is SKIPPED at runtime (deep_embedding=True),
            # because nn.LayerNorm with identity weights still normalizes:
            #   (x - mean) / std * 1 + 0 != x
            # So we absorb ln0 into the embedding matrix.
            emb_fused = F.layer_norm(emb.float(), (hidden_size,), weight=ln0_w.float(), bias=ln0_b.float())
            new_state_dict[emb_key] = emb_fused.to(emb.dtype)

            # ln0 weights are kept (loaded but unused at runtime)
            new_state_dict[ln0_w_key] = torch.ones(hidden_size, dtype=ln0_w.dtype)
            new_state_dict[ln0_b_key] = torch.zeros(hidden_size, dtype=ln0_b.dtype)

            logger.info("Applied deep embedding: fused emb.weight with blocks.0.ln0.")

    # Report on missing/extra v0/v1/v2 parameters
    for block_idx in range(config.num_hidden_layers):
        v0_key = f"rwkv7.blocks.{block_idx}.att.v0"
        v1_key = f"rwkv7.blocks.{block_idx}.att.v1"
        v2_key = f"rwkv7.blocks.{block_idx}.att.v2"
        missing = []
        for k in [v0_key, v1_key, v2_key]:
            if k not in new_state_dict:
                missing.append(k)
        if 0 < len(missing) < 3:
            # Fill missing v* params with default zeros (compatible with older checkpoints)
            for k in missing:
                if k.endswith(".v0"):
                    new_state_dict[k] = torch.zeros(1, 1, hidden_size)
                elif k.endswith(".v1"):
                    d = max(32, int(round((1.7 * (hidden_size ** 0.5)) / 32) * 32))
                    new_state_dict[k] = torch.zeros(hidden_size, d)
                elif k.endswith(".v2"):
                    d = max(32, int(round((1.7 * (hidden_size ** 0.5)) / 32) * 32))
                    new_state_dict[k] = torch.zeros(d, hidden_size)

    return new_state_dict


def convert_rwkv7_checkpoint_to_hf_format(
    repo_id: str,
    checkpoint_file: str,
    output_dir: str,
    size: str = None,
    tokenizer_file: str = None,
    push_to_hub: bool = False,
    model_name: str = None,
    apply_deep_embedding: bool = True,
):
    """
    Convert a BlinkDL RWKV-7 checkpoint to HuggingFace format.

    Args:
        repo_id: HuggingFace Hub repo ID where the checkpoint is stored.
        checkpoint_file: Name of the checkpoint file (e.g., 'RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth').
        output_dir: Directory to save the converted model.
        size: Model size (e.g., '0.1B', '1.5B'). Inferred from checkpoint name if not provided.
        tokenizer_file: Path to tokenizer file. If None, uses the default GPT-NeoX tokenizer.
        push_to_hub: Whether to push the converted model to the Hub.
        model_name: Name for the model on the Hub (required if push_to_hub is True).
        apply_deep_embedding: Whether to fuse embedding with ln0.
    """
    # 1. Build the tokenizer
    if tokenizer_file is None:
        logger.info("No tokenizer_file provided, using default tokenizer (EleutherAI/gpt-neox-20b).")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
    tokenizer.save_pretrained(output_dir)

    # 2. Infer model size and build config
    if size is None:
        possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            raise ValueError(
                f"Could not infer model size from checkpoint name '{checkpoint_file}'. "
                f"Please provide --size (one of {possible_sizes})."
            )

    if size not in NUM_HIDDEN_LAYERS_MAPPING:
        raise ValueError(f"Unknown size '{size}'. Known sizes: {list(NUM_HIDDEN_LAYERS_MAPPING.keys())}")

    config = Rwkv7Config(
        vocab_size=VOCAB_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS_MAPPING[size],
        hidden_size=HIDDEN_SIZE_MAPPING[size],
        head_size=HEAD_SIZE,
        deep_embedding=apply_deep_embedding,
    )
    config.save_pretrained(output_dir)

    # 3. Download and convert the checkpoint
    model_file = hf_hub_download(repo_id, checkpoint_file)
    state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
    state_dict = convert_state_dict(state_dict, config, apply_deep_embedding=apply_deep_embedding)

    # 4. Split into shards and save
    state_dict_split = split_torch_state_dict_into_shards(state_dict)
    shards = {}
    for tensors in state_dict_split.filename_to_tensors.values():
        shards = {tensor: state_dict[tensor] for tensor in tensors}

    if state_dict_split.is_sharded:
        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }

    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))

    if state_dict_split.is_sharded:
        save_index_file = os.path.join(output_dir, WEIGHTS_INDEX_NAME)
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        # Clean up memory
        del state_dict
        del shards
        gc.collect()

        shard_files = list(shards.keys())
        for shard_file in shard_files:
            sd = torch.load(os.path.join(output_dir, shard_file), weights_only=True)
            torch.save({k: v.cpu().clone() for k, v in sd.items()}, os.path.join(output_dir, shard_file))

    del state_dict
    gc.collect()

    if push_to_hub:
        if model_name is None:
            raise ValueError("Please provide --model_name to push the model to the Hub.")
        model = AutoModelForCausalLM.from_pretrained(output_dir)
        model.push_to_hub(model_name, max_shard_size="2GB")
        tokenizer.push_to_hub(model_name)

    logger.info(f"Model successfully converted and saved to {output_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a BlinkDL RWKV-7 checkpoint to HuggingFace format.")
    parser.add_argument(
        "--repo_id", default=None, type=str, required=True,
        help="Repo ID from which to download the checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_file", default=None, type=str, required=True,
        help="Name of the checkpoint file in the repo.",
    )
    parser.add_argument(
        "--output_dir", default=None, type=str, required=True,
        help="Where to save the converted model.",
    )
    parser.add_argument(
        "--tokenizer_file", default=None, type=str,
        help="Path to the tokenizer file (defaults to EleutherAI/gpt-neox-20b).",
    )
    parser.add_argument(
        "--size", default=None, type=str,
        help="Model size (e.g., '0.1B', '1.5B'). Inferred from checkpoint name if not provided.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true",
        help="Push the converted model to HuggingFace Hub.",
    )
    parser.add_argument(
        "--model_name", default=None, type=str,
        help="Name for the pushed model on the Hub (required if --push_to_hub is set).",
    )
    parser.add_argument(
        "--no_deep_embedding", action="store_true",
        help="Do NOT fuse embedding weights with ln0.",
    )

    args = parser.parse_args()
    convert_rwkv7_checkpoint_to_hf_format(
        repo_id=args.repo_id,
        checkpoint_file=args.checkpoint_file,
        output_dir=args.output_dir,
        size=args.size,
        tokenizer_file=args.tokenizer_file,
        push_to_hub=args.push_to_hub,
        model_name=args.model_name,
        apply_deep_embedding=not args.no_deep_embedding,
    )
