#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Convert a canonical BlinkDL RWKV-7 `.pth` checkpoint to native Transformers Safetensors."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, save_torch_state_dict

from transformers import AutoTokenizer, GenerationConfig, RwkvConfig, RwkvForCausalLM


LAYER_PATTERN = re.compile(r"^blocks\.(\d+)\.")
TOKENIZER_SOURCE = "RWKV/RWKV7-1.5B-20260805"
TOKENIZER_REVISION = "bfb3a69a63e6681f729651c357f13ce0c774ea9c"
TOKENIZER_PROBES = ("RWKV-7 tokenizer", " hello\n", "你好，世界！", "é e\u0301 😀🧑\u200d🚀", "a\x00b\t\r\n")


def _canonical_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("RWKV-7 checkpoint must be a non-empty tensor dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("RWKV-7 checkpoint must contain only string keys and tensor values.")
    keys = tuple(checkpoint)
    has_orig_mod = [key.startswith("_orig_mod.") for key in keys]
    if any(has_orig_mod) and not all(has_orig_mod):
        raise ValueError("`_orig_mod.` must either prefix every tensor key or no tensor key.")
    if all(has_orig_mod):
        checkpoint = {key.removeprefix("_orig_mod."): value for key, value in checkpoint.items()}
    return checkpoint


def _require_shape(state: dict[str, torch.Tensor], key: str, dimensions: int) -> torch.Size:
    if key not in state:
        raise ValueError(f"Missing canonical RWKV-7 tensor `{key}`.")
    shape = state[key].shape
    if len(shape) != dimensions:
        raise ValueError(f"`{key}` must have {dimensions} dimensions, got shape {tuple(shape)}.")
    return shape


def _infer_low_rank_dim(
    state: dict[str, torch.Tensor], layer_ids: list[int], stem: str, hidden_size: int
) -> int:
    ranks = set()
    for layer_id in layer_ids:
        prefix = f"blocks.{layer_id}.att.{stem}"
        input_hidden, rank = _require_shape(state, f"{prefix}1", 2)
        output_rank, output_hidden = _require_shape(state, f"{prefix}2", 2)
        if input_hidden != hidden_size or output_hidden != hidden_size or output_rank != rank:
            raise ValueError(
                f"`{prefix}1` and `{prefix}2` must form hidden-to-rank-to-hidden projections; "
                f"got {tuple(state[f'{prefix}1'].shape)} and {tuple(state[f'{prefix}2'].shape)}."
            )
        ranks.add(rank)
    if len(ranks) != 1:
        raise ValueError(f"RWKV-7 `{stem}` low-rank dimensions must agree across all layers, got {sorted(ranks)}.")
    return ranks.pop()


def infer_config(state: dict[str, torch.Tensor], context_length: int) -> RwkvConfig:
    vocab_size, hidden_size = _require_shape(state, "emb.weight", 2)
    head_vocab, head_hidden = _require_shape(state, "head.weight", 2)
    if (head_vocab, head_hidden) != (vocab_size, hidden_size):
        raise ValueError("`emb.weight` and `head.weight` must describe the same vocabulary and hidden size.")
    layer_ids = sorted({int(match.group(1)) for key in state if (match := LAYER_PATTERN.match(key))})
    if layer_ids != list(range(len(layer_ids))):
        raise ValueError(f"RWKV-7 block indices must be contiguous from zero, got {layer_ids}.")
    if not layer_ids:
        raise ValueError("No canonical `blocks.<index>.*` tensors were found.")
    intermediate_size, ffn_hidden = _require_shape(state, "blocks.0.ffn.key.weight", 2)
    if ffn_hidden != hidden_size:
        raise ValueError("`blocks.0.ffn.key.weight` does not match the embedding hidden size.")
    decay_rank = _infer_low_rank_dim(state, layer_ids, "w", hidden_size)
    a_rank = _infer_low_rank_dim(state, layer_ids, "a", hidden_size)
    value_rank = _infer_low_rank_dim(state, layer_ids, "v", hidden_size)
    gate_rank = _infer_low_rank_dim(state, layer_ids, "g", hidden_size)
    return RwkvConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_size=hidden_size,
        num_hidden_layers=len(layer_ids),
        intermediate_size=intermediate_size,
        head_size=64,
        num_attention_heads=hidden_size // 64,
        decay_low_rank_dim=decay_rank,
        a_low_rank_dim=a_rank,
        v_low_rank_dim=value_rank,
        gate_low_rank_dim=gate_rank,
        architecture_version="rwkv7",
        wkv_state_dtype="float32",
    )


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, tensor in state.items():
        target_key = key if key.startswith("head.") else f"model.{key}"
        # Released `.pth` checkpoints can store contiguous tensor views backed by
        # a larger flat storage. Clone so each Safetensors entry owns exactly its
        # logical bytes instead of retaining the source storage span.
        converted[target_key] = tensor.detach().clone(memory_format=torch.contiguous_format)
    return converted


def _tokenizer_load_kwargs(tokenizer_source: str, tokenizer_revision: str | None) -> dict:
    if Path(tokenizer_source).is_dir():
        return {"local_files_only": True}
    return {"revision": tokenizer_revision}


def _validate_tokenizer(tokenizer, vocab_size: int) -> list[dict[str, object]]:
    if not tokenizer.is_fast:
        raise ValueError("RWKV-7 conversion requires a standard fast tokenizer.json artifact.")
    if len(tokenizer) != vocab_size:
        raise ValueError(f"Tokenizer vocabulary ({len(tokenizer)}) must equal checkpoint vocabulary ({vocab_size}).")
    special_ids = {
        name: getattr(tokenizer, f"{name}_token_id") for name in ("bos", "eos", "pad", "unk")
    }
    if set(special_ids.values()) != {0}:
        raise ValueError(f"RWKV World BOS/EOS/PAD/UNK must all use token ID 0, got {special_ids}.")
    expected = []
    reserved_ids = set(range(65530, 65536)) if vocab_size == 65536 else set()
    for text in TOKENIZER_PROBES:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if reserved_ids.intersection(token_ids):
            raise ValueError(f"RWKV World reserved token IDs became reachable for probe {text!r}.")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ValueError(f"RWKV tokenizer failed a byte-preserving round trip for probe {text!r}.")
        expected.append({"text": text, "token_ids": token_ids})
    probe = TOKENIZER_PROBES[0]
    if tokenizer.encode(probe, add_special_tokens=True) != tokenizer.encode(probe, add_special_tokens=False):
        raise ValueError("RWKV World tokenizer must not insert a BOS token during ordinary encoding.")
    return expected


def save_tokenizer(
    tokenizer_source: str,
    tokenizer_revision: str | None,
    output_dir: Path,
    vocab_size: int,
) -> None:
    load_kwargs = _tokenizer_load_kwargs(tokenizer_source, tokenizer_revision)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=False,
        **load_kwargs,
    )
    expected = _validate_tokenizer(tokenizer, vocab_size)
    tokenizer.save_pretrained(output_dir)

    chat_template = output_dir / "chat_template.jinja"
    if not chat_template.is_file():
        if Path(tokenizer_source).is_dir():
            source_template = Path(tokenizer_source) / "chat_template.jinja"
            if not source_template.is_file():
                source_template = Path(
                    hf_hub_download(
                        repo_id=TOKENIZER_SOURCE,
                        filename="chat_template.jinja",
                        revision=TOKENIZER_REVISION,
                    )
                )
        else:
            source_template = Path(
                hf_hub_download(
                    repo_id=tokenizer_source,
                    filename="chat_template.jinja",
                    revision=tokenizer_revision,
                )
            )
        if not source_template.is_file():
            raise ValueError(f"Tokenizer source does not provide `chat_template.jinja`: {tokenizer_source}")
        shutil.copyfile(source_template, chat_template)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer_config = output_dir / "tokenizer_config.json"
    if not tokenizer_json.is_file() or not tokenizer_config.is_file():
        raise ValueError("Tokenizer save must produce both tokenizer.json and tokenizer_config.json.")
    saved_config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    if saved_config.get("auto_map"):
        raise ValueError("RWKV tokenizer artifacts must not require `auto_map` or remote code.")

    reloaded = AutoTokenizer.from_pretrained(
        output_dir,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    actual = _validate_tokenizer(reloaded, vocab_size)
    if actual != expected:
        raise ValueError("RWKV tokenizer token IDs changed after save_pretrained()/from_pretrained().")


def convert_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    context_length: int,
    max_shard_size: str,
    tokenizer_source: str = TOKENIZER_SOURCE,
    tokenizer_revision: str | None = TOKENIZER_REVISION,
) -> None:
    source = _canonical_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    config = infer_config(source, context_length)
    converted = convert_state_dict(source)
    with init_empty_weights():
        expected = RwkvForCausalLM(config).state_dict()
    missing = sorted(set(expected) - set(converted))
    unexpected = sorted(set(converted) - set(expected))
    mismatched = sorted(
        key for key in set(expected).intersection(converted) if expected[key].shape != converted[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Converted tensor contract mismatch: missing={missing}, unexpected={unexpected}, mismatched={mismatched}."
        )
    source_parameters = sum(tensor.numel() for tensor in source.values())
    converted_parameters = sum(tensor.numel() for tensor in converted.values())
    if source_parameters != converted_parameters:
        raise ValueError(
            f"Parameter count changed during conversion: source={source_parameters}, converted={converted_parameters}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config.architectures = ["RwkvForCausalLM"]
    config.bos_token_id = 0
    config.eos_token_id = 0
    config.pad_token_id = 0
    config.save_pretrained(output_dir)
    GenerationConfig.from_model_config(config).save_pretrained(output_dir)
    save_torch_state_dict(
        converted,
        output_dir,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    save_tokenizer(tokenizer_source, tokenizer_revision, output_dir, config.vocab_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--tokenizer-source", default=TOKENIZER_SOURCE)
    parser.add_argument("--tokenizer-revision", default=TOKENIZER_REVISION)
    args = parser.parse_args()
    if args.context_length <= 0:
        parser.error("--context-length must be positive")
    convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        args.context_length,
        args.max_shard_size,
        tokenizer_source=args.tokenizer_source,
        tokenizer_revision=args.tokenizer_revision,
    )


if __name__ == "__main__":
    main()
