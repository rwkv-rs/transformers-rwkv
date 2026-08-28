#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Convert a canonical BlinkDL RWKV-7 checkpoint to Transformers Safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, split_torch_state_dict_into_shards
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, processors

from transformers import AutoTokenizer, GenerationConfig, RwkvConfig, RwkvForCausalLM, RwkvTokenizer


BLOCK_KEY = re.compile(r"^blocks\.(\d+)\.")
TOKENIZER_REPO = "rwkv-rs/rwkv7-g1-st"
TOKENIZER_FILE = "rwkv_vocab_v20230424.json"
TOKENIZER_REVISION = "fd122cc7244c28db19beceb398aa033c35576b71"
TOKENIZER_SHA256 = "0bc72a74aadcd4245878ce07618c77f9c366c485a259d0fa1e4448e50b77cfd7"
TOKENIZER_VOCAB_SIZE = 65530
MODEL_VOCAB_SIZE = 65536
BOS_EOS_TOKEN = "<|endoftext|>"
CHAT_TEMPLATE = Path(__file__).with_name("rwkv_chat_template.jinja")
STOP_STRINGS = ["✿", "\nUser:", "\n### User"]
LAYER_ZERO_UNUSED = {
    "blocks.0.att.v0",
    "blocks.0.att.v1",
    "blocks.0.att.v2",
}


def load_checkpoint(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("The RWKV-7 checkpoint must be a non-empty tensor dictionary.")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("The RWKV-7 checkpoint must contain only string keys and tensor values.")
    if any(key.startswith("_orig_mod.") for key in state):
        raise ValueError("Compiled `_orig_mod.` checkpoint keys are not part of the RWKV-7 release contract.")
    return state


def _shape(state: dict[str, torch.Tensor], key: str, dimensions: int) -> torch.Size:
    tensor = state.get(key)
    if tensor is None:
        raise ValueError(f"Missing canonical RWKV-7 tensor `{key}`.")
    if tensor.ndim != dimensions:
        raise ValueError(f"`{key}` must have rank {dimensions}, got shape {tuple(tensor.shape)}.")
    return tensor.shape


def _rank(
    state: dict[str, torch.Tensor], layer_ids: list[int], stem: str, hidden_size: int
) -> int:
    ranks = set()
    for layer_idx in layer_ids:
        first = f"blocks.{layer_idx}.att.{stem}1"
        second = f"blocks.{layer_idx}.att.{stem}2"
        input_hidden, rank = _shape(state, first, 2)
        output_rank, output_hidden = _shape(state, second, 2)
        if input_hidden != hidden_size or output_hidden != hidden_size or output_rank != rank:
            raise ValueError(
                f"`{first}` and `{second}` must form [hidden, rank] and [rank, hidden] projections."
            )
        ranks.add(rank)
    if len(ranks) != 1:
        raise ValueError(f"RWKV-7 `{stem}` rank must agree across layers, got {sorted(ranks)}.")
    return ranks.pop()


def infer_config(state: dict[str, torch.Tensor], context_length: int) -> RwkvConfig:
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    vocab_size, hidden_size = _shape(state, "emb.weight", 2)
    if _shape(state, "head.weight", 2) != torch.Size((vocab_size, hidden_size)):
        raise ValueError("`emb.weight` and `head.weight` must have matching shapes.")
    layer_ids = sorted({int(match.group(1)) for key in state if (match := BLOCK_KEY.match(key))})
    if layer_ids != list(range(len(layer_ids))) or len(layer_ids) < 2:
        raise ValueError(f"RWKV-7 block indices must be contiguous from zero and include layer 1, got {layer_ids}.")
    intermediate_size, ffn_hidden = _shape(state, "blocks.0.ffn.key.weight", 2)
    if ffn_hidden != hidden_size or intermediate_size != 4 * hidden_size:
        raise ValueError("RWKV-7 ChannelMix key weight must have shape [4 * hidden_size, hidden_size].")

    return RwkvConfig(
        architecture_version="rwkv7",
        vocab_size=vocab_size,
        context_length=context_length,
        hidden_size=hidden_size,
        num_hidden_layers=len(layer_ids),
        intermediate_size=intermediate_size,
        head_size=64,
        num_attention_heads=hidden_size // 64,
        decay_low_rank_dim=_rank(state, layer_ids, "w", hidden_size),
        a_low_rank_dim=_rank(state, layer_ids, "a", hidden_size),
        v_low_rank_dim=_rank(state, layer_ids[1:], "v", hidden_size),
        gate_low_rank_dim=_rank(state, layer_ids, "g", hidden_size),
        wkv_state_dtype="float32",
    )


def translate_key(source_key: str) -> str | None:
    if source_key in LAYER_ZERO_UNUSED:
        return None
    top_level = {
        "emb.weight": "model.embed_tokens.weight",
        "ln_out.weight": "model.norm.weight",
        "ln_out.bias": "model.norm.bias",
        "head.weight": "lm_head.weight",
    }
    if source_key in top_level:
        return top_level[source_key]

    parts = source_key.split(".")
    if len(parts) < 4 or parts[0] != "blocks" or not parts[1].isdigit():
        raise ValueError(f"Unsupported canonical RWKV-7 tensor `{source_key}`.")
    layer_idx, component, *suffix = parts[1:]
    if component == "ln0":
        if layer_idx != "0":
            raise ValueError(f"Only block 0 may contain ln0, got `{source_key}`.")
        return ".".join(("model", "embedding_norm", *suffix))

    component = {
        "ln1": "input_layernorm",
        "ln2": "post_attention_layernorm",
        "att": "linear_attn",
        "ffn": "mlp",
    }.get(component)
    if component is None:
        raise ValueError(f"Unsupported canonical RWKV-7 tensor `{source_key}`.")
    if component == "linear_attn":
        suffix[0] = {
            "receptance": "r_proj",
            "key": "k_proj",
            "value": "v_proj",
            "output": "o_proj",
            "ln_x": "g_norm",
        }.get(suffix[0], suffix[0])
    return ".".join(("model", "layers", layer_idx, component, *suffix))


def translation_plan(state: dict[str, torch.Tensor]) -> tuple[dict[str, str], set[str]]:
    target_to_source = {}
    dropped = set()
    for source_key in state:
        target_key = translate_key(source_key)
        if target_key is None:
            dropped.add(source_key)
        elif target_key in target_to_source:
            raise ValueError(f"Multiple checkpoint tensors map to `{target_key}`.")
        else:
            target_to_source[target_key] = source_key
    return target_to_source, dropped


def converted_tensor(tensor: torch.Tensor, expected_shape: torch.Size) -> torch.Tensor:
    if tensor.shape != expected_shape and tensor.ndim == 3 and tensor.shape[:2] == (1, 1):
        tensor = tensor.squeeze(0).squeeze(0)
    if tensor.shape != expected_shape:
        raise ValueError(f"Converted tensor shape {tuple(tensor.shape)} does not match {tuple(expected_shape)}.")
    return tensor.detach().clone(memory_format=torch.contiguous_format)


def expected_state_dict(config: RwkvConfig) -> dict[str, torch.Tensor]:
    with torch.device("meta"):
        return RwkvForCausalLM(config).state_dict()


def validate_plan(
    state: dict[str, torch.Tensor],
    target_to_source: dict[str, str],
    dropped: set[str],
    expected: dict[str, torch.Tensor],
) -> None:
    missing = sorted(set(expected) - set(target_to_source))
    unexpected = sorted(set(target_to_source) - set(expected))
    mismatched = []
    for target_key in set(expected).intersection(target_to_source):
        source_shape = state[target_to_source[target_key]].shape
        if source_shape != expected[target_key].shape:
            converted_shape = source_shape
            if len(source_shape) == 3 and source_shape[:2] == (1, 1):
                converted_shape = source_shape[2:]
            if converted_shape != expected[target_key].shape:
                mismatched.append((target_key, tuple(source_shape), tuple(expected[target_key].shape)))
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Converted state_dict contract mismatch: missing={missing}, unexpected={unexpected}, mismatched={mismatched}."
        )
    source_parameters = sum(tensor.numel() for tensor in state.values())
    dropped_parameters = sum(state[key].numel() for key in dropped)
    expected_parameters = sum(tensor.numel() for tensor in expected.values())
    if source_parameters - dropped_parameters != expected_parameters:
        raise ValueError(
            "Parameter count must differ only by block 0's unused value-rank tensors: "
            f"source={source_parameters}, dropped={dropped_parameters}, expected={expected_parameters}."
        )


def write_shards(
    state: dict[str, torch.Tensor],
    target_to_source: dict[str, str],
    expected: dict[str, torch.Tensor],
    output: Path,
    max_shard_size: str,
) -> None:
    planning_state = {target: state[source] for target, source in target_to_source.items()}
    split = split_torch_state_dict_into_shards(
        planning_state,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
    )
    del planning_state

    for filename, target_keys in split.filename_to_tensors.items():
        shard = {}
        for target_key in target_keys:
            source_key = target_to_source[target_key]
            shard[target_key] = converted_tensor(state.pop(source_key), expected[target_key].shape)
        save_file(shard, output / filename, metadata={"format": "pt"})
        del shard

    if split.is_sharded:
        index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def resolve_tokenizer_source(tokenizer_source: Path | None) -> Path:
    if tokenizer_source is None:
        tokenizer_source = Path(
            hf_hub_download(
                repo_id=TOKENIZER_REPO,
                filename=TOKENIZER_FILE,
                revision=TOKENIZER_REVISION,
            )
        )
    digest = hashlib.sha256(tokenizer_source.read_bytes()).hexdigest()
    if digest != TOKENIZER_SHA256:
        raise ValueError(
            f"RWKV tokenizer SHA-256 mismatch: expected {TOKENIZER_SHA256}, got {digest}."
        )
    return tokenizer_source


def build_tokenizer(tokenizer_source: Path, context_length: int) -> RwkvTokenizer:
    backend = Tokenizer(models.RwkvTrie.from_file(str(tokenizer_source)))
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{BOS_EOS_TOKEN} $A",
        pair=f"{BOS_EOS_TOKEN} $A $B",
        special_tokens=[(BOS_EOS_TOKEN, 0)],
    )
    tokenizer = RwkvTokenizer(
        tokenizer_object=backend,
        bos_token=BOS_EOS_TOKEN,
        eos_token=BOS_EOS_TOKEN,
        model_max_length=context_length,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = CHAT_TEMPLATE.read_text(encoding="utf-8")
    return tokenizer


def validate_tokenizer(tokenizer: RwkvTokenizer) -> None:
    if tokenizer.vocab_size != TOKENIZER_VOCAB_SIZE or len(tokenizer) != TOKENIZER_VOCAB_SIZE:
        raise ValueError("The fixed RWKV tokenizer must contain IDs 0 through 65529 exactly.")
    if set(tokenizer.get_vocab().values()) != set(range(TOKENIZER_VOCAB_SIZE)):
        raise ValueError("The fixed RWKV tokenizer IDs must be contiguous from 0 through 65529.")
    special_ids = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id)
    if special_ids != (0, 0, None, None):
        raise ValueError(f"RWKV requires BOS/EOS=0 and no PAD/UNK token, got {special_ids}.")
    for text in ("RWKV-7 tokenizer", " hello\n", "你好，世界！", "a\x00b\t\r\n"):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
            raise ValueError(f"RWKV tokenizer failed a byte-preserving round trip for {text!r}.")


def save_tokenizer(tokenizer_source: Path | None, output: Path, context_length: int) -> None:
    tokenizer = build_tokenizer(resolve_tokenizer_source(tokenizer_source), context_length)
    validate_tokenizer(tokenizer)
    tokenizer.save_pretrained(output)
    reloaded = AutoTokenizer.from_pretrained(output, local_files_only=True, trust_remote_code=False)
    if type(reloaded) is not RwkvTokenizer:
        raise ValueError(f"AutoTokenizer must reload RwkvTokenizer, got {type(reloaded).__name__}.")
    validate_tokenizer(reloaded)


def convert_checkpoint(
    checkpoint: Path,
    output: Path,
    context_length: int,
    max_shard_size: str,
    tokenizer_source: Path | None,
) -> None:
    state = load_checkpoint(checkpoint)
    config = infer_config(state, context_length)
    target_to_source, dropped = translation_plan(state)
    expected = expected_state_dict(config)
    validate_plan(state, target_to_source, dropped, expected)
    for source_key in dropped:
        state.pop(source_key)

    output.mkdir(parents=True, exist_ok=True)
    config.architectures = ["RwkvForCausalLM"]
    config.save_pretrained(output)
    generation_config = GenerationConfig.from_model_config(config)
    generation_config.stop_strings = STOP_STRINGS
    generation_config.save_pretrained(output)
    write_shards(state, target_to_source, expected, output, max_shard_size)
    save_tokenizer(tokenizer_source, output, context_length)
    if state:
        raise ValueError(f"Conversion left unconsumed checkpoint tensors: {sorted(state)}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--tokenizer-source", type=Path)
    args = parser.parse_args()
    convert_checkpoint(
        args.checkpoint,
        args.output,
        args.context_length,
        args.max_shard_size,
        args.tokenizer_source,
    )


if __name__ == "__main__":
    main()
