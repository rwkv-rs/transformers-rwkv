#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Convert a canonical BlinkDL RWKV-7 `.pth` checkpoint to native Transformers Safetensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download, save_torch_state_dict
from tokenizers import Tokenizer, decoders, models, processors

from transformers import AutoTokenizer, GenerationConfig, RwkvConfig, RwkvForCausalLM, RwkvTokenizerFast


LAYER_PATTERN = re.compile(r"^blocks\.(\d+)\.")
RWKV_VOCAB_REPO = "rwkv-rs/rwkv7-g1-st"
RWKV_VOCAB_FILENAME = "rwkv_vocab_v20230424.json"
RWKV_VOCAB_REVISION = "fd122cc7244c28db19beceb398aa033c35576b71"
RWKV_VOCAB_SHA256 = "0bc72a74aadcd4245878ce07618c77f9c366c485a259d0fa1e4448e50b77cfd7"
RWKV_TOKENIZER_VOCAB_SIZE = 65530
RWKV_MODEL_VOCAB_SIZE = 65536
RWKV_BOS_EOS_TOKEN = "<|endoftext|>"
RWKV_BOS_EOS_TOKEN_ID = 0
TOKENIZER_PROBES = ("RWKV-7 tokenizer", " hello\n", "你好，世界！", "é e\u0301 😀🧑\u200d🚀", "a\x00b\t\r\n")
DEFAULT_CHAT_TEMPLATE = Path(__file__).with_name("rwkv_chat_template.jinja")


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


def _infer_low_rank_dim(state: dict[str, torch.Tensor], layer_ids: list[int], stem: str, hidden_size: int) -> int:
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
    if intermediate_size != 4 * hidden_size:
        raise ValueError(
            "`blocks.0.ffn.key.weight` must use the canonical MLP width 4 * hidden_size; "
            f"got intermediate_size={intermediate_size} and hidden_size={hidden_size}."
        )
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


def _transformers_weight_name(key: str) -> str:
    if key.startswith("emb."):
        return key.replace("emb.", "model.embed_tokens.", 1)
    if key.startswith("ln_out."):
        return key.replace("ln_out.", "model.norm.", 1)
    if key.startswith("head."):
        return key.replace("head.", "lm_head.", 1)

    parts = key.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        raise ValueError(f"Unsupported canonical RWKV-7 tensor name `{key}`.")

    layer_idx, component, *suffix = parts[1:]
    if component == "ln0":
        if layer_idx != "0":
            raise ValueError(f"Unsupported canonical RWKV-7 tensor name `{key}`.")
        return ".".join(("model", "embedding_norm", *suffix))
    component = {
        "ln1": "input_layernorm",
        "ln2": "post_attention_layernorm",
        "att": "linear_attn",
        "ffn": "mlp",
    }.get(component)
    if component is None:
        raise ValueError(f"Unsupported canonical RWKV-7 tensor name `{key}`.")
    if component == "linear_attn":
        suffix[0] = {
            "receptance": "r_proj",
            "key": "k_proj",
            "value": "v_proj",
            "output": "o_proj",
            "ln_x": "g_norm",
        }.get(suffix[0], suffix[0])
    elif component == "mlp":
        suffix[0] = {"key": "up_proj", "value": "down_proj"}.get(suffix[0], suffix[0])
    return ".".join(("model", "layers", layer_idx, component, *suffix))


def convert_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = {}
    for key, tensor in state.items():
        target_key = _transformers_weight_name(key)
        # Released `.pth` checkpoints can store contiguous tensor views backed by
        # a larger flat storage. Clone so each Safetensors entry owns exactly its
        # logical bytes instead of retaining the source storage span.
        converted[target_key] = tensor.detach().clone(memory_format=torch.contiguous_format)
    return converted


def _verify_vocab_bytes(data: bytes, source: str) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if digest != RWKV_VOCAB_SHA256:
        raise ValueError(f"RWKV vocabulary SHA-256 mismatch for {source}: expected {RWKV_VOCAB_SHA256}, got {digest}.")


def resolve_vocab_json(vocab_json: Path | None) -> Path:
    if vocab_json is None:
        vocab_json = Path(
            hf_hub_download(
                repo_id=RWKV_VOCAB_REPO,
                filename=RWKV_VOCAB_FILENAME,
                revision=RWKV_VOCAB_REVISION,
            )
        )
    _verify_vocab_bytes(vocab_json.read_bytes(), str(vocab_json))
    return vocab_json


def build_tokenizer(vocab_json: Path, chat_template: Path, model_max_length: int) -> RwkvTokenizerFast:
    rwkv_trie = getattr(models, "RwkvTrie", None)
    from_file = getattr(rwkv_trie, "from_file", None)
    if from_file is None:
        raise ImportError(
            "RWKV tokenizer conversion requires `tokenizers.models.RwkvTrie.from_file()`; the installed tokenizers "
            "package does not provide it. Reinstall this Transformers checkout so its pinned tokenizers-rwkv Git "
            "dependency is applied. Python, WordPiece, and raw-vocabulary fallbacks are not supported."
        )
    backend = Tokenizer(from_file(str(vocab_json)))
    backend.decoder = decoders.ByteLevel()
    backend.post_processor = processors.TemplateProcessing(
        single=f"{RWKV_BOS_EOS_TOKEN} $A",
        pair=f"{RWKV_BOS_EOS_TOKEN} $A $B",
        special_tokens=[(RWKV_BOS_EOS_TOKEN, 0)],
    )
    tokenizer = RwkvTokenizerFast(
        tokenizer_object=backend,
        bos_token=RWKV_BOS_EOS_TOKEN,
        eos_token=RWKV_BOS_EOS_TOKEN,
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
    )
    tokenizer.chat_template = chat_template.read_text(encoding="utf-8")
    return tokenizer


def _validate_tokenizer(tokenizer, model_vocab_size: int) -> list[dict[str, object]]:
    if not tokenizer.is_fast:
        raise ValueError("RWKV-7 conversion requires a standard fast tokenizer.json artifact.")
    if model_vocab_size != RWKV_MODEL_VOCAB_SIZE:
        raise ValueError(f"RWKV-7 model vocabulary must be 65536, got {model_vocab_size}.")
    if tokenizer.vocab_size != RWKV_TOKENIZER_VOCAB_SIZE or len(tokenizer) != RWKV_TOKENIZER_VOCAB_SIZE:
        raise ValueError(
            f"RWKV tokenizer must contain IDs 0..65529 exactly, got vocab_size={tokenizer.vocab_size}, "
            f"len={len(tokenizer)}."
        )
    vocabulary_ids = set(tokenizer.get_vocab().values())
    if vocabulary_ids != set(range(RWKV_TOKENIZER_VOCAB_SIZE)):
        raise ValueError("RWKV tokenizer vocabulary IDs must cover 0..65529 exactly.")
    special_ids = {name: getattr(tokenizer, f"{name}_token_id") for name in ("bos", "eos", "pad", "unk")}
    if special_ids != {"bos": 0, "eos": 0, "pad": None, "unk": None}:
        raise ValueError(f"RWKV requires BOS/EOS ID 0 and no PAD/UNK token, got {special_ids}.")
    expected = []
    reserved_ids = set(range(RWKV_TOKENIZER_VOCAB_SIZE, RWKV_MODEL_VOCAB_SIZE))
    for text in TOKENIZER_PROBES:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if RWKV_BOS_EOS_TOKEN_ID in token_ids:
            raise ValueError(f"RWKV BOS/EOS token ID 0 became reachable for ordinary probe {text!r}.")
        if reserved_ids.intersection(token_ids):
            raise ValueError(f"RWKV World reserved token IDs became reachable for probe {text!r}.")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ValueError(f"RWKV tokenizer failed a byte-preserving round trip for probe {text!r}.")
        expected.append({"text": text, "token_ids": token_ids})
    probe = TOKENIZER_PROBES[0]
    payload = tokenizer.encode(probe, add_special_tokens=False)
    if tokenizer.encode(probe, add_special_tokens=True) != [0, *payload]:
        raise ValueError("RWKV encoding with special tokens must add exactly one leading BOS/EOS token ID 0.")
    if not tokenizer.chat_template or tokenizer.chat_template.strip() == "{# RWKV native chat template #}":
        raise ValueError("RWKV tokenizer requires an executable native chat template, not a placeholder.")
    return expected


def save_tokenizer(
    vocab_json: Path | None,
    chat_template: Path,
    output_dir: Path,
    model_vocab_size: int,
    model_max_length: int,
) -> None:
    resolved_vocab = resolve_vocab_json(vocab_json)
    tokenizer = build_tokenizer(resolved_vocab, chat_template, model_max_length)
    expected = _validate_tokenizer(tokenizer, model_vocab_size)
    tokenizer.save_pretrained(output_dir)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer_config = output_dir / "tokenizer_config.json"
    saved_chat_template = output_dir / "chat_template.jinja"
    if not tokenizer_json.is_file() or not tokenizer_config.is_file() or not saved_chat_template.is_file():
        raise ValueError("Tokenizer save must produce tokenizer.json, tokenizer_config.json, and chat_template.jinja.")
    saved_config = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    saved_config["chat_template"] = tokenizer.chat_template
    saved_config.pop("pad_token", None)
    saved_config.pop("unk_token", None)
    tokenizer_config.write_text(
        json.dumps(saved_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {"bos_token": RWKV_BOS_EOS_TOKEN, "eos_token": RWKV_BOS_EOS_TOKEN},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if saved_config.get("auto_map"):
        raise ValueError("RWKV tokenizer artifacts must not require `auto_map` or remote code.")
    if any(saved_config.get(name) is not None for name in ("pad_token", "unk_token")):
        raise ValueError("RWKV tokenizer artifacts must not declare PAD or UNK tokens.")

    reloaded = AutoTokenizer.from_pretrained(
        output_dir,
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
    )
    if not isinstance(reloaded, RwkvTokenizerFast):
        raise ValueError(f"AutoTokenizer must load RwkvTokenizerFast, got {type(reloaded).__name__}.")
    actual = _validate_tokenizer(reloaded, model_vocab_size)
    if actual != expected:
        raise ValueError("RWKV tokenizer token IDs changed after save_pretrained()/from_pretrained().")


def convert_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    context_length: int,
    max_shard_size: str,
    vocab_json: Path | None = None,
    chat_template: Path = DEFAULT_CHAT_TEMPLATE,
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
    config.pad_token_id = None
    config.save_pretrained(output_dir)
    GenerationConfig.from_model_config(config).save_pretrained(output_dir)
    save_torch_state_dict(
        converted,
        output_dir,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=max_shard_size,
        safe_serialization=True,
    )
    save_tokenizer(vocab_json, chat_template, output_dir, config.vocab_size, config.context_length)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--rwkv-vocab-json", type=Path)
    parser.add_argument("--chat-template", type=Path, default=DEFAULT_CHAT_TEMPLATE)
    args = parser.parse_args()
    if args.context_length <= 0:
        parser.error("--context-length must be positive")
    convert_checkpoint(
        args.checkpoint,
        args.output_dir,
        args.context_length,
        args.max_shard_size,
        vocab_json=args.rwkv_vocab_json,
        chat_template=args.chat_template,
    )


if __name__ == "__main__":
    main()
