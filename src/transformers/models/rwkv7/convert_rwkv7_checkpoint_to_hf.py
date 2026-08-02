# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Convert a legacy RWKV-7 raw PyTorch state dict to a Transformers artifact."""

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from ..auto.tokenization_auto import AutoTokenizer
from .configuration_rwkv7 import Rwkv7Config
from .modeling_rwkv7 import Rwkv7ForCausalLM


_SUPPORTED_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

_VALIDATION_MARKER = "RWKV7_ARTIFACT_VALIDATION="
_LEGACY_LOW_RANK_PATTERN = re.compile(r"^(blocks\.\d+\.att\.(?:w1|w2|a1|a2|v1|v2|g1|g2))\.weight$")
_VALIDATION_SCRIPT = f"""
import json
import sys

import torch

from transformers import AutoModelForCausalLM


artifact_dir, encoded_input_ids, encoded_max_new_tokens, encoded_device, encoded_dtype, expected_backend = sys.argv[1:]
dtype = None if encoded_dtype == "auto" else getattr(torch, encoded_dtype)
model, loading_info = AutoModelForCausalLM.from_pretrained(
    artifact_dir,
    dtype=dtype,
    output_loading_info=True,
)
load_errors = {{
    name: loading_info.get(name, [])
    for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    if loading_info.get(name)
}}
if load_errors:
    raise RuntimeError(f"RWKV-7 artifact did not strict-load: {{load_errors}}")
device = torch.device(encoded_device)
model.to(device).eval()
input_ids = torch.tensor([json.loads(encoded_input_ids)], dtype=torch.long, device=device)
generation_kwargs = {{
    "max_new_tokens": int(encoded_max_new_tokens),
    "do_sample": False,
    "use_cache": True,
}}
first = model.generate(input_ids, **generation_kwargs)
second = model.generate(input_ids, **generation_kwargs)
if not torch.equal(first, second):
    raise RuntimeError("RWKV-7 artifact generation is not deterministic.")
observed_backends = sorted({{block.att.last_wkv_backend for block in model.model.blocks}})
if expected_backend and observed_backends != [expected_backend]:
    raise RuntimeError(
        f"RWKV-7 artifact requested backend {{expected_backend!r}} but observed {{observed_backends!r}}."
    )
result = {{
    "architecture": type(model).__name__,
    "device": str(device),
    "dtype": encoded_dtype,
    "generated_ids": first.cpu().tolist(),
    "input_ids": input_ids.cpu().tolist(),
    "max_new_tokens": generation_kwargs["max_new_tokens"],
    "observed_wkv_backends": observed_backends,
    "requested_wkv_backend": model.config.wkv_backend,
    "strict_load": True,
}}
print("{_VALIDATION_MARKER}" + json.dumps(result, sort_keys=True))
"""


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_rwkv7_artifact_in_subprocess(
    artifact_dir: str | os.PathLike,
    *,
    input_ids: list[int] | tuple[int, ...] = (1, 2, 3),
    max_new_tokens: int = 4,
    device: str = "cpu",
    dtype: str = "auto",
    expected_wkv_backend: str | None = None,
) -> dict:
    """Strict-load and deterministically generate on the selected device in a fresh Python process."""
    if not input_ids:
        raise ValueError("Artifact validation requires at least one input token id.")
    if max_new_tokens < 1:
        raise ValueError("Artifact validation requires `max_new_tokens` to be positive.")
    if not device:
        raise ValueError("Artifact validation requires a non-empty `device`.")
    if dtype != "auto" and dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported validation dtype `{dtype}`. Choose from auto or {sorted(_SUPPORTED_DTYPES)}.")
    if expected_wkv_backend not in {None, "reference", "flash_rwkv"}:
        raise ValueError("Expected WKV backend must be `reference`, `flash_rwkv`, or omitted.")
    command = [
        sys.executable,
        "-c",
        _VALIDATION_SCRIPT,
        os.fspath(artifact_dir),
        json.dumps(list(input_ids)),
        str(max_new_tokens),
        device,
        dtype,
        expected_wkv_backend or "",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(f"RWKV-7 artifact validation failed in a fresh process: {details}") from error
    marker_lines = [line for line in completed.stdout.splitlines() if line.startswith(_VALIDATION_MARKER)]
    if len(marker_lines) != 1:
        raise RuntimeError("RWKV-7 artifact validation process did not emit exactly one result payload.")
    return json.loads(marker_lines[0].removeprefix(_VALIDATION_MARKER))


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


def _legacy_tensor_spec(model_name: str, tensor: torch.Tensor) -> tuple[str, tuple[int, ...]]:
    raw_name = _raw_name(model_name)
    if match := _LEGACY_LOW_RANK_PATTERN.fullmatch(raw_name):
        return match.group(1), tuple(reversed(tensor.shape))
    return raw_name, tuple(tensor.shape)


def _converted_tensor_spec(raw_name: str, tensor: torch.Tensor) -> tuple[str, torch.Tensor]:
    if _LEGACY_LOW_RANK_PATTERN.fullmatch(f"{raw_name}.weight"):
        return f"model.{raw_name}.weight", tensor.transpose(0, 1).contiguous()
    if raw_name == "emb.weight":
        return "model.embeddings.weight", tensor
    if raw_name == "head.weight":
        return raw_name, tensor
    return f"model.{raw_name}", tensor


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
    expected_shapes = {}
    for name, tensor in expected_model.state_dict().items():
        raw_name, expected_shape = _legacy_tensor_spec(name, tensor)
        expected_shapes[raw_name] = expected_shape
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
        target_name, target_tensor = _converted_tensor_spec(name, tensor)
        converted[target_name] = target_tensor.detach().clone(memory_format=torch.contiguous_format)
    return converted


def convert_rwkv7_checkpoint_to_hf_format(
    checkpoint_path: str,
    output_dir: str,
    *,
    fuse_embedding_layer_norm: bool = False,
    dtype: str | None = None,
    safe_serialization: bool = True,
    max_shard_size: str = "5GB",
    tokenizer_name_or_path: str | None = None,
    source_revision: str | None = None,
    wkv_backend: str = "auto",
    publication_ready: bool = False,
    model_card_path: str | None = None,
    license_path: str | None = None,
    hub_repo_id: str | None = None,
    validation_input_ids: list[int] | tuple[int, ...] = (1, 2, 3),
    validation_max_new_tokens: int = 4,
) -> dict:
    """Convert one local legacy raw ``.pth`` checkpoint and save a strict-loadable artifact."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if dtype is not None and dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype `{dtype}`. Choose from {sorted(_SUPPORTED_DTYPES)} or preserve it.")
    if wkv_backend not in {"auto", "reference", "flash_rwkv"}:
        raise ValueError("`wkv_backend` must be `auto`, `reference`, or `flash_rwkv`.")
    if publication_ready and (tokenizer_name_or_path is None or model_card_path is None or license_path is None):
        raise ValueError(
            "Publication-ready conversion requires tokenizer_name_or_path, model_card_path, and license_path."
        )

    raw_state_dict = _validate_tensor_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    config = infer_rwkv7_config(
        raw_state_dict,
        checkpoint_path,
        embedding_layer_norm_fused=fuse_embedding_layer_norm,
    )
    config.wkv_backend = wkv_backend
    converted = convert_state_dict(raw_state_dict, config, fuse_embedding_layer_norm)
    model = Rwkv7ForCausalLM(config)
    try:
        model.load_state_dict(converted, strict=True)
    except RuntimeError as error:
        raise ValueError(f"Checkpoint does not satisfy the RWKV-7 model contract: {error}") from error
    if dtype is not None:
        model.to(_SUPPORTED_DTYPES[dtype])
    config.architectures = ["Rwkv7ForCausalLM"]
    model.save_pretrained(
        output_dir,
        safe_serialization=safe_serialization,
        max_shard_size=max_shard_size,
    )
    model.generation_config.save_pretrained(output_dir)
    if tokenizer_name_or_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if len(tokenizer) != config.vocab_size:
            raise ValueError(
                f"Tokenizer vocabulary size {len(tokenizer)} does not match model vocabulary size {config.vocab_size}."
            )
        tokenizer.save_pretrained(output_dir)

    output_path = Path(output_dir)
    conversion = {
        "checkpoint_file": os.path.basename(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dtype": dtype or "preserved",
        "embedding_layer_norm_fused": fuse_embedding_layer_norm,
        "max_shard_size": max_shard_size,
        "safe_serialization": safe_serialization,
        "source_revision": source_revision,
        "tokenizer_source": tokenizer_name_or_path,
        "wkv_backend": wkv_backend,
    }
    (output_path / "rwkv7_conversion.json").write_text(
        json.dumps(conversion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = validate_rwkv7_artifact_in_subprocess(
        output_path,
        input_ids=validation_input_ids,
        max_new_tokens=validation_max_new_tokens,
        device="cuda" if wkv_backend == "flash_rwkv" else "cpu",
        dtype=dtype or "auto",
        expected_wkv_backend=None if wkv_backend == "auto" else wkv_backend,
    )
    validation["artifact_files"] = sorted(
        [path.name for path in output_path.iterdir() if path.is_file()] + ["rwkv7_validation.json"]
    )
    (output_path / "rwkv7_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    publication = None
    if publication_ready:
        from .prepare_rwkv7_hf_upload import prepare_rwkv7_hf_upload

        publication = prepare_rwkv7_hf_upload(
            output_path,
            model_card_path=model_card_path,
            license_path=license_path,
            hub_repo_id=hub_repo_id,
        )
    return {"conversion": conversion, "publication": publication, "validation": validation}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", required=True, help="Local legacy RWKV-7 raw .pth state dict.")
    parser.add_argument("--output_dir", required=True, help="Destination for the Transformers artifact.")
    parser.add_argument(
        "--dtype", choices=sorted(_SUPPORTED_DTYPES), help="Optional output dtype; preserve by default."
    )
    parser.add_argument("--fuse_embedding_layer_norm", action="store_true")
    parser.add_argument("--no_safe_serialization", action="store_true")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--tokenizer_name_or_path")
    parser.add_argument("--source_revision")
    parser.add_argument("--wkv_backend", choices=("auto", "reference", "flash_rwkv"), default="auto")
    parser.add_argument("--publication_ready", action="store_true")
    parser.add_argument("--model_card_path")
    parser.add_argument("--license_path")
    parser.add_argument("--hub_repo_id")
    parser.add_argument("--validation_input_ids", default="1,2,3")
    parser.add_argument("--validation_max_new_tokens", type=int, default=4)
    args = parser.parse_args(argv)
    validation_input_ids = [int(token_id) for token_id in args.validation_input_ids.split(",") if token_id]
    result = convert_rwkv7_checkpoint_to_hf_format(
        args.checkpoint_path,
        args.output_dir,
        fuse_embedding_layer_norm=args.fuse_embedding_layer_norm,
        dtype=args.dtype,
        safe_serialization=not args.no_safe_serialization,
        max_shard_size=args.max_shard_size,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        source_revision=args.source_revision,
        wkv_backend=args.wkv_backend,
        publication_ready=args.publication_ready,
        model_card_path=args.model_card_path,
        license_path=args.license_path,
        hub_repo_id=args.hub_repo_id,
        validation_input_ids=validation_input_ids,
        validation_max_new_tokens=args.validation_max_new_tokens,
    )
    print(f"RWKV7_CONVERSION={json.dumps(result, sort_keys=True)}")


if __name__ == "__main__":
    main()
