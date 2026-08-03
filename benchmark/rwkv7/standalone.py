# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Standalone CUDA benchmark for an explicitly selected RWKV7 Hugging Face artifact.

This entry point deliberately lives outside ``benchmark/benches`` so it is not
imported by the legacy benchmark discovery machinery.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.rwkv7 import validate_rwkv7_runtime_provenance


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
_REQUIRED_PROVIDER = "flash_rwkv"
_CANONICAL_MANIFEST_NAME = "rwkv7_hf_upload_manifest.json"
_TOKENIZER_FILES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
    }
)
_OPTIONAL_ARTIFACT_METADATA = frozenset(
    {
        "generation_config.json",
        "rwkv7_conversion.json",
        "rwkv7_validation.json",
    }
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Explicit local Hugging Face artifact path or Hub repository ID.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON result path.")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-ids",
        help="Optional comma-separated fixed token ids; its length must equal prompt_tokens + decode_tokens.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Pass local_files_only=True to the standard AutoConfig/AutoModel loaders.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.model.strip():
        raise ValueError("--model must name an explicit local artifact path or Hub repository ID")
    for name in ("batch_size", "prompt_tokens", "decode_tokens", "iterations"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA because latency is measured with CUDA events")


def _parse_fixed_input_ids(encoded: str, *, expected_length: int, vocab_size: int) -> list[int]:
    try:
        input_ids = [int(value) for value in encoded.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--input-ids must be a comma-separated list of integers") from error
    if len(input_ids) != expected_length:
        raise ValueError(
            f"--input-ids contains {len(input_ids)} ids; expected {expected_length} "
            "from --prompt-tokens + --decode-tokens"
        )
    if min(input_ids) < 0 or max(input_ids) >= vocab_size:
        raise ValueError(f"--input-ids must be in [0, {vocab_size})")
    return input_ids


def _quantile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _latency_summary(samples_ms: list[float], tokens_per_iteration: int) -> dict[str, Any]:
    p50_ms = _quantile(samples_ms, 0.5)
    return {
        "samples_ms": samples_ms,
        "p10_ms": _quantile(samples_ms, 0.1),
        "p50_ms": p50_ms,
        "p90_ms": _quantile(samples_ms, 0.9),
        "tokens_per_iteration": tokens_per_iteration,
        "tokens_per_second_at_p50": tokens_per_iteration * 1000.0 / p50_ms,
    }


def _clone_state(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(component.clone() for component in state)


def _observed_backends(model: torch.nn.Module) -> set[str]:
    return {str(module.last_wkv_backend) for module in model.modules() if hasattr(module, "last_wkv_backend")}


def _require_observed_backend(observed: set[str]) -> None:
    if not observed:
        raise RuntimeError("RWKV7 benchmark could not observe the selected WKV backend")
    if len(observed) != 1:
        raise RuntimeError(f"RWKV7 layers selected inconsistent WKV backends: {sorted(observed)}")
    if observed != {_REQUIRED_PROVIDER}:
        raise RuntimeError(f"FlashRWKV provider requirement failed closed; observed {sorted(observed)!r}")


def _capture_observed_backend(model: torch.nn.Module, stage: str) -> set[str]:
    observed = _observed_backends(model)
    try:
        _require_observed_backend(observed)
    except RuntimeError as error:
        raise RuntimeError(f"RWKV7 {stage} backend observation failed: {error}") from error
    return observed


def _require_consistent_backend_observations(observations: dict[str, set[str]]) -> set[str]:
    signatures = {tuple(sorted(observed)) for observed in observations.values()}
    if len(signatures) != 1:
        rendered = {stage: sorted(observed) for stage, observed in observations.items()}
        raise RuntimeError(f"RWKV7 benchmark stages selected inconsistent WKV backends: {rendered}")
    return set(next(iter(signatures)))


def _state_difference_summary(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    differences = [
        (actual_component.float() - expected_component.float()).abs()
        for actual_component, expected_component in zip(actual, expected, strict=True)
    ]
    total_values = sum(difference.numel() for difference in differences)
    return (
        max(difference.max().item() for difference in differences),
        sum(difference.sum().item() for difference in differences) / total_values,
    )


def _rrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(torch.finfo(torch.float32).eps)
    return (difference.square().mean().sqrt() / denominator).item()


def _correctness_gate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_tokens: int,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    one_shot = model(input_ids=input_ids, use_cache=True)
    one_shot_backend = _capture_observed_backend(model, "correctness one-shot")
    if one_shot.state is None:
        raise RuntimeError("RWKV7 one-shot correctness call did not return recurrent state with use_cache=True")
    prefix = model(input_ids=input_ids[:, :prompt_tokens], use_cache=True)
    prefix_backend = _capture_observed_backend(model, "correctness prefix prefill")
    if prefix.state is None:
        raise RuntimeError("RWKV7 prefill did not return recurrent state with use_cache=True")

    state = prefix.state
    prefix_state_snapshot = _clone_state(prefix.state)
    staged_logits = []
    decode_backend_observations = {}
    for token_index in range(prompt_tokens, input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, token_index : token_index + 1],
            state=state,
            use_cache=True,
        )
        decode_backend_observations[f"staged_decode_token_{token_index - prompt_tokens}"] = _capture_observed_backend(
            model,
            f"correctness staged decode token {token_index - prompt_tokens}",
        )
        if step.state is None:
            raise RuntimeError("RWKV7 decode did not return recurrent state with use_cache=True")
        state = step.state
        staged_logits.append(step.logits)

    staged = torch.cat(staged_logits, dim=1)
    expected = one_shot.logits[:, prompt_tokens:]
    difference = (staged.float() - expected.float()).abs()
    if dtype == torch.float32:
        # The public fp32 model contract uses the canonical fp32-state/fp16-I/O
        # WKV path.  Its staged-vs-one-shot comparison is therefore a numerical
        # equivalence check, rather than an elementwise fp32 identity check.
        rrmse_limit = 0.002
    elif dtype == torch.float16:
        rrmse_limit = 0.003
    else:
        rrmse_limit = 5e-2
    logits_rrmse = _rrmse(staged, expected)
    if logits_rrmse > rrmse_limit:
        raise AssertionError(
            "RWKV7 staged-vs-one-shot logits exceeded the numerical contract: "
            f"RRMSE={logits_rrmse:.6g} > {rrmse_limit:.6g}, "
            f"max_abs={difference.max().item():.6g}"
        )
    if len(state) != len(one_shot.state):
        raise RuntimeError("RWKV7 staged and one-shot calls returned different recurrent-state layouts")
    state_rrmse = 0.0
    for component_index, (staged_component, expected_component) in enumerate(zip(state, one_shot.state, strict=True)):
        component_rrmse = _rrmse(staged_component, expected_component)
        state_rrmse = max(state_rrmse, component_rrmse)
        if component_rrmse > rrmse_limit:
            raise AssertionError(
                "RWKV7 staged-vs-one-shot recurrent state exceeded the numerical contract: "
                f"component={component_index}, RRMSE={component_rrmse:.6g} > {rrmse_limit:.6g}"
            )
    for preserved_component, snapshot_component in zip(prefix.state, prefix_state_snapshot, strict=True):
        torch.testing.assert_close(preserved_component, snapshot_component, rtol=0, atol=0)

    backend_observations = {
        "one_shot": one_shot_backend,
        "prefix_prefill": prefix_backend,
        **decode_backend_observations,
    }
    observed = _require_consistent_backend_observations(backend_observations)
    state_max_abs_error, state_mean_abs_error = _state_difference_summary(state, one_shot.state)

    return (
        {
            "passed": True,
            "comparison": "staged recurrent decode logits and final state vs matching one-shot outputs",
            "compared_tokens": input_ids.shape[1] - prompt_tokens,
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "rrmse": logits_rrmse,
            "rrmse_limit": rrmse_limit,
            "state_components": len(state),
            "state_max_abs_error": state_max_abs_error,
            "state_mean_abs_error": state_mean_abs_error,
            "state_rrmse": state_rrmse,
            "input_state_preserved": True,
            "observed_backends": sorted(observed),
            "backend_observations": {
                "one_shot": sorted(one_shot_backend),
                "prefix_prefill": sorted(prefix_backend),
                "staged_decode": sorted(_require_consistent_backend_observations(decode_backend_observations)),
            },
        },
        prefix_state_snapshot,
    )


def _measure_cuda(
    operation,
    setup,
    *,
    model: torch.nn.Module,
    stage: str,
    warmup: int,
    iterations: int,
    tokens_per_iteration: int,
    device: torch.device,
) -> dict[str, Any]:
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    backend_observations = {}
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    for _ in range(warmup):
        payload = setup()
        torch.cuda.synchronize(device)
        result = operation(payload)
        backend_observations[f"warmup_{len(backend_observations)}"] = _capture_observed_backend(
            model, f"{stage} warmup"
        )
        torch.cuda.synchronize(device)
        del result, payload

    samples_ms = []
    resident_before_samples = []
    peak_allocated_samples = []
    peak_increment_samples = []
    for iteration, (start, end) in enumerate(events):
        payload = setup()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        resident_before_bytes = torch.cuda.memory_allocated(device)
        start.record()
        result = operation(payload)
        end.record()
        backend_observations[f"timed_{iteration}"] = _capture_observed_backend(
            model, f"{stage} timed iteration {iteration}"
        )
        torch.cuda.synchronize(device)
        samples_ms.append(start.elapsed_time(end))
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        resident_before_samples.append(resident_before_bytes)
        peak_allocated_samples.append(peak_allocated_bytes)
        peak_increment_samples.append(peak_allocated_bytes - resident_before_bytes)
        del result, payload

    observed = _require_consistent_backend_observations(backend_observations)
    summary = _latency_summary(samples_ms, tokens_per_iteration)
    summary.update(
        {
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "observed_backends": sorted(observed),
            "torch_allocator_memory": {
                "scope": "PyTorch allocator only; excludes state setup and may not include native external allocations",
                "resident_before_operation_bytes_samples": resident_before_samples,
                "peak_allocated_bytes_samples": peak_allocated_samples,
                "operation_peak_increment_bytes_samples": peak_increment_samples,
                "max_operation_peak_increment_bytes": max(peak_increment_samples),
            },
        }
    )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read RWKV7 artifact metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"RWKV7 artifact metadata must be a JSON object: {path}")
    return payload


def _local_artifact_files(artifact_dir: Path) -> tuple[list[Path], list[Path]]:
    config_path = artifact_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Local model artifact has no config.json: {artifact_dir}")

    index_path = artifact_dir / "model.safetensors.index.json"
    single_weight_path = artifact_dir / "model.safetensors"
    if index_path.is_file() and single_weight_path.is_file():
        raise ValueError("Local RWKV7 artifact ambiguously contains both sharded and single-file safetensors")
    if index_path.is_file():
        index = _read_json_object(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("model.safetensors.index.json requires a non-empty weight_map")
        weight_names = sorted(set(weight_map.values()))
        if not all(
            isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors")
            for name in weight_names
        ):
            raise ValueError("model.safetensors.index.json must reference top-level .safetensors shards")
        weight_paths = [artifact_dir / name for name in weight_names]
        missing = [path.name for path in weight_paths if not path.is_file()]
        if missing:
            raise ValueError(f"model.safetensors.index.json references missing weight shards: {missing}")
        index_files = [index_path]
    elif single_weight_path.is_file():
        weight_paths = [single_weight_path]
        index_files = []
    else:
        raise ValueError("Local RWKV7 artifact requires model.safetensors or a complete safetensors shard index")

    unreferenced_weights = sorted(set(artifact_dir.glob("*.safetensors")) - set(weight_paths))
    if unreferenced_weights:
        raise ValueError(
            "Local RWKV7 artifact contains unreferenced safetensors files: "
            f"{[path.name for path in unreferenced_weights]}"
        )
    tokenizer_paths = sorted(
        path for path in artifact_dir.iterdir() if path.is_file() and path.name in _TOKENIZER_FILES
    )
    metadata_paths = sorted(
        path for path in artifact_dir.iterdir() if path.is_file() and path.name in _OPTIONAL_ARTIFACT_METADATA
    )
    tracked_paths = [config_path, *index_files, *weight_paths, *tokenizer_paths, *metadata_paths]
    return tracked_paths, weight_paths


def _file_record(path: Path, artifact_dir: Path) -> dict[str, Any]:
    return {
        "name": path.relative_to(artifact_dir).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _validated_manifest_records(
    manifest_path: Path,
    artifact_dir: Path,
    required_names: set[str],
) -> list[dict[str, Any]]:
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("RWKV7 canonical upload manifest must use schema_version=1")
    artifact = manifest.get("artifact")
    files = artifact.get("files") if isinstance(artifact, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("RWKV7 canonical upload manifest requires a non-empty artifact.files list")

    records = []
    observed_names = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("RWKV7 canonical upload manifest file entries must be JSON objects")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("RWKV7 canonical upload manifest file entries require a name")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in observed_names:
            raise ValueError(f"RWKV7 canonical upload manifest contains an unsafe or duplicate path: {name!r}")
        path = (artifact_dir / relative).resolve()
        try:
            path.relative_to(artifact_dir)
        except ValueError as error:
            raise ValueError(f"RWKV7 canonical upload manifest path escapes the artifact: {name!r}") from error
        if not path.is_file():
            raise ValueError(f"RWKV7 canonical upload manifest references a missing file: {name!r}")
        record = _file_record(path, artifact_dir)
        if entry.get("sha256") != record["sha256"] or entry.get("size_bytes") != record["size_bytes"]:
            raise ValueError(f"RWKV7 canonical upload manifest digest changed for {name!r}")
        records.append(record)
        observed_names.add(name)

    missing = sorted(required_names - observed_names)
    if missing:
        raise ValueError(f"RWKV7 canonical upload manifest does not cover required benchmark files: {missing}")
    return sorted(records, key=lambda record: record["name"])


def _artifact_provenance(reference: str, config: Any) -> dict[str, Any]:
    local_path = Path(reference).expanduser()
    if local_path.exists():
        resolved = local_path.resolve()
        tracked_paths, weight_paths = _local_artifact_files(resolved)
        required_names = {path.relative_to(resolved).as_posix() for path in tracked_paths}
        manifest_path = resolved / _CANONICAL_MANIFEST_NAME
        if manifest_path.is_file():
            files = _validated_manifest_records(manifest_path, resolved, required_names)
            identity = {
                "method": "rwkv7_hf_upload_manifest",
                "manifest_name": manifest_path.name,
                "manifest_sha256": _sha256(manifest_path),
            }
        else:
            files = sorted((_file_record(path, resolved) for path in tracked_paths), key=lambda record: record["name"])
            identity = {"method": "direct_sha256"}
        return {
            "kind": "local_path",
            "input_reference": reference,
            "resolved_reference": str(resolved),
            "identity": identity,
            "files": files,
            "weight_files": [path.name for path in weight_paths],
            "hub_commit": getattr(config, "_commit_hash", None),
        }
    hub_commit = getattr(config, "_commit_hash", None)
    if not isinstance(hub_commit, str) or re.fullmatch(r"[0-9a-fA-F]{40}", hub_commit) is None:
        raise ValueError("Hub RWKV7 artifact did not resolve to an immutable 40-character commit")
    return {
        "kind": "hub_repo_id",
        "input_reference": reference,
        "resolved_reference": reference,
        "hub_commit": hub_commit.lower(),
    }


def _distribution_provenance(name: str) -> dict[str, Any] | None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None
    result = {"version": distribution.version}
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is not None:
        result["direct_url"] = json.loads(direct_url)
    return result


def _validated_operator_provenance(observed: set[str]) -> dict[str, str]:
    _require_observed_backend(observed)
    return dict(validate_rwkv7_runtime_provenance())


def _source_provenance(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--short")),
    }


def _hardware_provenance(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "device_index": device.index,
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    dtype = _DTYPES[args.dtype]
    device = torch.device("cuda", torch.cuda.current_device())

    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    if config.model_type != "rwkv7":
        raise ValueError(f"Expected an RWKV7 artifact, but model_type is {config.model_type!r}")
    if args.prompt_tokens + args.decode_tokens > config.context_length:
        raise ValueError(
            "The correctness workload exceeds config.context_length: "
            f"{args.prompt_tokens} + {args.decode_tokens} > {config.context_length}"
        )
    artifact_provenance = _artifact_provenance(args.model, config)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    if args.input_ids is None:
        generator = torch.Generator().manual_seed(args.seed)
        full_input_ids = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(args.batch_size, args.prompt_tokens + args.decode_tokens),
            generator=generator,
        ).to(device)
        input_ids_source = "seeded-random"
    else:
        fixed_input_ids = _parse_fixed_input_ids(
            args.input_ids,
            expected_length=args.prompt_tokens + args.decode_tokens,
            vocab_size=config.vocab_size,
        )
        full_input_ids = torch.tensor([fixed_input_ids] * args.batch_size, dtype=torch.long, device=device)
        input_ids_source = "fixed-cli"
    prompt_input_ids = full_input_ids[:, : args.prompt_tokens]
    decode_input_ids = full_input_ids[:, args.prompt_tokens : args.prompt_tokens + 1]

    with torch.inference_mode():
        correctness, prefix_state = _correctness_gate(
            model,
            full_input_ids,
            args.prompt_tokens,
            dtype,
        )
        correctness_backends = set(correctness["observed_backends"])
        operator_provenance = _validated_operator_provenance(correctness_backends)
        torch.cuda.synchronize(device)
        prefill = _measure_cuda(
            lambda _: model(input_ids=prompt_input_ids, use_cache=True),
            lambda: None,
            model=model,
            stage="prefill",
            warmup=args.warmup,
            iterations=args.iterations,
            tokens_per_iteration=args.batch_size * args.prompt_tokens,
            device=device,
        )
        decode = _measure_cuda(
            lambda state: model(input_ids=decode_input_ids, state=state, use_cache=True),
            lambda: _clone_state(prefix_state),
            model=model,
            stage="decode",
            warmup=args.warmup,
            iterations=args.iterations,
            tokens_per_iteration=args.batch_size,
            device=device,
        )
        observed = _require_consistent_backend_observations(
            {
                "correctness": correctness_backends,
                "prefill": set(prefill["observed_backends"]),
                "decode": set(decode["observed_backends"]),
            }
        )
        if not operator_provenance:
            raise RuntimeError("FlashRWKV benchmark result lacks validated public runtime provenance")

    repo_root = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": 2,
        "benchmark": "rwkv7_standalone_model_only",
        "scope": "standalone diagnostic; recorded hardware only, with no acceptance-hardware claim",
        "source": _source_provenance(repo_root),
        "artifact": artifact_provenance,
        "runtime": {
            "dtype": args.dtype,
            "gemm_accumulation": "fp32" if args.dtype == "float32" else "model default",
            "required_provider": _REQUIRED_PROVIDER,
            "observed_backends": sorted(observed),
            "wkv_input_output": "float16" if args.dtype == "float32" else args.dtype,
            "wkv_mode": "fp32io16",
            "wkv_state": "float32",
            "validated_operator_provenance": operator_provenance,
            "provider_packages": {
                name: provenance
                for name in ("flash-rwkv", "flash-linear-attention")
                if (provenance := _distribution_provenance(name)) is not None
            },
        },
        "hardware": _hardware_provenance(device),
        "shape": {
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "correctness_decode_tokens": args.decode_tokens,
            "timed_decode_tokens_per_iteration": 1,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "head_size": config.head_size,
            "input_ids": full_input_ids.cpu().tolist(),
            "input_ids_source": input_ids_source,
        },
        "correctness_gate": correctness,
        "measurements": {"prefill": prefill, "decode": decode},
        "measurement_boundary": {
            "clock": "CUDA events with explicit device synchronization",
            "included": ["AutoModelForCausalLM forward", "returned recurrent-state construction"],
            "excluded": [
                "model/config loading",
                "input construction",
                "correctness gate",
                "decode state cloning",
                "CUDA event construction",
                "JSON serialization and disk write",
            ],
            "decode_state": "each timed one-token decode starts from a clone of the same prefill state",
            "memory": (
                "per-iteration operation-only peak from the PyTorch allocator; setup state is in the baseline, "
                "and native external allocations may be absent"
            ),
        },
        "seed": args.seed,
        "local_files_only": args.local_files_only,
        "command": [sys.executable, *sys.argv],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
