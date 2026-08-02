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
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from transformers import AutoConfig, AutoModelForCausalLM


_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Explicit local Hugging Face artifact path or Hub repository ID.",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON result path.")
    parser.add_argument("--backend", choices=("auto", "reference", "flash_rwkv"), default="auto")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
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


def _require_observed_backend(requested: str, observed: set[str]) -> None:
    if not observed:
        raise RuntimeError("RWKV7 benchmark could not observe the selected WKV backend")
    if len(observed) != 1:
        raise RuntimeError(f"RWKV7 layers selected inconsistent WKV backends: {sorted(observed)}")
    if requested != "auto" and observed != {requested}:
        raise RuntimeError(f"Explicit {requested!r} backend request failed closed; observed {sorted(observed)!r}")


def _correctness_gate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_tokens: int,
    requested_backend: str,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
    one_shot = model(input_ids=input_ids, use_cache=True)
    prefix = model(input_ids=input_ids[:, :prompt_tokens], use_cache=True)
    if prefix.state is None:
        raise RuntimeError("RWKV7 prefill did not return recurrent state with use_cache=True")

    state = prefix.state
    staged_logits = []
    for token_index in range(prompt_tokens, input_ids.shape[1]):
        step = model(
            input_ids=input_ids[:, token_index : token_index + 1],
            state=state,
            use_cache=True,
        )
        if step.state is None:
            raise RuntimeError("RWKV7 decode did not return recurrent state with use_cache=True")
        state = step.state
        staged_logits.append(step.logits)

    staged = torch.cat(staged_logits, dim=1)
    expected = one_shot.logits[:, prompt_tokens:]
    difference = (staged.float() - expected.float()).abs()
    if dtype == torch.float32:
        rtol, atol = 1e-5, 1e-5
    elif dtype == torch.float16:
        rtol, atol = 5e-3, 5e-3
    else:
        rtol, atol = 5e-2, 5e-2
    torch.testing.assert_close(staged, expected, rtol=rtol, atol=atol)

    observed = _observed_backends(model)
    _require_observed_backend(requested_backend, observed)
    return (
        {
            "passed": True,
            "comparison": "staged recurrent decode logits vs matching one-shot logits",
            "compared_tokens": input_ids.shape[1] - prompt_tokens,
            "max_abs_error": difference.max().item(),
            "mean_abs_error": difference.mean().item(),
            "rtol": rtol,
            "atol": atol,
            "observed_backends": sorted(observed),
        },
        prefix.state,
    )


def _measure_cuda(
    operation,
    setup,
    *,
    warmup: int,
    iterations: int,
    tokens_per_iteration: int,
    device: torch.device,
) -> dict[str, Any]:
    for _ in range(warmup):
        payload = setup()
        torch.cuda.synchronize(device)
        result = operation(payload)
        torch.cuda.synchronize(device)
        del result, payload

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iterations)]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    resident_before_bytes = torch.cuda.memory_allocated(device)
    samples_ms = []
    for start, end in events:
        payload = setup()
        torch.cuda.synchronize(device)
        start.record()
        result = operation(payload)
        end.record()
        torch.cuda.synchronize(device)
        samples_ms.append(start.elapsed_time(end))
        del result, payload

    summary = _latency_summary(samples_ms, tokens_per_iteration)
    summary.update(
        {
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "resident_before_bytes": resident_before_bytes,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_increment_bytes": torch.cuda.max_memory_allocated(device) - resident_before_bytes,
        }
    )
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_provenance(reference: str, config: Any) -> dict[str, Any]:
    local_path = Path(reference).expanduser()
    if local_path.exists():
        resolved = local_path.resolve()
        config_path = resolved / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Local model artifact has no config.json: {resolved}")
        weight_files = sorted(resolved.glob("*.safetensors"))
        return {
            "kind": "local_path",
            "input_reference": reference,
            "resolved_reference": str(resolved),
            "config_sha256": _sha256(config_path),
            "weight_files": [{"name": path.name, "size_bytes": path.stat().st_size} for path in weight_files],
            "hub_commit": getattr(config, "_commit_hash", None),
        }
    return {
        "kind": "hub_repo_id",
        "input_reference": reference,
        "resolved_reference": reference,
        "hub_commit": getattr(config, "_commit_hash", None),
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
    config.wkv_backend = args.backend
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    model.eval()

    generator = torch.Generator().manual_seed(args.seed)
    full_input_ids = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(args.batch_size, args.prompt_tokens + args.decode_tokens),
        generator=generator,
    ).to(device)
    prompt_input_ids = full_input_ids[:, : args.prompt_tokens]
    decode_input_ids = full_input_ids[:, args.prompt_tokens : args.prompt_tokens + 1]

    with torch.inference_mode():
        correctness, prefix_state = _correctness_gate(
            model,
            full_input_ids,
            args.prompt_tokens,
            args.backend,
            dtype,
        )
        torch.cuda.synchronize(device)
        prefill = _measure_cuda(
            lambda _: model(input_ids=prompt_input_ids, use_cache=True),
            lambda: None,
            warmup=args.warmup,
            iterations=args.iterations,
            tokens_per_iteration=args.batch_size * args.prompt_tokens,
            device=device,
        )
        decode = _measure_cuda(
            lambda state: model(input_ids=decode_input_ids, state=state, use_cache=True),
            lambda: _clone_state(prefix_state),
            warmup=args.warmup,
            iterations=args.iterations,
            tokens_per_iteration=args.batch_size,
            device=device,
        )
        observed = _observed_backends(model)
        _require_observed_backend(args.backend, observed)

    repo_root = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": 1,
        "benchmark": "rwkv7_standalone_model_only",
        "scope": "standalone diagnostic; recorded hardware only, with no acceptance-hardware claim",
        "source": _source_provenance(repo_root),
        "artifact": _artifact_provenance(args.model, config),
        "runtime": {
            "dtype": args.dtype,
            "requested_backend": args.backend,
            "observed_backends": sorted(observed),
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
