# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Validate a real RWKV-7 artifact with a tokenizer-free fixed token workload.

This is deliberately a model-contract check rather than a text-generation demo.
The 1.5B checkpoint used by the RWKV-7 integration has no accepted publication
tokenizer, so the validation boundary is explicit token ids.  The model is
loaded through the public Auto classes in this fresh process, while the
private trace hook is enabled only for the independent WKV oracle comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.rwkv7.modeling_rwkv7 import (
    RWKV7_FLA_REVISION,
    RWKV7_FLASH_RWKV_REVISION,
    _load_fla_rwkv7_contract,
    _rwkv7_flash,
    rwkv7_reference,
    validate_rwkv7_runtime_provenance,
)


CHECKPOINT_SHA256 = "737079d81865801fd85e5459488d89a36d5304a524e890244eb83d44f531c89c"
SOURCE_REVISION = "bd552d5e6aaaad88196629f7eb8dc8e24a644484"
REFERENCE_REVISIONS = {
    "rwkv_lm": "bd552d5e6aaaad88196629f7eb8dc8e24a644484",
    "albatross": "ee3308f6922e59f2166c7fac3c5a192340a2b48e",
}

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-ids", default="1,7,11,3")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bfloat16")
    parser.add_argument("--optimizer-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return payload


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _rrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(torch.finfo(torch.float32).eps)
    return (difference.square().mean().sqrt() / denominator).item()


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.float()
    return {
        "max": values.max().item(),
        "mean": values.mean().item(),
        "min": values.min().item(),
        "norm": values.norm().item(),
    }


def _token_stats(tensor: torch.Tensor) -> list[dict[str, Any]]:
    return [
        {"token": token_index, "stats": _tensor_stats(tensor[:, token_index : token_index + 1])}
        for token_index in range(tensor.shape[1])
    ]


def _parse_input_ids(encoded: str, vocab_size: int) -> list[int]:
    try:
        input_ids = [int(value) for value in encoded.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError("--input-ids must be a comma-separated list of integers") from error
    if len(input_ids) < 3:
        raise ValueError("The fixed workload must contain at least three token ids")
    if min(input_ids) < 0 or max(input_ids) >= vocab_size:
        raise ValueError(f"Fixed token ids must be in [0, {vocab_size}); got {input_ids}")
    return input_ids


def _artifact_evidence(artifact: Path, checkpoint: Path) -> dict[str, Any]:
    conversion = _json(artifact / "rwkv7_conversion.json")
    if conversion.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("The artifact checkpoint binding does not match the accepted 1.5B checkpoint SHA-256")
    if conversion.get("source_revision") != SOURCE_REVISION:
        raise RuntimeError("The artifact source revision is not the accepted RWKV-LM revision")
    if conversion.get("tokenizer_files") != {}:
        raise RuntimeError("This fixed-token validation must not silently accept an unreviewed tokenizer")
    if conversion.get("wkv_provider") != "flash_rwkv":
        raise RuntimeError("RWKV-7 artifact does not declare the canonical FlashRWKV provider")
    config_payload = _json(artifact / "config.json")
    if config_payload.get("auto_map"):
        raise RuntimeError("RWKV-7 artifact must load without auto_map or remote code")
    if _sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("The supplied checkpoint SHA-256 does not match the accepted 1.5B checkpoint")
    tracked_files = {
        path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in sorted(artifact.iterdir())
        if path.is_file()
    }
    return {
        "path": str(artifact.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config": config_payload,
        "conversion": conversion,
        "files": tracked_files,
        "tokenizer_available": False,
    }


def _enable_trace(model: torch.nn.Module, traces: list[list[dict[str, torch.Tensor]]]) -> None:
    for block, trace in zip(model.model.blocks, traces, strict=True):
        block.att._rwkv7_trace = trace


def _disable_trace(model: torch.nn.Module) -> None:
    for block in model.model.blocks:
        block.att._rwkv7_trace = None


def _oracle_trace(
    traces: list[list[dict[str, Any]]],
    *,
    head_size: int,
    rrmse_limit: float = 0.008,
) -> dict[str, Any]:
    layer_records = []
    maximum_output_error = 0.0
    maximum_state_error = 0.0
    for layer_index, layer_trace in enumerate(traces):
        if not layer_trace:
            raise RuntimeError(f"RWKV-7 layer {layer_index} did not emit a trace record")
        for call_index, entry in enumerate(layer_trace):
            raw_decay_logits = entry["decay_logits"]
            log_decay = -math.exp(-0.5) * torch.sigmoid(raw_decay_logits)
            oracle_output, oracle_state = rwkv7_reference(
                entry["receptance"],
                log_decay,
                entry["key"],
                entry["value"],
                entry["a"],
                entry["b"],
                entry["initial_state"],
                head_size=head_size,
            )
            output_error = _rrmse(entry["wkv_output"], oracle_output)
            state_error = _rrmse(entry["final_state"], oracle_state)
            if entry["provider"] != "flash_rwkv":
                raise RuntimeError(
                    f"RWKV-7 independent oracle trace used provider={entry['provider']!r} at layer={layer_index}"
                )
            maximum_output_error = max(maximum_output_error, output_error)
            maximum_state_error = max(maximum_state_error, state_error)
            if output_error > rrmse_limit or state_error > rrmse_limit:
                raise RuntimeError(
                    "RWKV-7 independent oracle mismatch at "
                    f"layer={layer_index}, call={call_index}: "
                    f"output_rrmse={output_error}, state_rrmse={state_error}, limit={rrmse_limit}"
                )
            retention = torch.exp(-math.exp(-0.5) * torch.sigmoid(raw_decay_logits.float()))
            token_records = []
            for token_index in range(raw_decay_logits.shape[1]):
                token_slice = slice(token_index, token_index + 1)
                token_records.append(
                    {
                        "D": _tensor_stats(retention[:, token_slice]),
                        "a": _tensor_stats(entry["a"][:, token_slice]),
                        "b": _tensor_stats(entry["b"][:, token_slice]),
                        "k": _tensor_stats(entry["key"][:, token_slice]),
                        "layer_output": _tensor_stats(entry["layer_output"][:, token_slice]),
                        "r": _tensor_stats(entry["receptance"][:, token_slice]),
                        "raw_decay_logits": _tensor_stats(raw_decay_logits[:, token_slice]),
                        "token": token_index,
                        "v": _tensor_stats(entry["value"][:, token_slice]),
                        "wkv_output": _tensor_stats(entry["wkv_output"][:, token_slice]),
                    }
                )
            layer_records.append(
                {
                    "call": call_index,
                    "decay_logits": _tensor_stats(raw_decay_logits),
                    "effective_retention": _tensor_stats(retention),
                    "layer": layer_index,
                    "output_rrmse": output_error,
                    "output_stats": _tensor_stats(entry["wkv_output"]),
                    "state_rrmse": state_error,
                    "state_stats": _tensor_stats(entry["final_state"]),
                    "token_records": token_records,
                    "tokens": int(raw_decay_logits.shape[1]),
                }
            )
    return {
        "calls": layer_records,
        "max_output_rrmse": maximum_output_error,
        "max_state_rrmse": maximum_state_error,
        "oracle": "transformers.models.rwkv7.modeling_rwkv7.rwkv7_reference",
        "retention_formula": "exp(-exp(-0.5) * sigmoid(raw_decay_logits))",
    }


def _run_traced(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    state: tuple[torch.Tensor, ...] | None = None,
) -> tuple[Any, list[list[dict[str, Any]]]]:
    traces = [[] for _ in model.model.blocks]
    _enable_trace(model, traces)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, state=state, use_cache=True)
    finally:
        _disable_trace(model)
    return output, traces


def _state_rrmse(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    return max(_rrmse(left, right) for left, right in zip(actual, expected, strict=True))


def _clone_state(state: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(component.clone() for component in state)


def _continuation_trace_diagnostics(
    full_trace: list[list[dict[str, Any]]],
    staged_trace: list[list[dict[str, Any]]],
    *,
    split: int,
    rrmse_limit: float,
) -> dict[str, Any]:
    """Find the first layer/token/tensor that breaks staged continuation."""
    fields = (
        "receptance",
        "decay_logits",
        "key",
        "value",
        "a",
        "b",
        "wkv_output",
        "layer_output",
    )
    first_failure = None
    maximum = dict.fromkeys((*fields, "final_state"), 0.0)
    for layer_index, (full_layer, staged_layer) in enumerate(
        zip(full_trace, staged_trace, strict=True)
    ):
        if len(full_layer) != 1:
            raise RuntimeError(
                f"RWKV-7 continuation diagnostics expected one full trace call at layer={layer_index}"
            )
        full_entry = full_layer[0]
        for call_index, staged_entry in enumerate(staged_layer):
            token_index = split + call_index
            for field in fields:
                error = _rrmse(
                    staged_entry[field],
                    full_entry[field][:, token_index : token_index + 1],
                )
                maximum[field] = max(maximum[field], error)
                if first_failure is None and error > rrmse_limit:
                    first_failure = {
                        "layer": layer_index,
                        "rrmse": error,
                        "tensor": field,
                        "token": token_index,
                    }
            if call_index == len(staged_layer) - 1:
                error = _rrmse(staged_entry["final_state"], full_entry["final_state"])
                maximum["final_state"] = max(maximum["final_state"], error)
                if first_failure is None and error > rrmse_limit:
                    first_failure = {
                        "layer": layer_index,
                        "rrmse": error,
                        "tensor": "final_state",
                        "token": token_index,
                    }
    return {
        "first_failure": first_failure,
        "max_rrmse": maximum,
        "rrmse_limit": rrmse_limit,
    }


def _inference_contract(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    full, full_trace = _run_traced(model, input_ids)
    full_oracle = _oracle_trace(full_trace, head_size=model.config.head_size, rrmse_limit=0.002)
    if full.state is None:
        raise RuntimeError("RWKV-7 full fixed-token call did not return a recurrent state")

    split = input_ids.shape[1] - 2
    prefix, prefix_trace = _run_traced(model, input_ids[:, :split])
    if prefix.state is None:
        raise RuntimeError("RWKV-7 prefix call did not return a recurrent state")
    staged_outputs = []
    staged_traces: list[list[dict[str, Any]]] = [[] for _ in model.model.blocks]
    state = prefix.state
    for token_index in range(split, input_ids.shape[1]):
        step, step_trace = _run_traced(model, input_ids[:, token_index : token_index + 1], state=state)
        if step.state is None:
            raise RuntimeError("RWKV-7 staged decode did not return a recurrent state")
        staged_outputs.append(step.logits)
        state = step.state
        for layer_index, layer_trace in enumerate(step_trace):
            staged_traces[layer_index].extend(layer_trace)
    staged_logits = torch.cat(staged_outputs, dim=1)
    staged_oracle = _oracle_trace(staged_traces, head_size=model.config.head_size, rrmse_limit=0.002)
    continuation_rrmse = _rrmse(staged_logits, full.logits[:, split:])
    state_continuation_rrmse = _state_rrmse(state, full.state)
    continuation_diagnostics = _continuation_trace_diagnostics(
        full_trace,
        staged_traces,
        split=split,
        rrmse_limit=0.002,
    )
    if continuation_rrmse > 0.002 or state_continuation_rrmse > 0.002:
        raise RuntimeError(
            "RWKV-7 one-shot/staged continuation mismatch: "
            f"logits_rrmse={continuation_rrmse}, state_rrmse={state_continuation_rrmse}, "
            f"first_trace_failure={continuation_diagnostics['first_failure']}, "
            f"max_trace_rrmse={continuation_diagnostics['max_rrmse']}"
        )

    zero_state_changed = any(component.float().abs().max().item() > 0 for component in full.state)
    if not zero_state_changed:
        raise RuntimeError("RWKV-7 zero-initialized fixed-token workload did not update recurrent state")

    initial_state = model.model._init_state(1, model.dtype, device)
    nonzero_state = _clone_state(initial_state)
    nonzero_snapshot = _clone_state(nonzero_state)
    with torch.no_grad():
        nonzero_state[0].fill_(0.01)
        nonzero_state[1].fill_(0.02)
        nonzero_state[2].fill_(0.03)
    nonzero_input = input_ids[:, :2]
    nonzero, nonzero_trace = _run_traced(model, nonzero_input, state=nonzero_state)
    nonzero_oracle = _oracle_trace(nonzero_trace, head_size=model.config.head_size)
    if nonzero.state is None or _state_rrmse(nonzero.state, nonzero_snapshot) == 0:
        raise RuntimeError("RWKV-7 nonzero initial state was not propagated")

    with torch.inference_mode():
        first_generation = model.generate(input_ids[:, :split], max_new_tokens=2, do_sample=False, use_cache=True)
        second_generation = model.generate(input_ids[:, :split], max_new_tokens=2, do_sample=False, use_cache=True)
    if not torch.equal(first_generation, second_generation):
        raise RuntimeError("RWKV-7 fixed-token generation is not deterministic")

    observed_backends = sorted({block.att.last_wkv_backend for block in model.model.blocks})
    if observed_backends != ["flash_rwkv"]:
        raise RuntimeError(f"RWKV-7 fixed-token inference selected {observed_backends!r}")
    contract = {
        "device": str(device),
        "full_tokens": int(input_ids.shape[1]),
        "generation_ids": first_generation.cpu().tolist(),
        "nonzero_initial_state": {
            "oracle": nonzero_oracle,
            "tokens": int(nonzero_input.shape[1]),
        },
        "observed_backends": observed_backends,
        "one_shot_logits": _token_stats(full.logits),
        "one_shot_oracle": full_oracle,
        "staged_oracle": staged_oracle,
        "staged_trace_diagnostics": continuation_diagnostics,
        "staged_vs_one_shot_logits_rrmse": continuation_rrmse,
        "staged_vs_one_shot_state_rrmse": state_continuation_rrmse,
        "staged_logits": _token_stats(staged_logits),
        "zero_initial_state": {
            "oracle": full_oracle,
            "tokens": int(input_ids.shape[1]),
        },
    }
    return contract


def _finite_grad_norm(model: torch.nn.Module) -> tuple[float, bool]:
    squared_norm = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float64)
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        finite = finite and bool(torch.isfinite(parameter.grad).all())
        squared_norm += parameter.grad.float().square().sum().double()
    return squared_norm.sqrt().item(), finite


def _gradient_contract(*, device: torch.device, head_size: int) -> dict[str, Any]:
    """Compare native fp32io16 gradients with the independent Python recurrence."""
    torch.manual_seed(20260803)
    batch_size, sequence_length, num_heads = 1, 3, 2
    hidden_size = num_heads * head_size
    product_inputs = [
        torch.randn(
            batch_size,
            sequence_length,
            hidden_size,
            device=device,
            dtype=torch.float16,
            requires_grad=True,
        )
        for _ in range(6)
    ]
    product_state = torch.randn(
        batch_size,
        num_heads,
        head_size,
        head_size,
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    oracle_inputs = [tensor.detach().clone().requires_grad_() for tensor in product_inputs]
    oracle_state = product_state.detach().clone().requires_grad_()
    contract = _load_fla_rwkv7_contract()
    product_output, product_final_state = _rwkv7_flash(
        *product_inputs,
        product_state,
        head_size,
        contract=contract,
    )
    oracle_log_decay = -math.exp(-0.5) * torch.sigmoid(oracle_inputs[1].float())
    oracle_output, oracle_final_state = rwkv7_reference(
        oracle_inputs[0],
        oracle_log_decay,
        *oracle_inputs[2:],
        oracle_state,
        head_size=head_size,
    )
    upstream_output = torch.linspace(
        -1,
        1,
        product_output.numel(),
        device=device,
        dtype=product_output.dtype,
    ).reshape_as(product_output)
    upstream_state = torch.linspace(
        -1,
        1,
        product_final_state.numel(),
        device=device,
        dtype=product_final_state.dtype,
    ).reshape_as(product_final_state)
    (product_output * upstream_output).sum().add_((product_final_state * upstream_state).sum()).backward()
    (oracle_output * upstream_output).sum().add_((oracle_final_state * upstream_state).sum()).backward()
    gradient_records = {}
    for name, product, oracle in zip(
        ("dr", "dz", "dk", "dv", "da", "db"), product_inputs, oracle_inputs, strict=True
    ):
        error = _rrmse(product.grad, oracle.grad)
        gradient_records[name] = {
            "max_abs": (product.grad.float() - oracle.grad.float()).abs().max().item(),
            "rrmse": error,
        }
    state_error = _rrmse(product_state.grad, oracle_state.grad)
    gradient_records["dinitial_state"] = {
        "max_abs": (product_state.grad.float() - oracle_state.grad.float()).abs().max().item(),
        "rrmse": state_error,
    }
    output_error = _rrmse(product_output, oracle_output)
    state_forward_error = _rrmse(product_final_state, oracle_final_state)
    maximum_gradient_error = max(record["rrmse"] for record in gradient_records.values())
    if maximum_gradient_error > 2e-5:
        first_failure = next(
            name for name, record in gradient_records.items() if record["rrmse"] > 2e-5
        )
        raise RuntimeError(
            "RWKV-7 fp32io16 gradient mismatch: "
            f"tensor={first_failure}, rrmse={gradient_records[first_failure]['rrmse']}"
        )
    if contract.get_last_provider() != "flash_rwkv" or contract.get_last_kernel() != "pretrain_recurrent_fp32io16_forward":
        raise RuntimeError(
            "RWKV-7 gradient contract selected invalid runtime evidence: "
            f"provider={contract.get_last_provider()!r}, kernel={contract.get_last_kernel()!r}"
        )
    return {
        "forward_output_rrmse": output_error,
        "forward_state_rrmse": state_forward_error,
        "gradients": gradient_records,
        "gradient_rrmse_limit": 2e-5,
        "head_size": head_size,
        "kernel": contract.get_last_kernel(),
        "provider": contract.get_last_provider(),
        "sequence_length": sequence_length,
        "upstream_dout": "torch.linspace(-1, 1, output_and_state_numel)",
    }


def _training_contract(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=input_ids, use_cache=True)
        if output.state is None:
            raise RuntimeError("RWKV-7 training call did not return recurrent state")
        # A deterministic all-ones upstream dout keeps this check independent
        # of a tokenizer and includes both the output and final-state paths.
        loss = output.logits.float().mean() * 1e-6
        loss = loss + sum(component.float().mean() for component in output.state) * 1e-6
        loss.backward()
        grad_norm, gradients_finite = _finite_grad_norm(model)
        before = model.model.blocks[0].att.w0.detach().clone()
        contract = _load_fla_rwkv7_contract()
        provider = contract.get_last_provider()
        kernel = contract.get_last_kernel()
        optimizer.step()
        torch.cuda.synchronize()
        update_norm = (model.model.blocks[0].att.w0.detach() - before).float().norm().item()
        kernels = sorted({str(block.att.last_wkv_backend) for block in model.model.blocks})
        if not math.isfinite(loss.item()) or not gradients_finite or not math.isfinite(grad_norm):
            raise RuntimeError(f"RWKV-7 training step {step_index + 1} produced non-finite evidence")
        if (
            update_norm <= 0
            or kernels != ["flash_rwkv"]
            or provider != "flash_rwkv"
            or kernel != "pretrain_recurrent_fp32io16_forward"
        ):
            raise RuntimeError(
                f"RWKV-7 training step {step_index + 1} selected invalid runtime evidence: "
                f"provider={provider!r}, kernel={kernel!r}, backends={kernels!r}, update_norm={update_norm}"
            )
        history.append(
            {
                "backends": kernels,
                "grad_norm": grad_norm,
                "gradients_finite": gradients_finite,
                "kernel": kernel,
                "loss": loss.item(),
                "provider": provider,
                "step": step_index + 1,
                "w0_update_norm": update_norm,
            }
        )
    return {
        "fixed_upstream_dout": "ones, output and final recurrent state scaled by 1e-6",
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "steps": history,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.optimizer_steps < 3:
        raise ValueError("--optimizer-steps must be at least 3")
    artifact = args.artifact.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not artifact.is_dir():
        raise FileNotFoundError(f"RWKV-7 artifact directory does not exist: {artifact}")
    artifact_evidence = _artifact_evidence(artifact, checkpoint)
    config = AutoConfig.from_pretrained(artifact, local_files_only=True)
    input_ids = _parse_input_ids(args.input_ids, config.vocab_size)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("RWKV-7 real fixed-token validation requires CUDA")
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        artifact,
        config=config,
        dtype=_DTYPES[args.dtype],
        local_files_only=True,
        output_loading_info=True,
    )
    load_errors = {
        name: loading_info.get(name, [])
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    if load_errors:
        raise RuntimeError(f"RWKV-7 artifact did not strict-load: {load_errors}")
    if type(model).__name__ != "Rwkv7ForCausalLM" or type(config).__name__ != "Rwkv7Config":
        raise RuntimeError("RWKV-7 artifact did not resolve through the native Auto classes")
    model.to(device)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    provenance = validate_rwkv7_runtime_provenance()
    gradient = _gradient_contract(device=device, head_size=model.config.head_size)
    inference = _inference_contract(model, input_tensor, device=device)
    training = _training_contract(
        model,
        input_tensor,
        steps=args.optimizer_steps,
        learning_rate=args.learning_rate,
    )
    repo_root = Path(__file__).resolve().parents[3]
    report = {
        "artifact": artifact_evidence,
        "checkpoint": {"path": str(checkpoint), "sha256": CHECKPOINT_SHA256},
        "command": [sys.executable, *sys.argv],
        "hardware": {
            "compute_capability": ".".join(str(value) for value in torch.cuda.get_device_capability(device)),
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "input_ids": input_ids,
        "precision": {
            "dtype": args.dtype,
            "gemm_accumulation": "fp32" if args.dtype == "float32" else "model default",
            "wkv_input_output": "float16" if args.dtype == "float32" else args.dtype,
            "wkv_mode": "fp32io16",
            "wkv_state": "float32",
        },
        "references": REFERENCE_REVISIONS,
        "repository": {
            "dirty": bool(_git(repo_root, "status", "--short")),
            "revision": _git(repo_root, "rev-parse", "HEAD"),
        },
        "runtime": {
            "fla_revision": RWKV7_FLA_REVISION,
            "flash_rwkv_revision": RWKV7_FLASH_RWKV_REVISION,
            "provenance": provenance,
        },
        "status": "passed",
        "training": training,
        "inference": inference,
        "gradient": gradient,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
