# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Run RWKV-7 causal-LM pretraining through the standard ``run_clm.py`` path and validate its evidence."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import AutoConfig
from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import validate_rwkv7_artifact_in_subprocess
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME


_RUN_CLM_PATH = Path(__file__).parents[1] / "language-modeling" / "run_clm.py"
_VALIDATION_KEY = "rwkv7_validation"
_SUMMARY_NAME = "rwkv7_pretraining_validation.json"
_CONFIG_EXAMPLE = """
The JSON object is forwarded to examples/pytorch/language-modeling/run_clm.py after removing:

  "rwkv7_validation": {
    "checkpoint_resume_step": 100,
    "minimum_logged_steps": 2,
    "validation_input_ids": [1, 2, 3],
    "validation_max_new_tokens": 4,
    "device": "cuda",
    "dtype": "bfloat16"
  }

The RWKV-7 model always requires the pinned public FLA FlashRWKV provider. Typical run_clm keys include
model_name_or_path, train_file or dataset_name, output_dir, do_train=true, block_size,
per_device_train_batch_size, learning_rate, max_steps, logging_strategy="steps", and logging_steps.
"""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_CONFIG_EXAMPLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", type=Path, help="JSON config for run_clm.py plus an rwkv7_validation object.")
    parser.add_argument("--output-dir", type=Path, help="One-off override for output_dir.")
    parser.add_argument("--max-steps", type=int, help="One-off override for the final max_steps.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume one phase from this checkpoint instead of running the configured two-phase checkpoint test.",
    )
    return parser.parse_args(argv)


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read RWKV-7 pretraining config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("RWKV-7 pretraining config must be a JSON object.")
    run_config = deepcopy(payload)
    validation = run_config.pop(_VALIDATION_KEY, None)
    if not isinstance(validation, dict):
        raise ValueError(f"RWKV-7 pretraining config requires an `{_VALIDATION_KEY}` object.")
    if run_config.get("token"):
        raise ValueError(
            "Do not store a Hugging Face token in the pretraining config; use the authenticated environment."
        )
    return run_config, validation


def _positive_int(mapping: dict[str, Any], name: str, *, minimum: int = 1) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"`{name}` must be an integer greater than or equal to {minimum}.")
    return value


def _validate_config(run_config: dict[str, Any], validation: dict[str, Any]) -> None:
    for name in ("model_name_or_path", "output_dir"):
        if not isinstance(run_config.get(name), str) or not run_config[name].strip():
            raise ValueError(f"RWKV-7 pretraining requires a non-empty `{name}`.")
    if run_config.get("do_train") is not True:
        raise ValueError("RWKV-7 pretraining requires `do_train: true`.")
    final_steps = _positive_int(run_config, "max_steps", minimum=2)
    minimum_logged_steps = _positive_int(validation, "minimum_logged_steps", minimum=2)
    if minimum_logged_steps > final_steps:
        raise ValueError("`minimum_logged_steps` cannot exceed `max_steps`.")
    logging_steps = _positive_int(run_config, "logging_steps")
    if run_config.get("logging_strategy", "steps") != "steps":
        raise ValueError("RWKV-7 gradient evidence requires `logging_strategy: steps`.")
    if final_steps // logging_steps < minimum_logged_steps:
        raise ValueError(
            "`logging_steps` is too large to produce the required `minimum_logged_steps` gradient evidence."
        )

    model_config = AutoConfig.from_pretrained(run_config["model_name_or_path"])
    if model_config.model_type != "rwkv7":
        raise ValueError(f"Expected an RWKV-7 artifact, got model_type={model_config.model_type!r}.")
    if hasattr(model_config, "wkv_backend"):
        raise ValueError("RWKV-7 artifacts must not serialize a selectable `wkv_backend` runtime escape.")

    input_ids = validation.get("validation_input_ids", [1, 2, 3])
    if not isinstance(input_ids, list) or not input_ids or not all(isinstance(token, int) for token in input_ids):
        raise ValueError("`validation_input_ids` must be a non-empty list of integers.")
    if min(input_ids) < 0 or max(input_ids) >= model_config.vocab_size:
        raise ValueError("`validation_input_ids` contains a token outside the artifact vocabulary.")
    _positive_int(validation, "validation_max_new_tokens")
    if not isinstance(validation.get("device"), str) or not validation["device"]:
        raise ValueError("`device` must be a non-empty torch device string.")
    if validation.get("dtype") not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("`dtype` must be auto, bfloat16, float16, or float32.")

    checkpoint_step = validation.get("checkpoint_resume_step")
    if checkpoint_step is not None and (
        not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or not 1 <= checkpoint_step < final_steps
    ):
        raise ValueError("`checkpoint_resume_step` must be an integer between 1 and max_steps - 1.")


def _checkpoint_evidence(checkpoint: Path, expected_step: int | None = None) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise RuntimeError(f"RWKV-7 resume checkpoint does not exist: {checkpoint}")
    required_files = ("config.json", OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME)
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if not ((checkpoint / SAFE_WEIGHTS_NAME).is_file() or (checkpoint / WEIGHTS_NAME).is_file()):
        missing.append(f"{SAFE_WEIGHTS_NAME} or {WEIGHTS_NAME}")
    if not list(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if missing:
        raise RuntimeError(f"RWKV-7 checkpoint is incomplete: missing {missing} in {checkpoint}.")
    state = json.loads((checkpoint / TRAINER_STATE_NAME).read_text(encoding="utf-8"))
    global_step = state.get("global_step")
    if expected_step is not None and global_step != expected_step:
        raise RuntimeError(f"Checkpoint {checkpoint} has global_step={global_step}; expected {expected_step}.")
    return {
        "files": sorted(path.name for path in checkpoint.iterdir() if path.is_file()),
        "global_step": global_step,
        "path": str(checkpoint.resolve()),
    }


def _finite_positive_history(history: list[dict[str, Any]], key: str) -> list[float]:
    values = [entry[key] for entry in history if key in entry]
    if not values or not all(
        isinstance(value, int | float) and math.isfinite(value) and value > 0 for value in values
    ):
        raise RuntimeError(f"Trainer history must contain only finite positive `{key}` values; got {values!r}.")
    return [float(value) for value in values]


def _training_evidence(output_dir: Path, expected_steps: int, minimum_logged_steps: int) -> dict[str, Any]:
    state_path = output_dir / TRAINER_STATE_NAME
    metrics_path = output_dir / "train_results.json"
    for path in (state_path, metrics_path, output_dir / "config.json"):
        if not path.is_file():
            raise RuntimeError(f"RWKV-7 pretraining output is missing {path.name}: {output_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if state.get("global_step") != expected_steps:
        raise RuntimeError(f"Training stopped at global_step={state.get('global_step')}; expected {expected_steps}.")
    losses = _finite_positive_history(state.get("log_history", []), "loss")
    gradient_norms = _finite_positive_history(state.get("log_history", []), "grad_norm")
    if len(losses) < minimum_logged_steps or len(gradient_norms) < minimum_logged_steps:
        raise RuntimeError(
            f"Training logged {len(losses)} losses and {len(gradient_norms)} gradient norms; "
            f"expected at least {minimum_logged_steps} of each."
        )
    training_loss = metrics.get("train_loss")
    if not isinstance(training_loss, int | float) or not math.isfinite(training_loss) or training_loss <= 0:
        raise RuntimeError(f"Final train_loss must be finite and positive, got {training_loss!r}.")
    return {
        "final_global_step": state["global_step"],
        "gradient_norms": gradient_norms,
        "logged_losses": losses,
        "train_loss": float(training_loss),
        "train_runtime": metrics.get("train_runtime"),
    }


def _run_clm(run_config: dict[str, Any], phase_name: str) -> dict[str, Any]:
    if not _RUN_CLM_PATH.is_file():
        raise RuntimeError(f"Standard Transformers CLM entry point is missing: {_RUN_CLM_PATH}")
    with tempfile.TemporaryDirectory(prefix="rwkv7-run-clm-") as temporary_dir:
        config_path = Path(temporary_dir) / f"{phase_name}.json"
        config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        subprocess.run([sys.executable, str(_RUN_CLM_PATH), str(config_path)], check=True)
    return {
        "config_sha256": _sha256_json(run_config),
        "entrypoint": str(_RUN_CLM_PATH.resolve()),
        "phase": phase_name,
    }


def _run(run_config: dict[str, Any], validation: dict[str, Any], explicit_resume: Path | None) -> dict[str, Any]:
    output_dir = Path(run_config["output_dir"]).expanduser()
    final_steps = run_config["max_steps"]
    checkpoint_step = validation.get("checkpoint_resume_step")
    phase_results = []
    resume_evidence = None

    if explicit_resume is not None:
        resume_evidence = _checkpoint_evidence(explicit_resume)
        resumed_config = deepcopy(run_config)
        resumed_config["resume_from_checkpoint"] = str(explicit_resume.resolve())
        phase_results.append(_run_clm(resumed_config, "resume"))
    elif checkpoint_step is not None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to start the two-phase run in non-empty output_dir={output_dir}. "
                "Use --resume-from-checkpoint to continue an existing checkpoint."
            )
        initial_config = deepcopy(run_config)
        initial_config.update(
            {
                "max_steps": checkpoint_step,
                "save_strategy": "steps",
                "save_steps": checkpoint_step,
            }
        )
        phase_results.append(_run_clm(initial_config, "checkpoint"))
        checkpoint = output_dir / f"checkpoint-{checkpoint_step}"
        resume_evidence = _checkpoint_evidence(checkpoint, checkpoint_step)
        resumed_config = deepcopy(run_config)
        resumed_config["resume_from_checkpoint"] = str(checkpoint.resolve())
        phase_results.append(_run_clm(resumed_config, "resume"))
    else:
        phase_results.append(_run_clm(run_config, "train"))

    training = _training_evidence(output_dir, final_steps, validation["minimum_logged_steps"])
    artifact = validate_rwkv7_artifact_in_subprocess(
        output_dir,
        input_ids=validation.get("validation_input_ids", [1, 2, 3]),
        max_new_tokens=validation["validation_max_new_tokens"],
        device=validation["device"],
        dtype=validation["dtype"],
    )
    return {
        "artifact_validation": artifact,
        "model_name_or_path": run_config["model_name_or_path"],
        "output_dir": str(output_dir.resolve()),
        "phases": phase_results,
        "resume": {
            "checkpoint": resume_evidence,
            "validated": resume_evidence is not None,
        },
        "training": training,
    }


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_config, validation = _load_config(args.config)
    if args.output_dir is not None:
        run_config["output_dir"] = str(args.output_dir)
    if args.max_steps is not None:
        run_config["max_steps"] = args.max_steps
    _validate_config(run_config, validation)
    configured_resume = run_config.get("resume_from_checkpoint")
    explicit_resume = args.resume_from_checkpoint
    if explicit_resume is None and isinstance(configured_resume, str) and configured_resume:
        explicit_resume = Path(configured_resume)
    summary = _run(run_config, validation, explicit_resume)
    summary_path = Path(run_config["output_dir"]).expanduser() / _SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"RWKV7_PRETRAINING_VALIDATION={json.dumps(summary, sort_keys=True)}")


if __name__ == "__main__":
    main()
