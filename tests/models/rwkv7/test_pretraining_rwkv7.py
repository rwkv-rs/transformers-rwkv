# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.util
import json
from pathlib import Path

import pytest

from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import SAFE_WEIGHTS_NAME


_PRETRAINING_PATH = Path(__file__).parents[3] / "examples" / "pytorch" / "rwkv7" / "run_rwkv7_pretraining.py"
_SPEC = importlib.util.spec_from_file_location("rwkv7_pretraining", _PRETRAINING_PATH)
assert _SPEC is not None and _SPEC.loader is not None
pretraining = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pretraining)


def _artifact(path: Path) -> Path:
    model = Rwkv7ForCausalLM(
        Rwkv7Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            head_size=4,
            context_length=16,
        )
    )
    model.save_pretrained(path)
    return path


def _checkpoint(path: Path, step: int) -> None:
    path.mkdir(parents=True)
    for name in ("config.json", SAFE_WEIGHTS_NAME, OPTIMIZER_NAME, SCHEDULER_NAME, "rng_state.pth"):
        (path / name).write_bytes(b"test")
    (path / TRAINER_STATE_NAME).write_text(json.dumps({"global_step": step}), encoding="utf-8")


def _final_output(path: Path, steps: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    history = [{"step": step, "loss": 2.0 / step, "grad_norm": 0.5 / step} for step in range(1, steps + 1)]
    (path / TRAINER_STATE_NAME).write_text(
        json.dumps({"global_step": steps, "log_history": history}), encoding="utf-8"
    )
    (path / "train_results.json").write_text(json.dumps({"train_loss": 0.75, "train_runtime": 1.25}), encoding="utf-8")


def _validation() -> dict:
    return {
        "checkpoint_resume_step": 2,
        "device": "cpu",
        "dtype": "float32",
        "minimum_logged_steps": 2,
        "validation_input_ids": [1, 2, 3],
        "validation_max_new_tokens": 2,
    }


def test_rwkv7_pretraining_rejects_legacy_selectable_backend_in_artifact(tmp_path) -> None:
    artifact = _artifact(tmp_path / "artifact")
    run_config = {
        "do_train": True,
        "logging_steps": 1,
        "max_steps": 4,
        "model_name_or_path": str(artifact),
        "output_dir": str(tmp_path / "output"),
    }

    pretraining._validate_config(run_config, _validation())

    artifact_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    artifact_config["wkv_backend"] = "flash_rwkv"
    (artifact / "config.json").write_text(json.dumps(artifact_config), encoding="utf-8")
    with pytest.raises(ValueError, match="must not serialize a selectable"):
        pretraining._validate_config(run_config, _validation())


def test_rwkv7_pretraining_runs_checkpoint_then_resume_and_records_evidence(tmp_path, monkeypatch) -> None:
    output = tmp_path / "output"
    run_config = {
        "do_train": True,
        "logging_steps": 1,
        "max_steps": 4,
        "model_name_or_path": str(tmp_path / "artifact"),
        "output_dir": str(output),
    }

    def fake_run_clm(config, phase):
        if phase == "checkpoint":
            _checkpoint(output / "checkpoint-2", 2)
        else:
            assert config["resume_from_checkpoint"] == str((output / "checkpoint-2").resolve())
            _final_output(output, 4)
        return {"config_sha256": phase, "entrypoint": "run_clm.py", "phase": phase}

    monkeypatch.setattr(pretraining, "_run_clm", fake_run_clm)
    monkeypatch.setattr(
        pretraining,
        "validate_rwkv7_artifact_in_subprocess",
        lambda *args, **kwargs: {
            "observed_wkv_backends": ["flash_rwkv"],
            "strict_load": True,
        },
    )

    result = pretraining._run(run_config, _validation(), explicit_resume=None)

    assert [phase["phase"] for phase in result["phases"]] == ["checkpoint", "resume"]
    assert result["resume"]["validated"]
    assert result["resume"]["checkpoint"]["global_step"] == 2
    assert result["training"]["final_global_step"] == 4
    assert len(result["training"]["logged_losses"]) == 4
    assert len(result["training"]["gradient_norms"]) == 4
    assert result["artifact_validation"] == {"observed_wkv_backends": ["flash_rwkv"], "strict_load": True}


def test_rwkv7_pretraining_refuses_to_reuse_nonempty_two_phase_output(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "unrelated.txt").write_text("preserve me", encoding="utf-8")
    run_config = {
        "do_train": True,
        "logging_steps": 1,
        "max_steps": 4,
        "model_name_or_path": str(tmp_path / "artifact"),
        "output_dir": str(output),
    }

    with pytest.raises(RuntimeError, match="Refusing to start the two-phase run"):
        pretraining._run(run_config, _validation(), explicit_resume=None)

    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize(
    "history_key, history",
    [
        ("loss", [{"loss": float("nan"), "grad_norm": 1.0}, {"loss": 1.0, "grad_norm": 1.0}]),
        ("grad_norm", [{"loss": 1.0, "grad_norm": 0.0}, {"loss": 1.0, "grad_norm": 1.0}]),
    ],
)
def test_rwkv7_pretraining_rejects_non_finite_loss_or_zero_gradient(tmp_path, history_key, history) -> None:
    output = tmp_path / history_key
    output.mkdir()
    (output / "config.json").write_text("{}", encoding="utf-8")
    (output / TRAINER_STATE_NAME).write_text(json.dumps({"global_step": 2, "log_history": history}), encoding="utf-8")
    (output / "train_results.json").write_text(json.dumps({"train_loss": 1.0}), encoding="utf-8")

    with pytest.raises(RuntimeError, match=f"finite positive `{history_key}`"):
        pretraining._training_evidence(output, expected_steps=2, minimum_logged_steps=2)
