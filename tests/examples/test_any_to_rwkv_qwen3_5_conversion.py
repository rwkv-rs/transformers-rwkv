# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest

from transformers.models.rwkv7 import modeling_rwkv7


_EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples/pytorch/any_to_rwkv/qwen3_5_conversion.py"
_SPEC = importlib.util.spec_from_file_location("any_to_rwkv_qwen3_5_conversion", _EXAMPLE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
qwen3_5_conversion = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qwen3_5_conversion)


@pytest.mark.parametrize(
    ("scenario", "distribution_name", "repository", "error_match"),
    [
        pytest.param(
            "wrong-package",
            "official-flash-linear-attention",
            modeling_rwkv7.RWKV7_FLA_REPOSITORY,
            "distribution identity does not match",
            id="wrong-package",
        ),
        pytest.param(
            "fork",
            modeling_rwkv7.RWKV7_FLA_DISTRIBUTION,
            "https://github.com/attacker/fla-rwkv.git",
            "repository provenance mismatch",
            id="fork",
        ),
    ],
)
def test_any_to_rwkv_example_validates_runtime_provenance_before_public_api_import(
    monkeypatch,
    tmp_path,
    scenario,
    distribution_name,
    repository,
    error_match,
) -> None:
    module_origin = tmp_path / "fla" / "__init__.py"

    class ProvenanceDistribution:
        metadata = {"Name": distribution_name}
        version = "0.5.2"

        @staticmethod
        def read_text(_filename):
            return json.dumps(
                {
                    "url": repository,
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": modeling_rwkv7.RWKV7_FLA_REVISION,
                        "commit_id": modeling_rwkv7.RWKV7_FLA_REVISION,
                    },
                }
            )

        @staticmethod
        def locate_file(_filename):
            return module_origin

    imported_modules = []

    def unexpected_import(module_name):
        imported_modules.append(module_name)
        raise AssertionError(f"{scenario} provenance reached public API import")

    monkeypatch.setattr(
        modeling_rwkv7.importlib_metadata,
        "distribution",
        lambda _name: ProvenanceDistribution(),
    )
    monkeypatch.setattr(
        modeling_rwkv7.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(module_origin)),
    )
    monkeypatch.setattr(qwen3_5_conversion.importlib, "import_module", unexpected_import)

    with pytest.raises(RuntimeError, match=error_match):
        qwen3_5_conversion._load_public_recurrent_rwkv7()

    assert imported_modules == []
