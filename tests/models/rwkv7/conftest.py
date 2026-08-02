# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import os
from pathlib import Path

import pytest

import transformers.models.rwkv7.modeling_rwkv7 as modeling_rwkv7

from . import testing_utils


def _write_synthetic_distribution(site_packages, *, distribution, module, repository, revision) -> None:
    module_dir = site_packages / module
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    dist_info = site_packages / f"{distribution.replace('-', '_')}-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.0.0\n",
        encoding="utf-8",
    )
    (dist_info / "direct_url.json").write_text(
        json.dumps(
            {
                "url": repository,
                "vcs_info": {
                    "commit_id": revision,
                    "requested_revision": revision,
                    "vcs": "git",
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def synthetic_fla_public_contract(tmp_path, monkeypatch):
    """Provide the public FLA call shape on CPU; this is not a FlashRWKV operator E2E fixture."""
    site_packages = tmp_path / "synthetic-site-packages"
    _write_synthetic_distribution(
        site_packages,
        distribution=modeling_rwkv7.RWKV7_FLA_DISTRIBUTION,
        module="fla",
        repository=modeling_rwkv7.RWKV7_FLA_REPOSITORY,
        revision=modeling_rwkv7.RWKV7_FLA_REVISION,
    )
    _write_synthetic_distribution(
        site_packages,
        distribution=modeling_rwkv7.RWKV7_FLASH_RWKV_DISTRIBUTION,
        module="flash_rwkv",
        repository=modeling_rwkv7.RWKV7_FLASH_RWKV_REPOSITORY,
        revision=modeling_rwkv7.RWKV7_FLASH_RWKV_REVISION,
    )
    rwkv7_dir = site_packages / "fla" / "ops" / "rwkv7"
    rwkv7_dir.mkdir(parents=True)
    (site_packages / "fla" / "ops" / "__init__.py").write_text("", encoding="utf-8")
    (rwkv7_dir / "__init__.py").write_text(
        Path(testing_utils.__file__).read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(site_packages))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(site_packages) if not existing_pythonpath else f"{site_packages}:{existing_pythonpath}"
    monkeypatch.setenv("PYTHONPATH", pythonpath)
    monkeypatch.setattr(
        modeling_rwkv7,
        "_load_fla_rwkv7_contract",
        lambda: (testing_utils.recurrent_rwkv7, testing_utils.get_last_rwkv7_provider),
    )
