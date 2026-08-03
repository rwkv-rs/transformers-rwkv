# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import importlib
import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch

import transformers.models.rwkv7.modeling_rwkv7 as modeling_rwkv7
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, Trainer, TrainingArguments, is_torch_available
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import (
    Rwkv7ForCausalLM,
    Rwkv7Model,
    Rwkv7PreTrainedModel,
    rwkv7_reference,
)
from transformers.testing_utils import require_torch, run_test_using_subprocess, slow
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import SAFE_WEIGHTS_NAME

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from .testing_utils import get_last_rwkv7_kernel, get_last_rwkv7_provider, recurrent_rwkv7


class Rwkv7ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Rwkv7Model

    def __init__(self, parent):
        super().__init__(
            parent=parent,
            batch_size=2,
            seq_length=4,
            use_input_mask=False,
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            head_size=4,
            bos_token_id=0,
            eos_token_id=0,
            pad_token_id=0,
        )


@require_torch
class Rwkv7ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Rwkv7ModelTester
    _is_stateful = True

    def setUp(self):
        super().setUp()
        public_contract = mock.patch.object(
            modeling_rwkv7,
            "_load_fla_rwkv7_contract",
            return_value=modeling_rwkv7._FlaRwkv7Contract(
                recurrent_rwkv7=recurrent_rwkv7,
                flash_rwkv=None,
                can_use_flash_rwkv_inference=lambda *args, **kwargs: False,
                get_last_provider=get_last_rwkv7_provider,
                get_last_kernel=get_last_rwkv7_kernel,
            ),
        )
        public_contract.start()
        self.addCleanup(public_contract.stop)


def _tiny_config() -> Rwkv7Config:
    return Rwkv7Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        head_size=4,
        context_length=16,
    )


def _tiny_flash_config() -> Rwkv7Config:
    return Rwkv7Config(
        vocab_size=31,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        head_size=64,
        context_length=16,
    )


def _cpu_public_contract(
    recurrent=recurrent_rwkv7,
    *,
    provider=get_last_rwkv7_provider,
    kernel=get_last_rwkv7_kernel,
):
    return modeling_rwkv7._FlaRwkv7Contract(
        recurrent_rwkv7=recurrent,
        flash_rwkv=None,
        can_use_flash_rwkv_inference=lambda *args, **kwargs: False,
        get_last_provider=provider,
        get_last_kernel=kernel,
    )


def test_rwkv7_initializes_regular_linear_weights() -> None:
    model = Rwkv7PreTrainedModel(_tiny_config())
    linear = torch.nn.Linear(4, 4, bias=False)
    torch.nn.init.constant_(linear.weight, float("nan"))

    model._init_weights(linear)

    assert torch.isfinite(linear.weight).all()
    assert linear.weight.std() > 0


def test_rwkv7_skips_packed_linear_without_weight() -> None:
    class PackedLinear(torch.nn.Linear):
        def __init__(self):
            super().__init__(4, 4, bias=False)
            del self.weight
            self.register_buffer("weight_packed", torch.arange(8, dtype=torch.uint8))
            self.register_buffer("weight_scale", torch.arange(4, dtype=torch.float32))

    model = Rwkv7PreTrainedModel(_tiny_config())
    linear = PackedLinear()
    expected_packed = linear.weight_packed.clone()
    expected_scale = linear.weight_scale.clone()

    model._init_weights(linear)

    assert not hasattr(linear, "weight")
    torch.testing.assert_close(linear.weight_packed, expected_packed)
    torch.testing.assert_close(linear.weight_scale, expected_scale)


def test_rwkv7_low_rank_projections_are_standard_linear_modules_and_can_be_frozen(
    synthetic_fla_public_contract,
) -> None:
    model = Rwkv7ForCausalLM(_tiny_config()).train()
    time_mix = model.model.blocks[1].att
    projection_names = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")

    for name in projection_names:
        projection = getattr(time_mix, name)
        assert isinstance(projection, torch.nn.Linear)
        assert projection.bias is None
        assert torch.count_nonzero(projection.weight) == 0
        assert f"model.blocks.1.att.{name}.weight" in dict(model.named_parameters())
        projection.requires_grad_(False)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    loss = model(input_ids=input_ids, labels=input_ids).loss
    assert loss is not None
    loss.backward()

    for name in projection_names:
        assert getattr(time_mix, name).weight.grad is None
    assert time_mix.receptance.weight.grad is not None


@run_test_using_subprocess
def test_rwkv7_quantized_low_rank_initialization_preserves_packed_weights_in_fresh_process() -> None:
    initializer = Rwkv7PreTrainedModel(_tiny_config())
    time_mix = modeling_rwkv7.Rwkv7TimeMix(_tiny_config(), layer_id=1)
    packed_names = ("w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2")
    packed_state = {}
    for index, name in enumerate(packed_names):
        projection = getattr(time_mix, name)
        del projection.weight
        projection.register_buffer("weight_packed", torch.arange(8, dtype=torch.uint8) + index)
        projection.register_buffer("weight_scale", torch.arange(4, dtype=torch.float32) + index)
        packed_state[name] = (
            projection.weight_packed,
            projection.weight_packed.clone(),
            projection.weight_scale,
            projection.weight_scale.clone(),
        )

    # Match from_pretrained initialization order: quantized children first, then their TimeMix parent.
    for child in time_mix.children():
        initializer._init_weights(child)
    initializer._init_weights(time_mix)

    for name in packed_names:
        projection = getattr(time_mix, name)
        packed, expected_packed, scale, expected_scale = packed_state[name]
        assert not hasattr(projection, "weight")
        assert "weight" not in projection._parameters
        assert projection.weight_packed is packed
        assert projection.weight_scale is scale
        torch.testing.assert_close(projection.weight_packed, expected_packed)
        torch.testing.assert_close(projection.weight_scale, expected_scale)


def test_rwkv7_causal_lm_forward_backward_and_recurrent_state(synthetic_fla_public_contract) -> None:
    torch.manual_seed(0)
    model = Rwkv7ForCausalLM(_tiny_config())
    input_ids = torch.tensor([[1, 2, 3, 4]])

    output = model(input_ids, labels=input_ids)
    output.loss.backward()

    assert output.logits.shape == (1, 4, 31)
    assert model.head.weight.grad is not None
    prefix = model(input_ids[:, :3], use_cache=True)
    resumed = model(input_ids[:, 3:], state=prefix.state)
    torch.testing.assert_close(resumed.logits[:, -1], output.logits[:, -1])


def test_rwkv7_all_ones_attention_mask_matches_unmasked_input(synthetic_fla_public_contract) -> None:
    torch.manual_seed(0)
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    unmasked = model(input_ids)
    masked = model(input_ids, attention_mask=torch.ones_like(input_ids))

    torch.testing.assert_close(masked.logits, unmasked.logits)
    for masked_state, unmasked_state in zip(masked.state, unmasked.state, strict=True):
        torch.testing.assert_close(masked_state, unmasked_state)


def test_rwkv7_rejects_attention_mask_with_padding(synthetic_fla_public_contract) -> None:
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 0]])

    with pytest.raises(ValueError, match="does not yet support padding"):
        model(input_ids, attention_mask=torch.tensor([[1, 1, 0]]))


def test_rwkv7_cache_default_depends_on_training_mode(synthetic_fla_public_contract) -> None:
    model = Rwkv7ForCausalLM(_tiny_config())
    input_ids = torch.tensor([[1, 2, 3]])

    assert model(input_ids).state is None
    model.eval()
    assert model(input_ids).state is not None


def test_rwkv7_reference_matches_explicit_dplr_oracle() -> None:
    receptance = torch.tensor([[[0.2, -0.4], [0.3, 0.1]]])
    log_decay = torch.tensor([[[-0.5, -0.2], [-0.1, -0.7]]])
    key = torch.tensor([[[0.6, -0.3], [0.2, 0.8]]])
    value = torch.tensor([[[0.4, -0.5], [0.7, 0.9]]])
    a = torch.tensor([[[-0.2, 0.9], [0.5, -0.1]]])
    b = torch.tensor([[[0.7, 0.3], [-0.4, 0.6]]])
    initial_state = torch.tensor([[[[0.2, 0.1], [-0.3, 0.4]]]])

    output, final_state = rwkv7_reference(
        receptance,
        log_decay,
        key,
        value,
        a,
        b,
        initial_state,
        head_size=2,
    )
    state = initial_state
    expected_outputs = []
    for token in range(2):
        previous_state = state
        a_state = torch.einsum("bhk,bhkv->bhv", a[:, token : token + 1], previous_state)
        state = (
            log_decay[:, token : token + 1].exp().unsqueeze(-1) * previous_state
            + b[:, token : token + 1].unsqueeze(-1) * a_state.unsqueeze(-2)
            + key[:, token : token + 1].unsqueeze(-1) * value[:, token : token + 1].unsqueeze(-2)
        )
        expected_outputs.append(torch.einsum("bhk,bhkv->bhv", receptance[:, token : token + 1], state))
    expected_output = torch.stack(expected_outputs, dim=1).reshape_as(output)

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, state)


def test_rwkv7_decay_logit_semantics_match_primary_source_forward_and_gradient() -> None:
    raw_decay = torch.tensor([-40.0, -1.0, 0.0, 1.0, 40.0], requires_grad=True)
    expected_raw_decay = raw_decay.detach().clone().requires_grad_()

    actual = -torch.exp(torch.tensor(-0.5)) * torch.sigmoid(raw_decay)
    legacy_log_rate = -torch.nn.functional.softplus(-expected_raw_decay) - 0.5
    expected = -legacy_log_rate.exp()
    weights = torch.tensor([0.5, -0.75, 1.0, 1.25, -1.5])
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(raw_decay.grad, expected_raw_decay.grad, rtol=1e-6, atol=1e-6)
    assert not torch.isclose(actual[2].exp(), torch.exp(torch.tensor(-0.5)) * torch.sigmoid(torch.tensor(0.0)))


@pytest.mark.parametrize("nonzero_initial_state", [False, True], ids=("zero-state", "nonzero-state"))
def test_rwkv7_product_decay_and_recurrence_match_independent_oracle_with_gradients(
    nonzero_initial_state,
) -> None:
    torch.manual_seed(20260803)
    product_inputs = [(torch.randn(1, 3, 4) * 0.2).requires_grad_() for _ in range(6)]
    product_state = (
        torch.randn(1, 1, 4, 4) * 0.1 if nonzero_initial_state else torch.zeros(1, 1, 4, 4)
    ).requires_grad_()
    oracle_inputs = [tensor.detach().clone().requires_grad_() for tensor in product_inputs]
    oracle_state = product_state.detach().clone().requires_grad_()

    product_output, product_final_state = modeling_rwkv7._rwkv7_flash(
        *product_inputs,
        product_state,
        4,
        contract=_cpu_public_contract(),
    )
    oracle_log_decay = -torch.exp(torch.tensor(-0.5)) * torch.sigmoid(oracle_inputs[1])
    oracle_output, oracle_final_state = rwkv7_reference(
        oracle_inputs[0],
        oracle_log_decay,
        *oracle_inputs[2:],
        oracle_state,
        head_size=4,
    )
    output_gradient = torch.randn_like(product_output)
    state_gradient = torch.randn_like(product_final_state)
    product_loss = (product_output * output_gradient).sum() + (product_final_state * state_gradient).sum()
    oracle_loss = (oracle_output * output_gradient).sum() + (oracle_final_state * state_gradient).sum()
    product_loss.backward()
    oracle_loss.backward()

    torch.testing.assert_close(product_output, oracle_output, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(product_final_state, oracle_final_state, rtol=1e-6, atol=1e-6)
    for product, oracle in zip([*product_inputs, product_state], [*oracle_inputs, oracle_state], strict=True):
        torch.testing.assert_close(product.grad, oracle.grad, rtol=1e-6, atol=1e-6)


def test_rwkv7_reference_uses_fp32_state_with_half_io_and_gradients() -> None:
    torch.manual_seed(0)
    inputs = [torch.randn(1, 2, 4, dtype=torch.float16, requires_grad=True) for _ in range(6)]
    initial_state = torch.randn(1, 1, 4, 4, dtype=torch.float32, requires_grad=True)

    output, final_state = rwkv7_reference(*inputs, initial_state, head_size=4)
    output.float().square().mean().backward()

    assert output.dtype == torch.float16
    assert final_state.dtype == torch.float32
    for tensor in [*inputs, initial_state]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


_ASCII_CONTROL_REPOSITORY_URL_TEMPLATES = (
    " https://github.com/{repository_path}.git",
    "\thttps://github.com/{repository_path}.git",
    "\x00https://github.com/{repository_path}.git",
    "\x1fhttps://github.com/{repository_path}.git",
    "https://git\nhub.com/{repository_path}.git",
    "https://github.com/{repository_path}.git\n",
    "https://github.com/{repository_path}.git\x7f",
)
_ASCII_CONTROL_REPOSITORY_URL_IDS = (
    "leading-space",
    "leading-tab",
    "leading-nul",
    "leading-unit-separator",
    "embedded-newline",
    "trailing-newline",
    "delete",
)


@pytest.mark.parametrize(
    ("repository", "repository_path"),
    [
        pytest.param(modeling_rwkv7.RWKV7_FLA_REPOSITORY, "rwkv-rs/fla-rwkv", id="fla"),
        pytest.param(modeling_rwkv7.RWKV7_FLASH_RWKV_REPOSITORY, "rwkv-rs/FlashRWKV", id="flash-rwkv"),
    ],
)
@pytest.mark.parametrize(
    "url_template",
    [
        "https://github.com/{repository_path}",
        "https://github.com/{repository_path}/",
        "https://github.com/{uppercase_path}.git",
        "git+https://github.com/{repository_path}.git/",
    ],
    ids=("plain", "trailing-slash", "ascii-case-git", "git-prefix-git-trailing-slash"),
)
def test_rwkv7_github_repository_canonicalization_accepts_exact_repo(
    repository,
    repository_path,
    url_template,
) -> None:
    observed = url_template.format(
        repository_path=repository_path,
        uppercase_path=repository_path.upper(),
    )

    assert modeling_rwkv7._canonical_github_repository(observed) == modeling_rwkv7._canonical_github_repository(
        repository
    )


@pytest.mark.parametrize(
    ("repository", "repository_path"),
    [
        pytest.param(modeling_rwkv7.RWKV7_FLA_REPOSITORY, "rwkv-rs/fla-rwkv", id="fla"),
        pytest.param(modeling_rwkv7.RWKV7_FLASH_RWKV_REPOSITORY, "rwkv-rs/FlashRWKV", id="flash-rwkv"),
    ],
)
@pytest.mark.parametrize(
    "url_template",
    [
        *_ASCII_CONTROL_REPOSITORY_URL_TEMPLATES,
        "http://github.com/{repository_path}.git",
        "https://user@github.com/{repository_path}.git",
        "https://github.com:443/{repository_path}.git",
        "https://github.com/{repository_path}.git?ref=main",
        "https://github.com/{repository_path}.git#fragment",
        "https://github.com/{repository_path}.git;transport=ssh",
        "https://github.com/{owner}%2F{repository_name}.git",
        "https://github.com/rwkv-rſ/{repository_name}.git",
        "https://github.com//{repository_path}.git",
        "https://github.com/{repository_path}.git//",
        "https://gitlab.com/{repository_path}.git",
        "https://github.com/attacker/{repository_name}.git",
        "https://github.com/{repository_path}/extra.git",
        "https://github.com/{repository_path}.git.git",
    ],
    ids=(
        *_ASCII_CONTROL_REPOSITORY_URL_IDS,
        "non-https",
        "userinfo",
        "port",
        "query",
        "fragment",
        "path-params",
        "percent-encoding",
        "unicode-confusable",
        "repeated-leading-slash",
        "repeated-trailing-slash",
        "foreign-host",
        "fork",
        "extra-path",
        "repeated-git-suffix",
    ),
)
def test_rwkv7_github_repository_canonicalization_rejects_hostile_source(
    repository,
    repository_path,
    url_template,
) -> None:
    owner, repository_name = repository_path.split("/")
    observed = url_template.format(
        owner=owner,
        repository_name=repository_name,
        repository_path=repository_path,
    )

    assert modeling_rwkv7._canonical_github_repository(observed) != modeling_rwkv7._canonical_github_repository(
        repository
    )


@pytest.mark.parametrize(
    "url_template",
    _ASCII_CONTROL_REPOSITORY_URL_TEMPLATES,
    ids=_ASCII_CONTROL_REPOSITORY_URL_IDS,
)
def test_rwkv7_vcs_direct_url_rejects_ascii_controls(monkeypatch, tmp_path, url_template) -> None:
    module_origin = tmp_path / "fla" / "__init__.py"

    class HostileVcsDistribution:
        metadata = {"Name": modeling_rwkv7.RWKV7_FLA_DISTRIBUTION}
        version = "0.5.2"

        @staticmethod
        def read_text(_filename):
            return json.dumps(
                {
                    "url": url_template.format(repository_path="rwkv-rs/fla-rwkv"),
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

    monkeypatch.setattr(modeling_rwkv7.importlib_metadata, "distribution", lambda _name: HostileVcsDistribution())
    monkeypatch.setattr(
        modeling_rwkv7.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(module_origin)),
    )

    with pytest.raises(RuntimeError, match="repository provenance mismatch"):
        modeling_rwkv7._validate_rwkv7_distribution_provenance(
            distribution_name=modeling_rwkv7.RWKV7_FLA_DISTRIBUTION,
            module_name="fla",
            repository=modeling_rwkv7.RWKV7_FLA_REPOSITORY,
            revision=modeling_rwkv7.RWKV7_FLA_REVISION,
        )


@pytest.mark.parametrize(
    ("distribution_name", "module_name", "repository", "revision"),
    [
        pytest.param(
            modeling_rwkv7.RWKV7_FLA_DISTRIBUTION,
            "fla",
            modeling_rwkv7.RWKV7_FLA_REPOSITORY,
            modeling_rwkv7.RWKV7_FLA_REVISION,
            id="fla",
        ),
        pytest.param(
            modeling_rwkv7.RWKV7_FLASH_RWKV_DISTRIBUTION,
            "flash_rwkv",
            modeling_rwkv7.RWKV7_FLASH_RWKV_REPOSITORY,
            modeling_rwkv7.RWKV7_FLASH_RWKV_REVISION,
            id="flash-rwkv",
        ),
    ],
)
def test_rwkv7_editable_provenance_accepts_lowercase_repository_without_git_suffix(
    monkeypatch,
    tmp_path,
    distribution_name,
    module_name,
    repository,
    revision,
) -> None:
    source_dir = tmp_path / distribution_name
    module_origin = source_dir / module_name / "__init__.py"

    class EditableDistribution:
        metadata = {"Name": distribution_name}
        version = "0.0.0"

        @staticmethod
        def read_text(_filename):
            return json.dumps({"url": source_dir.as_uri(), "dir_info": {"editable": True}})

    def editable_git_value(observed_source_dir, *arguments):
        assert observed_source_dir == source_dir
        values = {
            ("remote", "get-url", "origin"): repository.removesuffix(".git").lower(),
            ("rev-parse", "HEAD"): revision,
            ("status", "--porcelain"): "",
        }
        return values[arguments]

    monkeypatch.setattr(modeling_rwkv7.importlib_metadata, "distribution", lambda _name: EditableDistribution())
    monkeypatch.setattr(
        modeling_rwkv7.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(module_origin)),
    )
    monkeypatch.setattr(modeling_rwkv7, "_editable_git_value", editable_git_value)

    assert modeling_rwkv7._validate_rwkv7_distribution_provenance(
        distribution_name=distribution_name,
        module_name=module_name,
        repository=repository,
        revision=revision,
    ) == {"source_kind": "editable", "version": "0.0.0"}


@run_test_using_subprocess
def test_rwkv7_runtime_provenance_is_fork_pinned_in_fresh_process() -> None:
    pytest.importorskip("fla")
    pytest.importorskip("flash_rwkv")
    from transformers.dependency_versions_table import deps
    from transformers.models.rwkv7 import RWKV7_FLA_REQUIREMENT, validate_rwkv7_runtime_provenance

    class PinnedVcsDistribution:
        def __init__(self, name, version, module_name, repository, revision):
            self.metadata = {"Name": name}
            self.version = version
            self.module_name = module_name
            self.direct_url = json.dumps(
                {
                    "url": repository,
                    "vcs_info": {"vcs": "git", "requested_revision": revision, "commit_id": revision},
                }
            )

        def read_text(self, _filename):
            return self.direct_url

        def locate_file(self, _filename):
            return modeling_rwkv7.importlib.util.find_spec(self.module_name).origin

    distributions = {
        "flash-linear-attention": PinnedVcsDistribution(
            "flash-linear-attention",
            "0.5.2",
            "fla",
            "https://github.com/rwkv-rs/fla-rwkv",
            "8173df6ab27adb1c160a59d84b4ee02b6c6d8926",
        ),
        "flash-rwkv": PinnedVcsDistribution(
            "flash-rwkv",
            "0.1.0",
            "flash_rwkv",
            "https://github.com/rwkv-rs/flashrwkv",
            "5410491f0d6cff6058e5bd21cbab900b5b54f220",
        ),
    }
    original_distribution = modeling_rwkv7.importlib_metadata.distribution
    modeling_rwkv7.importlib_metadata.distribution = distributions.__getitem__
    try:
        provenance = validate_rwkv7_runtime_provenance()
    finally:
        modeling_rwkv7.importlib_metadata.distribution = original_distribution

    assert deps["flash-linear-attention[flash-rwkv]"] == RWKV7_FLA_REQUIREMENT
    assert provenance == {
        "distribution": "flash-linear-attention",
        "distribution_version": "0.5.2",
        "extra": "flash-rwkv",
        "flash_rwkv_distribution": "flash-rwkv",
        "flash_rwkv_distribution_version": "0.1.0",
        "flash_rwkv_repository": "https://github.com/rwkv-rs/FlashRWKV.git",
        "flash_rwkv_revision": "5410491f0d6cff6058e5bd21cbab900b5b54f220",
        "flash_rwkv_source_kind": "vcs",
        "repository": "https://github.com/rwkv-rs/fla-rwkv.git",
        "requirement": "flash-linear-attention[flash-rwkv] @ git+https://github.com/rwkv-rs/fla-rwkv.git@8173df6ab27adb1c160a59d84b4ee02b6c6d8926",
        "revision": "8173df6ab27adb1c160a59d84b4ee02b6c6d8926",
        "source_kind": "vcs",
    }


@run_test_using_subprocess
def test_rwkv7_runtime_provenance_rejects_ascii_controls_in_fresh_process() -> None:
    module_origin = Path(__file__).resolve().parent / "synthetic_fla" / "__init__.py"

    class HostileVcsDistribution:
        metadata = {"Name": modeling_rwkv7.RWKV7_FLA_DISTRIBUTION}
        version = "0.5.2"
        repository = modeling_rwkv7.RWKV7_FLA_REPOSITORY

        def read_text(self, _filename):
            return json.dumps(
                {
                    "url": self.repository,
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

    distribution = HostileVcsDistribution()
    original_distribution = modeling_rwkv7.importlib_metadata.distribution
    original_find_spec = modeling_rwkv7.importlib.util.find_spec
    modeling_rwkv7.importlib_metadata.distribution = lambda _name: distribution
    modeling_rwkv7.importlib.util.find_spec = lambda name: importlib.machinery.ModuleSpec(
        name, loader=None, origin=str(module_origin)
    )
    try:
        for url_template in _ASCII_CONTROL_REPOSITORY_URL_TEMPLATES:
            distribution.repository = url_template.format(repository_path="rwkv-rs/fla-rwkv")
            with pytest.raises(RuntimeError, match="repository provenance mismatch"):
                modeling_rwkv7.validate_rwkv7_runtime_provenance()
    finally:
        modeling_rwkv7.importlib_metadata.distribution = original_distribution
        modeling_rwkv7.importlib.util.find_spec = original_find_spec


def test_rwkv7_runtime_provenance_rejects_same_name_registry_package(monkeypatch, request) -> None:
    class RegistryDistribution:
        metadata = {"Name": "flash-linear-attention"}
        version = "0.5.2"

        @staticmethod
        def read_text(filename):
            assert filename == "direct_url.json"
            return None

    monkeypatch.setattr(modeling_rwkv7.importlib_metadata, "distribution", lambda _name: RegistryDistribution())
    modeling_rwkv7._load_fla_rwkv7_contract.cache_clear()
    request.addfinalizer(modeling_rwkv7._load_fla_rwkv7_contract.cache_clear)

    with pytest.raises(RuntimeError, match="registry packages are rejected"):
        modeling_rwkv7._load_fla_rwkv7_contract()


def test_rwkv7_runtime_provenance_rejects_wrong_fork_revision(monkeypatch, tmp_path) -> None:
    module_origin = tmp_path / "fla" / "__init__.py"

    class WrongRevisionDistribution:
        metadata = {"Name": "flash-linear-attention"}
        version = "0.5.2"

        @staticmethod
        def read_text(_filename):
            return json.dumps(
                {
                    "url": "https://github.com/rwkv-rs/fla-rwkv.git",
                    "vcs_info": {"vcs": "git", "requested_revision": "0" * 40, "commit_id": "0" * 40},
                }
            )

        @staticmethod
        def locate_file(_filename):
            return module_origin

    monkeypatch.setattr(modeling_rwkv7.importlib_metadata, "distribution", lambda _name: WrongRevisionDistribution())
    monkeypatch.setattr(
        modeling_rwkv7.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(module_origin)),
    )

    with pytest.raises(RuntimeError, match="revision provenance mismatch"):
        modeling_rwkv7.validate_rwkv7_runtime_provenance()


def test_rwkv7_runtime_provenance_rejects_unproven_flash_rwkv(monkeypatch, tmp_path) -> None:
    fla_origin = tmp_path / "fla" / "__init__.py"

    class PinnedFlaDistribution:
        metadata = {"Name": "flash-linear-attention"}
        version = "0.5.2"

        @staticmethod
        def read_text(_filename):
            return json.dumps(
                {
                    "url": modeling_rwkv7.RWKV7_FLA_REPOSITORY,
                    "vcs_info": {
                        "vcs": "git",
                        "requested_revision": modeling_rwkv7.RWKV7_FLA_REVISION,
                        "commit_id": modeling_rwkv7.RWKV7_FLA_REVISION,
                    },
                }
            )

        @staticmethod
        def locate_file(_filename):
            return fla_origin

    class RegistryFlashRwkvDistribution:
        metadata = {"Name": "flash-rwkv"}
        version = "0.1.0"

        @staticmethod
        def read_text(_filename):
            return None

    monkeypatch.setattr(
        modeling_rwkv7.importlib_metadata,
        "distribution",
        lambda name: RegistryFlashRwkvDistribution() if name == "flash-rwkv" else PinnedFlaDistribution(),
    )
    monkeypatch.setattr(
        modeling_rwkv7.importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None, origin=str(fla_origin)),
    )

    with pytest.raises(RuntimeError, match="`flash-rwkv` provenance.*registry packages are rejected"):
        modeling_rwkv7.validate_rwkv7_runtime_provenance()


def test_rwkv7_missing_public_fla_contract_fails_closed(monkeypatch) -> None:
    def unavailable_contract():
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(modeling_rwkv7, "_load_fla_rwkv7_contract", unavailable_contract)
    model = Rwkv7ForCausalLM(_tiny_config()).eval()

    with pytest.raises(RuntimeError, match="FlashRWKV execution failed closed.*provider unavailable"):
        model(torch.tensor([[1, 2, 3]]))

    assert {block.att.last_wkv_backend for block in model.model.blocks} == {"uninitialized"}


def test_rwkv7_public_contract_does_not_fall_back_to_chunk(monkeypatch, request) -> None:
    class ChunkOnlyPublicModule:
        chunk_rwkv7 = staticmethod(recurrent_rwkv7)
        get_last_rwkv7_provider = staticmethod(get_last_rwkv7_provider)

    monkeypatch.setattr(modeling_rwkv7, "validate_rwkv7_runtime_provenance", lambda: {})
    monkeypatch.setattr(modeling_rwkv7.importlib, "import_module", lambda _name: ChunkOnlyPublicModule())
    modeling_rwkv7._load_fla_rwkv7_contract.cache_clear()
    request.addfinalizer(modeling_rwkv7._load_fla_rwkv7_contract.cache_clear)

    with pytest.raises(RuntimeError, match="must publicly expose recurrent_rwkv7"):
        modeling_rwkv7._load_fla_rwkv7_contract()


def test_rwkv7_public_contract_requires_fused_inference_surface(monkeypatch, request) -> None:
    class IncompleteFlashRwkv:
        infer_tmix_mix6_fp16 = staticmethod(lambda *args, **kwargs: None)

    class PublicRwkv7Module:
        recurrent_rwkv7 = staticmethod(recurrent_rwkv7)
        flash_rwkv = IncompleteFlashRwkv()
        get_last_rwkv7_provider = staticmethod(get_last_rwkv7_provider)
        get_last_rwkv7_kernel = staticmethod(get_last_rwkv7_kernel)

    class PublicInferenceModule:
        can_use_flash_rwkv_inference = staticmethod(lambda *args, **kwargs: False)

    monkeypatch.setattr(modeling_rwkv7, "validate_rwkv7_runtime_provenance", lambda: {})
    monkeypatch.setattr(
        modeling_rwkv7.importlib,
        "import_module",
        lambda name: PublicInferenceModule() if name.endswith(".inference") else PublicRwkv7Module(),
    )
    modeling_rwkv7._load_fla_rwkv7_contract.cache_clear()
    request.addfinalizer(modeling_rwkv7._load_fla_rwkv7_contract.cache_clear)

    with pytest.raises(RuntimeError, match="lacks public inference operators.*infer_cmix_mix_fp16"):
        modeling_rwkv7._load_fla_rwkv7_contract()


def test_rwkv7_public_recurrent_signature_is_importable_in_fresh_process(synthetic_fla_public_contract) -> None:
    code = """
import inspect
from fla.ops.rwkv7 import flash_rwkv, get_last_rwkv7_kernel, get_last_rwkv7_provider, recurrent_rwkv7
from fla.ops.rwkv7.inference import can_use_flash_rwkv_inference

required = {
    "decay_logits",
    "decay_bias",
    "elapsed_t",
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "state_indices",
    "mode",
}
assert required <= inspect.signature(recurrent_rwkv7).parameters.keys()
assert "log_decay" not in inspect.signature(recurrent_rwkv7).parameters
assert callable(get_last_rwkv7_provider)
assert callable(get_last_rwkv7_kernel)
assert callable(can_use_flash_rwkv_inference)
for operator in {
    "infer_cmix_mix_fp16",
    "infer_tmix_kk_a_gate_fp16",
    "infer_tmix_lnx_rkvres_xg_fp16",
    "infer_tmix_mix6_fp16",
    "infer_tmix_vres_gate_fp16",
}:
    assert callable(getattr(flash_rwkv, operator))
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rwkv7_public_recurrent_rejects_ambiguous_log_decay_signature(monkeypatch, request) -> None:
    def ambiguous_recurrent_rwkv7(
        r,
        log_decay,
        k,
        v,
        a,
        b,
        *,
        decay_logits=None,
        decay_bias=None,
        elapsed_t=None,
        initial_state=None,
        output_final_state=False,
        cu_seqlens=None,
        state_indices=None,
        mode="fp32io16",
    ):
        raise AssertionError("an ambiguous decay contract must not execute")

    class FlashRwkv:
        pass

    for operator in modeling_rwkv7._FLA_RWKV7_FUSED_INFERENCE_OPERATORS:
        setattr(FlashRwkv, operator, staticmethod(lambda *args, **kwargs: None))

    class PublicRwkv7Module:
        recurrent_rwkv7 = staticmethod(ambiguous_recurrent_rwkv7)
        flash_rwkv = FlashRwkv()
        get_last_rwkv7_provider = staticmethod(get_last_rwkv7_provider)
        get_last_rwkv7_kernel = staticmethod(get_last_rwkv7_kernel)

    class PublicInferenceModule:
        can_use_flash_rwkv_inference = staticmethod(lambda *args, **kwargs: False)

    monkeypatch.setattr(modeling_rwkv7, "validate_rwkv7_runtime_provenance", lambda: {})
    monkeypatch.setattr(
        modeling_rwkv7.importlib,
        "import_module",
        lambda name: PublicInferenceModule() if name.endswith(".inference") else PublicRwkv7Module(),
    )
    modeling_rwkv7._load_fla_rwkv7_contract.cache_clear()
    request.addfinalizer(modeling_rwkv7._load_fla_rwkv7_contract.cache_clear)

    with pytest.raises(RuntimeError, match="obsolete log_decay product boundary"):
        modeling_rwkv7._load_fla_rwkv7_contract()


@pytest.mark.parametrize("training", [False, True])
def test_rwkv7_monkeypatched_public_fla_contract_drives_two_layer_hf_calls(monkeypatch, training) -> None:
    """Validate the HF call contract on CPU; this does not execute the real FLA or FlashRWKV operator."""
    required_parameters = {
        "decay_logits",
        "decay_bias",
        "elapsed_t",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "state_indices",
        "mode",
    }
    calls = []
    telemetry = {}

    def public_recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        calls.append(
            (
                r,
                decay_logits,
                decay_bias,
                elapsed_t,
                k,
                v,
                a,
                b,
                initial_state,
                output_final_state,
                cu_seqlens,
                state_indices,
                mode,
            )
        )
        telemetry["kernel"] = (
            "pretrain_recurrent_fp32io16_from_decay_logits"
            if any(tensor.requires_grad for tensor in (r, decay_logits, k, v, a, b, initial_state))
            else "rwkv7_recurrent_from_decay_logits"
        )
        effective_decay_logits = (
            decay_logits if decay_bias is None else decay_logits + decay_bias.view(1, 1, *decay_bias.shape)
        )
        output = (r + effective_decay_logits + k + v + a + b) / 6
        if state_indices is not None:
            initial_state.add_(1)
            final_state = initial_state
        else:
            final_state = initial_state + torch.einsum("bthk,bthv->bhkv", k.float(), v.float())
        return output, final_state

    assert required_parameters <= inspect.signature(public_recurrent_rwkv7).parameters.keys()
    monkeypatch.setattr(
        modeling_rwkv7,
        "_load_fla_rwkv7_contract",
        lambda: _cpu_public_contract(
            public_recurrent_rwkv7,
            provider=lambda: "flash_rwkv",
            kernel=lambda: telemetry.get("kernel"),
        ),
    )
    model = Rwkv7ForCausalLM(_tiny_flash_config()).train(training)
    expected_bias = torch.linspace(-0.25, 0.25, model.config.hidden_size).view(1, 1, -1)
    with torch.no_grad():
        for block in model.model.blocks:
            block.att.w0.copy_(expected_bias)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    if training:
        output = model(input_ids=input_ids, labels=input_ids)
        assert output.loss is not None and torch.isfinite(output.loss)
        output.loss.backward()
        gradient = model.model.blocks[0].att.receptance.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        decay_bias_gradient = model.model.blocks[0].att.w0.grad
        assert decay_bias_gradient is not None and torch.isfinite(decay_bias_gradient).all()
    else:
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
        assert output.state is not None

    assert len(calls) == 2
    for call in calls:
        (
            r,
            decay_logits,
            decay_bias,
            elapsed_t,
            k,
            v,
            a,
            b,
            initial_state,
            output_final_state,
            cu_seqlens,
            state_indices,
            mode,
        ) = call
        assert all(tensor.shape == r.shape for tensor in (decay_logits, k, v, a, b))
        assert initial_state.shape == (2, 1, 64, 64)
        assert output_final_state is True
        assert mode == "fp32io16"
        assert r.shape == (2, 3, 1, 64)
        if training:
            assert decay_bias is None
            torch.testing.assert_close(decay_logits, expected_bias.expand_as(decay_logits))
        else:
            torch.testing.assert_close(decay_logits, torch.zeros_like(decay_logits))
            torch.testing.assert_close(decay_bias, expected_bias.view(1, 64))
        assert elapsed_t is None
        assert cu_seqlens is None
        assert state_indices is None
    assert {block.att.last_wkv_backend for block in model.model.blocks} == {"flash_rwkv"}


def test_rwkv7_eligible_inference_uses_public_fused_tmix_and_cmix_with_telemetry(monkeypatch) -> None:
    """Validate the public fused call contract on CPU; this does not execute real FlashRWKV operators."""
    calls = []
    telemetry = {}
    telemetry_checks = []

    def record(kernel, result):
        calls.append(kernel)
        telemetry.update(provider="flash_rwkv", kernel=kernel)
        return result

    class PublicFlashRwkv:
        @staticmethod
        def infer_tmix_mix6_fp16(hidden_states, shift_state, mixes):
            shift_state.copy_(hidden_states[:, -1])
            return record("infer_tmix_mix6_fp16", tuple(hidden_states + mix * 0 for mix in mixes))

        @staticmethod
        def infer_tmix_vres_gate_fp16(value, first_value, gate_bias, gate_delta):
            assert gate_bias.shape == (value.shape[-1],)
            del first_value, gate_delta
            return record("infer_tmix_vres_gate_fp16", value)

        @staticmethod
        def infer_tmix_kk_a_gate_fp16(key, key_scale, gate_bias, gate_delta, key_gate_scale):
            assert key_scale.shape == gate_bias.shape == key_gate_scale.shape == (key.shape[-1],)
            del gate_delta
            return record(
                "infer_tmix_kk_a_gate_fp16",
                (key, -torch.ones_like(key), torch.full_like(key, 0.25)),
            )

        @staticmethod
        def infer_tmix_lnx_rkvres_xg_fp16(
            output,
            receptance,
            key,
            value,
            residual_scale,
            norm_weight,
            norm_bias,
            gate,
        ):
            del receptance, key, value, residual_scale, norm_weight, norm_bias, gate
            return record("infer_tmix_lnx_rkvres_xg_fp16", output)

        @staticmethod
        def infer_cmix_mix_fp16(hidden_states, shift_state, mix):
            del mix
            shift_state.copy_(hidden_states[:, -1])
            return record("infer_cmix_mix_fp16", hidden_states)

    def public_recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        assert output_final_state is True
        assert cu_seqlens is None and state_indices is None and mode == "fp32io16"
        assert elapsed_t is None
        effective_decay_logits = (
            decay_logits if decay_bias is None else decay_logits + decay_bias.view(1, 1, *decay_bias.shape)
        )
        output = (r + effective_decay_logits + k + v + a + b) / 6
        final_state = initial_state + torch.einsum("bthk,bthv->bhkv", k.float(), v.float())
        return record("rwkv7_recurrent_from_decay_logits", (output, final_state))

    def get_last_provider():
        telemetry_checks.append(("provider", telemetry.get("provider")))
        return telemetry.get("provider")

    def get_last_kernel():
        telemetry_checks.append(("kernel", telemetry.get("kernel")))
        return telemetry.get("kernel")

    contract = modeling_rwkv7._FlaRwkv7Contract(
        recurrent_rwkv7=public_recurrent_rwkv7,
        flash_rwkv=PublicFlashRwkv,
        can_use_flash_rwkv_inference=lambda *args, **kwargs: True,
        get_last_provider=get_last_provider,
        get_last_kernel=get_last_kernel,
    )
    monkeypatch.setattr(modeling_rwkv7, "_load_fla_rwkv7_contract", lambda: contract)
    model = Rwkv7ForCausalLM(_tiny_flash_config()).eval()

    with torch.no_grad():
        output = model(torch.tensor([[1, 2, 3]]), use_cache=True)

    assert output.state is not None
    assert calls == [
        "infer_tmix_mix6_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "rwkv7_recurrent_from_decay_logits",
        "infer_tmix_lnx_rkvres_xg_fp16",
        "infer_cmix_mix_fp16",
        "infer_tmix_mix6_fp16",
        "infer_tmix_vres_gate_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "rwkv7_recurrent_from_decay_logits",
        "infer_tmix_lnx_rkvres_xg_fp16",
        "infer_cmix_mix_fp16",
    ]
    assert telemetry_checks == [item for kernel in calls for item in (("provider", "flash_rwkv"), ("kernel", kernel))]


def test_rwkv7_public_recurrent_packed_state_pool_is_updated_by_identity(monkeypatch) -> None:
    calls = []
    telemetry = {}

    def public_recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        assert decay_bias is None and elapsed_t is None
        calls.append((initial_state, output_final_state, cu_seqlens, state_indices, mode))
        telemetry["kernel"] = "rwkv7_recurrent_stateful_from_decay_logits"
        for state_index in state_indices.tolist():
            initial_state[state_index].add_(1)
        return torch.zeros_like(v), initial_state

    monkeypatch.setattr(
        modeling_rwkv7,
        "_load_fla_rwkv7_contract",
        lambda: _cpu_public_contract(
            public_recurrent_rwkv7,
            provider=lambda: "flash_rwkv",
            kernel=lambda: telemetry.get("kernel"),
        ),
    )
    inputs = [torch.randn(1, 5, 64) for _ in range(6)]
    state_pool = torch.zeros(4, 1, 64, 64)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    state_indices_storage = torch.tensor([3, -1, 1, -1], dtype=torch.int32)
    state_indices = state_indices_storage[::2]
    assert not state_indices.is_contiguous()

    output, final_state = modeling_rwkv7._rwkv7_flash(
        *inputs,
        state_pool,
        64,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
    )

    assert output.shape == (1, 5, 64)
    assert final_state is state_pool
    torch.testing.assert_close(state_pool[[3, 1]], torch.ones_like(state_pool[[3, 1]]))
    torch.testing.assert_close(state_pool[[0, 2]], torch.zeros_like(state_pool[[0, 2]]))
    assert len(calls) == 1
    observed_state, output_final_state, observed_cu_seqlens, observed_state_indices, mode = calls[0]
    assert observed_state is state_pool
    assert output_final_state is True
    assert observed_cu_seqlens is cu_seqlens
    assert observed_state_indices.tolist() == state_indices.tolist()
    assert observed_state_indices.is_contiguous()
    assert mode == "fp32io16"


def test_rwkv7_public_fla_contract_rejects_provider_fallback(monkeypatch) -> None:
    def public_recurrent_rwkv7(
        r,
        decay_logits,
        k,
        v,
        a,
        b,
        *,
        decay_bias,
        elapsed_t,
        initial_state,
        output_final_state,
        cu_seqlens,
        state_indices,
        mode,
    ):
        assert elapsed_t is None
        return torch.zeros_like(v), initial_state

    monkeypatch.setattr(
        modeling_rwkv7,
        "_load_fla_rwkv7_contract",
        lambda: _cpu_public_contract(
            public_recurrent_rwkv7,
            provider=lambda: "fla",
            kernel=lambda: "rwkv7_recurrent_from_decay_logits",
        ),
    )
    model = Rwkv7ForCausalLM(_tiny_flash_config()).eval()

    with torch.no_grad(), pytest.raises(RuntimeError, match="telemetry did not report"):
        model(torch.tensor([[1, 2, 3]]))


def test_rwkv7_generate_updates_recurrent_state(synthetic_fla_public_contract) -> None:
    torch.manual_seed(0)
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    cached = model.generate(input_ids, max_new_tokens=3, do_sample=False, use_cache=True)
    uncached = model.generate(input_ids, max_new_tokens=3, do_sample=False, use_cache=False)

    assert torch.equal(cached, uncached)


def test_rwkv7_save_reload_and_auto_classes(tmp_path, synthetic_fla_public_contract) -> None:
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = model(input_ids).logits
    model.save_pretrained(tmp_path)

    auto_model = AutoModel.from_pretrained(tmp_path)
    auto_causal_lm = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert isinstance(auto_model, Rwkv7Model)
    assert isinstance(auto_causal_lm, Rwkv7ForCausalLM)
    torch.testing.assert_close(auto_causal_lm(input_ids).logits, expected)


@slow
def test_rwkv7_auto_classes_survive_fla_backend_import(tmp_path) -> None:
    pytest.importorskip("fla")
    importlib.import_module("fla.ops.rwkv7.backends.flash_rwkv")
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    model.save_pretrained(tmp_path)

    auto_config = AutoConfig.from_pretrained(tmp_path)
    auto_model = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert isinstance(auto_config, Rwkv7Config)
    assert isinstance(auto_model, Rwkv7ForCausalLM)


class _TinyCausalLMDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.examples = [
            torch.tensor(tokens, dtype=torch.long)
            for tokens in (
                [1, 2, 3, 4, 5],
                [5, 4, 3, 2, 1],
                [2, 4, 6, 8, 10],
                [10, 8, 6, 4, 2],
                [3, 6, 9, 12, 15],
                [15, 12, 9, 6, 3],
                [7, 11, 13, 17, 19],
                [19, 17, 13, 11, 7],
            )
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        input_ids = self.examples[index]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def _training_arguments(output_dir, *, max_steps: int, use_cpu: bool = True) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=1,
        learning_rate=1e-3,
        lr_scheduler_type="linear",
        warmup_steps=0,
        optim="adamw_torch",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=2,
        report_to="none",
        disable_tqdm=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        seed=17,
        data_seed=23,
        use_cpu=use_cpu,
    )


def test_rwkv7_trainer_checkpoint_resume_matches_uninterrupted_training(
    tmp_path, synthetic_fla_public_contract
) -> None:
    dataset = _TinyCausalLMDataset()

    torch.manual_seed(11)
    gradient_model = Rwkv7ForCausalLM(_tiny_config()).train()
    gradient_batch = {name: tensor.unsqueeze(0) for name, tensor in dataset[0].items()}
    gradient_output = gradient_model(**gradient_batch)
    assert gradient_output.state is None
    assert gradient_output.loss is not None and torch.isfinite(gradient_output.loss)
    gradient_output.loss.backward()
    for name, parameter in gradient_model.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
    for name in (
        "model.embeddings.weight",
        "model.blocks.0.ffn.key.weight",
        "model.blocks.0.ffn.value.weight",
        "model.blocks.1.ffn.key.weight",
        "model.blocks.1.ffn.value.weight",
        "head.weight",
    ):
        assert gradient_model.get_parameter(name).grad.abs().sum() > 0

    torch.manual_seed(29)
    reference_model = Rwkv7ForCausalLM(_tiny_config())
    initial_parameters = {name: parameter.detach().clone() for name, parameter in reference_model.named_parameters()}
    reference_trainer = Trainer(
        model=reference_model,
        args=_training_arguments(tmp_path / "reference", max_steps=4),
        train_dataset=dataset,
    )
    reference_result = reference_trainer.train()

    assert torch.isfinite(torch.tensor(reference_result.training_loss))
    logged_losses = [entry["loss"] for entry in reference_trainer.state.log_history if "loss" in entry]
    assert len(logged_losses) == 4
    assert torch.isfinite(torch.tensor(logged_losses)).all()
    for name in (
        "model.blocks.0.ffn.key.weight",
        "model.blocks.1.ffn.value.weight",
        "head.weight",
    ):
        assert not torch.equal(reference_model.get_parameter(name), initial_parameters[name])

    checkpoint = tmp_path / "reference" / "checkpoint-2"
    for filename in (
        SAFE_WEIGHTS_NAME,
        "config.json",
        OPTIMIZER_NAME,
        SCHEDULER_NAME,
        "rng_state.pth",
        TRAINER_STATE_NAME,
    ):
        assert (checkpoint / filename).is_file(), f"missing checkpoint state: {filename}"
    checkpoint_state = json.loads((checkpoint / TRAINER_STATE_NAME).read_text(encoding="utf-8"))
    assert checkpoint_state["global_step"] == 2

    torch.manual_seed(999)
    resumed_model = Rwkv7ForCausalLM(_tiny_config())
    resumed_trainer = Trainer(
        model=resumed_model,
        args=_training_arguments(tmp_path / "resumed", max_steps=4),
        train_dataset=dataset,
    )
    resumed_trainer.train(resume_from_checkpoint=checkpoint)

    assert resumed_trainer.state.global_step == reference_trainer.state.global_step == 4
    assert resumed_trainer.lr_scheduler.state_dict() == reference_trainer.lr_scheduler.state_dict()
    for name, reference_parameter in reference_model.named_parameters():
        torch.testing.assert_close(
            resumed_model.get_parameter(name),
            reference_parameter,
            rtol=0,
            atol=0,
            msg=lambda message, name=name: f"resume diverged for {name}: {message}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FlashRWKV training")
def test_rwkv7_flash_trainer_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch) -> None:
    pytest.importorskip("flash_rwkv")
    monkeypatch.setenv("FLA_FLASH_RWKV", "1")
    dataset = _TinyCausalLMDataset()

    torch.manual_seed(11)
    gradient_model = Rwkv7ForCausalLM(_tiny_flash_config()).to(device="cuda", dtype=torch.bfloat16).train()
    gradient_batch = {name: tensor.unsqueeze(0).cuda() for name, tensor in dataset[0].items()}
    gradient_loss = gradient_model(**gradient_batch).loss
    assert gradient_loss is not None and torch.isfinite(gradient_loss) and gradient_loss > 0
    gradient_loss.backward()
    assert {block.att.last_wkv_backend for block in gradient_model.model.blocks} == {"flash_rwkv"}
    for name in (
        "model.embeddings.weight",
        "model.blocks.0.att.g2.weight",
        "model.blocks.0.ffn.key.weight",
        "model.blocks.1.att.g2.weight",
        "head.weight",
    ):
        gradient = gradient_model.get_parameter(name).grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0, f"zero gradient for {name}"

    torch.manual_seed(29)
    reference_model = Rwkv7ForCausalLM(_tiny_flash_config()).to(device="cuda", dtype=torch.bfloat16)
    reference_trainer = Trainer(
        model=reference_model,
        args=_training_arguments(tmp_path / "flash-reference", max_steps=4, use_cpu=False),
        train_dataset=dataset,
    )
    reference_result = reference_trainer.train()
    reference_losses = [entry["loss"] for entry in reference_trainer.state.log_history if "loss" in entry]
    assert torch.isfinite(torch.tensor(reference_result.training_loss))
    assert len(reference_losses) == 4 and all(loss > 0 for loss in reference_losses)
    assert torch.isfinite(torch.tensor(reference_losses)).all()
    assert {block.att.last_wkv_backend for block in reference_model.model.blocks} == {"flash_rwkv"}

    checkpoint = tmp_path / "flash-reference" / "checkpoint-2"
    for filename in (
        SAFE_WEIGHTS_NAME,
        "config.json",
        OPTIMIZER_NAME,
        SCHEDULER_NAME,
        "rng_state.pth",
        TRAINER_STATE_NAME,
    ):
        assert (checkpoint / filename).is_file(), f"missing checkpoint state: {filename}"
    assert json.loads((checkpoint / TRAINER_STATE_NAME).read_text())["global_step"] == 2

    torch.manual_seed(999)
    resumed_model = Rwkv7ForCausalLM(_tiny_flash_config()).to(device="cuda", dtype=torch.bfloat16)
    resumed_trainer = Trainer(
        model=resumed_model,
        args=_training_arguments(tmp_path / "flash-resumed", max_steps=4, use_cpu=False),
        train_dataset=dataset,
    )
    resumed_trainer.train(resume_from_checkpoint=checkpoint)

    assert resumed_trainer.state.global_step == reference_trainer.state.global_step == 4
    assert resumed_trainer.lr_scheduler.state_dict() == reference_trainer.lr_scheduler.state_dict()
    assert (
        resumed_trainer.optimizer.state_dict()["param_groups"]
        == reference_trainer.optimizer.state_dict()["param_groups"]
    )
    for resumed_state, reference_state in zip(
        resumed_trainer.optimizer.state_dict()["state"].values(),
        reference_trainer.optimizer.state_dict()["state"].values(),
        strict=True,
    ):
        assert resumed_state.keys() == reference_state.keys()
        for key in resumed_state:
            torch.testing.assert_close(resumed_state[key], reference_state[key], rtol=0, atol=0)
    for name, reference_parameter in reference_model.named_parameters():
        torch.testing.assert_close(resumed_model.get_parameter(name), reference_parameter, rtol=0, atol=0)
    assert {block.att.last_wkv_backend for block in resumed_model.model.blocks} == {"flash_rwkv"}
