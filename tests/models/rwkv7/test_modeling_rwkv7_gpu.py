# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import inspect
import json
import math
import subprocess
import sys
from importlib import metadata as importlib_metadata

import torch

import transformers.models.rwkv7.modeling_rwkv7 as modeling_rwkv7
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM
from transformers.testing_utils import require_torch_gpu


FLA_REVISION = "606752b7dff79eb326eeebf2d046102027da5306"
FLASH_RWKV_REVISION = "8b3d08a9a9430df23fb9da9b35fb0aa625faa1fb"
TORCH_VERSION = "2.11.0"


def _rrmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    denominator = expected.float().square().mean().sqrt().clamp_min(torch.finfo(torch.float32).eps)
    return (difference.square().mean().sqrt() / denominator).item()


def _gpu_config() -> Rwkv7Config:
    return Rwkv7Config(
        vocab_size=31,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        head_size=64,
        context_length=16,
    )


@require_torch_gpu
def test_rwkv7_canonical_runtime_provenance_in_fresh_process() -> None:
    code = f"""
import json
from importlib import metadata as importlib_metadata
from transformers.models.rwkv7.modeling_rwkv7 import validate_rwkv7_runtime_provenance

provenance = validate_rwkv7_runtime_provenance()
assert provenance["revision"] == {FLA_REVISION!r}
assert provenance["flash_rwkv_revision"] == {FLASH_RWKV_REVISION!r}
assert importlib_metadata.version("torch") == {TORCH_VERSION!r}
print(json.dumps(provenance, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    provenance = json.loads(result.stdout)
    assert provenance["revision"] == FLA_REVISION
    assert provenance["flash_rwkv_revision"] == FLASH_RWKV_REVISION
    assert importlib_metadata.version("torch") == TORCH_VERSION


@require_torch_gpu
def test_rwkv7_public_recurrent_uses_raw_decay_logits_contract() -> None:
    contract = modeling_rwkv7._load_fla_rwkv7_contract()
    parameters = inspect.signature(contract.recurrent_rwkv7).parameters

    assert "decay_logits" in parameters
    assert "decay_bias" in parameters
    assert "elapsed_t" in parameters
    assert "validated_metadata" in parameters
    assert "log_decay" not in parameters


@require_torch_gpu
def test_rwkv7_real_recurrent_matches_independent_nonzero_state_oracle_and_gradients() -> None:
    torch.manual_seed(20260803)
    product_inputs = [
        (torch.randn(1, 3, 64, device="cuda", dtype=torch.float16) * 0.05).requires_grad_() for _ in range(6)
    ]
    product_state = (torch.randn(1, 1, 64, 64, device="cuda") * 0.02).requires_grad_()
    oracle_inputs = [tensor.detach().clone().requires_grad_() for tensor in product_inputs]
    oracle_state = product_state.detach().clone().requires_grad_()

    product_output, product_final_state = modeling_rwkv7._rwkv7_flash(
        *product_inputs,
        product_state,
        64,
    )
    oracle_log_decay = -math.exp(-0.5) * torch.sigmoid(oracle_inputs[1])
    oracle_output, oracle_final_state = modeling_rwkv7.rwkv7_reference(
        oracle_inputs[0],
        oracle_log_decay,
        *oracle_inputs[2:],
        oracle_state,
        head_size=64,
    )
    output_gradient = torch.randn_like(product_output)
    state_gradient = torch.randn_like(product_final_state)
    ((product_output * output_gradient).sum() + (product_final_state * state_gradient).sum()).backward()
    ((oracle_output * output_gradient).sum() + (oracle_final_state * state_gradient).sum()).backward()
    torch.cuda.synchronize()

    contract = modeling_rwkv7._load_fla_rwkv7_contract()
    assert contract.get_last_provider() == "flash_rwkv"
    assert contract.get_last_kernel() == "pretrain_recurrent_fp32io16_forward"
    assert _rrmse(product_output, oracle_output) <= 0.007
    assert _rrmse(product_final_state, oracle_final_state) <= 0.008
    for product, oracle in zip([*product_inputs, product_state], [*oracle_inputs, oracle_state], strict=True):
        assert _rrmse(product.grad, oracle.grad) <= 0.008


@require_torch_gpu
def test_rwkv7_real_provider_inference_and_training() -> None:
    torch.manual_seed(20260802)
    model = Rwkv7ForCausalLM(_gpu_config()).cuda().half()
    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda")

    model.eval()
    contract = modeling_rwkv7._load_fla_rwkv7_contract()
    fused_calls = []
    fused_names = (
        "infer_tmix_mix6_fp16",
        "infer_tmix_vres_gate_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "infer_tmix_lnx_rkvres_xg_fp16",
        "infer_cmix_mix_fp16",
    )
    originals = {name: getattr(contract.flash_rwkv, name) for name in fused_names}

    for name, operator in originals.items():

        def observe(*args, _name=name, _operator=operator, **kwargs):
            result = _operator(*args, **kwargs)
            assert contract.get_last_provider() == "flash_rwkv"
            assert contract.get_last_kernel() == _name
            fused_calls.append(_name)
            return result

        setattr(contract.flash_rwkv, name, observe)

    try:
        with torch.no_grad():
            output = model(input_ids=input_ids, use_cache=True)
    finally:
        for name, operator in originals.items():
            setattr(contract.flash_rwkv, name, operator)

    assert output.state is not None
    assert torch.isfinite(output.logits).all()
    assert fused_calls == [
        "infer_tmix_mix6_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "infer_tmix_lnx_rkvres_xg_fp16",
        "infer_cmix_mix_fp16",
        "infer_tmix_mix6_fp16",
        "infer_tmix_vres_gate_fp16",
        "infer_tmix_kk_a_gate_fp16",
        "infer_tmix_lnx_rkvres_xg_fp16",
        "infer_cmix_mix_fp16",
    ]
    assert {block.att.last_wkv_backend for block in model.model.blocks} == {"flash_rwkv"}

    model.bfloat16().train()
    training_output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert training_output.loss is not None and torch.isfinite(training_output.loss)
    assert training_output.logits.dtype == torch.bfloat16
    training_output.loss.backward()
    assert contract.get_last_provider() == "flash_rwkv"
    assert contract.get_last_kernel() == "pretrain_recurrent_fp32io16_forward"
    gradient = model.model.blocks[0].att.receptance.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    decay_bias_gradient = model.model.blocks[0].att.w0.grad
    assert decay_bias_gradient is not None and torch.isfinite(decay_bias_gradient).all()


@require_torch_gpu
def test_rwkv7_real_provider_packed_noncontiguous_state_pool() -> None:
    torch.manual_seed(20260803)
    inputs = tuple((torch.randn(1, 3, 64, device="cuda", dtype=torch.float16) * 0.02).contiguous() for _ in range(6))
    state_pool = torch.zeros(6, 1, 64, 64, device="cuda", dtype=torch.float32)
    state_pool_before = state_pool.clone()
    untouched_before = state_pool_before[[0, 2, 3, 5]].clone()
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices_storage = torch.tensor([4, -1, 1, -1], device="cuda", dtype=torch.int32)
    state_indices = state_indices_storage[::2]
    assert not state_indices.is_contiguous()

    expected_outputs = []
    expected_state_pool = state_pool_before.clone()
    for sequence_index, state_index in enumerate(state_indices.tolist()):
        start = int(cu_seqlens[sequence_index])
        end = int(cu_seqlens[sequence_index + 1])
        sequence_inputs = [tensor[:, start:end] for tensor in inputs]
        log_decay = -math.exp(-0.5) * torch.sigmoid(sequence_inputs[1])
        expected_output, expected_state = modeling_rwkv7.rwkv7_reference(
            sequence_inputs[0],
            log_decay,
            *sequence_inputs[2:],
            expected_state_pool[state_index : state_index + 1],
            head_size=64,
        )
        expected_outputs.append(expected_output)
        expected_state_pool[state_index].copy_(expected_state[0])

    with torch.no_grad():
        output, final_state = modeling_rwkv7._rwkv7_flash(
            *inputs,
            state_pool,
            64,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
        )
    torch.cuda.synchronize()

    contract = modeling_rwkv7._load_fla_rwkv7_contract()
    assert final_state is state_pool
    assert contract.get_last_provider() == "flash_rwkv"
    assert contract.get_last_kernel() == "rwkv7_recurrent_stateful"
    assert torch.isfinite(output).all()
    assert _rrmse(output, torch.cat(expected_outputs, dim=1)) <= 0.002
    assert _rrmse(state_pool[[4, 1]], expected_state_pool[[4, 1]]) <= 0.002
    assert not torch.equal(state_pool[[4, 1]], torch.zeros_like(state_pool[[4, 1]]))
    torch.testing.assert_close(state_pool[[0, 2, 3, 5]], untouched_before, rtol=0, atol=0)
