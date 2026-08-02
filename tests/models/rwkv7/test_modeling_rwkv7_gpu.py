# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import json
import subprocess
import sys

import torch

import transformers.models.rwkv7.modeling_rwkv7 as modeling_rwkv7
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import Rwkv7ForCausalLM
from transformers.testing_utils import require_torch_gpu


FLA_REVISION = "8173df6ab27adb1c160a59d84b4ee02b6c6d8926"
FLASH_RWKV_REVISION = "5410491f0d6cff6058e5bd21cbab900b5b54f220"


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
from transformers.models.rwkv7.modeling_rwkv7 import validate_rwkv7_runtime_provenance

provenance = validate_rwkv7_runtime_provenance()
assert provenance["revision"] == {FLA_REVISION!r}
assert provenance["flash_rwkv_revision"] == {FLASH_RWKV_REVISION!r}
print(json.dumps(provenance, sort_keys=True))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    provenance = json.loads(result.stdout)
    assert provenance["revision"] == FLA_REVISION
    assert provenance["flash_rwkv_revision"] == FLASH_RWKV_REVISION


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

    model.train()
    training_output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    assert training_output.loss is not None and torch.isfinite(training_output.loss)
    training_output.loss.backward()
    assert contract.get_last_provider() == "flash_rwkv"
    assert contract.get_last_kernel() == "pretrain_recurrent_fp32io16_forward"
    gradient = model.model.blocks[0].att.receptance.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()


@require_torch_gpu
def test_rwkv7_real_provider_packed_noncontiguous_state_pool() -> None:
    torch.manual_seed(20260803)
    inputs = tuple((torch.randn(1, 3, 64, device="cuda", dtype=torch.float16) * 0.02).contiguous() for _ in range(6))
    state_pool = torch.zeros(6, 1, 64, 64, device="cuda", dtype=torch.float32)
    untouched_before = state_pool[[0, 2, 3, 5]].clone()
    cu_seqlens = torch.tensor([0, 2, 3], device="cuda", dtype=torch.int32)
    state_indices_storage = torch.tensor([4, -1, 1, -1], device="cuda", dtype=torch.int32)
    state_indices = state_indices_storage[::2]
    assert not state_indices.is_contiguous()

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
    assert not torch.equal(state_pool[[4, 1]], torch.zeros_like(state_pool[[4, 1]]))
    torch.testing.assert_close(state_pool[[0, 2, 3, 5]], untouched_before, rtol=0, atol=0)
