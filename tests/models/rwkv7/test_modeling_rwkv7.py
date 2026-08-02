# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch

from transformers import AutoModel, AutoModelForCausalLM
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import (
    Rwkv7ForCausalLM,
    Rwkv7Model,
    rwkv7_reference,
)


def _tiny_config() -> Rwkv7Config:
    return Rwkv7Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        head_size=4,
        context_length=16,
    )


def test_rwkv7_causal_lm_forward_backward_and_recurrent_state() -> None:
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


def test_rwkv7_all_ones_attention_mask_matches_unmasked_input() -> None:
    torch.manual_seed(0)
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    unmasked = model(input_ids)
    masked = model(input_ids, attention_mask=torch.ones_like(input_ids))

    torch.testing.assert_close(masked.logits, unmasked.logits)
    for masked_state, unmasked_state in zip(masked.state, unmasked.state, strict=True):
        torch.testing.assert_close(masked_state, unmasked_state)


def test_rwkv7_rejects_attention_mask_with_padding() -> None:
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 0]])

    with pytest.raises(ValueError, match="does not yet support padding"):
        model(input_ids, attention_mask=torch.tensor([[1, 1, 0]]))


def test_rwkv7_cache_default_depends_on_training_mode() -> None:
    model = Rwkv7ForCausalLM(_tiny_config())
    input_ids = torch.tensor([[1, 2, 3]])

    assert model(input_ids).state is None
    model.eval()
    assert model(input_ids).state is not None


def test_rwkv7_reference_matches_explicit_dplr_oracle() -> None:
    receptance = torch.tensor([[[0.2, -0.4], [0.3, 0.1]]])
    raw_decay = torch.tensor([[[0.5, -0.2], [0.1, 0.7]]])
    key = torch.tensor([[[0.6, -0.3], [0.2, 0.8]]])
    value = torch.tensor([[[0.4, -0.5], [0.7, 0.9]]])
    a = torch.tensor([[[-0.2, 0.9], [0.5, -0.1]]])
    b = torch.tensor([[[0.7, 0.3], [-0.4, 0.6]]])
    initial_state = torch.tensor([[[[0.2, 0.1], [-0.3, 0.4]]]])

    output, final_state = rwkv7_reference(
        receptance,
        raw_decay,
        key,
        value,
        a,
        b,
        initial_state,
        head_size=2,
    )
    state = initial_state
    expected_outputs = []
    log_decay = -torch.nn.functional.softplus(-raw_decay) - 0.5
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


def test_rwkv7_generate_updates_recurrent_state() -> None:
    torch.manual_seed(0)
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    cached = model.generate(input_ids, max_new_tokens=3, do_sample=False, use_cache=True)
    uncached = model.generate(input_ids, max_new_tokens=3, do_sample=False, use_cache=False)

    assert torch.equal(cached, uncached)


def test_rwkv7_save_reload_and_auto_classes(tmp_path) -> None:
    model = Rwkv7ForCausalLM(_tiny_config()).eval()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = model(input_ids).logits
    model.save_pretrained(tmp_path)

    auto_model = AutoModel.from_pretrained(tmp_path)
    auto_causal_lm = AutoModelForCausalLM.from_pretrained(tmp_path)

    assert isinstance(auto_model, Rwkv7Model)
    assert isinstance(auto_causal_lm, Rwkv7ForCausalLM)
    torch.testing.assert_close(auto_causal_lm(input_ids).logits, expected)
