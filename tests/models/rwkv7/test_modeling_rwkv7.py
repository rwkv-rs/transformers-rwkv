# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch

from transformers import AutoModel, AutoModelForCausalLM
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import (
    Rwkv7ForCausalLM,
    Rwkv7Model,
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
    prefix = model(input_ids[:, :3])
    resumed = model(input_ids[:, 3:], state=prefix.state)
    torch.testing.assert_close(resumed.logits[:, -1], output.logits[:, -1])


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
