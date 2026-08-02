# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0 (the "License");

import json

import pytest
import torch

import transformers.models.rwkv7.modeling_rwkv7 as modeling_rwkv7
from transformers import AutoModel, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers.models.rwkv7.configuration_rwkv7 import Rwkv7Config
from transformers.models.rwkv7.modeling_rwkv7 import (
    Rwkv7ForCausalLM,
    Rwkv7Model,
    Rwkv7PreTrainedModel,
    rwkv7_reference,
)
from transformers.trainer import OPTIMIZER_NAME, SCHEDULER_NAME, TRAINER_STATE_NAME
from transformers.utils import SAFE_WEIGHTS_NAME


def _tiny_config() -> Rwkv7Config:
    return Rwkv7Config(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        head_size=4,
        context_length=16,
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


@pytest.mark.parametrize("backend", ["auto", "flash_rwkv"])
def test_rwkv7_accelerated_backend_falls_back_on_cpu(backend) -> None:
    config = _tiny_config()
    config.wkv_backend = backend
    model = Rwkv7ForCausalLM(config).eval()

    model(torch.tensor([[1, 2, 3]]))

    assert {block.att.last_wkv_backend for block in model.model.blocks} == {"reference"}


def test_rwkv7_accelerated_backend_selection_is_observable(monkeypatch) -> None:
    def accelerated(*inputs):
        return rwkv7_reference(*inputs), "flash_rwkv"

    monkeypatch.setattr(modeling_rwkv7, "_rwkv7_flash", accelerated)
    config = _tiny_config()
    config.wkv_backend = "flash_rwkv"
    model = Rwkv7ForCausalLM(config).eval()

    model(torch.tensor([[1, 2, 3]]))

    assert {block.att.last_wkv_backend for block in model.model.blocks} == {"flash_rwkv"}


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


def _training_arguments(output_dir, *, max_steps: int) -> TrainingArguments:
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
        use_cpu=True,
    )


def test_rwkv7_trainer_checkpoint_resume_matches_uninterrupted_training(tmp_path) -> None:
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
