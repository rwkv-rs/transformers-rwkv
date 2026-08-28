# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transformers import is_torch_available
from transformers.testing_utils import require_accelerate, require_peft, require_torch


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import RwkvConfig, RwkvForCausalLM
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.rwkv.modeling_rwkv import RwkvCache


def tiny_config() -> RwkvConfig:
    return RwkvConfig(
        vocab_size=128,
        context_length=32,
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=256,
        head_size=64,
        decay_low_rank_dim=32,
        a_low_rank_dim=32,
        v_low_rank_dim=32,
        gate_low_rank_dim=32,
    )


class FakeFlashRwkv2:
    @staticmethod
    def _mix(x, initial, parameters):
        previous = torch.cat((initial[:, None], x[:, :-1]), dim=1)
        delta = previous - x
        outputs = tuple((x + delta * parameter).contiguous() for parameter in parameters)
        return (*outputs, x[:, -1].contiguous())

    def pretrain_tmix_tokenshift_bf16(self, x, *parameters):
        initial = torch.zeros_like(x[:, 0])
        return self._mix(x, initial, parameters)[:6]

    def statetune_tmix_tokenshift_bf16(self, x, initial, *parameters):
        return self._mix(x, initial, parameters)

    @staticmethod
    def pretrain_tmix_a_gate_bf16(a0, a12):
        return torch.sigmoid(a0 + a12).contiguous()

    @staticmethod
    def pretrain_tmix_vres_gate_bf16(value, v_first, v0, v12):
        return (value + (v_first - value) * torch.sigmoid(v0 + v12)).contiguous()

    @staticmethod
    def pretrain_tmix_kk_pre_bf16(key, key_scale, learning_rate, learning_rate_scale, *, head_size=64):
        batch_size, sequence_length, channels = key.shape
        heads = channels // head_size
        direction = F.normalize(
            (key * key_scale).view(batch_size, sequence_length, heads, head_size).float(),
            dim=-1,
        ).to(key.dtype)
        prepared_key = key * (1 + (learning_rate - 1) * learning_rate_scale)
        return (
            prepared_key.contiguous(),
            -direction.view_as(key).contiguous(),
            (direction * learning_rate.view_as(direction)).view_as(key).contiguous(),
        )

    @staticmethod
    def _recurrence(initial_state, receptance, decay_logits, key, value, recurrent_a, recurrent_b):
        batch_size, sequence_length, heads, head_size = receptance.shape
        outputs = []
        final_states = []
        for batch_idx in range(batch_size):
            state = initial_state[batch_idx]
            rows = []
            for token_idx in range(sequence_length):
                decay = torch.exp(-0.6065306597126334 * torch.sigmoid(decay_logits[batch_idx, token_idx].float()))
                a = recurrent_a[batch_idx, token_idx].float()
                b = recurrent_b[batch_idx, token_idx].float()
                k = key[batch_idx, token_idx].float()
                v = value[batch_idx, token_idx].float()
                dot = (a.unsqueeze(-1) * state).sum(dim=-2)
                state = (
                    decay.unsqueeze(-1) * state
                    + b.unsqueeze(-1) * dot.unsqueeze(-2)
                    + k.unsqueeze(-1) * v.unsqueeze(-2)
                )
                r = receptance[batch_idx, token_idx].float()
                rows.append((r.unsqueeze(-1) * state).sum(dim=-2).to(value.dtype))
            outputs.append(torch.stack(rows))
            final_states.append(state)
        return torch.stack(outputs), torch.stack(final_states)

    def pretrain_tmix_wkv7_recurrent_bf16(self, receptance, decay_logits, key, value, a, b, *, head_size=64):
        batch_size, sequence_length, channels = receptance.shape
        heads = channels // head_size
        values = [
            tensor.view(batch_size, sequence_length, heads, head_size)
            for tensor in (receptance, decay_logits, key, value, a, b)
        ]
        initial = torch.zeros(batch_size, heads, head_size, head_size, dtype=torch.float32)
        output, _ = self._recurrence(initial, *values)
        return output.view(batch_size, sequence_length, channels)

    def statetune_tmix_wkv7_recurrent_fp32io16(
        self,
        initial_state,
        sequence_chunk_offsets,
        chunk_token_starts,
        chunk_token_ends,
        receptance,
        decay_logits,
        key,
        value,
        a,
        b,
        *,
        scale=1.0,
    ):
        batch_size, heads, head_size, _ = initial_state.shape
        sequence_length = receptance.shape[0] // batch_size
        values = [
            tensor.view(batch_size, sequence_length, heads, head_size)
            for tensor in (receptance, decay_logits, key, value, a, b)
        ]
        output, final_state = self._recurrence(initial_state, *values)
        output = (output * scale).view_as(value)
        boundaries = initial_state[:, None].expand(-1, sequence_chunk_offsets[1].item(), -1, -1, -1)
        return output, final_state, boundaries.reshape(-1, heads, head_size, head_size), torch.zeros_like(value)

    @staticmethod
    def pretrain_tmix_readout_bf16(x, r, k, v, residual_scale, weight, bias, g, *, head_size=64):
        batch_size, sequence_length, channels = x.shape
        heads = channels // head_size
        normalized = F.group_norm(x.view(-1, channels), heads, weight, bias, 64e-5).view_as(x)
        residual = (
            (
                r.view(batch_size, sequence_length, heads, head_size)
                * k.view(batch_size, sequence_length, heads, head_size)
                * residual_scale
            ).sum(dim=-1, keepdim=True)
            * v.view(batch_size, sequence_length, heads, head_size)
        ).view_as(x)
        return ((normalized + residual) * g).contiguous()

    @staticmethod
    def pretrain_cmix_bf16(x, x_k, key_weight, value_weight):
        previous = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), dim=1)
        mixed = x + (previous - x) * x_k
        return F.linear(F.relu(F.linear(mixed, key_weight)).square(), value_weight)

    def statetune_cmix_bf16(self, x, initial, x_k, key_weight, value_weight):
        previous = torch.cat((initial[:, None], x[:, :-1]), dim=1)
        mixed = x + (previous - x) * x_k
        output = F.linear(F.relu(F.linear(mixed, key_weight)).square(), value_weight)
        return output, x[:, -1].contiguous()

    @staticmethod
    def infer_embedding_ln0_forward_varlen(embedding, weight, bias, *, eps=1e-5):
        return F.layer_norm(embedding, (embedding.shape[-1],), weight, bias, eps)

    @staticmethod
    def prepare_tmix_wkv7_recurrent_metadata(*args, **kwargs):
        return object()

    def infer_tmix_postnorm_tokenshift_forward_varlen(
        self,
        x,
        residual,
        weight,
        bias,
        *parameters,
        shift_state_pool,
        cu_seqlens,
        state_indices,
        max_seqlen=None,
        eps=1e-5,
        validated_metadata=None,
    ):
        batch_size = state_indices.numel()
        sequence_length = x.shape[0] // batch_size
        residual = x + residual
        normalized = F.layer_norm(residual, (x.shape[-1],), weight, bias, eps).view(batch_size, sequence_length, -1)
        outputs = self._mix(normalized, shift_state_pool, parameters)
        shift_state_pool.copy_(outputs[-1])
        return (residual, *(value.view_as(x) for value in outputs[:6]))

    @staticmethod
    def infer_tmix_wkv_prepare_forward_varlen(
        xr,
        xw,
        xk,
        xv,
        xa,
        xg,
        receptance_weight,
        key_weight,
        value_weight,
        w1,
        a1,
        g1,
        v1,
        w2,
        a2,
        g2,
        v2,
        v0,
        k_k,
        a0,
        k_a,
        *,
        v_first=None,
        w1_runtime=None,
        a1_runtime=None,
        g1_runtime=None,
        v1_runtime=None,
        w2_runtime=None,
        a2_runtime=None,
        g2_runtime=None,
        v2_runtime=None,
        receptance_lora_a=None,
        receptance_lora_b=None,
        receptance_lora_scale=1.0,
        key_lora_a=None,
        key_lora_b=None,
        key_lora_scale=1.0,
        value_lora_a=None,
        value_lora_b=None,
        value_lora_scale=1.0,
        head_size=64,
        batch_size=1,
        max_seqlen=None,
    ):
        def project(source, weight, lora_a, lora_b, scale):
            output = F.linear(source, weight)
            if lora_a is not None:
                output = output + F.linear(F.linear(source, lora_a), lora_b) * scale
            return output

        receptance = project(xr, receptance_weight, receptance_lora_a, receptance_lora_b, receptance_lora_scale)
        key = project(xk, key_weight, key_lora_a, key_lora_b, key_lora_scale)
        value = project(xv, value_weight, value_lora_a, value_lora_b, value_lora_scale)
        decay_logits = torch.tanh(xw @ w1_runtime) @ w2_runtime
        recurrent_gate = torch.sigmoid(a0 + (xa @ a1_runtime) @ a2_runtime)
        gate = torch.sigmoid(xg @ g1_runtime) @ g2_runtime
        if v_first is None:
            v_first = value
        else:
            value = value + (v_first - value) * torch.sigmoid(v0 + (xv @ v1_runtime) @ v2_runtime)
        heads = key.shape[-1] // head_size
        direction = F.normalize((key * k_k).view(-1, heads, head_size).float(), dim=-1).to(key.dtype)
        key = key * (1 + (recurrent_gate - 1) * k_a)
        return (
            receptance.contiguous(),
            decay_logits.contiguous(),
            key.contiguous(),
            value.contiguous(),
            -direction.view_as(key).contiguous(),
            (direction * recurrent_gate.view_as(direction)).view_as(key).contiguous(),
            gate.contiguous(),
            v_first.contiguous(),
        )

    def infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
        self,
        receptance,
        decay_logits,
        key,
        value,
        a,
        b,
        *,
        state_pool,
        cu_seqlens,
        state_indices,
        scale=1.0,
        decay_bias=None,
        max_seqlen=None,
        validated_metadata=None,
    ):
        batch_size = state_indices.numel()
        sequence_length = receptance.shape[0] // batch_size
        values = [
            tensor.view(batch_size, sequence_length, *tensor.shape[1:])
            for tensor in (receptance, decay_logits, key, value, a, b)
        ]
        if decay_bias is not None:
            values[1] = values[1] + decay_bias.view(1, 1, *values[1].shape[2:])
        output, final_state = self._recurrence(state_pool, *values)
        state_pool.copy_(final_state)
        return (output * scale).view_as(value)

    @staticmethod
    def infer_tmix_readout_forward_varlen(
        x,
        r,
        k,
        v,
        residual_scale,
        weight,
        bias,
        g,
        output_weight,
        *,
        output_lora_a=None,
        output_lora_b=None,
        output_lora_scale=1.0,
        head_size=64,
        batch_size=1,
        max_seqlen=None,
    ):
        channels = x.shape[-1]
        heads = channels // head_size
        x = x.view(-1, channels)
        normalized = F.group_norm(x, heads, weight, bias, 64e-5)
        residual = (
            (r.view(-1, heads, head_size) * k.view(-1, heads, head_size) * residual_scale.view(heads, head_size))
            .sum(dim=-1, keepdim=True)
            .mul(v.view(-1, heads, head_size))
            .view_as(x)
        )
        mixed = (normalized + residual) * g
        output = F.linear(mixed, output_weight)
        if output_lora_a is not None:
            output = output + F.linear(F.linear(mixed, output_lora_a), output_lora_b) * output_lora_scale
        return output

    def infer_cmix_forward_varlen(
        self,
        x,
        residual,
        weight,
        bias,
        x_k,
        key_weight,
        value_weight,
        *,
        shift_state_pool,
        cu_seqlens,
        state_indices,
        max_seqlen=None,
        eps=1e-5,
        validated_metadata=None,
        deterministic=False,
    ):
        batch_size = state_indices.numel()
        sequence_length = x.shape[0] // batch_size
        residual = x + residual
        normalized = F.layer_norm(residual, (x.shape[-1],), weight, bias, eps).view(batch_size, sequence_length, -1)
        previous = torch.cat((shift_state_pool[:, None], normalized[:, :-1]), dim=1)
        mixed = normalized + (previous - normalized) * x_k
        shift_state_pool.copy_(normalized[:, -1])
        activation = F.relu(F.linear(mixed, key_weight)).square()
        output = activation.view(-1, activation.shape[-1]) @ value_weight
        return residual, output

    @staticmethod
    def infer_post_norm_output_forward_varlen(x, residual, weight, bias, *, eps=1e-5):
        return F.layer_norm(x + residual, (x.shape[-1],), weight, bias, eps)

    @staticmethod
    def infer_head_linear_all_forward_varlen(x, weight):
        return F.linear(x, weight)

    @staticmethod
    def infer_head_linear_last_forward_varlen(x, weight, *, tokens_count):
        return F.linear(x, weight)


def fake_recurrent_metadata(self, flash_rwkv2, batch_size, sequence_length, device):
    cu_seqlens = torch.arange(0, (batch_size + 1) * sequence_length, sequence_length, dtype=torch.int32)
    state_indices = torch.arange(batch_size, dtype=torch.int32)
    return cu_seqlens, state_indices, sequence_length, object()


@require_torch
class RwkvMockProviderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        self.config = tiny_config()
        self.fake = FakeFlashRwkv2()
        self.loader = mock.patch(
            "transformers.models.rwkv.modeling_rwkv.load_flash_rwkv2",
            return_value=self.fake,
        )
        self.metadata = mock.patch.object(RwkvCache, "recurrent_metadata", fake_recurrent_metadata)
        self.loader.start()
        self.metadata.start()

    def tearDown(self):
        self.metadata.stop()
        self.loader.stop()

    def test_stateless_training_outputs_loss_backward_and_tuple(self):
        model = RwkvForCausalLM(self.config).to(dtype=torch.bfloat16).train()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 7))
        labels = torch.randint(0, self.config.vocab_size, (2, 7))
        outputs = model(input_ids, labels=labels, output_hidden_states=True)
        self.assertIsInstance(outputs, CausalLMOutputWithPast)
        self.assertEqual(outputs.logits.shape, (2, 7, self.config.vocab_size))
        self.assertEqual(len(outputs.hidden_states), self.config.num_hidden_layers + 1)
        self.assertIsNone(outputs.past_key_values)
        outputs.loss.backward()
        self.assertIsNotNone(model.lm_head.weight.grad)
        self.assertIsNotNone(model.model.layers[0].linear_attn.o_proj.weight.grad)
        tuple_output = model(input_ids, logits_to_keep=2, return_dict=False)
        self.assertEqual(tuple_output[0].shape, (2, 2, self.config.vocab_size))

    def test_bfloat16_evaluation_uses_stateless_training_operators(self):
        model = RwkvForCausalLM(self.config).to(dtype=torch.bfloat16).train()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 7))
        training_logits = model(input_ids, use_cache=False).logits

        model.eval()
        with torch.no_grad():
            evaluation = model(input_ids, use_cache=False)

        torch.testing.assert_close(evaluation.logits, training_logits)
        self.assertIsNone(evaluation.past_key_values)

    def test_stateful_training_matches_full_sequence_and_keeps_graph(self):
        model = RwkvForCausalLM(self.config).to(dtype=torch.bfloat16).train()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 7))
        full_cache = RwkvCache(self.config)
        full = model(input_ids, past_key_values=full_cache, use_cache=True).logits

        chunk_cache = RwkvCache(self.config)
        first = model(input_ids[:, :3], past_key_values=chunk_cache, use_cache=True).logits
        second = model(input_ids[:, 3:], past_key_values=chunk_cache, use_cache=True).logits
        torch.testing.assert_close(torch.cat((first, second), dim=1), full, atol=2e-2, rtol=2e-2)
        self.assertEqual(chunk_cache.get_seq_length(), 7)
        for full_layer, chunk_layer in zip(full_cache.layers, chunk_cache.layers, strict=True):
            torch.testing.assert_close(full_layer.conv_states[0], chunk_layer.conv_states[0])
            torch.testing.assert_close(full_layer.conv_states[1], chunk_layer.conv_states[1])
            torch.testing.assert_close(
                full_layer.recurrent_states[0], chunk_layer.recurrent_states[0], atol=1e-5, rtol=1e-5
            )
            self.assertIsNotNone(chunk_layer.recurrent_states[0].grad_fn)

    def test_inference_full_chunk_temporary_cache_hidden_states_and_last_logits(self):
        model = RwkvForCausalLM(self.config).half().eval()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 7))
        full_cache = RwkvCache(self.config)
        full = model(
            input_ids,
            past_key_values=full_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        self.assertEqual(len(full.hidden_states), self.config.num_hidden_layers + 1)
        state_keys = set(model.state_dict())
        for layer in model.model.layers:
            self.assertIsNotNone(layer.linear_attn._low_rank_canonical)
            self.assertIsNotNone(layer.mlp._value_runtime)
            self.assertEqual(layer.mlp.value.weight.device.type, "cpu")
        self.assertFalse(any("runtime" in key for key in state_keys))

        model.to(dtype=torch.float16)
        chunk_cache = RwkvCache(self.config)
        first = model(input_ids[:, :3], past_key_values=chunk_cache, use_cache=True).logits
        second = model(input_ids[:, 3:], past_key_values=chunk_cache, use_cache=True).logits
        torch.testing.assert_close(torch.cat((first, second), dim=1), full.logits, atol=2e-3, rtol=2e-3)
        for full_layer, chunk_layer in zip(full_cache.layers, chunk_cache.layers, strict=True):
            torch.testing.assert_close(full_layer.conv_states[0], chunk_layer.conv_states[0])
            torch.testing.assert_close(full_layer.conv_states[1], chunk_layer.conv_states[1])
            torch.testing.assert_close(
                full_layer.recurrent_states[0], chunk_layer.recurrent_states[0], atol=1e-5, rtol=1e-5
            )

        model.to(dtype=torch.float16)
        temporary = model(input_ids, use_cache=False, logits_to_keep=1)
        self.assertIsNone(temporary.past_key_values)
        self.assertEqual(temporary.logits.shape, (2, 1, self.config.vocab_size))
        embeddings = model.model.embed_tokens(input_ids).to(torch.float16)
        embedded = model(inputs_embeds=embeddings, use_cache=False)
        self.assertEqual(embedded.logits.shape, full.logits.shape)

        model.to(dtype=torch.bfloat16).train()
        for layer in model.model.layers:
            self.assertIsNone(layer.linear_attn._low_rank_canonical)
            self.assertIsNone(layer.mlp._value_runtime)
            self.assertEqual(layer.mlp.value.weight.dtype, torch.bfloat16)

    def test_gradient_checkpointing_preserves_state_and_gradients(self):
        plain = RwkvForCausalLM(self.config).to(dtype=torch.bfloat16).train()
        checkpointed = RwkvForCausalLM(self.config).to(dtype=torch.bfloat16).train()
        checkpointed.load_state_dict(plain.state_dict())
        checkpointed.gradient_checkpointing_enable()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 7))
        labels = torch.randint(0, self.config.vocab_size, (2, 7))

        plain_cache = RwkvCache(self.config)
        plain_output = plain(
            input_ids,
            labels=labels,
            past_key_values=plain_cache,
            use_cache=True,
        )
        plain_output.loss.backward()

        checkpointed_cache = RwkvCache(self.config)
        checkpointed_output = checkpointed(
            input_ids,
            labels=labels,
            past_key_values=checkpointed_cache,
            use_cache=True,
        )
        checkpointed_output.loss.backward()

        torch.testing.assert_close(checkpointed_output.logits, plain_output.logits)
        torch.testing.assert_close(checkpointed_output.loss, plain_output.loss)
        for plain_layer, checkpointed_layer in zip(plain_cache.layers, checkpointed_cache.layers, strict=True):
            torch.testing.assert_close(plain_layer.conv_states[0], checkpointed_layer.conv_states[0])
            torch.testing.assert_close(plain_layer.conv_states[1], checkpointed_layer.conv_states[1])
            torch.testing.assert_close(plain_layer.recurrent_states[0], checkpointed_layer.recurrent_states[0])
        for name in (
            "lm_head.weight",
            "model.layers.0.linear_attn.o_proj.weight",
            "model.layers.1.mlp.value.weight",
        ):
            plain_gradient = dict(plain.named_parameters())[name].grad
            checkpointed_gradient = dict(checkpointed.named_parameters())[name].grad
            torch.testing.assert_close(checkpointed_gradient, plain_gradient)

    def test_standard_greedy_and_beam_generation(self):
        model = RwkvForCausalLM(self.config).half().eval()
        input_ids = torch.randint(1, self.config.vocab_size, (1, 4))
        greedy = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
        )
        self.assertEqual(greedy.shape, (1, 7))

        model.to(dtype=torch.float16)
        beam = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            num_beams=2,
            eos_token_id=None,
        )
        self.assertEqual(beam.shape, (1, 7))

    @require_peft
    def test_lora_active_disabled_and_merged_inference(self):
        from peft import LoraConfig, get_peft_model

        model = (
            get_peft_model(
                RwkvForCausalLM(self.config),
                LoraConfig(
                    r=4,
                    lora_alpha=8,
                    target_modules=["r_proj", "k_proj", "v_proj", "o_proj"],
                ),
            )
            .half()
            .eval()
        )
        for module in model.modules():
            if hasattr(module, "lora_B") and "default" in module.lora_B:
                torch.nn.init.constant_(module.lora_B["default"].weight, 0.05)
        input_ids = torch.randint(1, self.config.vocab_size, (1, 5))

        active = model(input_ids, use_cache=False).logits
        model.to(dtype=torch.float16)
        with model.disable_adapter():
            disabled = model(input_ids, use_cache=False).logits
        self.assertFalse(torch.equal(active, disabled))

        model.to(dtype=torch.float16)
        model.merge_adapter()
        merged = model(input_ids, use_cache=False).logits
        torch.testing.assert_close(merged, active, atol=2e-3, rtol=2e-3)

    @require_accelerate
    def test_trainer_train_eval_save_reload_and_resume(self):
        from torch.utils.data import Dataset

        from transformers import Trainer, TrainingArguments

        class TinyDataset(Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                input_ids = torch.tensor([index + 1, 2, 3, 4, 5, 6, 7])
                return {"input_ids": input_ids, "labels": input_ids.clone()}

        dataset = TinyDataset()
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                max_steps=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                eval_strategy="steps",
                eval_steps=1,
                save_steps=1,
                logging_strategy="no",
                use_cpu=True,
                dataloader_pin_memory=False,
                report_to=[],
                disable_tqdm=True,
            )
            trainer = Trainer(
                model=RwkvForCausalLM(self.config).to(dtype=torch.bfloat16),
                args=arguments,
                train_dataset=dataset,
                eval_dataset=dataset,
            )
            self.assertEqual(trainer.train().global_step, 1)
            self.assertIn("eval_loss", trainer.evaluate())

            checkpoint = Path(directory) / "checkpoint-1"
            reloaded = RwkvForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16)
            self.assertEqual(reloaded(torch.tensor([[1, 2, 3]]), use_cache=False).logits.shape, (1, 3, 128))

            resumed_arguments = TrainingArguments(
                output_dir=directory,
                max_steps=2,
                per_device_train_batch_size=2,
                save_steps=1,
                logging_strategy="no",
                use_cpu=True,
                dataloader_pin_memory=False,
                report_to=[],
                disable_tqdm=True,
            )
            resumed = Trainer(model=reloaded, args=resumed_arguments, train_dataset=dataset)
            self.assertEqual(resumed.train(resume_from_checkpoint=checkpoint).global_step, 2)


if __name__ == "__main__":
    unittest.main()
