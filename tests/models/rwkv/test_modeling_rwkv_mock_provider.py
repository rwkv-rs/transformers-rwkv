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
import threading
import unittest
from pathlib import Path
from unittest import mock

from transformers import is_torch_available
from transformers.testing_utils import require_accelerate, require_peft, require_torch, require_torch_gpu


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import RwkvConfig, RwkvForCausalLM
    from transformers.generation import EosTokenCriteria, LogitsProcessor, MaxLengthCriteria, StopStringCriteria
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.rwkv.generation_rwkv import _rwkv_prefill_lengths
    from transformers.models.rwkv.modeling_rwkv import RwkvCache


def tiny_config(**kwargs) -> RwkvConfig:
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
        **kwargs,
    )


def single_token_stop_criteria(vocab_size: int, token_id: int) -> StopStringCriteria:
    criteria = object.__new__(StopStringCriteria)
    criteria.stop_strings = ("x",)
    criteria.maximum_token_len = 1
    criteria.num_stop_strings = 1
    criteria.max_valid_positions = 1
    criteria.max_valid_end_lens = 1
    criteria.target_lens = torch.tensor([1], dtype=torch.int32)
    criteria.embedding_vec = torch.full((vocab_size + 1, 3), -1, dtype=torch.int32)
    criteria.embedding_vec[:, -1] = 1
    criteria.embedding_vec[token_id, 1] = 1
    return criteria


class FakeRecurrentState:
    def __init__(self, state_pool, elapsed_state_pool, sequence_capacity):
        self._state_pool = state_pool
        self._elapsed_state_pool = elapsed_state_pool
        self._sequence_capacity = sequence_capacity

    def clone(self):
        elapsed_state_pool = None if self._elapsed_state_pool is None else self._elapsed_state_pool.clone()
        return type(self)(self._state_pool.clone(), elapsed_state_pool, self._sequence_capacity)

    def copy_(self, other):
        self._state_pool.copy_(other._state_pool)
        if self._elapsed_state_pool is not None:
            self._elapsed_state_pool.copy_(other._elapsed_state_pool)
        return self

    def zero_(self):
        self._state_pool.zero_()
        if self._elapsed_state_pool is not None:
            self._elapsed_state_pool.zero_()
        return self


class RecordingStreamer:
    def __init__(self):
        self.values = []
        self.thread_ids = []
        self.end_thread_id = None

    def put(self, value):
        self.values.append(value.clone())
        self.thread_ids.append(threading.get_ident())

    def end(self):
        self.end_thread_id = threading.get_ident()


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

    @staticmethod
    def prepare_tmix_wkv7_recurrent_fp16_state(
        state_pool_size, channels, *, sequence_capacity, head_size=64, device=None
    ):
        state_pool = torch.zeros(
            state_pool_size,
            channels // head_size,
            head_size,
            head_size,
            dtype=torch.float16,
            device=device,
        )
        elapsed_state_pool = torch.zeros(state_pool_size, dtype=torch.int32, device=device)
        return FakeRecurrentState(state_pool, elapsed_state_pool, sequence_capacity)

    @staticmethod
    def prepare_tmix_wkv7_recurrent_fp32io16_state(
        state_pool_size, channels, *, sequence_capacity, head_size=64, device=None
    ):
        state_pool = torch.zeros(
            state_pool_size,
            channels // head_size,
            head_size,
            head_size,
            dtype=torch.float32,
            device=device,
        )
        return FakeRecurrentState(state_pool, None, sequence_capacity)

    @staticmethod
    def prepare_tmix_wkv7_recurrent_fp32io16_state_from_tensor(state_pool):
        return FakeRecurrentState(state_pool, None, state_pool.shape[0])

    @staticmethod
    def setup_sampling_states(seed, num_slots):
        return torch.full((num_slots, 1), seed, dtype=torch.int64, device="cuda")

    @staticmethod
    def infer_sampling_temperature_topk_topp_forward_varlen(
        logits,
        states,
        slot_indices,
        *,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
    ):
        _ = slot_indices, temperature, top_p
        candidate_count = logits.shape[-1] if top_k == -1 else top_k
        candidates = torch.topk(logits, k=candidate_count, dim=-1).indices
        choices = torch.remainder(states[:, 0], candidate_count)
        sampled = candidates.gather(1, choices[:, None]).squeeze(1)
        states.add_(1)
        return sampled

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
        layer_index = getattr(self, "layer_index", 0)
        if layer_index:
            expected_x, expected_residual = self.expected_tmix_inputs
            if x.data_ptr() != expected_x.data_ptr() or residual.data_ptr() != expected_residual.data_ptr():
                raise AssertionError("TimeMix must preserve ChannelMix's hidden and residual stream order.")
        self.layer_index = (layer_index + 1) % self.num_layers
        batch_size = state_indices.numel()
        sequence_length = x.shape[0] // batch_size
        residual = x + residual
        normalized = F.layer_norm(residual, (x.shape[-1],), weight, bias, eps).view(batch_size, sequence_length, -1)
        outputs = self._mix(normalized, shift_state_pool, parameters)
        shift_state_pool.copy_(outputs[-1])
        self.expected_cmix_input = residual
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
        state,
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
        output, final_state = self._recurrence(state._state_pool, *values)
        state._state_pool.copy_(final_state)
        return (output * scale).view_as(value)

    def infer_tmix_wkv7_recurrent_fp16_forward_varlen(
        self,
        receptance,
        decay_logits,
        key,
        value,
        a,
        b,
        *,
        state,
        cu_seqlens,
        state_indices,
        scale=1.0,
        decay_bias=None,
        max_seqlen=None,
        validated_metadata=None,
    ):
        output = self.infer_tmix_wkv7_recurrent_fp32io16_forward_varlen(
            receptance,
            decay_logits,
            key,
            value,
            a,
            b,
            state=state,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            scale=scale,
            decay_bias=decay_bias,
            max_seqlen=max_seqlen,
            validated_metadata=validated_metadata,
        )
        state._elapsed_state_pool.add_(max_seqlen)
        return output

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
        if x.data_ptr() != self.expected_cmix_input.data_ptr():
            raise AssertionError("ChannelMix must consume the layer input returned by TimeMix.")
        batch_size = state_indices.numel()
        sequence_length = x.shape[0] // batch_size
        residual = x + residual
        normalized = F.layer_norm(residual, (x.shape[-1],), weight, bias, eps).view(batch_size, sequence_length, -1)
        previous = torch.cat((shift_state_pool[:, None], normalized[:, :-1]), dim=1)
        mixed = normalized + (previous - normalized) * x_k
        shift_state_pool.copy_(normalized[:, -1])
        activation = F.relu(F.linear(mixed, key_weight)).square()
        output = activation.view(-1, activation.shape[-1]) @ value_weight
        self.expected_tmix_inputs = (residual, output)
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
        self.fake.num_layers = self.config.num_hidden_layers
        self.loader = mock.patch(
            "transformers.models.rwkv.modeling_rwkv.load_flash_rwkv2",
            return_value=self.fake,
        )
        self.generation_loader = mock.patch(
            "transformers.models.rwkv.generation_rwkv.load_flash_rwkv2",
            return_value=self.fake,
        )
        self.metadata = mock.patch.object(RwkvCache, "recurrent_metadata", fake_recurrent_metadata)
        self.loader.start()
        self.generation_loader.start()
        self.metadata.start()

    def tearDown(self):
        self.metadata.stop()
        self.generation_loader.stop()
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

    def test_graph_prefill_consumes_every_prompt_token_except_the_last_once(self):
        self.assertEqual(_rwkv_prefill_lengths(1, None), ())
        self.assertEqual(_rwkv_prefill_lengths(9, None), (8,))
        self.assertEqual(_rwkv_prefill_lengths(9, 4), (4, 4))
        self.assertEqual(_rwkv_prefill_lengths(10, 4), (4, 4, 1))
        self.assertEqual(_rwkv_prefill_lengths(12, 4), (4, 4, 3))

    @require_torch_gpu
    def test_cuda_graph_generation_reuses_prefill_and_decode_graphs(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(1, self.config.vocab_size, (2, 6), device="cuda")

        reference_cache = RwkvCache(self.config)
        model.model._forward_state_only(input_ids[:, :-1], None, reference_cache, False)
        reference_token = input_ids[:, -1:]
        reference_completion = []
        for _ in range(3):
            reference_logits = model(
                reference_token,
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits[:, -1]
            reference_token = torch.argmax(reference_logits.float(), dim=-1, keepdim=True)
            reference_completion.append(reference_token)
        expected = torch.cat((input_ids, *reference_completion), dim=-1)

        first = model.generate(
            input_ids,
            max_new_tokens=3,
            prefill_chunk_size=2,
            do_sample=False,
            eos_token_id=None,
        )
        runtime = next(iter(model._rwkv_generation_graphs.values()))
        second = model.generate(
            input_ids,
            max_new_tokens=3,
            prefill_chunk_size=2,
            do_sample=False,
            eos_token_id=None,
        )

        torch.testing.assert_close(first, expected)
        torch.testing.assert_close(second, first)
        self.assertEqual(len(model._rwkv_generation_graphs), 1)
        self.assertEqual(runtime.prefill_lengths, (2, 2, 1))
        self.assertEqual(reference_cache.get_seq_length(), 8)
        for layer_idx, reference_layer in enumerate(reference_cache.layers):
            torch.testing.assert_close(
                runtime.state.attention_shift[layer_idx], reference_layer.conv_states[0].squeeze(-1)
            )
            torch.testing.assert_close(
                runtime.state.feed_forward_shift[layer_idx], reference_layer.conv_states[1].squeeze(-1)
            )
            torch.testing.assert_close(
                runtime.state.recurrent_states[layer_idx]._state_pool, reference_layer.recurrent_states[0]
            )

    @require_torch_gpu
    def test_single_token_prompt_starts_with_decode_graph_and_exposes_first_logits(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(1, self.config.vocab_size, (2, 1), device="cuda")
        reference_cache = RwkvCache(self.config)
        reference_logits = (
            model(
                input_ids,
                past_key_values=reference_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            .logits[:, -1]
            .float()
        )

        generated = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            eos_token_id=None,
        )
        runtime = next(iter(model._rwkv_generation_graphs.values()))

        self.assertEqual(runtime.prefill_lengths, ())
        torch.testing.assert_close(runtime.decode_graph.logits, reference_logits)
        torch.testing.assert_close(generated[:, 1], torch.argmax(reference_logits, dim=-1))

    @require_torch_gpu
    def test_cuda_graph_sampling_uses_fixed_seed_and_rapid_sampling_parameters(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(1, self.config.vocab_size, (4, 3), device="cuda")
        sampling_kwargs = {
            "max_new_tokens": 4,
            "do_sample": True,
            "temperature": 0.7,
            "top_k": 8,
            "top_p": 0.9,
            "eos_token_id": None,
        }

        torch.manual_seed(1234)
        first = model.generate(input_ids, **sampling_kwargs)
        torch.manual_seed(1234)
        second = model.generate(input_ids, **sampling_kwargs)
        torch.manual_seed(4321)
        third = model.generate(input_ids, **sampling_kwargs)

        torch.testing.assert_close(second, first)
        self.assertFalse(torch.equal(third, first))
        self.assertEqual(len(model._rwkv_generation_graphs), 1)

    @require_torch_gpu
    def test_cuda_graph_standard_stopping_criteria_override_config(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(1, self.config.vocab_size, (1, 4), device="cuda")
        eos_token_id = 1
        fixed_logits = torch.full((1, self.config.vocab_size), -1e4, device="cuda")
        fixed_logits[:, eos_token_id] = 1

        with mock.patch.object(self.fake, "infer_head_linear_last_forward_varlen", return_value=fixed_logits):
            generated = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                eos_token_id=2,
                stopping_criteria=[EosTokenCriteria(eos_token_id)],
            )
        runtime = next(iter(model._rwkv_generation_graphs.values()))

        self.assertEqual(generated.shape, (1, 5))
        self.assertEqual(generated[0, -1].item(), eos_token_id)
        self.assertEqual(runtime.decode_graph.completion_lengths.item(), 1)
        self.assertFalse(runtime.decode_graph.unfinished.item())

        model._rwkv_generation_graphs.clear()
        with mock.patch.object(self.fake, "infer_head_linear_last_forward_varlen", return_value=fixed_logits):
            generated = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                eos_token_id=None,
                stopping_criteria=[MaxLengthCriteria(input_ids.shape[1] + 2)],
            )
        self.assertEqual(generated.shape[1], input_ids.shape[1] + 2)

    @require_torch_gpu
    def test_finished_rows_are_padded_while_other_batch_rows_continue(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(2, self.config.vocab_size, (2, 4), device="cuda")
        stop_token = 1
        fixed_logits = torch.full((2, self.config.vocab_size), -1e4, device="cuda")
        fixed_logits[0, stop_token] = 1
        fixed_logits[1, 2] = 1

        first_criteria = single_token_stop_criteria(self.config.vocab_size, stop_token)
        second_criteria = single_token_stop_criteria(self.config.vocab_size, stop_token)
        tokenizer = object()
        with mock.patch.object(self.fake, "infer_head_linear_last_forward_varlen", return_value=fixed_logits):
            generated = model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
                stopping_criteria=[first_criteria],
                tokenizer=tokenizer,
            )
            repeated = model.generate(
                input_ids,
                max_new_tokens=4,
                do_sample=False,
                eos_token_id=None,
                stopping_criteria=[second_criteria],
                tokenizer=tokenizer,
            )
        runtime = next(iter(model._rwkv_generation_graphs.values()))
        completion_lengths = runtime.decode_graph.completion_lengths.tolist()

        self.assertEqual(completion_lengths[0], 1)
        self.assertGreater(completion_lengths[1], 1)
        torch.testing.assert_close(generated[0, 5:], torch.zeros_like(generated[0, 5:]))
        torch.testing.assert_close(repeated, generated)
        self.assertEqual(len(model._rwkv_generation_graphs), 1)

    @require_torch_gpu
    def test_cuda_graph_fp16_mode_uses_provider_managed_state(self):
        config = tiny_config(wkv_mode="fp16")
        model = RwkvForCausalLM(config).half().eval().cuda()
        input_ids = torch.randint(1, config.vocab_size, (2, 3), device="cuda")

        generated = model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
            eos_token_id=None,
        )
        runtime = next(iter(model._rwkv_generation_graphs.values()))

        self.assertEqual(generated.shape, (2, 5))
        self.assertEqual(runtime.state.recurrent_states[0]._state_pool.dtype, torch.float16)

    @require_torch_gpu
    def test_streamer_uses_async_pinned_completion_copies(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(1, self.config.vocab_size, (2, 3), device="cuda")
        streamer = RecordingStreamer()
        calling_thread = threading.get_ident()

        generated = model.generate(
            input_ids,
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=None,
            streamer=streamer,
        )
        self.assertEqual(len(streamer.values), 5)
        torch.testing.assert_close(streamer.values[0], input_ids.cpu())
        streamed_completion = torch.stack(streamer.values[1:], dim=1)
        torch.testing.assert_close(streamed_completion, generated[:, 3:].cpu())
        self.assertEqual(streamer.thread_ids[0], calling_thread)
        self.assertTrue(all(thread_id != calling_thread for thread_id in streamer.thread_ids[1:]))
        self.assertNotEqual(streamer.end_thread_id, calling_thread)

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
        layer_zero_v = model.model.layers[0].linear_attn._layer_zero_v
        self.assertEqual(
            tuple(tensor.shape for tensor in layer_zero_v),
            (
                (self.config.v_low_rank_dim, self.config.hidden_size),
                (self.config.hidden_size, self.config.v_low_rank_dim),
                (self.config.hidden_size,),
                (self.config.hidden_size, self.config.v_low_rank_dim),
                (self.config.v_low_rank_dim, self.config.hidden_size),
            ),
        )
        self.assertEqual(layer_zero_v[0].data_ptr(), layer_zero_v[4].data_ptr())
        self.assertEqual(layer_zero_v[1].data_ptr(), layer_zero_v[3].data_ptr())
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

    def test_state_only_inference_skips_final_norm_and_head(self):
        model = RwkvForCausalLM(self.config).half().eval()
        input_ids = torch.randint(0, self.config.vocab_size, (2, 5))
        cache = RwkvCache(self.config)

        with (
            mock.patch.object(
                self.fake,
                "infer_post_norm_output_forward_varlen",
                side_effect=AssertionError("state-only inference must skip final norm"),
            ),
            mock.patch.object(
                self.fake,
                "infer_head_linear_last_forward_varlen",
                side_effect=AssertionError("state-only inference must skip the LM head"),
            ),
            mock.patch.object(
                self.fake,
                "infer_sampling_temperature_topk_topp_forward_varlen",
                side_effect=AssertionError("state-only inference must skip sampling"),
            ),
        ):
            hidden_states, residual, all_hidden_states = model.model._forward_state_only(input_ids, None, cache, False)

        self.assertEqual(hidden_states.shape, (10, self.config.hidden_size))
        self.assertEqual(residual.shape, hidden_states.shape)
        self.assertIsNone(all_hidden_states)
        self.assertEqual(cache.get_seq_length(), 5)

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

    def test_unsupported_generation_modes_fail_closed(self):
        model = RwkvForCausalLM(self.config).half().eval()
        input_ids = torch.randint(1, self.config.vocab_size, (1, 4))
        with self.assertRaisesRegex(ValueError, "static input batch on one CUDA device"):
            model.generate(
                input_ids,
                max_new_tokens=3,
                do_sample=False,
                eos_token_id=None,
            )

    @require_torch_gpu
    def test_unsupported_cuda_graph_inputs_fail_closed(self):
        model = RwkvForCausalLM(self.config).half().eval().cuda()
        input_ids = torch.randint(1, self.config.vocab_size, (2, 4), device="cuda")

        cases = (
            (
                "padding or ragged batches",
                {"attention_mask": torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]], device="cuda")},
            ),
            ("custom logits processors", {"logits_processor": [LogitsProcessor()]}),
            ("detailed generation outputs", {"return_dict_in_generate": True}),
            ("generation mode 'GenerationMode.BEAM_SEARCH'", {"num_beams": 2}),
        )
        for error, kwargs in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                model.generate(
                    input_ids,
                    max_new_tokens=2,
                    do_sample=False,
                    eos_token_id=None,
                    **kwargs,
                )

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
