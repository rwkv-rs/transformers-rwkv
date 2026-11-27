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

import importlib.util
import itertools
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from transformers import RwkvConfig, is_torch_available
from transformers.testing_utils import (
    require_accelerate,
    require_bitsandbytes,
    require_peft,
    require_torch,
    require_torch_gpu,
    slow,
)

from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import RwkvForCausalLM, RwkvModel
    from transformers.models.rwkv.modeling_rwkv import RwkvCache

    all_model_classes = (RwkvModel, RwkvForCausalLM)


def tiny_config(**kwargs) -> RwkvConfig:
    values = {
        "vocab_size": 128,
        "context_length": 32,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "head_size": 64,
        "decay_low_rank_dim": 32,
        "a_low_rank_dim": 32,
        "v_low_rank_dim": 32,
        "gate_low_rank_dim": 32,
    }
    values.update(kwargs)
    return RwkvConfig(**values)


@require_torch
class RwkvConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = ConfigTester(self, config_class=RwkvConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_rwkv4_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires an RWKV-7 checkpoint"):
            RwkvConfig.from_dict({"model_type": "rwkv", "rescale_every": 6})

    def test_mtp_config_validation(self):
        self.assertEqual(tiny_config().mtp_intermediate_size, 224)
        for kwargs in (
            {"mtp_depth": -1},
            {"mtp_intermediate_size": 0},
            {"mtp_loss_scaling_factor": -1.0},
            {"train_mtp": True},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                tiny_config(**kwargs)

    def test_legacy_checkpoint_and_irwkv_sidecar_loading(self):
        model = RwkvForCausalLM(tiny_config())
        reference = RwkvForCausalLM(tiny_config(mtp_depth=3, mtp_intermediate_size=96, mtp_norm_epsilon=2e-5))
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as integrated:
            model.save_pretrained(directory)
            model_dir = Path(directory)
            config_path = model_dir / "config.json"
            config = json.loads(config_path.read_text())
            config_path.write_text(json.dumps({key: value for key, value in config.items() if "mtp" not in key}))
            loaded, info = RwkvForCausalLM.from_pretrained(directory, output_loading_info=True)
            self.assertEqual(info["missing_keys"], set())
            self.assertEqual(info["unexpected_keys"], set())
            self.assertEqual(loaded.config.mtp_depth, 0)
            self.assertIsNone(loaded.mtp)
            torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)

            (model_dir / "mtp_config.json").write_text(
                json.dumps(
                    {
                        "mtp_type": "lightmlp",
                        "mtp_depth": 3,
                        "mtp_loop": True,
                        "hidden_size": 64,
                        "intermediate_size": 96,
                        "norm_eps": 2e-5,
                        "num_blocks": 1,
                        "shared_lm_head": True,
                        "weight_file": "mtp.safetensors",
                    }
                )
            )
            save_file(reference.mtp.state_dict(prefix="mtp.blocks.0."), model_dir / "mtp.safetensors")
            loaded.load_mtp_sidecar(model_dir)
            self.assertEqual(loaded.config.mtp_depth, 3)
            self.assertEqual(loaded.mtp.norm_out.eps, 2e-5)
            torch.testing.assert_close(loaded.mtp.state_dict(), reference.mtp.state_dict(), rtol=0, atol=0)
            torch.testing.assert_close(loaded.lm_head.weight, model.lm_head.weight, rtol=0, atol=0)
            loaded.save_pretrained(integrated)
            restored = RwkvForCausalLM.from_pretrained(integrated)
            self.assertEqual(restored.mtp.norm_out.eps, 2e-5)
            torch.testing.assert_close(restored.state_dict(), loaded.state_dict(), rtol=0, atol=0)


@require_torch_gpu
class RwkvBF16ProviderTest(unittest.TestCase):
    def test_lightmtp_training(self):
        torch.manual_seed(117)
        model = RwkvForCausalLM(tiny_config(mtp_depth=3, train_mtp=True)).cuda().bfloat16().train()
        ids = torch.randint(1, model.config.vocab_size, (2, 9), device="cuda")
        labels = ids.clone()
        labels[0, -2:] = -100
        output = model(ids, labels=labels, use_cache=False)
        self.assertTrue(output.loss.isfinite())
        output.loss.backward()
        for parameter in model.mtp.parameters():
            self.assertTrue(parameter.grad.isfinite().all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.model.parameters()))
        self.assertIsNone(model.lm_head.weight.grad)

        model.zero_grad(set_to_none=True)
        base_loss = model(ids, labels=labels, use_cache=False, train_mtp=False).loss
        model.config.mtp_loss_scaling_factor = 0.4
        joint = model(ids, labels=labels, use_cache=False, train_mtp=False)
        torch.testing.assert_close(joint.loss, base_loss + 0.4 * output.loss, rtol=0, atol=0)
        joint.loss.backward()
        self.assertGreater(model.lm_head.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.mtp.proj.weight.grad.abs().sum().item(), 0)

    @torch.no_grad()
    def test_lightmtp_batch_generation_and_state_memory(self):
        torch.manual_seed(117)
        config = tiny_config(hidden_size=128, num_hidden_layers=8, mtp_depth=3, mtp_intermediate_size=128)
        model = RwkvForCausalLM(config).cuda().bfloat16().eval()
        # The base predicts its input token. Bias the drafter toward token 1 so only the first row accepts drafts.
        for name, parameter in model.named_parameters():
            if name.startswith("model.layers.") and parameter.ndim == 2:
                parameter.normal_(std=0.005)
        for weight in (model.model.embed_tokens.weight, model.lm_head.weight):
            weight.copy_(torch.eye(model.config.vocab_size, config.hidden_size, device="cuda", dtype=torch.bfloat16))
        model.mtp.proj.weight.zero_()
        model.mtp.down.weight.zero_()
        model.mtp.norm_out.bias[1] = 20
        for batch_size, mode in itertools.product((1, 4), ("fp32io16", "fp16")):
            with self.subTest(batch_size=batch_size, mode=mode):
                config.wkv_mode = mode
                ids = torch.arange(1, batch_size + 1, device="cuda")[:, None].expand(-1, 4).contiguous()
                kwargs = {"max_new_tokens": 8, "do_sample": False, "eos_token_id": None, "prefill_chunk_size": 2}
                expected = model.generate(ids, use_mtp=False, **kwargs)
                torch.testing.assert_close(expected[:, ids.shape[1] :], ids[:, :1].expand(-1, 8), rtol=0, atol=0)
                runner = next(iter(model._rwkv_generation_graphs.values()))
                expected_wkv = [
                    runner.state.get_recurrent_state(layer).clone() for layer in range(config.num_hidden_layers)
                ]
                expected_attention = runner.state.attention_shift.clone()
                expected_ffn = runner.state.feed_forward_shift.clone()
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                actual = model.generate(ids, use_mtp=True, **kwargs)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - allocated
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(len(model._rwkv_generation_graphs), 1)

                verify = runner.mtp_graph.state
                width = config.mtp_depth + 1
                scratch_bytes = batch_size * config.hidden_size * config.head_size * (2 if mode == "fp16" else 4)
                saved_bytes = 7 * config.num_hidden_layers * batch_size * width * config.hidden_size * 2
                self.assertEqual(verify.scratch_wkv.memory_layout["total_nbytes"], scratch_bytes)
                self.assertEqual(
                    verify.saved_wkv.nbytes + verify.saved_tmix_norm.nbytes + verify.saved_cmix_norm.nbytes,
                    saved_bytes,
                )
                self.assertGreaterEqual(peak, scratch_bytes + saved_bytes)
                # The graphs also allocate transient activations and TileLang workspaces.
                self.assertLess(peak, scratch_bytes + saved_bytes + 2 * 1024**2)
                for layer, state in enumerate(expected_wkv):
                    dtype = torch.int16 if mode == "fp16" else torch.int32
                    self.assertTrue(
                        torch.equal(runner.state.get_recurrent_state(layer).view(dtype), state.view(dtype))
                    )
                torch.testing.assert_close(runner.state.attention_shift, expected_attention, rtol=0, atol=0)
                torch.testing.assert_close(runner.state.feed_forward_shift, expected_ffn, rtol=0, atol=0)

                replay = runner.mtp_graph.replay
                first_prefixes = []

                def record_prefixes(hidden_states):
                    if first_prefixes:
                        return replay(hidden_states)
                    lengths = runner.decode_graph.completion_lengths.clone()
                    hidden_states = replay(hidden_states)
                    first_prefixes.extend((runner.decode_graph.completion_lengths - lengths).tolist())
                    return hidden_states

                with mock.patch.object(runner.mtp_graph, "replay", side_effect=record_prefixes):
                    torch.testing.assert_close(model.generate(ids, **kwargs), expected, rtol=0, atol=0)
                self.assertEqual(first_prefixes, [width] + [1] * (batch_size - 1))
                model._rwkv_generation_graphs.clear()

    def test_fp16_state_inference_preserves_fp32_training(self):
        torch.manual_seed(117)
        model = RwkvForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 16), device="cuda")
        reference = model(ids, labels=ids, use_cache=True)
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(reference.loss, parameters)
        model.config.wkv_mode = "fp16"
        training = model(ids, labels=ids, use_cache=True)
        torch.testing.assert_close(training.logits, reference.logits, atol=0, rtol=0)
        for actual, expected in zip(torch.autograd.grad(training.loss, parameters), gradients, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for actual, expected in zip(training.cache.layers, reference.cache.layers, strict=True):
            self.assertEqual(actual.recurrent_states[0].dtype, torch.float32)
            torch.testing.assert_close(actual.recurrent_states[0], expected.recurrent_states[0], atol=0, rtol=0)
        model.eval()
        # Grad-enabled evaluation still uses differentiable FP32-state operators.
        evaluated = model(ids, use_cache=True)
        torch.testing.assert_close(evaluated.logits, reference.logits, atol=0, rtol=0)
        self.assertEqual(evaluated.cache.layers[0].recurrent_states[0].dtype, torch.float32)
        with torch.no_grad():
            full = model(ids, use_cache=True)
            self.assertEqual(full.logits.dtype, torch.float32)
            cache = RwkvCache(model.config)
            chunks = [model(ids[:, :7], cache=cache).logits]
            for position in range(7, ids.shape[1]):
                chunks.append(model(ids[:, position : position + 1], cache=cache).logits)
            torch.testing.assert_close(torch.cat(chunks, dim=1), full.logits, atol=0, rtol=0)
            single = model(ids[:1], use_cache=True)
            torch.testing.assert_close(single.logits, full.logits[:1], atol=0, rtol=0)
            for whole, chunked, one in zip(full.cache.layers, cache.layers, single.cache.layers, strict=True):
                self.assertEqual(whole.recurrent_states[0].dtype, torch.float16)
                torch.testing.assert_close(chunked.recurrent_states[0], whole.recurrent_states[0], atol=0, rtol=0)
                torch.testing.assert_close(one.recurrent_states[0], whole.recurrent_states[0][:1], atol=0, rtol=0)
                for key in (0, 1):
                    self.assertEqual(whole.conv_states[key].dtype, torch.bfloat16)
                    torch.testing.assert_close(chunked.conv_states[key], whole.conv_states[key], atol=0, rtol=0)
            copied = cache.detach()
            copied.batch_repeat_interleave(2)
            copied.reorder_cache(torch.tensor([3, 0], device="cuda"))
            next_ids = ids[[1, 0], :1]
            actual = model(next_ids, cache=copied).logits
            expected = model(ids[:, :1], cache=cache).logits[[1, 0]]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            copied.reset()
            torch.testing.assert_close(model(ids, cache=copied).logits, full.logits, atol=0, rtol=0)
        model.train()
        with self.assertRaisesRegex(ValueError, "dtype torch.float32"):
            model(ids[:, :1], cache=full.cache)

    def test_grad_enabled_eval_backpropagates_through_fp32_cache(self):
        torch.manual_seed(122)
        model = RwkvForCausalLM(tiny_config(wkv_mode="fp16")).to(device="cuda", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
        cache = model(ids[:, :3], use_cache=True).cache
        layer = cache.layers[0]
        states = (layer.conv_states[0], layer.recurrent_states[0], layer.conv_states[1])
        self.assertEqual(states[1].dtype, torch.float32)
        loss = model(ids[:, 3:], labels=ids[:, 3:], cache=cache).loss
        for grad in torch.autograd.grad(loss, (*states, model.model.embed_tokens.weight)):
            self.assertTrue(grad.isfinite().all())
            self.assertGreater(grad.count_nonzero().item(), 0)

    @slow
    @torch.no_grad()
    def test_fp16_deferred_state_across_decode_clone_and_beam_reorder(self):
        torch.manual_seed(121)
        for heads in (40, 64):
            with self.subTest(heads=heads):
                config = tiny_config(
                    hidden_size=64 * heads, intermediate_size=None, num_hidden_layers=1, wkv_mode="fp16"
                )
                model = RwkvForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
                for parameter in model.parameters():
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.01)
                ids = torch.randint(1, config.vocab_size, (16, 11), device="cuda")
                cache, reference = RwkvCache(config), RwkvCache(config)
                model(ids[:, :3], cache=cache)
                model(ids[:, :3], cache=reference)
                handle = next(iter(cache._inference_recurrent_states.values()))
                self.assertGreater(handle.memory_layout["private_bytes_per_slot"], 0)

                chunks = ids[:, 3:9].split((1, 1, 1, 2, 1), dim=1)
                expected_logits = []
                for chunk in chunks:
                    expected_logits.append(model(chunk, cache=reference).logits)
                    reference.get_recurrent_state(0)
                with mock.patch.object(
                    type(handle), "materialize_slots_", side_effect=AssertionError("hot-path flush")
                ):
                    for chunk, expected in zip(chunks, expected_logits, strict=True):
                        actual = model(chunk, cache=cache).logits
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertIs(next(iter(cache._inference_recurrent_states.values())), handle)

                cloned = cache.clone()
                torch.testing.assert_close(
                    cloned.get_recurrent_state(0), reference.get_recurrent_state(0), atol=0, rtol=0
                )
                for current in (cache, reference):
                    model(ids[:, 9:10], cache=current)
                beams = torch.arange(15, -1, -1, device="cuda")
                beams[1] = beams[0]
                expected_state = reference.get_recurrent_state(0)[beams]
                cache.reorder_cache(beams)
                torch.testing.assert_close(cache.layers[0].recurrent_states[0], expected_state, atol=0, rtol=0)
                actual = model(ids[beams, 10:], cache=cache).logits
                expected = model(ids[:, 10:], cache=reference).logits
                torch.testing.assert_close(actual, expected[beams], atol=0, rtol=0)

                generated = model.generate(
                    ids[:, :4], max_new_tokens=3, prefill_chunk_size=3, do_sample=False, eos_token_id=None
                )
                eager = RwkvCache(config)
                model(ids[:, :3], cache=eager)
                for pos in range(3, 6):
                    logits = model(generated[:, pos : pos + 1], cache=eager).logits
                    torch.testing.assert_close(logits[:, -1].argmax(-1), generated[:, pos + 1], atol=0, rtol=0)
                runtime = next(reversed(model._rwkv_generation_graphs.values()))
                torch.testing.assert_close(
                    runtime.state.get_recurrent_state(0), eager.get_recurrent_state(0), atol=0, rtol=0
                )

    def test_training_prefill_decode_batch_state_and_optimizer(self):
        torch.manual_seed(115)
        model = RwkvForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 17), device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for step in range(2):
            model.train()
            training = model(ids, labels=ids, use_cache=False)
            self.assertEqual(training.logits.dtype, torch.float32)
            self.assertEqual(training.loss.dtype, torch.float32)
            training_cache = RwkvCache(model.config)
            stateful = model(ids, labels=ids, cache=training_cache, use_cache=True)
            torch.testing.assert_close(stateful.logits, training.logits, atol=0, rtol=0)
            model.eval()
            with torch.no_grad():
                full_cache = RwkvCache(model.config)
                full = model(ids, cache=full_cache, use_cache=True).logits
                torch.testing.assert_close(full, training.logits, atol=0, rtol=0)
                chunk_cache = RwkvCache(model.config)
                chunks = [model(ids[:, :7], cache=chunk_cache).logits]
                for position in range(7, ids.shape[1]):
                    chunks.append(model(ids[:, position : position + 1], cache=chunk_cache).logits)
                torch.testing.assert_close(torch.cat(chunks, dim=1), full, atol=0, rtol=0)
                single = model(ids[:1], use_cache=False).logits
                torch.testing.assert_close(single, full[:1], atol=0, rtol=0)
                for trained, whole, chunked in zip(
                    training_cache.layers, full_cache.layers, chunk_cache.layers, strict=True
                ):
                    for key in (0, 1):
                        self.assertEqual(whole.conv_states[key].dtype, torch.bfloat16)
                        torch.testing.assert_close(trained.conv_states[key], whole.conv_states[key], atol=0, rtol=0)
                        torch.testing.assert_close(chunked.conv_states[key], whole.conv_states[key], atol=0, rtol=0)
                    self.assertEqual(whole.recurrent_states[0].dtype, torch.float32)
                    torch.testing.assert_close(trained.recurrent_states[0], whole.recurrent_states[0], atol=0, rtol=0)
                    torch.testing.assert_close(chunked.recurrent_states[0], whole.recurrent_states[0], atol=0, rtol=0)
            if step == 0:
                parameters = tuple(model.parameters())
                expected_gradients = torch.autograd.grad(training.loss, parameters, retain_graph=True)
                stateful_gradients = torch.autograd.grad(stateful.loss, parameters)
                for expected_gradient, stateful_gradient in zip(expected_gradients, stateful_gradients, strict=True):
                    torch.testing.assert_close(stateful_gradient, expected_gradient, atol=1e-4, rtol=0.02)
                training.loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def test_cuda_graph_generation_matches_eager_bf16(self):
        torch.manual_seed(116)
        model = RwkvForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
        for mode in ("fp32io16", "fp16"):
            model.config.wkv_mode = mode
            cache = RwkvCache(model.config)
            logits = model(ids, cache=cache, logits_to_keep=1).logits[:, -1]
            token = logits.argmax(-1)
            completion = [token]
            for _ in range(2):
                logits = model(token[:, None], cache=cache).logits[:, -1]
                token = logits.argmax(-1)
                completion.append(token)
            expected = torch.cat((ids, torch.stack(completion, dim=1)), dim=1)
            for _ in range(2):
                actual = model.generate(
                    ids, max_new_tokens=3, prefill_chunk_size=3, do_sample=False, eos_token_id=None
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                runtime = next(reversed(model._rwkv_generation_graphs.values()))
                torch.testing.assert_close(runtime.decode_graph.logits, logits, atol=0, rtol=0)
                for layer_idx, layer in enumerate(cache.layers):
                    torch.testing.assert_close(
                        runtime.state.get_recurrent_state(layer_idx),
                        layer.recurrent_states[0],
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        runtime.state.attention_shift[layer_idx], layer.conv_states[0].squeeze(-1), atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        runtime.state.feed_forward_shift[layer_idx], layer.conv_states[1].squeeze(-1), atol=0, rtol=0
                    )
        self.assertEqual(len(model._rwkv_generation_graphs), 2)
        with self.assertRaisesRegex(ValueError, "does not accept an external cache"):
            model.generate(ids, cache=cache, max_new_tokens=1)

        head = torch.nn.Linear(
            model.config.hidden_size, model.vocab_size, bias=False, device="cuda", dtype=torch.bfloat16
        )
        model.set_output_embeddings(head)
        expected_tokens = model(ids, logits_to_keep=1).logits[:, -1].argmax(dim=-1)
        actual = model.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
        torch.testing.assert_close(actual[:, -1], expected_tokens)
        self.assertEqual(len(model._rwkv_generation_graphs), 1)

    def test_generation_streams_and_pads_finished_rows(self):
        class Streamer:
            def __init__(self):
                self.values = []
                self.threads = []

            def put(self, value):
                self.values.append(value)
                self.threads.append(threading.get_ident())

            def end(self):
                self.threads.append(threading.get_ident())

        torch.manual_seed(116)
        model = RwkvForCausalLM(tiny_config(mtp_depth=2)).cuda().bfloat16().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
            ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
            first_tokens = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].argmax(-1)
        self.assertNotEqual(first_tokens[0].item(), first_tokens[1].item())
        completions = []
        for use_mtp in (False, True):
            streamer = Streamer()
            generated = model.generate(
                ids,
                max_new_tokens=4,
                prefill_chunk_size=3,
                do_sample=False,
                use_mtp=use_mtp,
                eos_token_id=first_tokens[0].item(),
                pad_token_id=0,
                streamer=streamer,
            )
            completion = generated[:, ids.shape[1] :].cpu()
            completions.append(completion)
            self.assertGreater(completion.shape[1], 1)
            torch.testing.assert_close(completion[:, 0], first_tokens.cpu())
            torch.testing.assert_close(completion[0, 1:], torch.zeros_like(completion[0, 1:]))
            torch.testing.assert_close(streamer.values[0], ids.cpu())
            self.assertTrue(all(value.is_pinned() for value in streamer.values[1:]))
            torch.testing.assert_close(torch.stack(streamer.values[1:], dim=1), completion)
            self.assertEqual(streamer.threads[0], threading.get_ident())
            self.assertTrue(all(thread != threading.get_ident() for thread in streamer.threads[1:]))
        torch.testing.assert_close(*completions, rtol=0, atol=0)

    def test_packing_matches_independent_sequences(self):
        from transformers import DataCollatorWithFlattening

        torch.manual_seed(120)
        model = RwkvForCausalLM(tiny_config()).cuda().bfloat16()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        lengths = (4, 1, 7, 3)
        examples = [{"input_ids": torch.randint(1, model.config.vocab_size, (length,))} for length in lengths]
        collator = DataCollatorWithFlattening(return_flash_attn_kwargs=True)
        batch = {
            key: value.cuda() if isinstance(value, torch.Tensor) else value
            for key, value in collator(examples).items()
        }
        sequences = batch["input_ids"].split(lengths, dim=1)
        parameters = tuple(model.parameters())

        with self.assertRaisesRegex(ValueError, "packing cannot be used with cache"):
            model(**batch, cache=RwkvCache(model.config))
        with (
            self.assertWarnsRegex(FutureWarning, r"Use `cache` instead"),
            self.assertRaisesRegex(ValueError, "packing cannot be used with cache"),
        ):
            model(**batch, past_key_values=RwkvCache(model.config))

        for fused in (True, False):
            with (
                self.subTest(fused=fused),
                mock.patch.object(model.model, "_supports_fused_projections", return_value=fused),
            ):
                model.train()
                packed = model(**batch, use_cache=True)
                self.assertIsNone(packed.cache)
                separate = [model(ids, labels=ids if ids.shape[1] > 1 else None, use_cache=False) for ids in sequences]
                reference_logits = torch.cat([item.logits for item in separate], dim=1)
                torch.testing.assert_close(packed.logits, reference_logits, atol=0, rtol=0)
                reference_loss = sum(
                    item.loss * (length - 1) for item, length in zip(separate, lengths, strict=True) if length > 1
                ) / (sum(lengths) - len(lengths))
                torch.testing.assert_close(packed.loss, reference_loss)
                for actual, expected in zip(
                    torch.autograd.grad(packed.loss, parameters),
                    torch.autograd.grad(reference_loss, parameters),
                    strict=True,
                ):
                    torch.testing.assert_close(actual.float(), expected.float(), atol=0.01, rtol=0.04)
                with torch.no_grad():
                    evaluated = model.eval()(**batch)
                self.assertIsNone(evaluated.cache)
                torch.testing.assert_close(evaluated.logits, packed.logits, atol=0, rtol=0)

    @require_accelerate
    def test_trainer_with_packing(self):
        from transformers import DataCollatorWithFlattening, Trainer, TrainingArguments

        torch.manual_seed(121)
        model = RwkvForCausalLM(tiny_config()).cuda().bfloat16()
        dataset = [{"input_ids": list(range(1, length))} for length in range(4, 10)]
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory,
                max_steps=3,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                activation_checkpointing=True,
                save_strategy="no",
                report_to=[],
                disable_tqdm=True,
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=dataset,
                eval_dataset=dataset,
                data_collator=DataCollatorWithFlattening(return_flash_attn_kwargs=True),
            )
            self.assertEqual(trainer.train().global_step, 3)
            self.assertTrue(torch.isfinite(torch.tensor(trainer.evaluate()["eval_loss"])))

    def test_right_padding_preserves_loss_and_gradients(self):
        torch.manual_seed(117)
        model = RwkvForCausalLM(tiny_config(pad_token_id=0)).cuda().bfloat16()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
        ids[0, 4:] = 0
        mask = ids.ne(0)
        batched = model(ids, attention_mask=mask, labels=ids.masked_fill(~mask, -100), use_cache=False)
        first = model(ids[:1, :4], labels=ids[:1, :4], use_cache=False)
        second = model(ids[1:], labels=ids[1:], use_cache=False)
        reference_loss = (first.loss * 3 + second.loss * 6) / 9
        torch.testing.assert_close(batched.loss, reference_loss)
        parameters = tuple(model.parameters())
        for actual, expected in zip(
            torch.autograd.grad(batched.loss, parameters), torch.autograd.grad(reference_loss, parameters), strict=True
        ):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.01, rtol=0.04)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "unpadded"):
            model.eval()(ids, attention_mask=mask)

    @require_accelerate
    def test_trainer_right_padding_eval_save_reload_and_resume(self):
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        from .test_tokenization_rwkv import make_tokenizer

        tokenizer = make_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        dataset = [{"input_ids": list(range(index + 1, 10))} for index in range(4)]
        eval_dataset = [{"input_ids": [index + 1, 6, 7, 8]} for index in range(2)]
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(
                output_dir=directory,
                max_steps=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                eval_strategy="steps",
                eval_steps=1,
                save_steps=1,
                logging_strategy="no",
                report_to=[],
                disable_tqdm=True,
            )
            trainer = Trainer(
                model=RwkvForCausalLM(tiny_config(pad_token_id=tokenizer.pad_token_id)).cuda().bfloat16(),
                args=args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
            )
            self.assertEqual(trainer.train().global_step, 1)
            self.assertIn("eval_loss", trainer.evaluate())
            checkpoint = Path(directory) / "checkpoint-1"
            reloaded = RwkvForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).cuda()
            ids = torch.tensor([[1, 2, 3]], device="cuda")
            with torch.no_grad():
                torch.testing.assert_close(reloaded(ids).logits, trainer.model.eval()(ids).logits)
            args.max_steps = 2
            resumed = Trainer(
                model=reloaded, args=args, train_dataset=dataset, eval_dataset=eval_dataset, data_collator=collator
            )
            self.assertEqual(resumed.train(resume_from_checkpoint=checkpoint).global_step, 2)


@require_torch_gpu
@require_peft
class RwkvNoRATrainingTest(unittest.TestCase):
    def test_lora_nora_checkpoint_gradients_adapter_reload_and_merge(self):
        import copy

        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if "use_nora" not in LoraConfig.__dataclass_fields__:
            self.skipTest("requires Joluck/NoRA")
        for use_nora, wkv_mode in itertools.product((False, True), ("fp32io16", "fp16")):
            torch.manual_seed(113)
            base = RwkvForCausalLM(tiny_config(wkv_mode=wkv_mode)).to(device="cuda", dtype=torch.bfloat16).eval()
            # Nonzero output and gate matrices make every adapted projection observable.
            with torch.no_grad():
                for parameter in base.parameters():
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.03)
            restored_base = copy.deepcopy(base)
            ids = torch.randint(1, base.config.vocab_size, (2, 16), device="cuda")
            positions = torch.tensor([[7], [15]], device="cuda")
            with torch.no_grad():
                reference = base(ids, use_cache=False, logits_to_keep=positions).logits
            model = get_peft_model(
                base,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16,
                    lora_alpha=16,
                    lora_dropout=0.0,
                    use_nora=use_nora,
                    target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value", "lm_head"],
                ),
            )
            model.train()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(std=0.01)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
            logits = model(ids, use_cache=False, logits_to_keep=positions).logits
            logits.float().square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                    self.assertGreater(parameter.grad.float().norm().item(), 0, name)
                else:
                    self.assertIsNone(parameter.grad, name)
            optimizer.step()
            with torch.no_grad():
                trained = model(ids, use_cache=False, logits_to_keep=positions).logits
            model.eval()
            with torch.no_grad():
                adapted = model(ids, use_cache=False, logits_to_keep=positions).logits
                if wkv_mode == "fp32io16":
                    torch.testing.assert_close(adapted, trained, atol=0, rtol=0)
                cache = RwkvCache(model.config)
                first = model(ids[:, :7], cache=cache, use_cache=True).logits
                last = model(ids[:, 7:], cache=cache, use_cache=True).logits
                full = model(ids, use_cache=False).logits
                torch.testing.assert_close(torch.cat((first, last), dim=1), full, atol=0, rtol=0)
                generated = model.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
                torch.testing.assert_close(generated[:, -1], full[:, -1].argmax(-1))
                with model.disable_adapter():
                    disabled = model(ids, use_cache=False, logits_to_keep=positions).logits
                    disabled_full = model(ids, use_cache=False).logits
                    disabled_generated = model.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
                    torch.testing.assert_close(disabled_generated[:, -1], disabled_full[:, -1].argmax(-1))
                repeated = model.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
                torch.testing.assert_close(repeated, generated)
                self.assertEqual(len(model.base_model.model._rwkv_generation_graphs), 2)
                torch.testing.assert_close(disabled, reference, atol=0, rtol=0)
                full = model(ids, use_cache=False).logits
                torch.testing.assert_close(adapted[:, 0], full[torch.arange(2, device="cuda"), positions[:, 0]])
                with tempfile.TemporaryDirectory() as directory:
                    model.save_pretrained(directory)
                    restored = PeftModel.from_pretrained(restored_base, directory).eval()
                    actual = restored(ids, use_cache=False, logits_to_keep=positions).logits
                    torch.testing.assert_close(actual, adapted, atol=0, rtol=0)
                    restored.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
                    restored.merge_adapter(safe_merge=True)
                    merged_logits = restored(ids, use_cache=False, logits_to_keep=1).logits
                    merged_tokens = restored.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
                    torch.testing.assert_close(merged_tokens[:, -1], merged_logits[:, -1].argmax(-1))
                    restored.unmerge_adapter()
                    unmerged_logits = restored(ids, use_cache=False, logits_to_keep=1).logits
                    unmerged_tokens = restored.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
                    torch.testing.assert_close(unmerged_tokens[:, -1], unmerged_logits[:, -1].argmax(-1))
                    merged = restored.merge_and_unload(safe_merge=True)
                    actual = merged(ids, use_cache=False, logits_to_keep=positions).logits
                    torch.testing.assert_close(actual.float(), adapted.float(), atol=0.04, rtol=0.04)

    def test_unfused_lora_matches_fused_logits_and_gradients(self):
        from peft import LoraConfig, TaskType, get_peft_model

        torch.manual_seed(119)
        base = RwkvForCausalLM(tiny_config()).cuda().bfloat16()
        with torch.no_grad():
            for parameter in base.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=8,
                target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value", "lm_head"],
            ),
        )
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.01)

        ids = torch.randint(1, base.config.vocab_size, (2, 7), device="cuda")
        params = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
        fused = model(ids, labels=ids, use_cache=False)
        with mock.patch.object(base.model, "_supports_fused_projections", return_value=False):
            unfused = model(ids, labels=ids, use_cache=False)
        torch.testing.assert_close(unfused.logits.float(), fused.logits.float(), atol=0.02, rtol=0.03)
        for actual, expected in zip(
            torch.autograd.grad(unfused.loss, params), torch.autograd.grad(fused.loss, params), strict=True
        ):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.002, rtol=0.08)

    def test_unfused_dropout_dora_and_multiple_adapters_train(self):
        from peft import LoraConfig, TaskType, get_peft_model

        torch.manual_seed(113)
        for options in ({"lora_dropout": 0.05}, {"use_dora": True}, {}):
            with self.subTest(options=options):
                base = RwkvForCausalLM(tiny_config()).cuda().bfloat16()
                with torch.no_grad():
                    for parameter in base.parameters():
                        if parameter.ndim >= 2:
                            parameter.normal_(std=0.03)
                config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, target_modules=["r_proj", "key"], **options)
                model = get_peft_model(base, config)
                if not options:
                    model.add_adapter("second", config)
                    model.base_model.set_adapter(["default", "second"])
                model.train()
                self.assertFalse(base.model._supports_fused_projections())
                ids = torch.randint(1, base.config.vocab_size, (2, 7), device="cuda")
                optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
                loss = model(ids, labels=ids, use_cache=False).loss
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.grad is not None, "lora_" in name, name)
                optimizer.step()
                if options.get("use_dora"):
                    model.eval()
                    with torch.no_grad():
                        before_merge = model(ids, use_cache=False).logits
                        with self.assertRaisesRegex(ValueError, "Merge unsupported PEFT adapters"):
                            model.generate(ids, max_new_tokens=1)
                        merged = model.merge_and_unload()
                        self.assertTrue(merged.model._supports_fused_projections())
                        after_merge = merged(ids, use_cache=False).logits
                        torch.testing.assert_close(before_merge.float(), after_merge.float(), atol=0.03, rtol=0.04)

    @require_bitsandbytes
    @require_accelerate
    def test_qlora_4bit_training(self):
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

        from transformers import BitsAndBytesConfig

        with tempfile.TemporaryDirectory() as directory:
            base = RwkvForCausalLM(tiny_config()).bfloat16()
            with torch.no_grad():
                for parameter in base.parameters():
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.03)
            base.save_pretrained(directory)
            quantized = RwkvForCausalLM.from_pretrained(
                directory,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
                device_map={"": "cuda"},
                dtype=torch.bfloat16,
            )
            prepare_model_for_kbit_training(quantized, gradient_checkpointing_kwargs={"use_reentrant": True})
            model = get_peft_model(
                quantized, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, target_modules=["r_proj", "key"])
            ).train()
            self.assertFalse(quantized.model._supports_fused_projections())
            ids = torch.randint(1, base.config.vocab_size, (2, 7), device="cuda")
            loss = model(ids, labels=ids, use_cache=False).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, parameter in model.named_parameters():
                self.assertEqual(parameter.grad is not None, "lora_" in name, name)
            torch.optim.AdamW(p for p in model.parameters() if p.requires_grad).step()


@require_torch
class RwkvCacheTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.hidden = torch.zeros(2, 3, self.config.hidden_size, dtype=torch.bfloat16)

    def test_initial_state_shapes_and_length(self):
        cache = RwkvCache(self.config)
        self.assertTrue(all(layer.number_of_states == 2 for layer in cache.layers))
        attention_shift, wkv_state, mlp_shift = cache.layer_states(0, self.hidden)
        self.assertEqual(attention_shift.shape, (2, 64))
        self.assertEqual(mlp_shift.shape, (2, 64))
        self.assertEqual(wkv_state.shape, (2, 1, 64, 64))
        self.assertEqual(wkv_state.dtype, torch.float32)
        cache.advance(3, 2, self.hidden.device)
        self.assertEqual(cache.get_seq_length(), 3)
        self.assertEqual(cache.stream_lengths.tolist(), [3, 3])

    def test_training_update_preserves_autograd_and_detach_cuts_it(self):
        cache = RwkvCache(self.config)
        _, wkv_state, _ = cache.layer_states(0, self.hidden)
        attention_shift = (self.hidden[:, -1] + 1).requires_grad_()
        next_wkv = wkv_state.clone().requires_grad_() + 1
        mlp_shift = (self.hidden[:, -1] + 2).requires_grad_()
        cache.update_layer(0, attention_shift, next_wkv, mlp_shift, training=True)
        self.assertIsNotNone(cache.layers[0].conv_states[0].grad_fn)
        self.assertIsNotNone(cache.layers[0].recurrent_states[0].grad_fn)
        cloned = cache.clone()
        detached = cache.detach()
        self.assertIsNotNone(cloned.layers[0].conv_states[0].grad_fn)
        self.assertIsNone(detached.layers[0].conv_states[0].grad_fn)
        self.assertNotEqual(cloned.layers[0].conv_states[0].data_ptr(), cache.layers[0].conv_states[0].data_ptr())

    def test_reset_repeat_select_and_reorder(self):
        cache = RwkvCache(self.config)
        cache.layer_states(0, self.hidden)
        cache.advance(3, 2, self.hidden.device)
        cache.batch_repeat_interleave(2)
        self.assertEqual(cache.batch_size, 4)
        cache.batch_select_indices(torch.tensor([3, 1]))
        self.assertEqual(cache.batch_size, 2)
        cache.reset(torch.tensor([0]))
        self.assertEqual(cache.stream_lengths.tolist(), [0, 3])
        cache.reorder_cache(torch.tensor([1, 0]))
        self.assertEqual(cache.stream_lengths.tolist(), [3, 0])
        cache.reset()
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertEqual(cache.stream_lengths.tolist(), [0, 0])


@require_torch_gpu
class RwkvConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[3] / "temp" / "rwkv_pth2st.py"
        spec = importlib.util.spec_from_file_location("rwkv_pth2st", path)
        cls.converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.converter)

    @staticmethod
    def canonical_state_dict(model: RwkvForCausalLM) -> dict[str, torch.Tensor]:
        config = model.config
        source = {}
        top_level = {
            "model.embed_tokens.weight": "emb.weight",
            "model.embedding_norm.weight": "blocks.0.ln0.weight",
            "model.embedding_norm.bias": "blocks.0.ln0.bias",
            "model.norm.weight": "ln_out.weight",
            "model.norm.bias": "ln_out.bias",
            "lm_head.weight": "head.weight",
        }
        projection = {
            "r_proj": "receptance",
            "k_proj": "key",
            "v_proj": "value",
            "o_proj": "output",
            "g_norm": "ln_x",
        }
        for target_key, tensor in model.state_dict().items():
            if target_key in top_level:
                source_key = top_level[target_key]
            else:
                parts = target_key.split(".")
                layer_idx, component = parts[2], parts[3]
                suffix = parts[4:]
                component = {
                    "linear_attn": "att",
                    "mlp": "ffn",
                    "input_layernorm": "ln1",
                    "post_attention_layernorm": "ln2",
                }[component]
                if component == "att":
                    suffix[0] = projection.get(suffix[0], suffix[0])
                source_key = ".".join(("blocks", layer_idx, component, *suffix))
            value = tensor.detach().clone()
            if target_key.split(".")[-1] in {
                "x_r",
                "x_w",
                "x_k",
                "x_v",
                "x_a",
                "x_g",
                "w0",
                "a0",
                "v0",
                "k_k",
                "k_a",
            }:
                value = value.view(1, 1, -1)
            source[source_key] = value
        source["blocks.0.att.v0"] = torch.zeros(1, 1, config.hidden_size)
        source["blocks.0.att.v1"] = torch.zeros(config.hidden_size, config.v_low_rank_dim)
        source["blocks.0.att.v2"] = torch.zeros(config.v_low_rank_dim, config.hidden_size)
        return source

    def test_converted_checkpoint_matches_reference_logits(self):
        torch.manual_seed(113)
        reference = RwkvForCausalLM(tiny_config()).bfloat16().eval()
        with torch.no_grad():
            for parameter in reference.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "test.pth"
            output = Path(directory) / "converted"
            torch.save(self.canonical_state_dict(reference), checkpoint)
            with mock.patch.object(self.converter, "save_tokenizer"):
                self.converter.convert_checkpoint(checkpoint, output, 32, "20KB", None)
            model = RwkvForCausalLM.from_pretrained(output, dtype=torch.bfloat16).cuda().eval()
            reference.cuda()
            ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
            with torch.no_grad():
                torch.testing.assert_close(model(ids).logits, reference(ids).logits, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
