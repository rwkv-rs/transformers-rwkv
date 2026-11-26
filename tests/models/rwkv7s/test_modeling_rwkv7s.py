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

import itertools
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from transformers import Rwkv7sConfig, is_torch_available
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

    from transformers import Rwkv7sForCausalLM, Rwkv7sModel
    from transformers.models.rwkv7s.modeling_rwkv7s import Rwkv7sCache

    all_model_classes = (Rwkv7sModel, Rwkv7sForCausalLM)


def tiny_config(**kwargs) -> Rwkv7sConfig:
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
    return Rwkv7sConfig(**values)


def lm_parameters(model: "Rwkv7sForCausalLM") -> tuple["torch.nn.Parameter", ...]:
    return tuple(parameter for name, parameter in model.named_parameters() if not name.startswith("autodeco."))


@require_torch
class Rwkv7sConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = ConfigTester(self, config_class=Rwkv7sConfig, hidden_size=64)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_rwkv4_config_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires an RWKV-7S checkpoint"):
            Rwkv7sConfig.from_dict({"model_type": "rwkv7s", "rescale_every": 6})


@require_torch_gpu
class Rwkv7sBF16ProviderTest(unittest.TestCase):
    def test_fp16_state_inference_preserves_fp32_training(self):
        torch.manual_seed(117)
        model = Rwkv7sForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 16), device="cuda")
        reference = model(ids, labels=ids, use_cache=True)
        parameters = lm_parameters(model)
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
            cache = Rwkv7sCache(model.config)
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
        model = Rwkv7sForCausalLM(tiny_config(wkv_mode="fp16")).to(device="cuda", dtype=torch.bfloat16).eval()
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
                model = Rwkv7sForCausalLM(config).to(device="cuda", dtype=torch.bfloat16).eval()
                for parameter in model.parameters():
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.01)
                ids = torch.randint(1, config.vocab_size, (16, 11), device="cuda")
                cache, reference = Rwkv7sCache(config), Rwkv7sCache(config)
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
                eager = Rwkv7sCache(config)
                model(ids[:, :3], cache=eager)
                for pos in range(3, 6):
                    logits = model(generated[:, pos : pos + 1], cache=eager).logits
                    torch.testing.assert_close(logits[:, -1].argmax(-1), generated[:, pos + 1], atol=0, rtol=0)
                runtime = next(reversed(model._rwkv7s_generation_graphs.values()))
                torch.testing.assert_close(
                    runtime.state.get_recurrent_state(0), eager.get_recurrent_state(0), atol=0, rtol=0
                )

    def test_training_prefill_decode_batch_state_and_optimizer(self):
        torch.manual_seed(115)
        model = Rwkv7sForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16)
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
            training_cache = Rwkv7sCache(model.config)
            stateful = model(ids, labels=ids, cache=training_cache, use_cache=True)
            torch.testing.assert_close(stateful.logits, training.logits, atol=0, rtol=0)
            model.eval()
            with torch.no_grad():
                full_cache = Rwkv7sCache(model.config)
                full = model(ids, cache=full_cache, use_cache=True).logits
                torch.testing.assert_close(full, training.logits, atol=0, rtol=0)
                chunk_cache = Rwkv7sCache(model.config)
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
                parameters = lm_parameters(model)
                expected_gradients = torch.autograd.grad(training.loss, parameters, retain_graph=True)
                stateful_gradients = torch.autograd.grad(stateful.loss, parameters)
                for expected_gradient, stateful_gradient in zip(expected_gradients, stateful_gradients, strict=True):
                    torch.testing.assert_close(stateful_gradient, expected_gradient, atol=1e-4, rtol=0.02)
                training.loss.backward()
                for name, parameter in model.named_parameters():
                    if name.startswith("autodeco."):
                        self.assertIsNone(parameter.grad, name)
                    else:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def test_cuda_graph_generation_matches_eager_bf16(self):
        torch.manual_seed(116)
        model = Rwkv7sForCausalLM(tiny_config()).to(device="cuda", dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
        for mode in ("fp32io16", "fp16"):
            model.config.wkv_mode = mode
            cache = Rwkv7sCache(model.config)
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
                runtime = next(reversed(model._rwkv7s_generation_graphs.values()))
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
        self.assertEqual(len(model._rwkv7s_generation_graphs), 2)
        with self.assertRaisesRegex(ValueError, "does not accept an external cache"):
            model.generate(ids, cache=cache, max_new_tokens=1)

        head = torch.nn.Linear(
            model.config.hidden_size, model.vocab_size, bias=False, device="cuda", dtype=torch.bfloat16
        )
        model.set_output_embeddings(head)
        expected_tokens = model(ids, logits_to_keep=1).logits[:, -1].argmax(dim=-1)
        actual = model.generate(ids, max_new_tokens=1, do_sample=False, eos_token_id=None)
        torch.testing.assert_close(actual[:, -1], expected_tokens)
        self.assertEqual(len(model._rwkv7s_generation_graphs), 1)

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
        model = Rwkv7sForCausalLM(tiny_config()).cuda().bfloat16().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
            ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
            first_tokens = model(ids, use_cache=False, logits_to_keep=1).logits[:, -1].argmax(-1)
        self.assertNotEqual(first_tokens[0].item(), first_tokens[1].item())
        streamer = Streamer()
        generated = model.generate(
            ids,
            max_new_tokens=4,
            prefill_chunk_size=3,
            do_sample=False,
            eos_token_id=first_tokens[0].item(),
            pad_token_id=0,
            streamer=streamer,
        )
        completion = generated[:, ids.shape[1] :].cpu()
        self.assertGreater(completion.shape[1], 1)
        torch.testing.assert_close(completion[:, 0], first_tokens.cpu())
        torch.testing.assert_close(completion[0, 1:], torch.zeros_like(completion[0, 1:]))
        torch.testing.assert_close(streamer.values[0], ids.cpu())
        self.assertTrue(all(value.is_pinned() for value in streamer.values[1:]))
        torch.testing.assert_close(torch.stack(streamer.values[1:], dim=1), completion)
        self.assertEqual(streamer.threads[0], threading.get_ident())
        self.assertTrue(all(thread != threading.get_ident() for thread in streamer.threads[1:]))

    def test_packing_matches_independent_sequences(self):
        from transformers import DataCollatorWithFlattening

        torch.manual_seed(120)
        model = Rwkv7sForCausalLM(tiny_config()).cuda().bfloat16()
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
        parameters = lm_parameters(model)

        with self.assertRaisesRegex(ValueError, "packing cannot be used with cache"):
            model(**batch, cache=Rwkv7sCache(model.config))
        with (
            self.assertWarnsRegex(FutureWarning, r"Use `cache` instead"),
            self.assertRaisesRegex(ValueError, "packing cannot be used with cache"),
        ):
            model(**batch, past_key_values=Rwkv7sCache(model.config))

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
        model = Rwkv7sForCausalLM(tiny_config()).cuda().bfloat16()
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
        model = Rwkv7sForCausalLM(tiny_config(pad_token_id=0)).cuda().bfloat16()
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
        parameters = lm_parameters(model)
        for actual, expected in zip(
            torch.autograd.grad(batched.loss, parameters), torch.autograd.grad(reference_loss, parameters), strict=True
        ):
            torch.testing.assert_close(actual.float(), expected.float(), atol=0.01, rtol=0.04)
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "unpadded"):
            model.eval()(ids, attention_mask=mask)

    @require_accelerate
    def test_trainer_right_padding_eval_save_reload_and_resume(self):
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

        from .test_tokenization_rwkv7s import make_tokenizer

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
                model=Rwkv7sForCausalLM(tiny_config(pad_token_id=tokenizer.pad_token_id)).cuda().bfloat16(),
                args=args,
                train_dataset=dataset,
                eval_dataset=eval_dataset,
                data_collator=collator,
            )
            self.assertEqual(trainer.train().global_step, 1)
            self.assertIn("eval_loss", trainer.evaluate())
            checkpoint = Path(directory) / "checkpoint-1"
            reloaded = Rwkv7sForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).cuda()
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
class Rwkv7sNoRATrainingTest(unittest.TestCase):
    def test_lora_nora_checkpoint_gradients_adapter_reload_and_merge(self):
        import copy

        from peft import LoraConfig, PeftModel, TaskType, get_peft_model

        if "use_nora" not in LoraConfig.__dataclass_fields__:
            self.skipTest("requires Joluck/NoRA")
        for use_nora, wkv_mode in itertools.product((False, True), ("fp32io16", "fp16")):
            torch.manual_seed(113)
            base = Rwkv7sForCausalLM(tiny_config(wkv_mode=wkv_mode)).to(device="cuda", dtype=torch.bfloat16).eval()
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
                cache = Rwkv7sCache(model.config)
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
                self.assertEqual(len(model.base_model.model._rwkv7s_generation_graphs), 2)
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
        base = Rwkv7sForCausalLM(tiny_config()).cuda().bfloat16()
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
                base = Rwkv7sForCausalLM(tiny_config()).cuda().bfloat16()
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
            base = Rwkv7sForCausalLM(tiny_config()).bfloat16()
            with torch.no_grad():
                for parameter in base.parameters():
                    if parameter.ndim >= 2:
                        parameter.normal_(std=0.03)
            base.save_pretrained(directory)
            quantized = Rwkv7sForCausalLM.from_pretrained(
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
class Rwkv7sCacheTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.hidden = torch.zeros(2, 3, self.config.hidden_size, dtype=torch.bfloat16)

    def test_initial_state_shapes_and_length(self):
        cache = Rwkv7sCache(self.config)
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
        cache = Rwkv7sCache(self.config)
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
        cache = Rwkv7sCache(self.config)
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
class Rwkv7sAutoSaveLoadTest(unittest.TestCase):
    def test_auto_save_reload_matches_reference_logits(self):
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        torch.manual_seed(113)
        reference = Rwkv7sForCausalLM(tiny_config()).bfloat16().eval()
        with torch.no_grad():
            for parameter in reference.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.03)
        with tempfile.TemporaryDirectory() as directory:
            reference.save_pretrained(directory)
            config = AutoConfig.from_pretrained(directory, local_files_only=True)
            self.assertIsInstance(config, Rwkv7sConfig)
            base_model = AutoModel.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
            self.assertIsInstance(base_model, Rwkv7sModel)
            model = (
                AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16, local_files_only=True)
                .cuda()
                .eval()
            )
            self.assertIsInstance(model, Rwkv7sForCausalLM)
            reference.cuda()
            ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")
            with torch.no_grad():
                actual_out = model(ids)
                expected_out = reference(ids)
                torch.testing.assert_close(actual_out.logits, expected_out.logits, atol=0, rtol=0)
                torch.testing.assert_close(actual_out.temp_logits, expected_out.temp_logits, atol=0, rtol=0)
                torch.testing.assert_close(actual_out.top_p_logits, expected_out.top_p_logits, atol=0, rtol=0)


@require_torch_gpu
class Rwkv7sAutoDecoTest(unittest.TestCase):
    def test_autodeco_forward_and_training(self):
        torch.manual_seed(131)
        model = Rwkv7sForCausalLM(tiny_config(num_hidden_layers=3)).to(device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim >= 2:
                    parameter.normal_(std=0.05)
        ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")

        # Forward uses normalized second-to-last layer hidden states and supports toggling off.
        model.eval()
        with torch.no_grad():
            base = model.model(ids, output_hidden_states=True, use_cache=False)
            expected_prev = torch.nn.functional.layer_norm(
                base.hidden_states[-2].float(),
                (model.config.hidden_size,),
                model.model.norm.weight.float(),
                model.model.norm.bias.float(),
                model.config.layer_norm_epsilon,
            ).to(torch.bfloat16)
            torch.testing.assert_close(base.autodeco_hidden_state, expected_prev, atol=2e-2, rtol=2e-2)
            out = model(ids, use_cache=False)
            self.assertEqual(out.temp_logits.shape, (2, 7, 1))
            self.assertEqual(out.top_p_logits.shape, (2, 7, 1))
            kept = model(ids, use_cache=False, logits_to_keep=1)
            torch.testing.assert_close(kept.temp_logits, out.temp_logits[:, -1:], atol=0, rtol=0)
            torch.testing.assert_close(kept.top_p_logits, out.top_p_logits[:, -1:], atol=0, rtol=0)

        disabled = Rwkv7sForCausalLM(tiny_config(use_autodeco=False)).to(device="cuda", dtype=torch.bfloat16).eval()
        self.assertIsNone(disabled.autodeco)
        with torch.no_grad():
            self.assertIsNone(disabled(ids, use_cache=False).temp_logits)
        with self.assertRaisesRegex(ValueError, "requires `config.use_autodeco=True`"):
            disabled(ids, labels=ids, train_autodeco=True)

        # AutoDeco training (with FP32 heads over BF16 backbone) updates only autodeco parameters.
        model.train()
        model.autodeco.float()
        train_out = model(ids, labels=ids, use_cache=False, train_autodeco=True)
        self.assertTrue(torch.isfinite(train_out.loss))
        train_out.loss.backward()
        for name, parameter in model.named_parameters():
            if name.startswith("autodeco."):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.float().norm().item(), 0.0, name)
            else:
                self.assertIsNone(parameter.grad, name)

    @torch.no_grad()
    def test_autodeco_cuda_graph_generation_matches_eager(self):
        from transformers.integrations.flash_rwkv2 import load_flash_rwkv2

        torch.manual_seed(134)
        model = Rwkv7sForCausalLM(tiny_config(num_hidden_layers=3)).to(device="cuda", dtype=torch.bfloat16).eval()
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                parameter.normal_(std=0.05)
        ids = torch.randint(1, model.config.vocab_size, (2, 7), device="cuda")

        flash_rwkv2 = load_flash_rwkv2()
        seed = 4242
        with mock.patch("torch.Tensor.random_", return_value=torch.tensor(seed, dtype=torch.int64)):
            generated = model.generate(
                ids, max_new_tokens=3, prefill_chunk_size=3, do_sample=True, use_autodeco=True, eos_token_id=None
            )

        cache = Rwkv7sCache(model.config)
        penalties = torch.zeros((2, model.vocab_size), dtype=torch.float32, device="cuda")
        sampling_states = flash_rwkv2.setup_sampling_states(seed, 2)
        slots = torch.arange(2, dtype=torch.int32, device="cuda")

        out = model(ids, cache=cache, logits_to_keep=1)
        expected_tokens = []
        for _ in range(3):
            step_temp = out.temp_logits[:, -1].float().clamp_min(1e-2).view(2).contiguous()
            step_top_p = out.top_p_logits[:, -1].float().view(2).contiguous()
            token = flash_rwkv2.infer_sampling_six_parameter_forward_varlen(
                out.logits[:, -1].float().contiguous(),
                penalties,
                sampling_states,
                slots,
                temperature=step_temp,
                top_p=step_top_p,
            ).to(dtype=torch.long)
            expected_tokens.append(token)
            if len(expected_tokens) < 3:
                out = model(token[:, None], cache=cache)

        torch.testing.assert_close(
            generated, torch.cat((ids, torch.stack(expected_tokens, dim=1)), dim=1), atol=0, rtol=0
        )
        runtime = next(reversed(model._rwkv7s_generation_graphs.values()))
        torch.testing.assert_close(runtime.decode_graph.autodeco_temperature, step_temp, atol=0, rtol=0)
        torch.testing.assert_close(runtime.decode_graph.autodeco_top_p, step_top_p, atol=0, rtol=0)

        model.generate(
            ids, max_new_tokens=3, prefill_chunk_size=3, do_sample=True, use_autodeco=False, eos_token_id=None
        )
        self.assertEqual(len(model._rwkv7s_generation_graphs), 2)
        self.assertIsNone(next(reversed(model._rwkv7s_generation_graphs.values())).decode_graph.autodeco_temperature)


if __name__ == "__main__":
    unittest.main()
