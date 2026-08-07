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

import hashlib
import importlib.util
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    RwkvConfig,
)
from transformers.testing_utils import require_torch, require_torch_gpu


if importlib.util.find_spec("torch") is not None:
    import torch

    from transformers import RwkvCache, RwkvForCausalLM, RwkvModel, RwkvTimeMix


FLASH_RWKV2_AVAILABLE = importlib.util.find_spec("flashrwkv2") is not None
require_flash_rwkv2 = unittest.skipUnless(FLASH_RWKV2_AVAILABLE, "test requires the FlashRWKV2 public package")


def tiny_config(**kwargs):
    values = {
        "vocab_size": 256,
        "context_length": 16,
        "hidden_size": 128,
        "num_hidden_layers": 2,
        "intermediate_size": 512,
        "head_size": 64,
        "decay_low_rank_dim": 32,
        "a_low_rank_dim": 32,
        "v_low_rank_dim": 32,
        "gate_low_rank_dim": 32,
    }
    values.update(kwargs)
    return RwkvConfig(**values)


def train_temp_tmix_init(module, config, layer_idx: int) -> None:
    channels = config.hidden_size
    ratio_0_to_1 = layer_idx / max(config.num_hidden_layers - 1, 1)
    ratio_1_to_almost0 = 1.0 - layer_idx / config.num_hidden_layers
    ddd = torch.arange(channels, dtype=torch.float32).view(1, 1, -1) / channels
    linear = torch.arange(channels, dtype=torch.float32) / max(channels - 1, 1) - 0.5
    head_index = torch.arange(channels, dtype=torch.float32) % config.head_size
    zigzag = (head_index - (config.head_size - 1) / 2) / ((config.head_size - 1) / 2)
    zigzag = zigzag * zigzag.abs()
    decay = -6 + 6 * (torch.arange(channels, dtype=torch.float32) / max(channels - 1, 1)).pow(1 + ratio_0_to_1**0.3)
    with torch.no_grad():
        for name, exponent in {"r": 0.2, "w": 0.9, "k": 0.7, "v": 0.7, "a": 0.9, "g": 0.2}.items():
            getattr(module, f"x_{name}").copy_(1.0 - ddd.pow(exponent * ratio_1_to_almost0))
        module.w0.copy_((decay + 0.5 + zigzag * 2.5).view(1, 1, -1))
        module.a0.copy_((-0.19 + zigzag * 0.3 + linear * 0.4).view(1, 1, -1))
        module.v0.copy_((0.73 - linear * 0.4).view(1, 1, -1))
        module.k_k.copy_((0.71 - linear * 0.1).view(1, 1, -1))
        module.k_a.fill_(1.02)
        module.r_k.fill_(-0.04)
        for name in ("w1", "a1", "v1", "g1"):
            getattr(module, name).zero_()
        for name in ("w2", "a2", "v2", "g2"):
            parameter = getattr(module, name)
            gain = math.sqrt(parameter.shape[0] / parameter.shape[1]) if parameter.shape[0] > parameter.shape[1] else 1
            torch.nn.init.orthogonal_(parameter, gain=gain * 0.1)
        torch.nn.init.orthogonal_(module.receptance.weight, gain=1.0)
        torch.nn.init.orthogonal_(module.key.weight, gain=0.1)
        torch.nn.init.orthogonal_(module.value.weight, gain=1.0)
        module.output.weight.zero_()
        module.ln_x.weight.fill_(((layer_idx + 1) / config.num_hidden_layers) ** 0.7)
        module.ln_x.bias.zero_()


def train_temp_model_init(model) -> None:
    config = model.config
    with torch.no_grad():
        model.model.emb.weight.uniform_(-1e-4, 1e-4)
        for layer_idx, block in enumerate(model.model.blocks):
            for layer_norm in (getattr(block, "ln0", None), block.ln1, block.ln2):
                if layer_norm is not None:
                    layer_norm.weight.fill_(1.0)
                    layer_norm.bias.zero_()
            train_temp_tmix_init(block.att, config, layer_idx)
            channels = config.hidden_size
            ratio_1_to_almost0 = 1.0 - layer_idx / config.num_hidden_layers
            ddd = torch.arange(channels, dtype=torch.float32).view(1, 1, -1) / channels
            block.ffn.x_k.copy_(1.0 - ddd.pow(ratio_1_to_almost0**4))
            torch.nn.init.orthogonal_(block.ffn.key.weight, gain=1.0)
            block.ffn.value.weight.zero_()
        model.model.ln_out.weight.fill_(1.0)
        model.model.ln_out.bias.zero_()
        gain = (
            0.5 * math.sqrt(config.vocab_size / config.hidden_size) if config.vocab_size > config.hidden_size else 0.5
        )
        torch.nn.init.orthogonal_(model.head.weight, gain=gain)


@require_torch
class Rwkv7ConfigurationTest(unittest.TestCase):
    def test_default_contract(self):
        config = RwkvConfig()
        self.assertEqual(config.model_type, "rwkv")
        self.assertEqual(config.architecture_version, "rwkv7")
        self.assertEqual(config.head_size, 64)
        self.assertEqual(config.num_attention_heads, config.hidden_size // 64)
        self.assertEqual(config.intermediate_size, int((3.5 * config.hidden_size) // 32 * 32))
        self.assertEqual(config.wkv_state_dtype, "float32")
        self.assertEqual(config.number_of_conv_states, 2)
        self.assertEqual(config.bos_token_id, config.eos_token_id)
        self.assertIsNone(config.pad_token_id)

    def test_train_temp_low_rank_formulas(self):
        expected = {
            768: (64, 64, 32, 128),
            1024: (64, 64, 64, 160),
            2048: (128, 128, 64, 224),
            2560: (128, 128, 96, 256),
            4096: (160, 160, 96, 320),
        }
        for hidden_size, ranks in expected.items():
            with self.subTest(hidden_size=hidden_size):
                config = RwkvConfig(hidden_size=hidden_size)
                self.assertEqual(
                    (
                        config.decay_low_rank_dim,
                        config.a_low_rank_dim,
                        config.v_low_rank_dim,
                        config.gate_low_rank_dim,
                    ),
                    ranks,
                )

    def test_explicit_low_rank_dimensions_are_never_recomputed(self):
        config = RwkvConfig(
            hidden_size=128,
            decay_low_rank_dim=17,
            a_low_rank_dim=19,
            v_low_rank_dim=23,
            gate_low_rank_dim=29,
        )
        reloaded = RwkvConfig.from_dict(config.to_dict())
        self.assertEqual(
            (
                reloaded.decay_low_rank_dim,
                reloaded.a_low_rank_dim,
                reloaded.v_low_rank_dim,
                reloaded.gate_low_rank_dim,
            ),
            (17, 19, 23, 29),
        )

    def test_rejects_noncanonical_architecture(self):
        with self.assertRaisesRegex(Exception, "head_size=64"):
            RwkvConfig(head_size=128)
        with self.assertRaisesRegex(Exception, "wkv_state_dtype='float32'"):
            RwkvConfig(wkv_state_dtype="float16")
        with self.assertRaisesRegex(Exception, "num_attention_heads"):
            RwkvConfig(num_attention_heads=1)

    def test_rejects_rwkv4_configuration(self):
        with self.assertRaisesRegex(ValueError, "Legacy RWKV-4"):
            RwkvConfig(rescale_every=6)
        with self.assertRaisesRegex(ValueError, "Legacy RWKV-4"):
            RwkvConfig(attention_hidden_size=4096)

    def test_auto_mappings_keep_rwkv_identity(self):
        config = tiny_config()
        config_dict = config.to_dict()
        config_dict.pop("model_type")
        self.assertIsInstance(AutoConfig.for_model("rwkv", **config_dict), RwkvConfig)
        self.assertIsInstance(AutoModel.from_config(config), RwkvModel)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), RwkvForCausalLM)


@require_torch
class Rwkv7ModelStructureTest(unittest.TestCase):
    all_model_classes = (RwkvModel, RwkvForCausalLM)

    def test_public_component_and_canonical_tensor_names(self):
        model = RwkvForCausalLM(tiny_config())
        self.assertIsInstance(model.model.blocks[0].att, RwkvTimeMix)
        state = model.state_dict()
        self.assertEqual(state["model.blocks.0.att.w1"].shape, (128, 32))
        self.assertEqual(state["model.blocks.0.att.w2"].shape, (32, 128))
        self.assertEqual(state["model.blocks.0.att.v0"].shape, (1, 1, 128))
        self.assertEqual(state["model.blocks.0.att.v1"].shape, (128, 32))
        self.assertEqual(state["model.blocks.1.att.v1"].shape, (128, 32))
        self.assertEqual(state["model.blocks.0.ffn.key.weight"].shape, (512, 128))
        self.assertEqual(state["model.blocks.0.ffn.value.weight"].shape, (128, 512))
        self.assertTrue(all(tensor.isfinite().all() for tensor in state.values()))

    def test_runtime_fails_closed_on_cpu(self):
        model = RwkvForCausalLM(tiny_config()).train()
        with self.assertRaisesRegex(RuntimeError, "no product fallback"):
            model(torch.ones(1, 16, dtype=torch.long), use_cache=False)

    def test_explicit_low_rank_dimensions_control_parameter_shapes(self):
        model = RwkvForCausalLM(
            tiny_config(decay_low_rank_dim=17, a_low_rank_dim=19, v_low_rank_dim=23, gate_low_rank_dim=29)
        )
        attention = model.model.blocks[0].att
        self.assertEqual(attention.w1.shape, (128, 17))
        self.assertEqual(attention.a1.shape, (128, 19))
        self.assertEqual(attention.v1.shape, (128, 23))
        self.assertEqual(attention.g1.shape, (128, 29))

    def test_initialization_matches_train_temp_tensor_by_tensor(self):
        config = tiny_config()
        actual = RwkvForCausalLM(config)
        expected = RwkvForCausalLM(config)
        torch.manual_seed(20260806)
        actual.model.reset_parameters()
        actual.reset_head_parameters()
        torch.manual_seed(20260806)
        train_temp_model_init(expected)
        for name, tensor in actual.state_dict().items():
            self.assertTrue(torch.equal(tensor, expected.state_dict()[name]), name)

    def test_rwkv_cache_uses_standard_cache_interface(self):
        cache = RwkvCache(tiny_config())
        self.assertEqual(cache.get_seq_length(), 0)
        self.assertEqual(len(cache.layers), 2)
        cache.reorder_cache(torch.tensor([0]))
        cache.reset()


@require_torch
class Rwkv7ConversionTest(unittest.TestCase):
    @staticmethod
    def _converter_module():
        path = Path(__file__).parents[3] / "temp" / "convert_rwkv7_checkpoint.py"
        spec = importlib.util.spec_from_file_location("convert_rwkv7_checkpoint", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_default_vocab_json_uses_pinned_hub_artifact(self):
        converter = self._converter_module()
        with tempfile.TemporaryDirectory() as directory:
            vocab_json = Path(directory) / converter.RWKV_VOCAB_FILENAME
            vocab_json.write_bytes(b"published vocabulary")
            digest = hashlib.sha256(vocab_json.read_bytes()).hexdigest()
            with (
                mock.patch.object(converter, "RWKV_VOCAB_SHA256", digest),
                mock.patch.object(converter, "hf_hub_download", return_value=str(vocab_json)) as download,
            ):
                self.assertEqual(converter.resolve_vocab_json(None), vocab_json)
        download.assert_called_once_with(
            repo_id="rwkv-rs/rwkv7-g1-st",
            filename="rwkv_vocab_v20230424.json",
            revision="fd122cc7244c28db19beceb398aa033c35576b71",
        )

    def test_local_vocab_json_is_verified_without_hub_download(self):
        converter = self._converter_module()
        with tempfile.TemporaryDirectory() as directory:
            vocab_json = Path(directory) / "local.json"
            vocab_json.write_bytes(b"published vocabulary")
            digest = hashlib.sha256(vocab_json.read_bytes()).hexdigest()
            with (
                mock.patch.object(converter, "RWKV_VOCAB_SHA256", digest),
                mock.patch.object(converter, "hf_hub_download") as download,
            ):
                self.assertEqual(converter.resolve_vocab_json(vocab_json), vocab_json)
                download.assert_not_called()

    def test_vocab_json_hash_mismatch_fails_closed(self):
        converter = self._converter_module()
        with tempfile.TemporaryDirectory() as directory:
            vocab_json = Path(directory) / "wrong.json"
            vocab_json.write_bytes(b"wrong vocabulary")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                converter.resolve_vocab_json(vocab_json)

    def test_build_tokenizer_delegates_json_loading_to_rwkv_trie(self):
        converter = self._converter_module()
        calls = []

        class FakeRwkvTrie:
            @staticmethod
            def from_file(path):
                calls.append(path)
                return converter.models.WordLevel({"<|endoftext|>": 0, "a": 1}, unk_token="<|endoftext|>")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocab_json = root / "vocab.json"
            vocab_json.write_text("{}", encoding="utf-8")
            chat_template = root / "chat_template.jinja"
            chat_template.write_text("{{ messages }}", encoding="utf-8")
            with mock.patch.object(converter.models, "RwkvTrie", FakeRwkvTrie, create=True):
                tokenizer = converter.build_tokenizer(vocab_json, chat_template, model_max_length=128)
        self.assertEqual(calls, [str(vocab_json)])
        self.assertEqual(tokenizer.model_max_length, 128)

    def test_build_tokenizer_requires_rwkv_trie_from_file(self):
        converter = self._converter_module()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(converter.models, "RwkvTrie", object(), create=True),
        ):
            root = Path(directory)
            vocab_json = root / "vocab.json"
            vocab_json.write_text("{}", encoding="utf-8")
            chat_template = root / "chat_template.jinja"
            chat_template.write_text("{{ messages }}", encoding="utf-8")
            with self.assertRaisesRegex(ImportError, "RwkvTrie.from_file"):
                converter.build_tokenizer(vocab_json, chat_template, model_max_length=128)

    def test_minimal_conversion_and_fresh_model_process(self):
        converter = self._converter_module()
        model = RwkvForCausalLM(tiny_config())
        source = {
            key.removeprefix("model.") if key.startswith("model.") else key: value
            for key, value in model.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "rwkv7.pth"
            output = root / "converted"
            torch.save(source, checkpoint)
            original_save_tokenizer = converter.save_tokenizer
            converter.save_tokenizer = lambda *args, **kwargs: None
            try:
                converter.convert_checkpoint(checkpoint, output, context_length=16, max_shard_size="20MB")
            finally:
                converter.save_tokenizer = original_save_tokenizer
            loaded = AutoModelForCausalLM.from_pretrained(output)
            self.assertIsInstance(loaded, RwkvForCausalLM)
            self.assertEqual(set(loaded.state_dict()), set(model.state_dict()))
            self.assertTrue(
                all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
            )
            command = [
                str(Path(__file__).parents[3] / ".venv" / "bin" / "python"),
                "-c",
                (
                    "from transformers import AutoConfig,AutoModel,AutoModelForCausalLM,AutoTokenizer; "
                    f"p={str(output)!r}; "
                    "assert type(AutoConfig.from_pretrained(p,trust_remote_code=False)).__name__=='RwkvConfig'; "
                    "assert type(AutoModel.from_pretrained(p,trust_remote_code=False)).__name__=='RwkvModel'; "
                    "assert type(AutoModelForCausalLM.from_pretrained(p,trust_remote_code=False)).__name__"
                    "=='RwkvForCausalLM'"
                ),
            ]
            subprocess.run(command, check=True)

    def test_converter_preserves_nonformula_low_rank_dimensions(self):
        converter = self._converter_module()
        config = tiny_config(decay_low_rank_dim=17, a_low_rank_dim=19, v_low_rank_dim=23, gate_low_rank_dim=29)
        model = RwkvForCausalLM(config)
        source = {
            key.removeprefix("model.") if key.startswith("model.") else key: value
            for key, value in model.state_dict().items()
        }
        inferred = converter.infer_config(source, context_length=16)
        self.assertEqual(
            (
                inferred.decay_low_rank_dim,
                inferred.a_low_rank_dim,
                inferred.v_low_rank_dim,
                inferred.gate_low_rank_dim,
            ),
            (17, 19, 23, 29),
        )

    def test_converter_rejects_cross_layer_low_rank_drift(self):
        converter = self._converter_module()
        model = RwkvForCausalLM(tiny_config())
        source = {
            key.removeprefix("model.") if key.startswith("model.") else key: value
            for key, value in model.state_dict().items()
        }
        source["blocks.1.att.w1"] = torch.empty(128, 31)
        source["blocks.1.att.w2"] = torch.empty(31, 128)
        with self.assertRaisesRegex(ValueError, "must agree across all layers"):
            converter.infer_config(source, context_length=16)

    def test_official_standard_tokenizer_contract(self):
        source = os.environ.get("RWKV7_TOKENIZER_PATH")
        if not source:
            self.skipTest("set RWKV7_TOKENIZER_PATH to rwkv_vocab_v20230424.json")
        converter = self._converter_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            converter.save_tokenizer(
                Path(source),
                Path(__file__).parents[3] / "temp" / "rwkv_chat_template.jinja",
                output,
                model_vocab_size=65536,
                model_max_length=4096,
            )
            tokenizer = AutoTokenizer.from_pretrained(output, trust_remote_code=False, use_fast=True)
            self.assertTrue(tokenizer.is_fast)
            self.assertEqual(len(tokenizer), 65530)
            self.assertEqual((tokenizer.bos_token_id, tokenizer.eos_token_id), (0, 0))
            self.assertIsNone(tokenizer.pad_token_id)
            self.assertIsNone(tokenizer.unk_token_id)
            self.assertNotIn("auto_map", tokenizer.init_kwargs)
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}], tokenize=False, add_generation_prompt=True
            )
            self.assertEqual(rendered, "User✿hello✿\nBot✿<think")

    def test_converter_rejects_mixed_orig_mod_prefix(self):
        converter = self._converter_module()
        with self.assertRaisesRegex(ValueError, "must either prefix every"):
            converter._canonical_state_dict({"_orig_mod.emb.weight": torch.empty(1), "head.weight": torch.empty(1)})

    def test_converter_detaches_larger_source_storage(self):
        source_storage = torch.arange(16, dtype=torch.float32)
        converted = self._converter_module().convert_state_dict({"emb.weight": source_storage[4:8]})
        tensor = converted["model.emb.weight"]
        self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
        self.assertEqual(tensor.tolist(), [4.0, 5.0, 6.0, 7.0])


@require_torch
@require_torch_gpu
@require_flash_rwkv2
class Rwkv7FlashRwkv2Test(unittest.TestCase):
    def test_inference_preparation_offloads_only_canonical_ffn_down_layout(self):
        model = RwkvForCausalLM(tiny_config(hidden_size=1024, intermediate_size=4096)).cuda().eval()
        expected = model.model.blocks[0].ffn.value.weight.detach().cpu().half().clone()

        model.prepare_for_inference().prepare_for_inference()

        channel_mix = model.model.blocks[0].ffn
        self.assertEqual(channel_mix.value.weight.device.type, "cpu")
        self.assertEqual(channel_mix._value_runtime.device.type, "cuda")
        torch.testing.assert_close(channel_mix.value.weight, expected, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = RwkvForCausalLM.from_pretrained(directory, dtype=torch.float16)
        torch.testing.assert_close(reloaded.model.blocks[0].ffn.value.weight, expected, atol=0, rtol=0)

    def test_training_forward_backward_uses_public_train_temp_family(self):
        config = tiny_config()
        model = RwkvForCausalLM(config).cuda().to(torch.bfloat16).train()
        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
        outputs = model(input_ids, labels=input_ids, use_cache=False)
        self.assertEqual(outputs.logits.shape, (1, 16, config.vocab_size))
        self.assertEqual(outputs.logits.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(outputs.loss))
        outputs.loss.backward()
        self.assertIsNotNone(model.model.blocks[0].att.output.weight.grad)
        self.assertGreater(model.model.blocks[0].att.output.weight.grad.abs().max().item(), 0)

    def test_inference_prefill_decode_and_continuation(self):
        config = tiny_config(hidden_size=1024, intermediate_size=4096)
        model = RwkvForCausalLM(config).cuda().eval().prepare_for_inference()
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device="cuda")
        with torch.no_grad():
            whole = model(input_ids, use_cache=True)
            first = model(input_ids[:, :4], use_cache=True)
            staged = model(input_ids[:, 4:], past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(whole.past_key_values.get_seq_length(), 5)
        self.assertEqual(staged.past_key_values.get_seq_length(), 5)
        torch.testing.assert_close(whole.logits[:, -1], staged.logits[:, -1], atol=2e-2, rtol=2e-2)
        self.assertEqual(whole.past_key_values.layers[0].recurrent_states[0].dtype, torch.float32)
        staged.past_key_values.batch_repeat_interleave(2)
        self.assertEqual(staged.past_key_values.batch_size, 2)
        staged.past_key_values.batch_select_indices(torch.tensor([1], device="cuda"))
        self.assertEqual(staged.past_key_values.batch_size, 1)

    def test_inference_can_compute_only_last_logits(self):
        config = tiny_config(hidden_size=1024, intermediate_size=4096)
        model = RwkvForCausalLM(config).cuda().eval().prepare_for_inference()
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device="cuda")
        with torch.no_grad():
            all_logits = model(input_ids, use_cache=False).logits
            last_logits = model(input_ids, use_cache=False, logits_to_keep=1).logits
        self.assertEqual(last_logits.shape, (1, 1, config.vocab_size))
        torch.testing.assert_close(last_logits[:, 0], all_logits[:, -1], atol=2e-2, rtol=2e-2)

    def test_greedy_generation(self):
        config = tiny_config(
            hidden_size=1024,
            intermediate_size=4096,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = RwkvForCausalLM(config).cuda().eval().prepare_for_inference()
        input_ids = torch.randint(1, config.vocab_size, (1, 3), device="cuda")
        with torch.no_grad():
            generated = model.generate(input_ids, max_new_tokens=2, do_sample=False)
        self.assertEqual(generated.shape, (1, 5))

    def test_small_unsupported_inference_shape_fails_closed(self):
        model = RwkvForCausalLM(tiny_config()).cuda().eval().prepare_for_inference()
        input_ids = torch.randint(0, model.config.vocab_size, (1, 1), device="cuda")
        with self.assertRaisesRegex(RuntimeError, "supports only K>=1024"):
            model(input_ids, use_cache=True)
