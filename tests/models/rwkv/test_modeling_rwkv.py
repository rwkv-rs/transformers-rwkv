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
import types
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
from transformers.dependency_versions_table import deps
from transformers.testing_utils import require_peft, require_torch, require_torch_gpu


if importlib.util.find_spec("torch") is not None:
    import torch

    from transformers import RwkvCache, RwkvForCausalLM, RwkvModel, RwkvTimeMix, RwkvTrainingState
    from transformers.models.rwkv.modeling_rwkv import (
        _INFERENCE_OPERATORS,
        _cache_states,
        _infer_tmix_projection_spec,
        _load_flash_rwkv2,
        _stateful_training_metadata,
    )


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
    def test_flashrwkv2_a7_inference_operator_contract(self):
        self.assertEqual(deps["FlashRWKV2"], "FlashRWKV2==0.1.0a7")
        expected = {
            "infer_embedding_ln0_forward_varlen",
            "infer_tmix_postnorm_tokenshift_forward_varlen",
            "infer_tmix_wkv_prepare_forward_varlen",
            "infer_tmix_wkv7_recurrent_fp16_forward_varlen",
            "infer_tmix_wkv7_recurrent_fp32io16_forward_varlen",
            "infer_tmix_wkv7_chunk_bf16_forward_varlen",
            "infer_tmix_readout_forward_varlen",
            "infer_cmix_forward_varlen",
            "infer_post_norm_output_forward_varlen",
            "infer_head_linear_all_forward_varlen",
            "infer_head_linear_last_forward_varlen",
            "prepare_tmix_wkv7_recurrent_metadata",
        }
        self.assertEqual(set(_INFERENCE_OPERATORS), expected)
        self.assertEqual(len(_INFERENCE_OPERATORS), len(expected))

    def test_flashrwkv2_contract_fails_closed_on_missing_operator(self):
        missing = "infer_cmix_forward_varlen"
        module = types.SimpleNamespace(
            **{name: (lambda: None) for name in _INFERENCE_OPERATORS if name != missing},
            __version__="0.1.0a7",
            __file__="fake/flashrwkv2/__init__.py",
        )
        with (
            mock.patch("transformers.models.rwkv.modeling_rwkv.importlib.import_module", return_value=module),
            self.assertRaisesRegex(RuntimeError, missing),
        ):
            _load_flash_rwkv2("inference")

    def test_default_contract(self):
        config = RwkvConfig()
        self.assertEqual(config.model_type, "rwkv")
        self.assertEqual(config.architecture_version, "rwkv7")
        self.assertEqual(config.head_size, 64)
        self.assertEqual(config.num_attention_heads, config.hidden_size // 64)
        self.assertEqual(config.intermediate_size, 4 * config.hidden_size)
        self.assertEqual(config.wkv_state_dtype, "float32")
        self.assertEqual(config.number_of_conv_states, 2)
        self.assertEqual(config.bos_token_id, config.eos_token_id)
        self.assertIsNone(config.pad_token_id)

    def test_training_state_contract_and_selective_reset(self):
        config = tiny_config()
        state = RwkvTrainingState.zeros(config, 3, device="cpu", dtype=torch.bfloat16)
        self.assertEqual(state.time_mix_shift.shape, (2, 3, 128))
        self.assertEqual(state.wkv.shape, (2, 3, 2, 64, 64))
        self.assertEqual(state.wkv.dtype, torch.float32)
        state.validate(config, batch_size=3, device="cpu", dtype=torch.bfloat16)
        state.time_mix_shift.fill_(1)
        state.wkv.fill_(2)
        state.channel_mix_shift.fill_(3)
        cloned = state.clone_detach()
        cloned.reset_([1], wkv=False)
        self.assertEqual(cloned.time_mix_shift[:, 1].count_nonzero(), 0)
        self.assertEqual(cloned.channel_mix_shift[:, 1].count_nonzero(), 0)
        self.assertTrue(torch.all(cloned.wkv == 2))
        self.assertTrue(torch.all(state.time_mix_shift == 1))

    def test_training_state_rejects_shape_dtype_and_device_mismatch(self):
        config = tiny_config()
        state = RwkvTrainingState.zeros(config, 2, device="cpu", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "must have shape"):
            state.clone().validate(config, batch_size=1, device="cpu", dtype=torch.bfloat16)
        state.wkv = state.wkv.to(torch.bfloat16)
        with self.assertRaisesRegex(TypeError, "torch.float32"):
            state.validate(config, batch_size=2, device="cpu", dtype=torch.bfloat16)

    def test_stateful_training_metadata_uses_canonical_replay_boundaries(self):
        offsets, starts, ends = _stateful_training_metadata(2, 33, torch.device("cpu"))
        self.assertEqual(offsets.tolist(), [0, 3, 6])
        self.assertEqual(starts.tolist(), [0, 16, 32, 33, 49, 65])
        self.assertEqual(ends.tolist(), [16, 32, 33, 49, 65, 66])

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

    def test_nontrivial_attention_mask_still_fails_closed(self):
        model = RwkvForCausalLM(tiny_config()).eval()
        with self.assertRaisesRegex(ValueError, "all-ones mask"):
            model(
                torch.ones(1, 2, dtype=torch.long),
                attention_mask=torch.tensor([[1, 0]]),
                use_cache=False,
            )

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
        cache._rwkv_metadata_key = (1, 1, "cuda", 0)
        cache._rwkv_metadata = object()
        cache.reset()
        self.assertIsNone(cache._rwkv_metadata_key)
        self.assertIsNone(cache._rwkv_metadata)

    def test_generation_inputs_support_first_step_inputs_embeds(self):
        model = RwkvForCausalLM(tiny_config())
        input_ids = torch.ones(1, 2, dtype=torch.long)
        inputs_embeds = torch.zeros(1, 2, model.config.hidden_size)
        prepared = model.prepare_inputs_for_generation(input_ids, inputs_embeds=inputs_embeds)
        self.assertNotIn("input_ids", prepared)
        self.assertIs(prepared["inputs_embeds"], inputs_embeds)

    def test_cache_state_shape_is_owned_by_each_time_mix_layer(self):
        cache_config = mock.Mock(num_hidden_layers=2, number_of_conv_states=2)
        cache = RwkvCache(cache_config)
        hidden_states = torch.zeros(1, 3, 512)
        layer_config = mock.Mock(num_attention_heads=4, head_size=128)
        _, wkv_state, _ = _cache_states(cache, 0, hidden_states, layer_config)
        self.assertEqual(wkv_state.shape, (1, 4, 128, 128))


@require_torch
class Rwkv7ConversionTest(unittest.TestCase):
    @staticmethod
    def _converter_module():
        path = Path(__file__).parents[3] / "temp" / "rwkv_pth2st.py"
        spec = importlib.util.spec_from_file_location("rwkv_pth2st", path)
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
    @require_peft
    def test_unmerged_lora_projection_spec_matches_peft(self):
        from peft import LoraConfig

        config = tiny_config(hidden_size=1024, intermediate_size=4096)
        model = RwkvForCausalLM(config).cuda().eval()
        targets = ["receptance", "key", "value", "output"]
        model.add_adapter(
            LoraConfig(r=8, lora_alpha=16, target_modules=targets, init_lora_weights=False),
            adapter_name="first",
        )
        model.set_adapter("first")
        model.prepare_for_inference()
        x = torch.randn(5, config.hidden_size, device="cuda", dtype=torch.float16)

        for name in targets:
            projection = getattr(model.model.blocks[0].att, name)
            with torch.no_grad():
                expected = projection(x)
                weight, lora_a, lora_b, scale = _infer_tmix_projection_spec(projection)
                actual = torch.nn.functional.linear(x, weight)
                if lora_a is not None:
                    actual = actual + torch.nn.functional.linear(torch.nn.functional.linear(x, lora_a), lora_b) * scale
            torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

        input_ids = torch.randint(0, config.vocab_size, (1, 5), device="cuda")
        with torch.no_grad():
            output = model(input_ids, use_cache=False)
        self.assertTrue(torch.isfinite(output.logits).all())

        projection = model.model.blocks[0].att.receptance
        model.disable_adapters()
        with torch.no_grad():
            weight, lora_a, lora_b, scale = _infer_tmix_projection_spec(projection)
            disabled = torch.nn.functional.linear(x, weight)
            base = projection.get_base_layer()(x)
        self.assertIsNone(lora_a)
        self.assertIsNone(lora_b)
        self.assertEqual(scale, 1.0)
        torch.testing.assert_close(disabled, base, atol=0, rtol=0)

        model.enable_adapters()
        model.add_adapter(
            LoraConfig(r=16, lora_alpha=8, target_modules=targets, init_lora_weights=False),
            adapter_name="second",
        )
        model.set_adapter(["first", "second"])
        with self.assertRaisesRegex(RuntimeError, "exactly one active"):
            _infer_tmix_projection_spec(projection)

    def test_stateful_chunk_forward_matches_single_stateful_call(self):
        config = tiny_config()
        model = RwkvForCausalLM(config).cuda().to(torch.bfloat16).train()
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device="cuda")
        initial = RwkvTrainingState.zeros(config, 2, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            whole = model(input_ids, training_state=initial.clone(), use_cache=False)
            first = model(input_ids[:, :7], training_state=initial.clone(), use_cache=False)
            second = model(input_ids[:, 7:], training_state=first.training_state, use_cache=False)
        torch.testing.assert_close(
            whole.logits,
            torch.cat((first.logits, second.logits), dim=1),
            atol=4e-2,
            rtol=4e-2,
        )
        for actual, expected in zip(second.training_state.tensors(), whole.training_state.tensors(), strict=True):
            torch.testing.assert_close(actual, expected, atol=4e-2, rtol=4e-2)

    def test_stateful_training_backward_and_fp32_wkv_contract(self):
        config = tiny_config()
        model = RwkvForCausalLM(config).cuda().to(torch.bfloat16).train()
        input_ids = torch.randint(0, config.vocab_size, (1, 7), device="cuda")
        state = RwkvTrainingState.zeros(config, 1, device="cuda", dtype=torch.bfloat16)
        outputs = model(input_ids, labels=input_ids, training_state=state, use_cache=False)
        self.assertEqual(outputs.training_state.wkv.dtype, torch.float32)
        outputs.loss.backward()
        self.assertIsNotNone(model.model.blocks[0].att.receptance.weight.grad)
        self.assertTrue(torch.isfinite(model.model.blocks[0].att.receptance.weight.grad).all())

    def test_inference_preparation_offloads_only_canonical_ffn_down_layout(self):
        model = RwkvForCausalLM(tiny_config(hidden_size=1024, intermediate_size=4096)).cuda().eval()
        expected = model.model.blocks[0].ffn.value.weight.detach().cpu().half().clone()

        model.prepare_for_inference().prepare_for_inference()

        channel_mix = model.model.blocks[0].ffn
        self.assertEqual(channel_mix.value.weight.device.type, "cpu")
        self.assertEqual(channel_mix._value_runtime.device.type, "cuda")
        torch.testing.assert_close(channel_mix.value.weight, expected, atol=0, rtol=0)
        with torch.no_grad():
            channel_mix.value.weight.add_(1)
        model.prepare_for_inference()
        torch.testing.assert_close(
            channel_mix._value_runtime,
            channel_mix.value.weight.T.cuda(),
            atol=0,
            rtol=0,
        )
        expected = channel_mix.value.weight.detach().clone()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = RwkvForCausalLM.from_pretrained(directory, dtype=torch.float16)
        torch.testing.assert_close(reloaded.model.blocks[0].ffn.value.weight, expected, atol=0, rtol=0)

    def test_recurrent_metadata_is_scoped_to_cuda_stream(self):
        class FakeFlashRwkv2:
            def __init__(self):
                self.calls = 0

            def prepare_tmix_wkv7_recurrent_metadata(self, *args, **kwargs):
                self.calls += 1
                return object()

        flash = FakeFlashRwkv2()
        cache = RwkvCache(tiny_config())
        device = torch.device("cuda")
        first_stream = torch.cuda.current_stream()
        first = cache.recurrent_metadata(flash, 1, 4, device)
        self.assertIs(first, cache.recurrent_metadata(flash, 1, 4, device))
        with torch.cuda.stream(torch.cuda.Stream()):
            second = cache.recurrent_metadata(flash, 1, 4, device)
        self.assertIsNot(first, second)
        with torch.cuda.stream(first_stream):
            third = cache.recurrent_metadata(flash, 1, 4, device)
        self.assertIsNot(second, third)
        self.assertEqual(flash.calls, 3)

    def test_cuda_graph_capture_uses_prepared_stream_metadata(self):
        config = tiny_config(hidden_size=1024, intermediate_size=4096)
        model = RwkvForCausalLM(config).cuda().eval().prepare_for_inference()
        input_ids = torch.ones((1, 1), device="cuda", dtype=torch.long)
        cache = RwkvCache(config)
        with torch.no_grad():
            model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache.reset()

        flash = importlib.import_module("flashrwkv2")
        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            cache.recurrent_metadata(flash, 1, 1, input_ids.device)
        graph_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=graph_stream):
            logits = model(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1).logits
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(logits.shape, (1, 1, config.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

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

    def test_outputs_use_one_rwkv_type_and_standard_loss_hook(self):
        config = tiny_config()
        model = RwkvForCausalLM(config).cuda().to(torch.bfloat16).train()
        input_ids = torch.randint(0, config.vocab_size, (1, 16), device="cuda")
        calls = []

        def loss_function(*, logits, labels, vocab_size, **kwargs):
            calls.append((logits, labels, vocab_size, kwargs))
            return logits.float().sum() * 0

        model.loss_function = loss_function
        outputs = model(input_ids, labels=input_ids, use_cache=False, output_hidden_states=True)
        self.assertEqual(type(outputs).__name__, "RwkvCausalLMOutput")
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        self.assertEqual(calls[0][2], config.vocab_size)
        tuple_outputs = model(input_ids, use_cache=False, return_dict=False)
        self.assertEqual(len(tuple_outputs), 1)
        self.assertIsInstance(tuple_outputs[0], torch.Tensor)

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
        first_layer = whole.past_key_values.layers[0]
        self.assertEqual(first_layer.number_of_states, 2)
        self.assertTrue(first_layer.is_conv_states_initialized[0])
        self.assertTrue(first_layer.is_conv_states_initialized[1])
        self.assertTrue(first_layer.is_recurrent_states_initialized[0])
        self.assertFalse(first_layer.is_recurrent_states_initialized[1])
        self.assertEqual(first_layer.recurrent_states[0].dtype, torch.float32)
        self.assertEqual(
            first_layer.recurrent_states[0].shape,
            (1, config.num_attention_heads, config.head_size, config.head_size),
        )
        staged.past_key_values.batch_repeat_interleave(2)
        self.assertEqual(staged.past_key_values.batch_size, 2)
        staged.past_key_values.batch_select_indices(torch.tensor([1], device="cuda"))
        self.assertEqual(staged.past_key_values.batch_size, 1)

    def test_inference_output_hidden_states_preserves_block_boundaries(self):
        config = tiny_config(hidden_size=1024, intermediate_size=4096)
        model = RwkvForCausalLM(config).cuda().eval().prepare_for_inference()
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device="cuda")
        with torch.no_grad():
            outputs = model(input_ids, use_cache=False, output_hidden_states=True)
        self.assertEqual(len(outputs.hidden_states), config.num_hidden_layers + 1)
        for hidden_state in outputs.hidden_states:
            self.assertEqual(hidden_state.shape, (1, 5, config.hidden_size))
            self.assertTrue(torch.isfinite(hidden_state).all())

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
