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
import json
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path

from packaging.requirements import Requirement

from transformers import GenerationConfig, RwkvConfig, is_torch_available
from transformers.dependency_versions_table import deps
from transformers.testing_utils import require_peft, require_torch

from ...test_configuration_common import ConfigTester


if is_torch_available():
    import torch

    from transformers import RwkvForCausalLM, RwkvModel
    from transformers.cache_utils import DynamicCache, LinearAttentionLayer
    from transformers.integrations.flash_rwkv2 import flash_rwkv2_linear_spec, load_flash_rwkv2
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.rwkv.modeling_rwkv import (
        RwkvAttention,
        RwkvCache,
        RwkvDecoderLayer,
        RwkvFeedForward,
        RwkvPreTrainedModel,
    )

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

    def test_rank_formulas_and_fixed_contract(self):
        config = RwkvConfig(hidden_size=1024, num_hidden_layers=24)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.intermediate_size, 4096)
        self.assertEqual(config.decay_low_rank_dim, 64)
        self.assertEqual(config.a_low_rank_dim, 64)
        self.assertEqual(config.v_low_rank_dim, 64)
        self.assertEqual(config.gate_low_rank_dim, 160)
        self.assertEqual(config.wkv_mode, "fp32io16")
        self.assertEqual(tiny_config(wkv_mode="fp16").wkv_mode, "fp16")

    def test_rwkv4_and_noncanonical_fields_fail_closed(self):
        invalid = (
            {"attention_hidden_size": 64},
            {"rescale_every": 6},
            {"head_size": 128},
            {"intermediate_size": 320},
            {"wkv_state_dtype": "float32"},
            {"wkv_mode": "deltalog"},
            {"pad_token_id": 0},
            {"tie_word_embeddings": True},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                tiny_config(**override)


@require_torch
class RwkvStructureTest(unittest.TestCase):
    def test_public_types_and_standard_outputs(self):
        self.assertTrue(issubclass(RwkvPreTrainedModel, torch.nn.Module))
        self.assertTrue(issubclass(RwkvAttention, torch.nn.Module))
        self.assertTrue(issubclass(RwkvFeedForward, torch.nn.Module))
        self.assertTrue(issubclass(RwkvDecoderLayer, torch.nn.Module))
        self.assertTrue(issubclass(RwkvCache, object))
        self.assertNotIn("RwkvTrainingState", __import__("transformers").__dict__)
        self.assertNotIn("RwkvModelOutput", __import__("transformers").__dict__)
        self.assertIs(BaseModelOutputWithPast, type(BaseModelOutputWithPast()))
        self.assertIs(CausalLMOutputWithPast, type(CausalLMOutputWithPast()))

    def test_qwen_style_module_and_state_dict_names(self):
        model = RwkvForCausalLM(tiny_config())
        layer = model.model.layers[0]
        self.assertEqual(
            set(layer._modules),
            {"linear_attn", "mlp", "input_layernorm", "post_attention_layernorm"},
        )
        keys = set(model.state_dict())
        required = {
            "model.embed_tokens.weight",
            "model.embedding_norm.weight",
            "model.layers.0.linear_attn.r_proj.weight",
            "model.layers.0.mlp.key.weight",
            "model.layers.0.mlp.value.weight",
            "model.norm.weight",
            "lm_head.weight",
        }
        self.assertTrue(required.issubset(keys))
        self.assertFalse(any(key.startswith("model.layers.0.linear_attn.v0") for key in keys))
        self.assertFalse(any(key.startswith("model.layers.0.linear_attn.v1") for key in keys))
        self.assertFalse(any(key.startswith("model.layers.0.linear_attn.v2") for key in keys))
        self.assertIn("model.layers.1.linear_attn.v0", keys)
        self.assertFalse(any("time_mix" in key or "receptance" in key for key in keys))

    def test_generation_config_defaults_disable_rapid_sampling_penalties(self):
        generation_config = RwkvForCausalLM(tiny_config()).generation_config
        self.assertEqual(generation_config.presence_penalty, 0.0)
        self.assertEqual(generation_config.frequency_penalty, 0.0)
        self.assertEqual(generation_config.penalty_decay, 0.996)

    def test_initialization_matches_train_temp_formulas(self):
        torch.manual_seed(11)
        config = tiny_config()
        model = RwkvForCausalLM(config)
        attention = model.model.layers[1].linear_attn
        channels = config.hidden_size
        position = torch.arange(channels, dtype=torch.float32)
        ddd = position / channels
        ratio = 1.0 - 1 / config.num_hidden_layers
        expected_x_r = 1 - ddd.pow(0.2 * ratio)
        torch.testing.assert_close(attention.x_r.float(), expected_x_r)
        linear = position / (channels - 1) - 0.5
        torch.testing.assert_close(attention.v0.float(), 0.73 - linear * 0.4)
        self.assertTrue(torch.equal(attention.w1, torch.zeros_like(attention.w1)))
        self.assertTrue(torch.equal(attention.a1, torch.zeros_like(attention.a1)))
        self.assertTrue(torch.equal(attention.v1, torch.zeros_like(attention.v1)))
        self.assertTrue(torch.equal(attention.g1, torch.zeros_like(attention.g1)))
        self.assertTrue(torch.equal(attention.o_proj.weight, torch.zeros_like(attention.o_proj.weight)))
        self.assertTrue(
            torch.equal(
                model.model.layers[0].mlp.value.weight, torch.zeros_like(model.model.layers[0].mlp.value.weight)
            )
        )
        self.assertLessEqual(float(model.model.embed_tokens.weight.detach().abs().max()), 1e-4)

    def test_cpu_mask_and_cache_boundaries_fail_closed(self):
        model = RwkvModel(tiny_config()).half().eval()
        input_ids = torch.tensor([[1, 2]])
        with self.assertRaisesRegex(ValueError, "padding or ragged"):
            model(input_ids, attention_mask=torch.tensor([[1, 0]]))
        with self.assertRaisesRegex(TypeError, "only RwkvCache"):
            model(input_ids, past_key_values=DynamicCache())
        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model(input_ids)

    def test_provider_pin_and_requested_operator_validation(self):
        requirement = Requirement(deps["FlashRWKV2"])
        self.assertEqual(str(requirement.specifier), "==0.1.0a11")
        self.assertEqual(version("FlashRWKV2"), "0.1.0a11")
        module = load_flash_rwkv2(
            (
                "pretrain_tmix_wkv7_recurrent_bf16",
                "statetune_tmix_wkv7_recurrent_fp32io16",
                "infer_tmix_wkv_prepare_forward_varlen",
                "infer_sampling_six_parameter_forward_varlen",
            )
        )
        self.assertEqual(module.__version__, "0.1.0a11")
        with self.assertRaisesRegex(RuntimeError, "public operators"):
            load_flash_rwkv2("operator_that_does_not_exist")

    def test_plain_linear_spec_and_bias_failure(self):
        projection = torch.nn.Linear(8, 8, bias=False)
        weight, lora_a, lora_b, scale = flash_rwkv2_linear_spec(projection)
        self.assertIs(weight, projection.weight)
        self.assertIsNone(lora_a)
        self.assertIsNone(lora_b)
        self.assertEqual(scale, 1.0)
        with self.assertRaisesRegex(RuntimeError, "base bias"):
            flash_rwkv2_linear_spec(torch.nn.Linear(8, 8, bias=True))

    @require_peft
    def test_peft_linear_spec_active_disabled_merged_and_unsupported(self):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            RwkvForCausalLM(tiny_config()),
            LoraConfig(r=4, lora_alpha=8, target_modules=["r_proj"]),
        )
        projection = model.base_model.model.model.layers[0].linear_attn.r_proj
        weight, lora_a, lora_b, scale = flash_rwkv2_linear_spec(projection)
        self.assertIs(weight, projection.get_base_layer().weight)
        self.assertIs(lora_a, projection.lora_A["default"].weight)
        self.assertIs(lora_b, projection.lora_B["default"].weight)
        self.assertEqual(scale, 2.0)

        model.disable_adapter_layers()
        self.assertEqual(flash_rwkv2_linear_spec(projection)[1:], (None, None, 1.0))
        model.enable_adapter_layers()
        model.merge_adapter()
        self.assertEqual(flash_rwkv2_linear_spec(projection)[1:], (None, None, 1.0))
        model.unmerge_adapter()

        model.add_adapter("second", LoraConfig(r=4, lora_alpha=4, target_modules=["r_proj"]))
        projection.set_adapter(["default", "second"])
        with self.assertRaisesRegex(RuntimeError, "exactly one active"):
            flash_rwkv2_linear_spec(projection)

        dora_model = get_peft_model(
            RwkvForCausalLM(tiny_config()),
            LoraConfig(r=4, lora_alpha=8, use_dora=True, target_modules=["r_proj"]),
        )
        dora_projection = dora_model.base_model.model.model.layers[0].linear_attn.r_proj
        with self.assertRaisesRegex(RuntimeError, "vanilla LoRA"):
            flash_rwkv2_linear_spec(dora_projection)


@require_torch
class RwkvCacheTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        self.hidden = torch.zeros(2, 3, self.config.hidden_size, dtype=torch.bfloat16)

    def test_layout_length_and_linear_attention_layers(self):
        cache = RwkvCache(self.config)
        self.assertTrue(all(isinstance(layer, LinearAttentionLayer) for layer in cache.layers))
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

    def test_training_chunk_metadata_covers_arbitrary_lengths(self):
        cache = RwkvCache(self.config)
        for sequence_length in (1, 7, 15, 16, 17, 31):
            offsets, starts, ends = cache.training_metadata(2, sequence_length, self.hidden.device)
            self.assertEqual(offsets[0].item(), 0)
            self.assertEqual(offsets[-1].item(), len(starts))
            self.assertEqual(starts[0].item(), 0)
            self.assertEqual(ends[-1].item(), 2 * sequence_length)
            self.assertTrue(torch.all(ends > starts))
            self.assertTrue(torch.all(ends - starts <= 16))


@require_torch
class RwkvConverterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).parents[3] / "temp" / "rwkv_pth2st.py"
        spec = importlib.util.spec_from_file_location("rwkv_pth2st", path)
        cls.converter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.converter)

    @staticmethod
    def canonical_state_dict(config: RwkvConfig) -> dict[str, torch.Tensor]:
        model_state = RwkvForCausalLM(config).state_dict()
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
        for target_key, tensor in model_state.items():
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

    def test_inference_translation_and_layer_zero_drop(self):
        config = tiny_config(decay_low_rank_dim=17, a_low_rank_dim=19, v_low_rank_dim=23, gate_low_rank_dim=29)
        source = self.canonical_state_dict(config)
        inferred = self.converter.infer_config(source, context_length=16)
        self.assertEqual(
            (
                inferred.decay_low_rank_dim,
                inferred.a_low_rank_dim,
                inferred.v_low_rank_dim,
                inferred.gate_low_rank_dim,
            ),
            (17, 19, 23, 29),
        )
        plan, dropped = self.converter.translation_plan(source)
        self.assertEqual(dropped, self.converter.LAYER_ZERO_UNUSED)
        expected = self.converter.expected_state_dict(inferred)
        self.converter.validate_plan(source, plan, dropped, expected)
        self.assertEqual(set(plan), set(expected))
        self.assertEqual(plan["model.layers.0.mlp.key.weight"], "blocks.0.ffn.key.weight")

    def test_converter_rejects_compiled_and_cross_layer_rank_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "compiled.pth"
            torch.save({"_orig_mod.emb.weight": torch.empty(1)}, checkpoint)
            with self.assertRaisesRegex(ValueError, "_orig_mod"):
                self.converter.load_checkpoint(checkpoint)

        source = self.canonical_state_dict(tiny_config())
        source["blocks.1.att.w1"] = torch.empty(64, 31)
        source["blocks.1.att.w2"] = torch.empty(31, 64)
        with self.assertRaisesRegex(ValueError, "rank must agree"):
            self.converter.infer_config(source, context_length=16)

    def test_converter_rejects_noncanonical_broad_squeeze_shapes(self):
        config = tiny_config()
        source = self.canonical_state_dict(config)
        source["blocks.0.att.x_r"] = source["blocks.0.att.x_r"].reshape(1, config.hidden_size, 1)
        plan, dropped = self.converter.translation_plan(source)
        with self.assertRaisesRegex(ValueError, "mismatched"):
            self.converter.validate_plan(source, plan, dropped, self.converter.expected_state_dict(config))

    def test_generation_configs_define_open_fake_and_tools_profiles(self):
        expected = {
            "generation_config.json": (0.96, 0.76, 32, 1.0, 0.1, 0.988),
            "fake_think_generation_config.json": (1.0, 0.28, 32, 0.0, 0.0, 0.996),
            "tools_generation_config.json": (0.96, 0.76, 32, 0.0, 0.0, 0.996),
        }
        with tempfile.TemporaryDirectory() as directory:
            self.converter.save_generation_configs(tiny_config(), Path(directory))
            for config_file_name, values in expected.items():
                generation_config = GenerationConfig.from_pretrained(
                    directory,
                    config_file_name=config_file_name,
                    local_files_only=True,
                )
                self.assertTrue(generation_config.do_sample)
                self.assertEqual(
                    (
                        generation_config.temperature,
                        generation_config.top_p,
                        generation_config.top_k,
                        generation_config.presence_penalty,
                        generation_config.frequency_penalty,
                        generation_config.penalty_decay,
                    ),
                    values,
                )
                self.assertEqual(generation_config.stop_strings, self.converter.STOP_STRINGS)

    def test_shard_writer_consumes_source_and_writes_reloadable_values(self):
        from safetensors import safe_open

        config = tiny_config()
        source = self.canonical_state_dict(config)
        expected_embedding = source["emb.weight"].clone()
        storage = torch.cat([tensor.flatten() for tensor in source.values()])
        offset = 0
        for key, tensor in source.items():
            source[key] = storage[offset : offset + tensor.numel()].view_as(tensor)
            offset += tensor.numel()
        plan, dropped = self.converter.translation_plan(source)
        expected = self.converter.expected_state_dict(config)
        for source_key in dropped:
            source.pop(source_key)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.converter.write_shards(source, plan, expected, output, "20KB")
            index_path = output / "model.safetensors.index.json"
            self.assertTrue(index_path.is_file())
            index = json.loads(index_path.read_text())
            self.assertEqual(set(index["weight_map"]), set(expected))
            embedding_file = output / index["weight_map"]["model.embed_tokens.weight"]
            with safe_open(embedding_file, framework="pt") as shard:
                actual = shard.get_tensor("model.embed_tokens.weight")
            torch.testing.assert_close(actual, expected_embedding)
        self.assertFalse(source)


if __name__ == "__main__":
    unittest.main()
