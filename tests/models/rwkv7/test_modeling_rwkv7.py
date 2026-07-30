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

import copy
import math
import tempfile
import unittest

from transformers import Rwkv7Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch
    from safetensors.torch import load_file
    from torch.nn import functional as F

    from transformers import Rwkv7ForCausalLM, Rwkv7Model
    from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import (
        convert_rwkv7_checkpoint_to_hf_format,
        convert_state_dict,
        infer_rwkv7_config,
    )
    from transformers.models.rwkv7.modeling_rwkv7 import register_rwkv7_wkv_backend

    def rwkv7_reference_backend(
        receptance,
        raw_decay,
        key,
        value,
        negative_key,
        scaled_key,
        state,
        attention_mask=None,
        cu_seq_lens=None,
        head_size=64,
    ):
        batch_size, sequence_length, hidden_size = receptance.shape
        num_heads = hidden_size // head_size
        if cu_seq_lens is not None:
            boundaries = cu_seq_lens.tolist()
            outputs = []
            final_state = None
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                segment_inputs = [
                    tensor[:, start:end] for tensor in (receptance, raw_decay, key, value, negative_key, scaled_key)
                ]
                segment_state = state.new_zeros(1, num_heads, head_size, head_size)
                segment_output, final_state = rwkv7_reference_backend(
                    *segment_inputs,
                    state=segment_state,
                    head_size=head_size,
                )
                outputs.append(segment_output)
            return torch.cat(outputs, dim=1), final_state

        matrix_state = state
        head_shape = (batch_size, sequence_length, num_heads, head_size)
        receptance, raw_decay, key, value, negative_key, scaled_key = (
            tensor.view(head_shape) for tensor in (receptance, raw_decay, key, value, negative_key, scaled_key)
        )
        decay = torch.exp(-math.exp(-0.5) * torch.sigmoid(raw_decay))
        outputs = []
        for token_index in range(sequence_length):
            candidate_state = (
                matrix_state * decay[:, token_index].unsqueeze(-2)
                + matrix_state @ negative_key[:, token_index].unsqueeze(-1) @ scaled_key[:, token_index].unsqueeze(-2)
                + value[:, token_index].unsqueeze(-1) @ key[:, token_index].unsqueeze(-2)
            )
            if attention_mask is not None:
                update_mask = attention_mask[:, token_index].bool().view(batch_size, 1, 1, 1)
                matrix_state = torch.where(update_mask, candidate_state, matrix_state)
            else:
                matrix_state = candidate_state
            token_output = (matrix_state @ receptance[:, token_index].unsqueeze(-1)).squeeze(-1)
            if attention_mask is not None:
                token_output = token_output * attention_mask[:, token_index].view(batch_size, 1, 1)
            outputs.append(token_output)
        output = torch.stack(outputs, dim=1).reshape(batch_size, sequence_length, hidden_size)
        return output, matrix_state

    register_rwkv7_wkv_backend("reference", rwkv7_reference_backend)


def get_config(embedding_layer_norm_fused=False, wkv_backend="reference"):
    return Rwkv7Config(
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        head_size=8,
        context_length=32,
        embedding_layer_norm_fused=embedding_layer_norm_fused,
        wkv_backend=wkv_backend,
        rescale_every=0,
    )


class Rwkv7ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        intermediate_size=64,
        head_size=8,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.head_size = head_size
        self.num_attention_heads = hidden_size // head_size
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.is_training = True

    def get_config(self):
        return Rwkv7Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            head_size=self.head_size,
            context_length=32,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            wkv_backend="reference",
            rescale_every=0,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        return config

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones_like(input_ids)
        return config, input_ids, attention_mask

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Rwkv7Model(config).to(torch_device).eval()
        result = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (self.batch_size, self.seq_length, self.hidden_size),
        )
        self.parent.assertEqual(len(result.hidden_states), config.num_hidden_layers + 1)

    def create_and_check_causal_lm(self, config, input_ids, attention_mask):
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()
        result = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        self.parent.assertEqual(result.loss.shape, ())
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Rwkv7ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Rwkv7Model, Rwkv7ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Rwkv7Model, "text-generation": Rwkv7ForCausalLM} if is_torch_available() else {}
    )
    has_attentions = False
    _is_stateful = True

    def setUp(self):
        self.model_tester = Rwkv7ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Rwkv7Config,
            hidden_size=32,
            head_size=8,
            common_properties=["hidden_size", "num_hidden_layers"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        self.model_tester.create_and_check_model(*self.model_tester.prepare_config_and_inputs())

    def test_causal_lm(self):
        self.model_tester.create_and_check_causal_lm(*self.model_tester.prepare_config_and_inputs())

    def test_config_defaults_and_validation(self):
        config = Rwkv7Config(hidden_size=64, head_size=8)
        self.assertEqual(config.intermediate_size, 256)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertFalse(config.embedding_layer_norm_fused)
        with self.assertRaises(ValueError):
            Rwkv7Config(hidden_size=30, head_size=8)
        with self.assertRaises(ValueError):
            Rwkv7Config(hidden_size=32, head_size=8, intermediate_size=0)

        with self.assertRaises(ValueError):
            Rwkv7Config(hidden_size=32, head_size=8, wkv_backend="")
        with self.assertRaises(ValueError):
            Rwkv7Config(hidden_size=32, head_size=8, wkv_state_dtype="float16")

    def test_model_package_does_not_export_internal_kernels(self):
        import transformers.models.rwkv7 as rwkv7_module

        self.assertFalse(hasattr(rwkv7_module, "rwkv7_recurrent"))

    def test_block_zero_has_no_value_residual_parameters(self):
        model = Rwkv7Model(get_config())
        self.assertFalse(any(hasattr(model.blocks[0].att, name) for name in ("v0", "v1", "v2")))
        self.assertTrue(all(hasattr(model.blocks[1].att, name) for name in ("v0", "v1", "v2")))
        fused_model = Rwkv7Model(get_config(embedding_layer_norm_fused=True))
        self.assertFalse(hasattr(fused_model.blocks[0], "ln0"))

    def test_forward_and_state_shapes(self):
        config = get_config()
        model = Rwkv7Model(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)
        outputs = model(input_ids, use_cache=True)
        self.assertEqual(outputs.last_hidden_state.shape, (2, 5, config.hidden_size))
        self.assertEqual(outputs.state[0].shape, (config.num_hidden_layers, 2, config.hidden_size))
        self.assertEqual(
            outputs.state[1].shape,
            (config.num_hidden_layers, 2, config.num_attention_heads, config.head_size, config.head_size),
        )
        self.assertEqual(outputs.state[2].shape, (config.num_hidden_layers, 2, config.hidden_size))
        invalid_state = [outputs.state[0].half(), outputs.state[1], outputs.state[2]]
        with self.assertRaisesRegex(ValueError, "dtype"):
            model(input_ids, state=invalid_state)

    def test_save_load_roundtrip_preserves_all_parameters(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.uniform_(-0.25, 0.25)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = Rwkv7ForCausalLM.from_pretrained(directory)

        for name, parameter in model.state_dict().items():
            self.assertTrue(torch.equal(parameter, reloaded.state_dict()[name]), name)

    def test_prefill_decode_state_equivalence(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 7), device=torch_device)
        with torch.no_grad():
            full_logits = model(input_ids).logits
            prefill = model(input_ids[:, :4], use_cache=True)
            continuation = model(input_ids[:, 4:], state=prefill.state, use_cache=True)
            continuation_without_return = model(input_ids[:, 4:], state=prefill.state, use_cache=False)
        self.assertTrue(torch.allclose(continuation.logits, full_logits[:, 4:], atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(continuation_without_return.logits, full_logits[:, 4:], atol=2e-5, rtol=2e-5))
        self.assertIsNone(continuation_without_return.state)

    def test_attention_mask_preserves_recurrent_state(self):
        config = get_config()
        model = Rwkv7Model(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], device=torch_device)
        with torch.no_grad():
            batched = model(input_ids, attention_mask=attention_mask, use_cache=True)
            first = model(input_ids[:1, :3], use_cache=True)
            second = model(input_ids[1:], use_cache=True)
        for state_index in range(3):
            self.assertTrue(torch.allclose(batched.state[state_index][:, :1], first.state[state_index], atol=2e-5))
            self.assertTrue(torch.allclose(batched.state[state_index][:, 1:], second.state[state_index], atol=2e-5))

    def test_left_padding_and_all_ones_mask(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()
        short = torch.randint(1, config.vocab_size, (1, 3), device=torch_device)
        long = torch.randint(1, config.vocab_size, (1, 5), device=torch_device)
        padded = torch.cat((F.pad(short, (2, 0)), long), dim=0)
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]], device=torch_device)
        with torch.no_grad():
            batched = model(padded, attention_mask=mask, use_cache=True)
            short_output = model(short, use_cache=True)
            long_output = model(long, use_cache=True)
            unmasked = model(long)
            all_ones = model(long, attention_mask=torch.ones_like(long))
        self.assertTrue(torch.allclose(batched.logits[0, -1], short_output.logits[0, -1], atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.allclose(batched.logits[1, -1], long_output.logits[0, -1], atol=2e-5, rtol=2e-5))
        for state_index in range(3):
            self.assertTrue(
                torch.allclose(batched.state[state_index][:, :1], short_output.state[state_index], atol=2e-5)
            )
            self.assertTrue(
                torch.allclose(batched.state[state_index][:, 1:], long_output.state[state_index], atol=2e-5)
            )
        self.assertTrue(torch.equal(unmasked.logits, all_ones.logits))

    def test_embedding_layer_norm_fusion(self):
        regular = Rwkv7Model(get_config(embedding_layer_norm_fused=False)).to(torch_device).eval()
        fused = Rwkv7Model(get_config(embedding_layer_norm_fused=True)).to(torch_device).eval()
        fused_state_dict = {
            name: copy.deepcopy(tensor)
            for name, tensor in regular.state_dict().items()
            if not name.startswith("blocks.0.ln0.")
        }
        fused.load_state_dict(fused_state_dict)
        with torch.no_grad():
            embedding = fused.embeddings.weight
            fused.embeddings.weight.copy_(
                F.layer_norm(
                    embedding.float(),
                    (fused.config.hidden_size,),
                    regular.blocks[0].ln0.weight.float(),
                    regular.blocks[0].ln0.bias.float(),
                ).to(embedding.dtype)
            )
        input_ids = torch.randint(0, regular.config.vocab_size, (2, 5), device=torch_device)
        with torch.no_grad():
            regular_output = regular(input_ids).last_hidden_state
            fused_output = fused(input_ids).last_hidden_state
        self.assertTrue(torch.allclose(regular_output, fused_output, atol=2e-5, rtol=2e-5))

    def test_conversion_detaches_checkpoint_storage_views(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config)
        raw_state_dict = {}
        for name, tensor in model.state_dict().items():
            if name == "head.weight":
                raw_name = name
            elif name == "rwkv7.embeddings.weight":
                raw_name = "emb.weight"
            else:
                raw_name = name.removeprefix("rwkv7.")
            raw_state_dict[raw_name] = tensor

        norm_weight = raw_state_dict["blocks.0.ln1.weight"]
        oversized_storage = torch.empty(norm_weight.numel() + 8, dtype=norm_weight.dtype)
        oversized_storage[: norm_weight.numel()].copy_(norm_weight)
        raw_state_dict["blocks.0.ln1.weight"] = oversized_storage[: norm_weight.numel()]

        converted = convert_state_dict(raw_state_dict, config, fuse_embedding_layer_norm=False)
        converted_norm_weight = converted["rwkv7.blocks.0.ln1.weight"]
        self.assertEqual(
            converted_norm_weight.untyped_storage().nbytes(),
            converted_norm_weight.numel() * converted_norm_weight.element_size(),
        )

    def test_conversion_discards_redundant_block_zero_value_parameters(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config)
        raw_state_dict = {}
        for name, tensor in model.state_dict().items():
            if name == "head.weight":
                raw_name = name
            elif name == "rwkv7.embeddings.weight":
                raw_name = "emb.weight"
            else:
                raw_name = name.removeprefix("rwkv7.")
            raw_state_dict[raw_name] = tensor

        for name in ("v0", "v1", "v2"):
            raw_state_dict[f"blocks.0.att.{name}"] = raw_state_dict[f"blocks.1.att.{name}"].clone()

        converted = convert_state_dict(raw_state_dict, config, fuse_embedding_layer_norm=False)
        self.assertFalse(any(f"rwkv7.blocks.0.att.{name}" in converted for name in ("v0", "v1", "v2")))

    def test_conversion_rejects_incompatible_low_rank_shapes(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config)
        raw_state_dict = {}
        for name, tensor in model.state_dict().items():
            if name == "head.weight":
                raw_name = name
            elif name == "rwkv7.embeddings.weight":
                raw_name = "emb.weight"
            else:
                raw_name = name.removeprefix("rwkv7.")
            raw_state_dict[raw_name] = tensor
        raw_state_dict["blocks.0.att.w1"] = torch.empty(config.hidden_size, 31)

        with self.assertRaisesRegex(ValueError, "low-rank tensors"):
            infer_rwkv7_config(raw_state_dict)

    def test_conversion_saves_requested_dtype_and_metadata(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config)
        raw_state_dict = {}
        for name, tensor in model.state_dict().items():
            if name == "head.weight":
                raw_name = name
            elif name == "rwkv7.embeddings.weight":
                raw_name = "emb.weight"
            else:
                raw_name = name.removeprefix("rwkv7.")
            raw_state_dict[raw_name] = tensor

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = f"{temporary_directory}/rwkv7-ctx32.pth"
            output_directory = f"{temporary_directory}/converted"
            torch.save(raw_state_dict, checkpoint_path)
            convert_rwkv7_checkpoint_to_hf_format(
                output_dir=output_directory,
                checkpoint_path=checkpoint_path,
                fuse_embedding_layer_norm=False,
                dtype="float16",
            )

            converted_config = Rwkv7Config.from_pretrained(output_directory)
            converted_state_dict = load_file(f"{output_directory}/model.safetensors")
            converted_model = Rwkv7ForCausalLM.from_pretrained(output_directory)

        self.assertEqual(converted_config.architectures, ["Rwkv7ForCausalLM"])
        self.assertEqual(converted_config.dtype, torch.float16)
        self.assertEqual({tensor.dtype for tensor in converted_state_dict.values()}, {torch.float16})
        for name, parameter in converted_model.named_parameters():
            expected_dtype = torch.float32 if name.endswith("att.w0") else torch.float16
            self.assertEqual(parameter.dtype, expected_dtype)

    def test_backward(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config).to(torch_device).train()
        input_ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)
        loss = model(input_ids, labels=input_ids).loss
        loss.backward()
        parameters_without_grad = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None
        }
        self.assertEqual(parameters_without_grad, set())

    def test_tuple_output_order(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 4), device=torch_device)
        outputs = model(input_ids, use_cache=False, output_hidden_states=True, return_dict=False)
        self.assertEqual(outputs[0].shape, (2, 4, config.vocab_size))
        self.assertEqual(len(outputs[1]), config.num_hidden_layers + 1)

    def test_generation_with_beam_state_reordering(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()
        input_ids = torch.randint(1, config.vocab_size, (2, 4), device=torch_device)
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                max_new_tokens=2,
                num_beams=2,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
            )
        self.assertEqual(generated.shape, (2, 6))

    def test_generation_continuation_drops_attention_mask(self):
        model = Rwkv7ForCausalLM(get_config()).to(torch_device).eval()
        input_ids = torch.randint(1, model.config.vocab_size, (1, 4), device=torch_device)
        with torch.no_grad():
            state = model(input_ids[:, :3], use_cache=True).state
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            state=state,
            attention_mask=torch.ones_like(input_ids),
        )
        self.assertEqual(model_inputs["input_ids"].shape, (1, 1))
        self.assertIsNone(model_inputs["attention_mask"])

    def test_packed_model_matches_individual_sequences(self):
        config = get_config()
        model = Rwkv7ForCausalLM(config).to(torch_device).eval()
        first = torch.randint(1, config.vocab_size, (1, 4), device=torch_device)
        second = torch.randint(1, config.vocab_size, (1, 7), device=torch_device)
        packed = torch.cat((first, second), dim=1)
        cu_seq_lens = torch.tensor([0, first.shape[1], packed.shape[1]], device=torch_device)
        with torch.no_grad():
            packed_output = model(packed, cu_seq_lens=cu_seq_lens, use_cache=True)
            first_output = model(first, use_cache=True)
            second_output = model(second, use_cache=True)
        expected_logits = torch.cat((first_output.logits, second_output.logits), dim=1)
        self.assertTrue(torch.allclose(packed_output.logits, expected_logits, atol=2e-5, rtol=2e-5))
        for packed_state, expected_state in zip(packed_output.state, second_output.state):
            self.assertTrue(torch.allclose(packed_state, expected_state, atol=2e-5, rtol=2e-5))
        with self.assertRaisesRegex(ValueError, "existing recurrent state"):
            model(packed, cu_seq_lens=cu_seq_lens, state=second_output.state)


if __name__ == "__main__":
    unittest.main()
