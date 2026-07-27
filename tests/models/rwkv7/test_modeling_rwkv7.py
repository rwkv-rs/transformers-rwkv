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

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        Rwkv7Block,
        Rwkv7ChannelMix,
        Rwkv7Config,
        Rwkv7DeepEmbedding,
        Rwkv7ForCausalLM,
        Rwkv7Model,
        Rwkv7TimeMix,
    )
    from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import (
        convert_rwkv7_checkpoint_to_hf,
        convert_state_dict,
        infer_rwkv7_config,
    )


def get_config(**kwargs):
    defaults = {
        "vocab_size": 37,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "head_dim": 4,
        "attention_hidden_size": 16,
        "intermediate_size": 24,
        "decay_low_rank_dim": 4,
        "a_low_rank_dim": 4,
        "gate_low_rank_dim": 6,
        "value_low_rank_dim": 3,
    }
    return Rwkv7Config(**(defaults | kwargs))


def to_official_state_dict(model):
    state_dict = {}
    for name, tensor in model.state_dict().items():
        name = name.removeprefix("model.") if name != "head.weight" else name
        state_dict[name] = tensor.detach().clone()
    return state_dict


class Rwkv7ModelTester:
    def __init__(self, parent):
        self.parent = parent
        self.batch_size = 3
        self.seq_length = 5
        self.vocab_size = 37
        self.hidden_size = 16
        self.num_hidden_layers = 2
        self.num_labels = 3
        self.num_choices = 4
        self.is_training = True

    def get_config(self):
        return get_config()

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        return self.get_config(), {"input_ids": input_ids}


@require_torch
class Rwkv7ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (Rwkv7Model, Rwkv7ForCausalLM) if is_torch_available() else ()
    has_attentions = False
    test_missing_keys = False

    def setUp(self):
        self.model_tester = Rwkv7ModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=Rwkv7Config,
            n_embd=16,
            common_properties=["hidden_size", "num_hidden_layers"],
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def _check_caches_are_equal(self, cache1, cache2):
        self.assertEqual(cache1.seen_tokens, cache2.seen_tokens)
        for name in ("recurrent_state", "time_mix_state", "channel_mix_state"):
            for state1, state2 in zip(getattr(cache1, name), getattr(cache2, name)):
                torch.testing.assert_close(state1, state2)
        torch.testing.assert_close(cache1.v_first, cache2.v_first)

    def test_auto_classes(self):
        config_dict = get_config().to_dict()
        config_dict.pop("model_type")
        config = AutoConfig.for_model("rwkv7", **config_dict)
        self.assertIsInstance(config, Rwkv7Config)
        self.assertIsInstance(AutoModel.from_config(config), Rwkv7Model)
        self.assertIsInstance(AutoModelForCausalLM.from_config(config), Rwkv7ForCausalLM)

    def test_bo_names_and_replaceable_components(self):
        class TestTimeMix(Rwkv7TimeMix):
            pass

        class TestChannelMix(Rwkv7ChannelMix):
            pass

        class TestBlock(Rwkv7Block):
            time_mix_class = TestTimeMix
            channel_mix_class = TestChannelMix

        class TestModel(Rwkv7Model):
            block_class = TestBlock

        model = TestModel(get_config())
        self.assertIsInstance(model.blocks[0].att, TestTimeMix)
        self.assertIsInstance(model.blocks[0].ffn, TestChannelMix)
        keys = set(model.state_dict())
        self.assertIn("blocks.0.att.w1", keys)
        self.assertIn("blocks.0.att.receptance.weight", keys)
        self.assertNotIn("blocks.0.att.v1", keys)
        self.assertIn("blocks.1.att.v1", keys)
        self.assertFalse(any("lora" in key or "r_proj" in key for key in keys))

    def test_chunked_recurrent_cache_parity_and_precision(self):
        torch.manual_seed(0)
        input_ids = torch.randint(0, 37, (2, 6))
        for wkv_mode, dtype in (("fp32io16", torch.float32), ("fp16", torch.float16)):
            model = Rwkv7ForCausalLM(get_config(wkv_mode=wkv_mode)).to(dtype=dtype).eval()
            with torch.no_grad():
                whole = model(input_ids).logits
                prefix = model(input_ids[:, :4])
                suffix = model(input_ids[:, 4:], past_key_values=prefix.past_key_values)
            self.assertEqual(
                prefix.past_key_values.recurrent_state[0].dtype, torch.float32 if wkv_mode == "fp32io16" else dtype
            )
            self.assertTrue(
                torch.allclose(torch.cat((prefix.logits, suffix.logits), dim=1), whole, atol=1e-3, rtol=1e-3)
            )

    def test_masked_batch_matches_unpadded_sequence(self):
        torch.manual_seed(1)
        model = Rwkv7ForCausalLM(get_config()).eval()
        padded = torch.tensor([[0, 0, 3, 4, 5], [6, 7, 8, 9, 10]])
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        with torch.no_grad():
            batched = model(padded, attention_mask=mask).logits[0, -3:]
            unpadded = model(padded[0, -3:].unsqueeze(0)).logits[0]
        self.assertTrue(torch.allclose(batched, unpadded, atol=1e-6, rtol=1e-5))

    def test_cache_batch_operations(self):
        model = Rwkv7Model(get_config()).to(torch_device).eval()
        cache = model(torch.tensor([[1, 2], [3, 4]], device=torch_device)).past_key_values
        original = cache.recurrent_state[0].clone()
        cache.reorder_cache(torch.tensor([1, 0]))
        self.assertTrue(torch.equal(cache.recurrent_state[0], original.flip(0)))
        cache.batch_repeat_interleave(2)
        self.assertEqual(cache.batch_size, 4)
        cache.batch_select_indices(torch.tensor([3, 0]))
        self.assertEqual(cache.batch_size, 2)

    def test_deep_embedding_factorization_matches_merged_table(self):
        torch.manual_seed(2)
        vocab_size, hidden_size, deep_size = 11, 8, 3
        embedding_table = torch.randn(vocab_size, hidden_size)
        direct_embedding = torch.nn.Embedding(vocab_size, deep_size * deep_size)
        residual_projection = torch.nn.Linear(hidden_size, deep_size * deep_size, bias=False)
        input_ids = torch.tensor([[1, 4, 7], [3, 2, 9]])
        factorized = Rwkv7DeepEmbedding()(
            input_ids,
            embedding_table[input_ids],
            direct_embedding,
            residual_projection,
        )
        merged_table = direct_embedding.weight + embedding_table @ residual_projection.weight.T
        self.assertTrue(torch.allclose(factorized, merged_table[input_ids]))

    def test_deep_embedding_model_cache_and_resize(self):
        model = Rwkv7ForCausalLM(get_config(deep_embedding_size=2)).eval()
        input_ids = torch.randint(0, 37, (2, 5))
        with torch.no_grad():
            whole = model(input_ids).logits
            prefix = model(input_ids[:, :3])
            suffix = model(input_ids[:, 3:], past_key_values=prefix.past_key_values).logits
        self.assertTrue(torch.allclose(whole, torch.cat((prefix.logits, suffix), dim=1), atol=1e-6))

        model.resize_token_embeddings(41)
        self.assertEqual(model.model.emb.num_embeddings, 41)
        self.assertEqual(model.head.out_features, 41)
        self.assertTrue(all(block.ffn.s_emb.num_embeddings == 41 for block in model.model.blocks))

    def test_loss_backward(self):
        model = Rwkv7ForCausalLM(get_config())
        input_ids = torch.randint(0, 37, (2, 5))
        loss = model(input_ids, labels=input_ids, use_cache=False).loss
        loss.backward()
        self.assertIsNotNone(model.model.blocks[0].att.receptance.weight.grad)

    def test_official_checkpoint_conversion_roundtrip(self):
        torch.manual_seed(3)
        original = Rwkv7ForCausalLM(get_config()).eval()
        official = to_official_state_dict(original)
        official["blocks.0.att.r_k"] = official["blocks.0.att.r_k"].reshape(1, 1, 4, 4)
        inferred = infer_rwkv7_config(official)
        self.assertEqual(inferred.hidden_size, 16)
        self.assertEqual(inferred.num_hidden_layers, 2)
        self.assertEqual(inferred.head_dim, 4)
        expected_shapes = {name: tensor.shape for name, tensor in original.state_dict().items()}
        converted = convert_state_dict(official, expected_shapes)
        self.assertEqual(set(converted), set(original.state_dict()))

        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint = Path(tmp_dir) / "rwkv7.pth"
            output_dir = Path(tmp_dir) / "hf"
            torch.save(official, checkpoint)
            convert_rwkv7_checkpoint_to_hf(str(checkpoint), str(output_dir))
            restored = Rwkv7ForCausalLM.from_pretrained(output_dir).eval()
        for name, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(tensor, restored.state_dict()[name]), name)


if __name__ == "__main__":
    unittest.main()
