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

from transformers import AutoConfig, Rwkv7Config

from ...test_configuration_common import ConfigTester


class Rwkv7ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config_tester = ConfigTester(
            self,
            config_class=Rwkv7Config,
            vocab_size=99,
            context_length=32,
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            head_size=8,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_derived_dimensions(self):
        config = Rwkv7Config(hidden_size=64, head_size=8)

        self.assertEqual(config.intermediate_size, 256)
        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.max_position_embeddings, config.context_length)

    def test_auto_config(self):
        config = AutoConfig.for_model("rwkv7", hidden_size=64, head_size=8)

        self.assertIsInstance(config, Rwkv7Config)
        self.assertEqual(config.num_attention_heads, 8)

    def test_auto_config_save_load_round_trip(self):
        config = Rwkv7Config(
            vocab_size=128,
            context_length=256,
            hidden_size=64,
            num_hidden_layers=3,
            head_size=8,
            wkv_backend="reference",
        )

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            reloaded = AutoConfig.from_pretrained(directory)

        self.assertIsInstance(reloaded, Rwkv7Config)
        self.assertEqual(reloaded.to_dict(), config.to_dict())
