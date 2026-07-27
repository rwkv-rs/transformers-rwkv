# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the RWKV-7 model.

The interesting failure modes of a recurrent model are not shape errors, they are
state errors: a carried state that is stale, mixed between batch rows, or simply
not equivalent to having read the whole prefix. Those are what these tests pin.
"""

import unittest

from transformers import Rwkv7Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device


if is_torch_available():
    import torch

    from transformers import Rwkv7ForCausalLM, Rwkv7Model


def _tiny_config(**kwargs):
    defaults = {
        "vocab_size": 256,
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "head_dim": 8,
        "num_heads": 4,
        "decay_low_rank_dim": 8,
        "a_low_rank_dim": 8,
        "v_low_rank_dim": 8,
        "gate_low_rank_dim": 8,
        "intermediate_size": 64,
    }
    defaults.update(kwargs)
    return Rwkv7Config(**defaults)


def _randomised(model):
    """Break the zero-init so a dropped term cannot pass by cancelling out."""
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model.eval()


@require_torch
class Rwkv7ModelTest(unittest.TestCase):
    def test_config_rejects_inconsistent_heads(self):
        with self.assertRaises(ValueError):
            _tiny_config(num_heads=3)
        with self.assertRaises(ValueError):
            _tiny_config(hidden_size=30, head_dim=8, num_heads=4)

    def test_forward_shapes_and_state(self):
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 7), device=torch_device)

        out = model(input_ids=input_ids, use_cache=True)
        self.assertEqual(out.last_hidden_state.shape, (2, 7, config.hidden_size))
        att_shift, ffn_shift, wkv = out.state
        self.assertEqual(att_shift.shape, (config.num_hidden_layers, 2, config.hidden_size))
        self.assertEqual(ffn_shift.shape, (config.num_hidden_layers, 2, config.hidden_size))
        self.assertEqual(
            wkv.shape,
            (config.num_hidden_layers, 2, config.num_heads, config.head_dim, config.head_dim),
        )

    def test_prefill_matches_incremental_decoding(self):
        """Reading a prefix in one go must equal reading it one token at a time.

        This is the property that makes the carried state a real cache rather than
        an approximation; it fails loudly if the token shift, the WKV recurrence or
        the v_first hand-off is wired wrong.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 9), device=torch_device)

        with torch.no_grad():
            full = model(input_ids=input_ids, use_cache=True).logits
            state, steps = None, []
            for position in range(input_ids.shape[1]):
                out = model(input_ids=input_ids[:, position : position + 1], state=state, use_cache=True)
                state, _ = out.state, steps.append(out.logits)
            incremental = torch.cat(steps, dim=1)

        torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-4)

    def test_batch_rows_are_independent(self):
        """A row's output must not depend on what shares its batch.

        Duplicating a sequence inside a batch is the sharpest version of this: the
        copies see identical inputs, so any difference between them is leaked state
        rather than numerics.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        single = torch.randint(0, config.vocab_size, (1, 6), device=torch_device)
        other = torch.randint(0, config.vocab_size, (1, 6), device=torch_device)

        with torch.no_grad():
            alone = model(input_ids=single).logits
            batched = model(input_ids=torch.cat([single, other, single], dim=0)).logits

        torch.testing.assert_close(batched[0], batched[2], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched[0], alone[0], rtol=1e-4, atol=1e-4)

    def test_layer_zero_value_residual_lora_is_unused(self):
        """Layer 0 produces `v_first`; it must never mix towards it.

        Reference checkpoints still ship `v0/v1/v2` on layer 0, and the `fla`
        layout drops them — so the two conversions disagree on exactly these
        tensors. Outputs must not.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)

        with torch.no_grad():
            before = model(input_ids=input_ids).logits
            for name in ("v0", "v1", "v2"):
                getattr(model.rwkv7.blocks[0].att, name).normal_(10.0, 1.0)
            after = model(input_ids=input_ids).logits

        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_deep_embed_hook(self):
        """The RWKV-8 DeepEmbed hook modulates the channel-mix, and is off by default."""
        config = _tiny_config(use_deep_embed=True)
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)
        layers, batch, seq = config.num_hidden_layers, 1, 4

        with torch.no_grad():
            plain = model(input_ids=input_ids).logits
            # a table of ones must be a no-op in the "1x" (output-modulating) form
            ones = torch.ones(layers, batch, seq, config.hidden_size, device=torch_device)
            neutral = model(input_ids=input_ids, deep_embeds=ones).logits
            # zeros silence the channel-mix entirely, which is an unambiguous
            # signal that the hook is actually applied (a scale near 1 is not: the
            # channel-mix is a small residual, so halving it barely moves logits)
            silenced = model(input_ids=input_ids, deep_embeds=torch.zeros_like(ones)).logits

        torch.testing.assert_close(plain, neutral, rtol=1e-5, atol=1e-5)
        self.assertGreater((plain - silenced).abs().max().item(), 0.0)

    def test_deep_embed_four_x_variant(self):
        """A table as wide as the channel-mix inner width modulates its input instead."""
        config = _tiny_config(use_deep_embed=True, deep_embed_size=64)
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)

        with torch.no_grad():
            plain = model(input_ids=input_ids).logits
            ones = torch.ones(config.num_hidden_layers, 1, 4, config.intermediate_size, device=torch_device)
            neutral = model(input_ids=input_ids, deep_embeds=ones).logits

        torch.testing.assert_close(plain, neutral, rtol=1e-5, atol=1e-5)

    def test_generate_is_deterministic_under_greedy(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)

        with torch.no_grad():
            first = model.generate(input_ids, max_new_tokens=8, do_sample=False)
            second = model.generate(input_ids, max_new_tokens=8, do_sample=False)

        self.assertTrue(torch.equal(first, second))

    def test_gradients_reach_every_parameter(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)
        model(input_ids=input_ids, labels=input_ids, use_cache=False).loss.backward()

        # layer 0's value-residual LoRA is intentionally dead (see the test above)
        dead = {f"rwkv7.blocks.0.att.{n}" for n in ("v0", "v1", "v2")}
        missing = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and name not in dead and (param.grad is None or param.grad.abs().sum() == 0)
        ]
        self.assertEqual(missing, [], f"parameters received no gradient: {missing}")

    def test_chunked_matches_sequential_recurrence(self):
        """The chunk-parallel prefill must reproduce the sequential recurrence.

        They are two forms of the same step, so any disagreement is a bug in the
        chunked derivation rather than a tolerance question. Several chunk sizes
        are checked, including ones that do not divide the sequence length.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_chunked, rwkv7_recurrent

        torch.manual_seed(0)
        batch, seq_len, heads, dim = 2, 37, 3, 8
        shape = (batch, seq_len, heads, dim)
        r = torch.randn(shape, device=torch_device)
        # w_log is a log-decay in (-e^-0.5, 0), as the decay LoRA produces
        w_log = -0.6065306597126334 * torch.sigmoid(torch.randn(shape, device=torch_device))
        k = torch.randn(shape, device=torch_device)
        v = torch.randn(shape, device=torch_device)
        kk = torch.nn.functional.normalize(torch.randn(shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device=torch_device))
        state = torch.randn(batch, heads, dim, dim, device=torch_device)

        reference, reference_state = rwkv7_recurrent(r, w_log, k, v, kk, a, state.clone())
        for chunk_size in (1, 4, 16, 64):
            out, out_state = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=chunk_size)
            torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-4, msg=f"chunk={chunk_size}")
            torch.testing.assert_close(
                out_state, reference_state, rtol=2e-4, atol=2e-4, msg=f"state chunk={chunk_size}"
            )
