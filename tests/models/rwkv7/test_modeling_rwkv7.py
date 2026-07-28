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
from transformers.testing_utils import require_torch, require_torch_gpu, torch_device


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

    def test_left_padded_row_matches_the_same_row_alone(self):
        """Padding must leave the recurrent state exactly where it found it.

        `generate` left-pads a batch, so the pads run through the recurrence
        *before* the real tokens. There is no per-position attention mask for an
        all-recurrent model to hide behind: unless the padding is neutralised, a
        short row starts from a state the pads have already moved. The second
        assertion keeps this test honest -- it fails if the padding happened to be
        harmless here, which would make the first assertion prove nothing. It is
        written as a ratio because an absolute floor only holds for one fixture:
        every weight here is drawn at 0.05, so the whole model works in small
        numbers and the untreated damage lands around 1e-6, not 1e-2.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        real = torch.randint(1, config.vocab_size, (1, 4), device=torch_device)
        pads = torch.zeros(1, 3, dtype=real.dtype, device=torch_device)
        padded = torch.cat([pads, real], dim=1)
        mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            alone = model(input_ids=real).logits[0, -1]
            masked = model(input_ids=padded, attention_mask=mask).logits[0, -1]
            ignored = model(input_ids=padded).logits[0, -1]

        treated = (masked - alone).abs().max().item()
        untreated = (ignored - alone).abs().max().item()
        torch.testing.assert_close(masked, alone, rtol=1e-5, atol=1e-5)
        self.assertGreater(untreated, 100 * max(treated, 1e-9))

    def test_packed_batch_matches_each_sequence_run_alone(self):
        """A varlen (packed) row must decode exactly like its sequences separately.

        Two things reach across a boundary and both have to be cut: the recurrent
        state, which restarts per segment, and the token shift, which otherwise
        hands a segment's first token the *previous* sequence's last hidden state.
        The control at the end is what makes this test mean something -- it fails
        if packing happened to be harmless, which would leave the first assertion
        proving nothing.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        lengths = [4, 3, 5]
        segments = [torch.randint(1, config.vocab_size, (1, n), device=torch_device) for n in lengths]
        packed = torch.cat(segments, dim=1)
        cu_seq_lens = torch.tensor([0, 4, 7, 12], device=torch_device)

        with torch.no_grad():
            together = model(input_ids=packed, cu_seq_lens=cu_seq_lens).logits[0]
            naive = model(input_ids=packed).logits[0]

        start, treated = 0, 0.0
        for segment, n in zip(segments, lengths):
            with torch.no_grad():
                alone = model(input_ids=segment).logits[0]
            torch.testing.assert_close(together[start : start + n], alone, rtol=1e-5, atol=1e-5)
            treated = max(treated, (together[start : start + n] - alone).abs().max().item())
            start += n

        # A ratio, not an absolute floor. Every weight in this fixture is drawn at
        # 0.05, so the untreated damage sits around 1e-6 and a bare 1e-6 threshold
        # would be a coin flip on the very quantity it is meant to bound. What has to
        # hold is that honouring the boundaries removes most of the damage, at
        # whatever scale the fixture happens to put it.
        untreated = (naive - together).abs().max().item()
        self.assertGreater(untreated, 100 * max(treated, 1e-9))

    def test_packed_batch_rejects_a_malformed_boundary_list(self):
        """A wrong `cu_seq_lens` must raise, not split the recurrence somewhere else.

        Nothing downstream can notice a bad boundary list: the model still returns
        fluent logits, just computed from states that restarted in the wrong places.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)

        for bad in ([1, 3, 6], [0, 3, 5], [0, 3, 8]):
            with self.assertRaises(ValueError):
                model(input_ids=ids, cu_seq_lens=torch.tensor(bad, device=torch_device))

        with self.assertRaises(ValueError):  # packing describes one row, not a batch
            model(
                input_ids=ids.repeat(2, 1),
                cu_seq_lens=torch.tensor([0, 3, 6], device=torch_device),
            )

    def test_wkv_implementation_is_selectable(self):
        """The recurrence is looked up by name, so a kernel can be dropped in.

        Registering a wrapper and selecting it must route every call through it and
        leave the output identical -- that is what makes the registry a seam rather
        than a second code path.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import RWKV7_WKV_FUNCTIONS, rwkv7_eager

        calls = []

        def counting_wkv(*args, **kwargs):
            calls.append(1)
            return rwkv7_eager(*args, **kwargs)

        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)
        with torch.no_grad():
            expected = model(input_ids=ids).logits

        RWKV7_WKV_FUNCTIONS["counting"] = counting_wkv
        try:
            model.config.wkv_implementation = "counting"
            with torch.no_grad():
                got = model(input_ids=ids).logits
        finally:
            del RWKV7_WKV_FUNCTIONS["counting"]

        self.assertEqual(len(calls), config.num_hidden_layers)
        self.assertTrue(torch.equal(expected, got))

    @require_torch_gpu
    def test_heavily_padded_fp16_batch_is_finite(self):
        """Padding in fp16, at a realistic pad fraction, must not produce NaN.

        This is the configuration every real user is in and the one the other padding
        test is not: it runs fp32 on a tiny model, where `F.normalize`'s 1e-12 epsilon
        is representable. In fp16 it is below the smallest subnormal, so normalising
        the zero vector a blanked pad position produces divides by a true zero and the
        whole row comes out NaN. Caught on real prompts, not here — hence the pad
        fraction below is 90%, like a short prompt batched with a long one, rather
        than the three pad tokens the other test uses.
        """
        config = _tiny_config(sparse_channel_mix=False)
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device).half()
        short = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)
        long = torch.randint(1, config.vocab_size, (1, 60), device=torch_device)
        width = 60
        padded = torch.cat(
            [torch.cat([torch.zeros(1, width - 6, dtype=short.dtype, device=torch_device), short], dim=1), long],
            dim=0,
        )
        mask = torch.zeros(2, width, dtype=torch.long, device=torch_device)
        mask[0, -6:] = 1
        mask[1, :] = 1

        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=mask).logits
            alone = model(input_ids=short).logits[0, -1]

        self.assertTrue(torch.isfinite(out).all(), "padded fp16 batch produced non-finite logits")
        self.assertEqual(int(out[0, -1].argmax()), int(alone.argmax()))

    def test_mask_without_padding_is_a_no_op(self):
        """An all-ones mask must not perturb the unpadded path at all.

        `generate` passes a mask on every call, padded or not, so the common case
        has to come out bit-identical -- not merely close.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)

        with torch.no_grad():
            without = model(input_ids=ids).logits
            with_ones = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

        self.assertTrue(torch.equal(without, with_ones))

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

    def test_reorder_cache_moves_the_state_onto_the_beams(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        state = model.rwkv7.allocate_state(4, device=torch_device)
        # Give every batch row a value only that row has, so a permutation that
        # lands on the wrong source cannot pass by looking plausible.
        for entry in state:
            for row in range(4):
                entry[:, row] = row + 1
        addresses = [entry.data_ptr() for entry in state]

        # 2 appears twice: a reorder that aliased its source into its destination
        # would corrupt the second copy.
        beams = [2, 2, 0, 3]
        reordered = model._reorder_cache(state, torch.tensor(beams, device=torch_device))

        for entry in reordered:
            for row, source in enumerate(beams):
                self.assertTrue(torch.all(entry[:, row] == source + 1))
        # In place, so the addresses `allocate_state` pinned are still pinned.
        self.assertEqual([entry.data_ptr() for entry in reordered], addresses)

    def test_beam_search_score_survives_an_independent_rescore(self):
        """Beam search accumulated a score through the state; rescore it fresh.

        "It ran without raising" is not evidence. Beam search keeps running when
        the state follows the wrong beam -- it just searches with a history that
        belongs to some other candidate, and reports a cumulative score built from
        that wrong history. So the check is a consistency one: `sequences_scores`
        is accumulated step by step through the recurrent state, and rescoring the
        very same tokens in one full forward has to reproduce it. Only a state that
        tracked the surviving beams makes those two numbers agree.

        Two things here were measured rather than assumed, both after a first
        version of this test passed with the reorder stubbed out entirely:

        * The obvious property -- beam scoring at least as well as greedy -- is not
          a theorem and is not used. Greedy's path can be pruned out of the top-k
          part way through and still be the better sequence at the end, which is
          exactly what this model does (beam -41.81 vs greedy -39.45, with a
          correct reorder). A control run on Llama in the same checkout happens to
          come out the other way, +3.37, which is what made the false premise look
          confirmed.
        * The sharper init is load-bearing. At `_randomised`'s 0.05 the tiny
          model's next-token distribution is nearly uniform, every beam is
          interchangeable, and a stubbed reorder still agrees to 1.5e-05 -- no
          discrimination at all. At 0.5 over 16 tokens and 4 beams: 3.8e-06 with
          the reorder, 15.9 without it. The 1e-3 tolerance sits between those.
        """
        config = _tiny_config()
        torch.manual_seed(0)
        model = Rwkv7ForCausalLM(config)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(0.0, 0.5)
        model = model.eval().to(torch_device)
        prompt = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)

        with torch.no_grad():
            beamed = model.generate(
                prompt,
                max_new_tokens=16,
                num_beams=4,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequence = beamed.sequences
            logits = model(sequence).logits.float()

        generated = sequence.shape[1] - prompt.shape[1]
        rescored = (
            torch.log_softmax(logits[:, :-1], dim=-1)
            .gather(-1, sequence[:, 1:, None])[:, prompt.shape[1] - 1 :]
            .sum()
            .item()
        )
        # `sequences_scores` is the length-normalised beam score; length_penalty
        # defaults to 1.0, so multiplying by the generated length undoes it.
        accumulated = beamed.sequences_scores.item() * generated

        self.assertAlmostEqual(accumulated, rescored, delta=1e-3)

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

    def test_wkv_state_dtype_is_configurable(self):
        """The state precision is independent of the activation dtype.

        The recurrence is unrolled over the whole sequence, so the state is where
        precision actually matters; the config exposes it separately for that
        reason. fp32 must stay closer to a fp64 rollout than fp16 does.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_recurrent

        torch.manual_seed(0)
        shape = (1, 24, 2, 8)
        r = torch.randn(shape, device=torch_device)
        w_log = -0.6065306597126334 * torch.sigmoid(torch.randn(shape, device=torch_device))
        k = torch.randn(shape, device=torch_device)
        v = torch.randn(shape, device=torch_device)
        kk = torch.nn.functional.normalize(torch.randn(shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device=torch_device))
        state = torch.zeros(1, 2, 8, 8, device=torch_device)

        truth, _ = rwkv7_recurrent(
            r.double(),
            w_log.double(),
            k.double(),
            v.double(),
            kk.double(),
            a.double(),
            state.double(),
            compute_dtype=torch.float64,
        )
        fp32, s32 = rwkv7_recurrent(r, w_log, k, v, kk, a, state.clone(), compute_dtype=torch.float32)
        fp16, s16 = rwkv7_recurrent(
            r.half(),
            w_log.half(),
            k.half(),
            v.half(),
            kk.half(),
            a.half(),
            state.clone().half(),
            compute_dtype=torch.float16,
        )
        self.assertEqual(s32.dtype, torch.float32)
        self.assertEqual(s16.dtype, torch.float16)
        err32 = (fp32.double() - truth).abs().max().item()
        err16 = (fp16.double() - truth).abs().max().item()
        self.assertLess(err32, err16, f"fp32 state ({err32:.3e}) should beat fp16 ({err16:.3e})")

        for bad in ("float8", "int8"):
            with self.assertRaises(ValueError):
                _tiny_config(wkv_state_dtype=bad)

    def test_sparse_scratch_does_not_shadow_the_projection(self):
        """The cache attributes must not collide with the methods around them.

        Runs on CPU deliberately: the sparse projection itself is CUDA-only, so
        every test that exercises it is skipped on a machine without a GPU -- and a
        scratch buffer named after the method that uses it then passes the whole
        suite and fails on the first real decode with `'Tensor' object is not
        callable`. Which is exactly what happened.
        """
        config = _tiny_config(sparse_channel_mix=True)
        model = Rwkv7ForCausalLM(config)
        ffn = model.rwkv7.blocks[0].ffn
        ffn.build_sparse_cache()

        self.assertTrue(callable(ffn._sparse_value))
        self.assertTrue(callable(ffn._project))
        for name in ("_compact_index", "_compact_value", "_compact_counter", "_accumulator"):
            self.assertIsInstance(getattr(ffn, name), torch.Tensor, name)

    @require_torch_gpu
    def test_allocate_state_warms_the_sparse_caches(self):
        """Everything the compiled decode needs must exist before compiling.

        The caches are built lazily by the projection, which is fine eagerly and a
        2.7x loss under compilation: allocated inside the traced region they cannot
        be pinned, so CUDA graphs are declined for mutating inputs -- a slowdown that
        announces itself only as a warning line. `allocate_state` is the one call the
        compiled path already requires, so it warms them too.
        """
        config = _tiny_config(sparse_channel_mix=True)
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device).half()

        for block in model.rwkv7.blocks:
            self.assertIsNone(block.ffn._value_t)

        model.rwkv7.allocate_state(1)

        for block in model.rwkv7.blocks:
            self.assertIsNotNone(block.ffn._value_t)
            self.assertIsNotNone(block.ffn._accumulator)
            # transposed: the sparse read needs one contiguous row per input channel
            self.assertEqual(tuple(block.ffn._value_t.shape), tuple(block.ffn.value.weight.shape[::-1]))

    def test_allocate_state_leaves_the_dense_path_cold(self):
        """With the sparse path off, nothing extra should be materialised -- the
        transposed copy costs about 30% more weight memory."""
        config = _tiny_config(sparse_channel_mix=False)
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)

        model.rwkv7.allocate_state(1)

        for block in model.rwkv7.blocks:
            self.assertIsNone(block.ffn._value_t)

    @require_torch_gpu
    def test_sparse_channel_mix_matches_dense(self):
        """Skipping zero channels must be exact, not merely close.

        `relu(x)**2` produces exact zeros and `0 * w == 0`, so the sparse path is
        the same sum with terms that contribute nothing left out. Only the
        accumulation dtype differs from the dense reference, so fp32 inputs let
        this be compared tightly.

        GPU-gated, and the gate is the point. `sparse_channel_mix_value` falls back
        to `F.linear(activation, weight_t.t())` when the Triton op is unavailable --
        character for character the `dense` expression below -- so without a GPU this
        compared an expression to itself and passed while measuring nothing. The
        kernel is also only registered as a side effect of some other test having
        called `build_sparse_cache`, so it was order-dependent as well; that is fixed
        by arming it here explicitly.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import _ensure_sparse_op, sparse_channel_mix_value

        self.assertTrue(_ensure_sparse_op(), "sparse op unavailable; this test would compare dense to dense")

        torch.manual_seed(0)
        inter, hidden = 512, 128
        weight_t = torch.randn(inter, hidden, device=torch_device) * 0.02
        act = torch.randn(inter, device=torch_device)
        act[torch.rand(inter, device=torch_device) > 0.07] = 0.0  # exact zeros, as relu^2 gives
        accumulator = torch.zeros(hidden, device=torch_device, dtype=torch.float32)
        index = torch.zeros(inter, device=torch_device, dtype=torch.int32)
        value = torch.zeros(inter, device=torch_device, dtype=torch.float32)
        counter = torch.zeros(1, device=torch_device, dtype=torch.int32)
        scratch = (accumulator, index, value, counter)

        dense = torch.nn.functional.linear(act, weight_t.t())
        sparse = sparse_channel_mix_value(act, weight_t, *scratch)
        torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-5)

        # an all-zero activation must give exactly zero, and leave every scratch
        # buffer clean -- the projection reuses them across layers and steps, so a
        # finalize pass that forgot to re-zero one would only show up on the call
        # after the one being checked
        zeros = sparse_channel_mix_value(torch.zeros_like(act), weight_t, *scratch)
        self.assertEqual(zeros.abs().max().item(), 0.0)
        self.assertEqual(accumulator.abs().max().item(), 0.0)
        self.assertEqual(counter.abs().max().item(), 0)

        # and a repeat of the first call must reproduce it exactly, which it cannot
        # if anything above was left dirty
        again = sparse_channel_mix_value(act, weight_t, *scratch)
        torch.testing.assert_close(again, sparse, rtol=0, atol=0)

    @require_torch_gpu
    def test_sparse_path_agrees_with_dense_including_deep_embed(self):
        """The sparse channel-mix must be a pure optimisation, DeepEmbed included.

        Both DeepEmbed widths are checked because they attach on opposite sides of
        the value projection: the `intermediate_size` table scales its input and the
        `hidden_size` one its output. An implementation that applies only the second
        passes a plain sparse-vs-dense check and is still wrong.
        """
        for width_name, width in (("1x", 32), ("4x", 64)):
            config = _tiny_config(use_deep_embed=True, deep_embed_size=width, sparse_channel_mix=True)
            model = _randomised(Rwkv7ForCausalLM(config)).to("cuda")
            input_ids = torch.randint(0, config.vocab_size, (1, 1), device="cuda")
            deep = torch.full((config.num_hidden_layers, 1, 1, width), 0.5, device="cuda")

            with torch.no_grad():
                model.config.sparse_channel_mix = False
                dense = model(input_ids=input_ids, deep_embeds=deep).logits
                model.config.sparse_channel_mix = True
                sparse = model(input_ids=input_ids, deep_embeds=deep).logits

            torch.testing.assert_close(sparse, dense, rtol=1e-3, atol=1e-3, msg=f"DeepEmbed {width_name}")

    @require_torch_gpu
    def test_sparse_cache_follows_weight_updates(self):
        """The transposed copy the sparse path keeps must not outlive its weight.

        A cache that silently survives a weight change turns every later forward
        into a wrong answer with no error, which is the worst failure mode
        available here.
        """
        config = _tiny_config(sparse_channel_mix=True)
        model = _randomised(Rwkv7ForCausalLM(config)).to("cuda")
        input_ids = torch.randint(0, config.vocab_size, (1, 1), device="cuda")

        with torch.no_grad():
            model(input_ids=input_ids)  # populates the cache
            model.rwkv7.blocks[0].ffn.value.weight.mul_(3.0)
            after_update = model(input_ids=input_ids).logits
            model.config.sparse_channel_mix = False
            dense = model(input_ids=input_ids).logits

        torch.testing.assert_close(after_update, dense, rtol=1e-3, atol=1e-3)
