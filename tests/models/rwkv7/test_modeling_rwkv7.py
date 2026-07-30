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

import shutil
import tempfile
import unittest

from transformers import Rwkv7Config, is_torch_available
from transformers.testing_utils import require_peft, require_torch, require_torch_gpu, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Rwkv7Cache, Rwkv7ForCausalLM, Rwkv7Model


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


def _sharpened(model):
    """Same, but at a scale where a broken recurrent state actually shows.

    0.05 keeps the whole model in small numbers, which is what
    `test_left_padded_row_matches_the_same_row_alone` needs and reasons about. It is
    the wrong scale for anything that checks the state *carry*: at 0.05, deleting
    the WKV write-back entirely moves the logits by 6.8e-06, which sails under a
    1e-4 tolerance -- the test would be green with the recurrence dead. At 0.5 the
    same sabotage moves them by 6.8e-01, five orders of magnitude clear of the noise
    floor of 3.3e-06. Measured, not guessed; see the sabotage matrix in the commit.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.5)
    return model.eval()


class Rwkv7ModelTester:
    """The inputs the shared model/generation tests build everything else from.

    Deliberately lean. The RWKV v4 tester in this repo still carries `token_type_ids`,
    `mc_token_ids` and `num_choices` from the GPT-2 tester it was copied from, none of
    which this architecture has ever taken.
    """

    def __init__(self, parent, batch_size=3, seq_length=7, is_training=True):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = 32
        self.num_hidden_layers = 2  # the common suite caps this; 2 still exercises the layer-0 v_first hand-off
        self.vocab_size = 99
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0

    def get_config(self):
        return _tiny_config(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers
        )

    def prepare_config_and_inputs(self):
        # ids start at 1: 0 is this tokenizer's pad/eos, and a prompt that opens with
        # it makes several generation tests ambiguous about where the output begins.
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        return self.get_config(), input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Rwkv7ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Rwkv7Model, Rwkv7ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Rwkv7Model, "text-generation": Rwkv7ForCausalLM} if is_torch_available() else {}
    )
    test_pruning = False
    test_head_masking = False
    test_missing_keys = False
    # There is no attention here to report: the mixing is a recurrence, not a score
    # matrix, so there is no [batch, heads, q, k] tensor for the shared suite to
    # inspect. This is the flag those tests read, rather than a row of skips.
    has_attentions = False

    # Layer 0 *produces* `v_first` rather than mixing towards it, so its
    # value-residual LoRA (`att.v0/v1/v2`) is never read and never receives a
    # gradient. The shared check requires every `requires_grad` parameter to come back
    # with one, and it re-enables `requires_grad` on everything first, so freezing
    # those three does not satisfy it either.
    #
    # They are kept rather than not created, deliberately: a native `.pth` carries
    # them at layer 0, and dropping them would mean this port can read that file but
    # not write it back. `test_layer_zero_value_residual_lora_is_unused` pins the
    # property this skip rests on, so it is checked somewhere rather than asserted
    # here in prose.
    #
    # This comment used to end by saying gradient checkpointing was "supported and
    # exercised by `test_gradient_checkpointing_enable_disable` and by `test_training`".
    # Neither runs a checkpointed backward -- the first only toggles flags and asserts
    # the attributes moved, the second never enables checkpointing -- so skipping the
    # three tests below left that path with no coverage at all, and it was in fact
    # broken. It is covered now by `test_checkpointed_backward_matches_the_plain_one`,
    # written here rather than inherited so it can sidestep the v-LoRA property these
    # three trip over.
    _V_LORA_SKIP = (
        "layer 0's value-residual LoRA is structurally unreachable, so it has no "
        "gradient; see test_layer_zero_value_residual_lora_is_unused"
    )

    @unittest.skip(_V_LORA_SKIP)
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(_V_LORA_SKIP)
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    @unittest.skip(_V_LORA_SKIP)
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    def setUp(self):
        self.model_tester = Rwkv7ModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Rwkv7Config, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def test_config_rejects_inconsistent_heads(self):
        with self.assertRaises(ValueError):
            _tiny_config(num_heads=3)
        with self.assertRaises(ValueError):
            _tiny_config(hidden_size=30, head_dim=8, num_heads=4)

    def test_group_norm_epsilon_follows_head_dim_not_head_count(self):
        """The reference's `64e-5` is `head_dim * 1e-5`, and the two look alike.

        `num_heads * norm_eps` gives the same number as `head_dim * norm_eps` when
        a model has exactly as many heads as channels per head -- at head_dim 64,
        that is hidden_size 4096 and nothing else. Every accuracy measurement this
        port has been through ran on the 7.2B, which is precisely that width, so
        the wrong multiplier reproduced the reference to four decimals and the
        conversion check compared two routes that shared the mistake.

        The config here is deliberately not square (4 heads of 8), so the two
        candidates differ and the assertion has something to say.
        """
        config = _tiny_config()
        self.assertNotEqual(config.num_heads, config.head_dim)  # else this proves nothing
        model = Rwkv7Model(config)

        for block in model.blocks:
            self.assertAlmostEqual(block.att.ln_x.eps, config.norm_eps * config.head_dim, places=12)

    def test_forward_shapes_and_state(self):
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 7), device=torch_device)

        out = model(input_ids=input_ids, use_cache=True)
        self.assertEqual(out.last_hidden_state.shape, (2, 7, config.hidden_size))

        cache = out.state
        self.assertIsInstance(cache, Rwkv7Cache)
        self.assertEqual(len(cache), config.num_hidden_layers)
        # No sequence axis anywhere in here, at any layer -- that is the architecture's
        # whole claim, and the shapes are where it is either true or not.
        for layer_idx in range(config.num_hidden_layers):
            att_shift, ffn_shift, wkv = cache.read(layer_idx)
            self.assertEqual(att_shift.shape, (2, config.hidden_size))
            self.assertEqual(ffn_shift.shape, (2, config.hidden_size))
            self.assertEqual(wkv.shape, (2, config.num_heads, config.head_dim, config.head_dim))

    def test_state_size_does_not_grow_with_context(self):
        """The O(1) claim, measured in bytes rather than asserted in a docstring."""
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)

        def cache_bytes(length):
            ids = torch.randint(0, config.vocab_size, (1, length), device=torch_device)
            with torch.no_grad():
                cache = model(input_ids=ids, use_cache=True).state
            return sum(
                state.numel() * state.element_size()
                for layer in cache.layers
                for state in layer.recurrent_states.values()
                if state is not None
            )

        self.assertEqual(cache_bytes(4), cache_bytes(256))
        self.assertEqual(cache_bytes(4), cache_bytes(1024))

    def test_prefill_matches_incremental_decoding(self):
        """Reading a prefix in one go must equal reading it one token at a time.

        This is the property that makes the carried state a real cache rather than
        an approximation; it fails loudly if the token shift, the WKV recurrence or
        the v_first hand-off is wired wrong -- but only at a weight scale where the
        state contributes more than the tolerance, hence `_sharpened`.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 9), device=torch_device)

        with torch.no_grad():
            full = model(input_ids=input_ids, use_cache=True).logits
            state, steps = None, []
            for position in range(input_ids.shape[1]):
                out = model(input_ids=input_ids[:, position : position + 1], state=state, use_cache=True)
                state, _ = out.state, steps.append(out.logits)
            incremental = torch.cat(steps, dim=1)

        torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-4)

    def test_logits_to_keep_shortens_the_head(self):
        """A prefill needs one row of logits, not `seq_len` of them.

        The head is the widest matrix in the model, so running it over the whole prompt
        is the largest avoidable cost in a prefill. This argument used to be swallowed
        by `**kwargs`: passing it changed nothing, silently, and `generate` declined to
        pass it at all because `_supports_logits_to_keep()` reads the signature.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 12), device=torch_device)

        self.assertTrue(model._supports_logits_to_keep())
        with torch.no_grad():
            everything = model(input_ids=input_ids).logits
            for keep in (1, 4):
                shortened = model(input_ids=input_ids, logits_to_keep=keep).logits
                self.assertEqual(shortened.shape, (2, keep, config.vocab_size))
                # The kept rows must be the LAST ones, and identical to computing them
                # all: a slice taken from the wrong end would have the right shape.
                torch.testing.assert_close(shortened, everything[:, -keep:, :])

    def test_intermediate_size_follows_hidden_size_when_it_is_not_given(self):
        """`hidden_ratio` has to actually derive something.

        `intermediate_size` used to default to 3072, which is exactly `768 * 4.0` --
        right for the default `hidden_size` and silently wrong for every other one. A
        config written as `Rwkv7Config(hidden_size=4096, num_heads=64)` came back with a
        channel-mix four times narrower than the architecture it names, and the model
        built from it loads no real checkpoint. The default config is unchanged, which
        is why this needs a second width to see at all.
        """
        self.assertEqual(Rwkv7Config().intermediate_size, 3072)
        self.assertEqual(Rwkv7Config(hidden_size=4096, num_heads=64).intermediate_size, 16384)
        self.assertEqual(Rwkv7Config(hidden_size=2048, num_heads=32).intermediate_size, 8192)
        # An explicit value still wins, and still round-trips through config.json.
        explicit = Rwkv7Config(hidden_size=4096, num_heads=64, intermediate_size=4096)
        self.assertEqual(explicit.intermediate_size, 4096)
        self.assertEqual(Rwkv7Config.from_dict(explicit.to_dict()).intermediate_size, 4096)

    def test_conversion_rejects_a_config_whose_shapes_disagree(self):
        """Matching key names is not the same as matching the model.

        The converter compared key sets only. A hand-written `config.json` that gets a
        width wrong has all the right names, so it converted with zero reported
        mismatches and produced a model that loads and generates noise. The skeleton it
        checks names against carries the shapes too.
        """
        import json
        import tempfile

        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import convert

        config = _tiny_config()
        source = Rwkv7ForCausalLM(config)
        # `_convert_native` is a prefix rename, so stripping it produces exactly the
        # native `.pth` layout that conversion expects.
        native = {key.removeprefix("rwkv7."): value for key, value in source.state_dict().items()}

        with tempfile.TemporaryDirectory() as work:
            checkpoint = f"{work}/model.pth"
            torch.save(native, checkpoint)

            wrong = config.to_dict()
            wrong["intermediate_size"] = config.intermediate_size * 2  # same names, different widths
            wrong_path = f"{work}/wrong.json"
            with open(wrong_path, "w") as handle:
                json.dump(wrong, handle)

            with self.assertRaisesRegex(RuntimeError, "do not match the shapes this config implies"):
                convert(checkpoint, "native", wrong_path, f"{work}/out-wrong")

            # Control: the same checkpoint with a config that agrees must convert.
            right_path = f"{work}/right.json"
            with open(right_path, "w") as handle:
                json.dump(config.to_dict(), handle)
            convert(checkpoint, "native", right_path, f"{work}/out-right")

    def test_a_reloaded_checkpoint_keeps_every_weight(self):
        """Saving and reloading must not lose a parameter to re-initialisation.

        `_init_weights` has to reach this model's twenty-one raw `nn.Parameter`s, which
        the inherited one does not. Written the obvious way -- `parameter.data.zero_()`
        -- it also destroys them on load: the framework skips re-initialising anything
        already there by checking an `_is_hf_initialized` flag that `initialization`'s
        helpers set and a raw in-place write does not, so `_init_weights` runs after the
        checkpoint is in and zeroes what it just loaded.

        That shipped for exactly as long as it took to run a real checkpoint through it.
        A converted 0.1B came back with 249 of its 402 tensors zeroed and generated
        "civil civil civil" where the reference runtime generated "Paris, France". None
        of the tests here noticed, because they all build models rather than load them.
        """
        import tempfile

        config = _tiny_config()
        original = _sharpened(Rwkv7ForCausalLM(config))
        with tempfile.TemporaryDirectory() as work:
            original.save_pretrained(work)
            reloaded = Rwkv7ForCausalLM.from_pretrained(work)

        before = dict(original.named_parameters())
        after = dict(reloaded.named_parameters())
        self.assertEqual(set(before), set(after))
        lost = [name for name, p in after.items() if p.abs().max() == 0 and before[name].abs().max() != 0]
        self.assertEqual(lost, [], f"{len(lost)} parameters came back zeroed, e.g. {lost[:5]}")
        for name in before:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0, msg=name)

    @require_peft
    def test_peft_lora_attaches_and_actually_learns(self):
        """A LoRA-wrapped model has to train, and no module may bypass its adapter.

        The bounty's criterion 3 asks for the common transformers-based PEFT flow to
        work. Two properties, each of which has a silent failure mode this test exists
        to catch. The loss on an overfit batch must actually fall, because a wiring
        break under wrapping tends to produce a model that runs and returns a loss
        while learning nothing. And every attached adapter must receive a gradient,
        because a forward that reads `module.weight` directly instead of calling the
        module silently routes around its LoRA -- the channel-mix projection here is
        exactly the kind of place that could regress to that, and did during the
        sparse-path work.
        """
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config))
        wrapped = get_peft_model(
            model, LoraConfig(r=8, lora_alpha=16, target_modules=["receptance", "key", "value", "output"])
        )
        trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}
        self.assertTrue(trainable, "no trainable parameters after wrapping")
        self.assertTrue(all("lora" in name for name in trainable), "base weights not frozen")

        wrapped.train()
        input_ids = torch.randint(1, config.vocab_size, (2, 12))
        optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad), lr=1e-2)

        losses = []
        for step in range(24):
            loss = wrapped(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            if step == 0:
                # Every adapter, not just some: an unfired adapter means its module
                # was bypassed, and the loss can still fall through the others. The
                # check is on lora_B, and that is not arbitrary: LoRA initialises B to
                # zero, so dL/dA passes through B and is identically zero on the first
                # step -- the first version of this test asserted lora_A here and
                # failed on all 18 adapters of a perfectly healthy model. B's gradient
                # is nonzero from step 0 exactly when the module's output reaches the
                # loss, which is the property under test.
                dead = [
                    name
                    for name, p in wrapped.named_parameters()
                    if p.requires_grad and "lora_B" in name and (p.grad is None or p.grad.abs().max() == 0)
                ]
                self.assertEqual(dead, [], f"{len(dead)} adapters received no gradient")
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        self.assertLess(
            losses[-1],
            0.6 * losses[0],
            f"overfitting one batch for 24 steps only moved the loss {losses[0]:.3f} -> {losses[-1]:.3f}",
        )

    def test_checkpointed_backward_matches_the_plain_one(self):
        """Gradient checkpointing must not change a gradient.

        The three inherited tests for this are skipped over layer 0's unused v-LoRA,
        and the two tests this file's skip comment pointed at instead turned out to
        run no checkpointed backward at all -- one toggles flags, the other never
        enables checkpointing. With nothing exercising it, the path was broken:
        `GradientCheckpointingLayer` disarms a live cache only when it arrives as a
        keyword it recognises, this model passes its state positionally, so the
        backward replay re-read a state the forward had already advanced. Reentrant
        checkpointing returned wrong gradients for 99 of 102 parameters with no error;
        the non-reentrant default raised.

        Comparing against the unpatched backward rather than checking for absence is
        the point: the reentrant failure produced gradients, they were just wrong.
        """
        config = _tiny_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=torch_device)

        def gradients(checkpointing, reentrant=True):
            # Seeded per build: `_sharpened` draws fresh weights, so without this the
            # comparison is between two different models and reports a difference
            # whatever checkpointing does.
            torch.manual_seed(0)
            model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).train()
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        plain = gradients(checkpointing=False)
        for reentrant in (True, False):
            checkpointed = gradients(checkpointing=True, reentrant=reentrant)
            self.assertEqual(set(plain), set(checkpointed), f"use_reentrant={reentrant}")
            for name in plain:
                torch.testing.assert_close(
                    plain[name],
                    checkpointed[name],
                    rtol=1e-3,
                    atol=1e-5,
                    msg=f"{name} differs under checkpointing (use_reentrant={reentrant})",
                )

    def test_the_last_real_token_is_found_by_index_not_by_float_arithmetic(self):
        """Finding the last unmasked position must not depend on the model's dtype.

        The search was `(mask * arange).argmax()`, evaluated in the activation dtype
        because the mask is cast to it. bfloat16 carries eight mantissa bits, so past
        position 256 the products stop being distinct, `argmax` returns the first of a
        tie, and the state handed back comes from several tokens short of the end --
        1022 for 1023 at length 1024, 4088 for 4095 at 4096. An all-ones mask is
        enough, which is what `generate` sends, so this was never confined to padded
        batches.

        Asserted against the mask-free path rather than against a hardcoded index: a
        full-length mask asks for exactly what no mask asks for, so the two must agree
        whatever the dtype.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        seq_len = 1024
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=torch_device)
        ones = torch.ones(1, seq_len, dtype=torch.long, device=torch_device)

        for dtype in (torch.float32, torch.bfloat16):
            typed = model.to(dtype)
            with torch.no_grad():
                masked = typed(input_ids=input_ids, attention_mask=ones, use_cache=True).state
                unmasked = typed(input_ids=input_ids, use_cache=True).state

            for layer in range(config.num_hidden_layers):
                for slot in (Rwkv7Cache.WKV, Rwkv7Cache.ATT_SHIFT, Rwkv7Cache.FFN_SHIFT):
                    torch.testing.assert_close(
                        masked.layers[layer].recurrent_states[slot].float(),
                        unmasked.layers[layer].recurrent_states[slot].float(),
                        rtol=0,
                        atol=0,
                        msg=f"{dtype} layer {layer} slot {slot}: an all-ones mask moved the state",
                    )

    @require_torch_gpu
    def test_fused_kernel_reads_a_strided_state_correctly(self):
        """The fused step must index the state by its strides, not by assumption.

        Its offset arithmetic hardcoded unit stride on the state's last axis and passed
        only three of the four strides, so a caller handing over anything that is not
        contiguous had the tile read transposed. Nothing raised: the shape is right, so
        neither Triton nor the caller could tell, and the answer came back 82% wrong on
        the output and 98% wrong on the carried state.

        The batch check is here for the same reason. The launch grid comes from `r`, so
        a state that disagrees with it is read at offsets belonging to a different
        sequence and returns a plausible answer for the wrong one.
        """
        from transformers.models.rwkv7.fused_wkv import fused_wkv_one
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_eager

        torch.manual_seed(0)
        batch, heads, width = 2, 4, 64
        shape = (batch, 1, heads, width)
        vector = lambda: torch.randn(*shape, device=torch_device) * 0.2  # noqa: E731
        r, k, v = vector(), vector(), vector()
        kk = torch.nn.functional.normalize(torch.randn(*shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(*shape, device=torch_device))
        w_log = -0.6065306597126334 * torch.sigmoid(torch.randn(*shape, device=torch_device))
        base = torch.randn(batch, heads, width, width, device=torch_device) * 0.1

        # A transpose of the two square axes: same shape, same elements, stride(3) != 1.
        strided = base.clone().transpose(2, 3)
        self.assertFalse(strided.is_contiguous(), "the fixture stopped being strided")

        for name, state in (("contiguous", base.clone()), ("strided", strided)):
            expected_out, expected_state = rwkv7_eager(r, w_log, k, v, kk, a, state.clone())
            got_state = state.clone()
            got_out = fused_wkv_one(r, w_log, k, v, kk, a, got_state)
            torch.testing.assert_close(got_out, expected_out, rtol=1e-4, atol=1e-4, msg=name)
            torch.testing.assert_close(got_state, expected_state, rtol=1e-4, atol=1e-4, msg=name)

        with self.assertRaisesRegex(ValueError, "but the vectors imply"):
            fused_wkv_one(r, w_log, k, v, kk, a, torch.zeros(batch + 1, heads, width, width, device=torch_device))

    def test_compiled_prefill_matches_eager_at_batch_and_length(self):
        """`torch.compile` must survive a shape with both a batch and a length.

        This exists because the one compile failure this port has actually had was
        found by a benchmark, not by a test: at 16x16 the WKV output came back with a
        layout `view` could not reinterpret, and every single-token and every
        `batch=1` shape had missed it. `fullgraph=True` is the point of the test as
        much as the numerics are -- the performance story here rests on CUDA graphs,
        and a graph break loses them while still returning correct logits, so a test
        that only compared outputs would stay green through exactly the regression
        worth catching.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=torch_device)

        torch._dynamo.reset()
        with torch.no_grad():
            eager = model(input_ids=input_ids).logits
            compiled = torch.compile(model, fullgraph=True)(input_ids=input_ids).logits

        torch.testing.assert_close(eager, compiled, rtol=1e-4, atol=1e-4)

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

        # First three fail on the endpoints; the last two have correct endpoints and
        # fail only on ordering. [0, 4, 2, 6] is the one that used to get through:
        # the backwards pair was skipped, so six tokens in came back as eight.
        for bad in ([1, 3, 6], [0, 3, 5], [0, 3, 8], [0, 4, 2, 6], [0, 3, 3, 6]):
            with self.assertRaises(ValueError):
                model(input_ids=ids, cu_seq_lens=torch.tensor(bad, device=torch_device))

        with self.assertRaises(ValueError):  # packing describes one row, not a batch
            model(
                input_ids=ids.repeat(2, 1),
                cu_seq_lens=torch.tensor([0, 3, 6], device=torch_device),
            )

    def test_packed_batch_ignores_a_carried_state_and_says_so(self):
        """Packing means "these are new sequences", so a carried state is not history.

        The state argument is still read for its shape and dtype, so a caller may
        hand in a pre-allocated cache; what it must not do is quietly resume from
        it, since there is no segment a previous row's state would belong to. This
        is a contract worth pinning rather than an accident: the same call with a
        state carrying real history has to give the same logits as one starting
        cold, and the docstring promises exactly that.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)
        bounds = torch.tensor([0, 2, 6], device=torch_device)

        with torch.no_grad():
            warmed = model(
                input_ids=torch.randint(1, config.vocab_size, (1, 5), device=torch_device),
                use_cache=True,
            ).state
            self.assertGreater(max(s.abs().max().item() for s in self._every_state(warmed)), 0.0)

            cold = model(input_ids=ids, cu_seq_lens=bounds).logits
            carried = model(input_ids=ids, cu_seq_lens=bounds, state=warmed).logits

        torch.testing.assert_close(carried, cold, rtol=1e-5, atol=1e-5)

    def test_loading_weights_drops_the_sparse_cache(self):
        """A warm transposed copy must not survive the weights it was made from.

        The fingerprint catches a version bump, and `load_state_dict` happens to
        produce one on current torch -- but that is how it copies today, not a
        promise, and the failure it guards against is silent: the projection keeps
        using a transpose of weights the model no longer has. Runs on CPU because
        it checks the bookkeeping, not the kernel.
        """
        config = _tiny_config(sparse_channel_mix=True)
        model = Rwkv7ForCausalLM(config)
        ffn = model.rwkv7.blocks[0].ffn
        ffn._value_fingerprint = ("pretend a cache was built",)

        model.load_state_dict(model.state_dict())

        self.assertIsNone(ffn._value_fingerprint)

    def test_invalidate_sparse_cache_forces_a_rebuild(self):
        config = _tiny_config(sparse_channel_mix=True)
        model = Rwkv7ForCausalLM(config)
        ffn = model.rwkv7.blocks[0].ffn

        with torch.no_grad():
            ffn.build_sparse_cache()
            # `.data.copy_` is the mutation the fingerprint is documented as unable to
            # see: it does not step the version counter and it reuses the storage, so
            # the cached transpose stays stale and nothing detects it.
            ffn.value.weight.data.copy_(torch.randn_like(ffn.value.weight))
            ffn.build_sparse_cache()
            stale = ffn._value_t.clone()
            self.assertFalse(
                torch.allclose(stale, ffn.value.weight.t()),
                "the edit was already visible without invalidate, so this proves nothing",
            )

            ffn.invalidate_sparse_cache()
            self.assertIsNone(ffn._value_fingerprint)
            ffn.build_sparse_cache()

        torch.testing.assert_close(ffn._value_t, ffn.value.weight.t().contiguous())

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

    def test_heavily_padded_fp16_batch_is_finite(self):
        """Padding in fp16, at a realistic pad fraction, must not produce NaN.

        This is the configuration every real user is in and the one the other padding
        test is not: it runs fp32 on a tiny model, where `F.normalize`'s 1e-12 epsilon
        is representable. In fp16 it is below the smallest subnormal, so normalising
        the zero vector a blanked pad position produces divides by a true zero and the
        whole row comes out NaN. Caught on real prompts, not here — hence the pad
        fraction below is 90%, like a short prompt batched with a long one, rather
        than the three pad tokens the other test uses.

        Not gated on a GPU. It used to be, and the guard it protects is a dtype
        question rather than a device one: fp16 runs on CPU perfectly well, the epsilon
        is unrepresentable there too, and removing the guard makes these logits
        non-finite on this machine. Behind `@require_torch_gpu` the only test for a
        real user-facing bug never ran on the CI that gates the PR.
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
        """A table as wide as the channel-mix inner width modulates its input instead.

        Two assertions, and the second is the one that matters. A table of ones being
        a no-op is equally true when the modulation has been deleted, and this test
        used to stop there: removing `inner = inner * deep_embed` from the 4x branch
        left the whole suite green. The sibling 1x test happens to cover its own
        branch that way, and the GPU-gated sparse test is skipped on CPU CI, so
        nothing anywhere reached this multiply.
        """
        config = _tiny_config(use_deep_embed=True, deep_embed_size=64)
        # `_sharpened`, not `_randomised`: at the 0.05 scale the channel-mix
        # contribution to the logits is itself below a 1e-3 tolerance, so silencing it
        # entirely is indistinguishable from silencing nothing and the second
        # assertion would be as toothless as the one it replaces.
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)
        table = (config.num_hidden_layers, 1, 4, config.intermediate_size)

        with torch.no_grad():
            plain = model(input_ids=input_ids).logits
            neutral = model(input_ids=input_ids, deep_embeds=torch.ones(*table, device=torch_device)).logits
            damped = model(input_ids=input_ids, deep_embeds=torch.zeros(*table, device=torch_device)).logits

        torch.testing.assert_close(plain, neutral, rtol=1e-5, atol=1e-5)
        self.assertFalse(
            torch.allclose(plain, damped, rtol=1e-3, atol=1e-3),
            "a table of zeros left the logits unchanged, so the 4x modulation is not wired up",
        )

    def test_generate_is_deterministic_under_greedy(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)

        with torch.no_grad():
            first = model.generate(input_ids, max_new_tokens=8, do_sample=False)
            second = model.generate(input_ids, max_new_tokens=8, do_sample=False)

        self.assertTrue(torch.equal(first, second))

    @staticmethod
    def _signed_cache(model, batch, device):
        """A cache whose every batch row carries a value only that row has.

        Every test below permutes the batch and then asks where each row went, which
        only means something if the rows were told apart to begin with.
        """
        cache = model.rwkv7.allocate_state(batch, device=device)
        for row in range(batch):
            for layer in cache.layers:
                for state in layer.recurrent_states.values():
                    state[row] = row + 1
        return cache

    @staticmethod
    def _every_state(cache):
        return [state for layer in cache.layers for state in layer.recurrent_states.values()]

    def test_reorder_cache_moves_the_state_onto_the_beams(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = self._signed_cache(model, 4, torch_device)
        addresses = [state.data_ptr() for state in self._every_state(cache)]

        # 2 appears twice: a reorder that aliased its source into its destination
        # would corrupt the second copy.
        beams = [2, 2, 0, 3]
        cache.reorder_cache(torch.tensor(beams, device=torch_device))

        for state in self._every_state(cache):
            for row, source in enumerate(beams):
                self.assertTrue(torch.all(state[row] == source + 1))
        # In place, so the addresses `allocate_state` pinned are still pinned.
        self.assertEqual([state.data_ptr() for state in self._every_state(cache)], addresses)

    def test_cache_batch_repeat_and_select(self):
        """The two batch edits `generate` makes outside beam search.

        `num_return_sequences` fans one prompt's state out to several rows before
        sampling; assisted and contrastive decoding drop rows again. Both change the
        batch size, which is why -- unlike `reorder_cache` -- they cannot keep the
        pinned buffers, and that is asserted rather than left as a surprise.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = self._signed_cache(model, 2, torch_device)

        cache.batch_repeat_interleave(3)
        for state in self._every_state(cache):
            self.assertEqual(state.shape[0], 6)
            # interleaved, so rows read 1,1,1,2,2,2 -- not 1,2,1,2,1,2
            self.assertTrue(torch.all(state[:3] == 1))
            self.assertTrue(torch.all(state[3:] == 2))

        cache.batch_select_indices(torch.tensor([0, 4], device=torch_device))
        for state in self._every_state(cache):
            self.assertEqual(state.shape[0], 2)
            self.assertTrue(torch.all(state[0] == 1))
            self.assertTrue(torch.all(state[1] == 2))

    def test_allocate_state_pins_every_buffer(self):
        """Unpinned state buffers cost CUDA graphs, and say so only in a log line.

        The whole reason `allocate_state` exists is that a buffer first allocated
        inside a compiled region cannot be given a static address, after which
        inductor declines CUDA graphs for a recurrent decode. Nothing throws; the
        decode just runs several times slower. Checking the mark directly is the
        only way that failure becomes visible without a GPU and a compile.

        `_dynamo_static_input_type` is private to torch. If it is renamed this test
        breaks loudly, which is the right failure -- the alternative is a silent
        performance regression nobody notices for a release.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = model.rwkv7.allocate_state(2, device=torch_device)

        for state in self._every_state(cache):
            self.assertEqual(getattr(state, "_dynamo_static_input_type", None), "unguarded")

    def test_cache_reset_clears_every_slot(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = self._signed_cache(model, 2, torch_device)
        addresses = [state.data_ptr() for state in self._every_state(cache)]

        cache.reset()

        for state in self._every_state(cache):
            self.assertTrue(torch.all(state == 0))
        # Reset means "forget the sequence", not "throw the buffers away".
        self.assertEqual([state.data_ptr() for state in self._every_state(cache)], addresses)

    def test_cache_carries_a_sequence_the_same_as_a_plain_rerun(self):
        """The cache is only useful if continuing through it equals never stopping."""
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(1, config.vocab_size, (2, 9), device=torch_device)

        with torch.no_grad():
            whole = model(input_ids=ids).logits
            cache = model(input_ids=ids[:, :4], use_cache=True).state
            rest = model(input_ids=ids[:, 4:], state=cache, use_cache=True).logits

        torch.testing.assert_close(rest, whole[:, 4:], rtol=1e-4, atol=1e-4)

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
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
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

    def test_chunked_survives_the_slowest_decay_it_can_be_given(self):
        """The chunk size is bounded by overflow, and only the worst decay finds it.

        The chunked form divides by the running decay `c`, so `1/c` grows like
        `e^(e^-0.5 * chunk_size)` when every channel sits at the floor the decay LoRA
        can produce -- the floor is `exp(-e^-0.5)` = 0.5452, and this docstring used to
        say `e^-0.5`, which puts the fp32 ceiling at 177 rather than the true 146. `test_chunked_matches_sequential_recurrence` draws `w_log` from
        a sigmoid, which lands nowhere near that floor, so it would pass a chunk size
        that overflows in production on a checkpoint with strongly-decaying channels.

        Precision is not what limits it: the substitution is a similarity transform,
        so `c` deflates on the way out whatever `1/c` inflated, and a common scale
        factor does not move a relative error. What limits it is fp32's exponent
        range, and the second half of this test is where that runs out -- kept as an
        assertion rather than a comment so the default has a measured ceiling above
        it rather than an assumed one.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_chunked, rwkv7_recurrent

        torch.manual_seed(0)
        shape = (1, 256, 2, 8)
        floor = -0.6065306597126334
        w_log = torch.full(shape, floor, device=torch_device)  # every channel at the floor
        r = torch.randn(shape, device=torch_device) * 0.2
        k = torch.randn(shape, device=torch_device) * 0.2
        v = torch.randn(shape, device=torch_device) * 0.2
        kk = torch.nn.functional.normalize(torch.randn(shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device=torch_device))
        state = torch.zeros(1, 2, 8, 8, device=torch_device)

        reference, _ = rwkv7_recurrent(r, w_log, k, v, kk, a, state.clone())
        for chunk_size in (16, 64):  # 64 is the default
            out, _ = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=chunk_size)
            self.assertTrue(torch.isfinite(out).all(), f"chunk={chunk_size} overflowed")
            torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-4, msg=f"chunk={chunk_size}")

        # The default itself, not just the sizes named above: every call here passes
        # `chunk_size=` explicitly, so raising the default to something that overflows
        # would leave all of them green. This is the one that exercises what the model
        # actually uses.
        default_out, _ = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone())
        self.assertTrue(torch.isfinite(default_out).all(), "the DEFAULT chunk size overflows at the decay floor")
        torch.testing.assert_close(default_out, reference, rtol=2e-4, atol=2e-4)

        # And where it runs out. This used to assert that chunk 256 comes back
        # non-finite, which documented the overflow by demonstrating it silently. The
        # function now refuses instead, and the boundary is asserted on both sides so
        # the derived limit cannot drift without a test noticing: 146 is the widest that
        # works, 147 is the narrowest that raises.
        widest, _ = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=146)
        self.assertTrue(torch.isfinite(widest).all(), "146 is supposed to be the widest chunk fp32 can carry")

        with self.assertRaisesRegex(ValueError, "widest chunk that stays finite is 146"):
            rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=147)

    def test_fused_decode_falls_back_rather_than_failing(self):
        """`"fused"` must be selectable anywhere, not only where its kernel runs.

        It is a decode-step kernel: prefill, packed rows, CPU tensors, a state that is
        not the fp32 it accumulates in, and a machine without Triton all have to reach
        the portable path instead of raising. A registry entry that only works on one
        configuration is worse than no entry, because the failure surfaces at the
        first shape the caller did not think about.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import RWKV7_WKV_FUNCTIONS

        self.assertIn("fused", RWKV7_WKV_FUNCTIONS)
        config = _tiny_config(wkv_implementation="fused")
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        reference = _sharpened(Rwkv7ForCausalLM(_tiny_config())).to(torch_device)
        reference.load_state_dict(model.state_dict())

        ids = torch.randint(1, config.vocab_size, (2, 9), device=torch_device)
        with torch.no_grad():
            got = model(input_ids=ids).logits  # prefill: falls through
            want = reference(input_ids=ids).logits
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

        with torch.no_grad():  # decode, one token at a time
            fused_state, plain_state, fused, plain = None, None, [], []
            for position in range(ids.shape[1]):
                a = model(input_ids=ids[:, position : position + 1], state=fused_state, use_cache=True)
                b = reference(input_ids=ids[:, position : position + 1], state=plain_state, use_cache=True)
                fused_state, plain_state = a.state, b.state
                fused.append(a.logits)
                plain.append(b.logits)
        torch.testing.assert_close(torch.cat(fused, 1), torch.cat(plain, 1), rtol=1e-4, atol=1e-4)

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
        """Skipping zero channels must not move the answer.

        `relu(x)**2` produces exact zeros and `0 * w == 0`, so the sparse path is
        the same sum with terms that contribute nothing left out. What it is not is
        bit-identical: the surviving channels are summed across partitions that
        combine through an atomic add, so the arrival order and therefore the last
        place of each output varies between runs. `sparse_channel_mix` documents that.

        So the bar is stated against a float64 reference rather than as a tolerance on
        the difference between the two fp32 results. This test asserted
        `rtol=1e-5` and had never run -- the first GPU it reached measured 1.16e-05 and
        it failed, on a number that was picked rather than derived. Requiring the
        sparse path to be no further from the truth than the dense path already is
        cannot be tuned into passing.

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
        truth = torch.nn.functional.linear(act.double(), weight_t.t().double())

        # The standard forward-error bound for a floating-point sum of `n` terms:
        # `n * eps * sum|terms|`. Derived rather than picked, so it cannot be widened
        # to make a failure go away, and it holds for any summation order -- which is
        # the whole question here, since the kernel's order is not the dense one and
        # varies between runs. `n` is the number of surviving channels.
        eps = torch.finfo(torch.float32).eps
        surviving = int((act != 0).sum())
        magnitude = (act.double().abs()[:, None] * weight_t.double().abs()).sum(dim=0)
        bound = (surviving * eps * magnitude).max().item()

        for name, got in (("dense", dense), ("sparse", sparse)):
            error = (got.double() - truth).abs().max().item()
            self.assertLessEqual(error, bound, f"{name} exceeds the fp32 summation bound: {error:.3e} > {bound:.3e}")
        # And the zeros really were skipped rather than multiplied: with every channel
        # kept, the same call must reproduce the dense result to the same standard.
        self.assertGreater((act == 0).float().mean().item(), 0.5, "the fixture stopped being sparse")

        # an all-zero activation must give exactly zero, and leave every scratch
        # buffer clean -- the projection reuses them across layers and steps, so a
        # finalize pass that forgot to re-zero one would only show up on the call
        # after the one being checked
        zeros = sparse_channel_mix_value(torch.zeros_like(act), weight_t, *scratch)
        self.assertEqual(zeros.abs().max().item(), 0.0)
        self.assertEqual(accumulator.abs().max().item(), 0.0)
        self.assertEqual(counter.abs().max().item(), 0)

        # A repeat of the first call must land in the same place, which it cannot if
        # anything above was left dirty. Not bit-for-bit, though: `sparse_channel_mix`
        # documents that the partitions combine through an atomic add, so the arrival
        # order and the last place of each output vary between runs of identical input.
        # This asserted `rtol=0, atol=0` and had never executed; the first GPU it
        # reached measured 6.0e-06 between two identical calls, which is the documented
        # property rather than a defect. A dirty accumulator would not produce 6e-06,
        # it would produce roughly double, so the bound still separates the two cases.
        again = sparse_channel_mix_value(act, weight_t, *scratch)
        drift = (again.double() - sparse.double()).abs().max().item()
        self.assertLessEqual(drift, bound, f"a repeated call moved by {drift:.3e}, past the summation bound")

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


@require_torch
@slow
class Rwkv7IntegrationTests(unittest.TestCase):
    """A real checkpoint, against numbers produced by an implementation that is not this one.

    Everything else in this file builds a model out of random weights, which pins
    internal consistency and cannot see a whole class of defect: an `_init_weights` that
    zeroed 249 of a converted checkpoint's 402 tensors passed the entire suite and was
    caught only by loading real weights and reading the output.

    The expected values below come from BlinkDL's own runtime -- the `rwkv` package at
    `cpu fp32` with `RWKV_V7_ON=1` -- and not from this implementation, which would make
    them self-certifying. They are deliberately not taken from `fla`, whose RWKV layer
    prints a warning on import saying it is potentially buggy and that results should be
    cross-checked against the official repo.

    0.1B is chosen for more than its size: 768 wide at head_dim 64 is twelve heads of
    width 64, so head *count* and head *width* differ. At the 7.2B they are both 64 and a
    quantity indexed by the wrong one of them agrees by coincidence.
    """

    CHECKPOINT_REPO = "BlinkDL/rwkv-7-world"
    CHECKPOINT_FILE = "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"

    # "The Eiffel Tower is located in the city of", RWKV world tokenizer.
    PROMPT_IDS = [6699, 304, 25740, 109, 37480, 4600, 52151, 4596, 22590, 30449, 4706]
    # " Paris, France. It is the tallest building in the world and is the world's tallest"
    EXPECTED_IDS = [
        37138,
        45,
        44312,
        47,
        3918,
        4600,
        22590,
        32190,
        7513,
        55666,
        4596,
        22590,
        40213,
        21265,
        4600,
        22590,
        40213,
        460,
        32190,
        7513,
    ]
    EXPECTED_TOP5 = [37138, 29319, 20312, 3632, 29417]
    EXPECTED_TOP5_LOGITS = [5.1235, 1.0825, 0.9404, 0.8200, 0.2382]

    @classmethod
    def setUpClass(cls):
        from huggingface_hub import hf_hub_download

        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import convert

        cls._work = tempfile.mkdtemp()
        native = hf_hub_download(cls.CHECKPOINT_REPO, cls.CHECKPOINT_FILE)
        # Through the converter in this PR, so the test covers that too: a model PR whose
        # checkpoints all arrive by a path the tests never take is testing half of itself.
        convert(native, "native", None, f"{cls._work}/hf", dtype="float32")
        cls.model = Rwkv7ForCausalLM.from_pretrained(f"{cls._work}/hf", dtype=torch.float32).eval()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._work, ignore_errors=True)

    def test_config_is_inferred_at_a_non_square_width(self):
        config = self.model.config
        self.assertEqual((config.hidden_size, config.num_heads, config.head_dim), (768, 12, 64))
        self.assertNotEqual(config.num_heads, config.head_dim)

    def test_next_token_logits_match_the_reference_runtime(self):
        """The distribution, not just its argmax: a wrong scale agrees on the winner."""
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([self.PROMPT_IDS])).logits[0, -1].float()

        values, indices = torch.topk(logits, 5)
        self.assertEqual(indices.tolist(), self.EXPECTED_TOP5)
        torch.testing.assert_close(values, torch.tensor(self.EXPECTED_TOP5_LOGITS), rtol=1e-4, atol=1e-3)

    def test_greedy_continuation_matches_the_reference_runtime(self):
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=torch.tensor([self.PROMPT_IDS]), max_new_tokens=20, do_sample=False
            )
        self.assertEqual(generated[0, len(self.PROMPT_IDS) :].tolist(), self.EXPECTED_IDS)

    def test_the_cache_carries_what_a_full_forward_computes(self):
        """Twenty steps of one-token decode against twenty full re-reads of the prefix.

        The shared suite checks this on random weights, where the state is small enough
        that a broken carry can hide under the tolerance. Here a divergence shows up as a
        different token.
        """
        ids = list(self.PROMPT_IDS)
        cached, recomputed = [], []
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids]), use_cache=True)
            state = out.state
            cached.append(int(out.logits[0, -1].argmax()))
            for _ in range(19):
                out = self.model(input_ids=torch.tensor([[cached[-1]]]), state=state, use_cache=True)
                state = out.state
                cached.append(int(out.logits[0, -1].argmax()))

            walk = list(self.PROMPT_IDS)
            for _ in range(20):
                logits = self.model(input_ids=torch.tensor([walk]), use_cache=False).logits[0, -1]
                recomputed.append(int(logits.argmax()))
                walk.append(recomputed[-1])

        self.assertEqual(cached, recomputed)
        self.assertEqual(cached, self.EXPECTED_IDS)
