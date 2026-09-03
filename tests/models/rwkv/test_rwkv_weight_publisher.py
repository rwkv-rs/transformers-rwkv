# Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors import safe_open

from temp.publish_rwkv7_weights import (
    ArtifactValidation,
    Checkpoint,
    latest_checkpoints,
    parse_checkpoint,
    parse_provenance,
    run,
    target_subfolders,
    update_readme,
)
from temp.rwkv_pth2st import convert_checkpoint, translate_key
from transformers import RwkvConfig, RwkvForCausalLM


def sibling(filename, *, sha256=None, size=1):
    lfs = None if sha256 is None else SimpleNamespace(sha256=sha256)
    return SimpleNamespace(rfilename=filename, lfs=lfs, size=size)


def canonical_key(target_key):
    top_level = {
        "model.embed_tokens.weight": "emb.weight",
        "model.embedding_norm.weight": "blocks.0.ln0.weight",
        "model.embedding_norm.bias": "blocks.0.ln0.bias",
        "model.norm.weight": "ln_out.weight",
        "model.norm.bias": "ln_out.bias",
        "lm_head.weight": "head.weight",
    }
    if target_key in top_level:
        return top_level[target_key]
    _, _, layer, component, *suffix = target_key.split(".")
    component = {
        "input_layernorm": "ln1",
        "post_attention_layernorm": "ln2",
        "linear_attn": "att",
        "mlp": "ffn",
    }[component]
    if component == "att":
        suffix[0] = {
            "r_proj": "receptance",
            "k_proj": "key",
            "v_proj": "value",
            "o_proj": "output",
            "g_norm": "ln_x",
        }.get(suffix[0], suffix[0])
    return ".".join(("blocks", layer, component, *suffix))


class RwkvWeightPublisherTest(unittest.TestCase):
    def test_parse_checkpoint_accepts_only_canonical_stable_names(self):
        checkpoint = parse_checkpoint(
            "rwkv7-g1j-7.2b-20260831-ctx16384.pth",
            sha256="abc",
            size=123,
        )
        assert checkpoint is not None
        self.assertEqual(checkpoint.name, "rwkv7-g1j-7.2b-20260831-ctx16384")
        self.assertEqual(checkpoint.context_length, 16384)
        for filename in (
            "rwkv7a-g1j-7.2b-20260831-ctx16384.pth",
            "rwkv7b-g1j-7.2b-20260831-ctx16384.pth",
            "rwkv-x070-test.pth",
            "states/rwkv7-g1j-7.2b-20260831-ctx16384.pth",
        ):
            self.assertIsNone(parse_checkpoint(filename, sha256="abc", size=123))

    def test_latest_checkpoints_chooses_latest_generation_and_size_order(self):
        siblings = [
            sibling("rwkv7-g1i-13.3b-20260805-ctx16384.pth", sha256="old"),
            sibling("rwkv7-g1j-13.3b-20260831-ctx16384.pth", sha256="large"),
            sibling("rwkv7-g1j-1.5b-20260831-ctx16384.pth", sha256="small"),
            sibling("rwkv7a-g1z-0.1b-20260901-ctx16384.pth", sha256="experimental"),
        ]
        self.assertEqual([item.size for item in latest_checkpoints(siblings)], ["1.5b", "13.3b"])

    def test_target_subfolders_ignores_root_files(self):
        siblings = [sibling("README.md"), sibling("rwkv7-g1j-1.5b/file.json")]
        self.assertEqual(target_subfolders(siblings), {"rwkv7-g1j-1.5b"})

    def test_parse_provenance_extracts_identity(self):
        provenance = parse_provenance(
            "- Source repository: `BlinkDL/rwkv7-g1`\n"
            "- Source revision: `0123`\n"
            "- Source file: `model.pth`\n"
            "- Source SHA256: `abcd`\n"
        )
        self.assertEqual(provenance["Source revision"], "0123")
        self.assertEqual(provenance["Source SHA256"], "abcd")

    def test_update_readme_is_deterministic(self):
        readme = (
            "# Models\n\n"
            "| Checkpoint | Parameters | Context | Subfolder |\n"
            "| --- | ---: | ---: | --- |\n"
            "| RWKV-7 G1i 7.2B | 7,199 | 16,384 | `rwkv7-g1i-7.2b-old-ctx16384` |\n"
            "\nAfter.\n"
        )
        checkpoint = Checkpoint("rwkv7-g1j-1.5b-20260831-ctx16384.pth", "j", "1.5b", "20260831", 16384, "x", 1)
        validation = ArtifactValidation(1, 1527404544, (("model.safetensors", "hash"),))
        updated = update_readme(readme, [(checkpoint, validation)])
        self.assertLess(updated.index("G1i 7.2B"), updated.index("G1j 1.5B"))
        self.assertEqual(updated, update_readme(updated, [(checkpoint, validation)]))

    def test_dry_run_has_no_download_or_mutation(self):
        source = SimpleNamespace(
            sha="source-sha",
            siblings=[sibling("rwkv7-g1j-1.5b-20260831-ctx16384.pth", sha256="source-lfs", size=10)],
        )
        target = SimpleNamespace(sha="target-sha", siblings=[sibling("README.md")])
        api = Mock()
        api.model_info.side_effect = [source, target]
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("temp.publish_rwkv7_weights.OUTPUT_ROOT", Path(temporary) / "outputs"),
        ):
            result = run(dry_run=True, source_revision=None, api=api)
            self.assertEqual(result["pending"], ["rwkv7-g1j-1.5b-20260831-ctx16384.pth"])
            self.assertFalse((Path(temporary) / "outputs").exists())
        api.hf_hub_download.assert_not_called()
        api.create_commit.assert_not_called()
        api.preupload_lfs_files.assert_not_called()

    def test_missing_token_fails_before_download_or_publish(self):
        source = SimpleNamespace(
            sha="source-sha",
            siblings=[sibling("rwkv7-g1j-1.5b-20260831-ctx16384.pth", sha256="source-lfs", size=10)],
        )
        target = SimpleNamespace(sha="target-sha", siblings=[sibling("README.md")])
        api = Mock()
        api.model_info.side_effect = [source, target]
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "HF_TOKEN"):
            run(dry_run=False, source_revision=None, api=api)
        api.hf_hub_download.assert_not_called()
        api.create_commit.assert_not_called()

    def test_conversion_failure_cannot_reach_upload(self):
        source = SimpleNamespace(
            sha="source-sha",
            siblings=[sibling("rwkv7-g1j-1.5b-20260831-ctx16384.pth", sha256="source-lfs", size=10)],
        )
        target = SimpleNamespace(sha="target-sha", siblings=[sibling("README.md")])
        api = Mock()
        api.model_info.side_effect = [source, target]
        with (
            patch.dict("os.environ", {"HF_TOKEN": "test-token"}, clear=True),
            patch("temp.publish_rwkv7_weights.validate_disk_space"),
            patch("temp.publish_rwkv7_weights.prepare_artifact", side_effect=RuntimeError("conversion failed")),
            self.assertRaisesRegex(RuntimeError, "conversion failed"),
        ):
            run(dry_run=False, source_revision=None, api=api)
        api.preupload_lfs_files.assert_not_called()
        api.create_commit.assert_not_called()

    def test_synthetic_checkpoint_conversion_is_exact(self):
        config = RwkvConfig(
            vocab_size=128,
            context_length=32,
            hidden_size=64,
            num_hidden_layers=2,
            intermediate_size=256,
            head_size=64,
            decay_low_rank_dim=32,
            a_low_rank_dim=32,
            v_low_rank_dim=32,
            gate_low_rank_dim=32,
        )
        target_state = RwkvForCausalLM(config).to(torch.bfloat16).state_dict()
        source_state = {canonical_key(key): tensor for key, tensor in target_state.items()}
        for name in ("v0", "v1", "v2"):
            source_state[f"blocks.0.att.{name}"] = source_state[f"blocks.1.att.{name}"].clone()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.pth"
            output = root / "output"
            torch.save(source_state, checkpoint)
            with patch("temp.rwkv_pth2st.save_tokenizer"):
                convert_checkpoint(checkpoint, output, 32, "1GB", tokenizer_source=None)
            with safe_open(output / "model.safetensors", framework="pt", device="cpu") as converted:
                self.assertEqual(set(converted.keys()), set(target_state))
                for source_key, tensor in source_state.items():
                    target_key = translate_key(source_key)
                    if target_key is not None:
                        torch.testing.assert_close(converted.get_tensor(target_key), tensor, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
