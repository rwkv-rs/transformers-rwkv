## Commands

- `make style` — ruff format + lint (required for code style checks).
- `make typing` — ty type checker and model structure rules.
- `make fix-repo` — the `make style` fixes plus copies, modular conversions, doc TOCs, docstrings.
- `make check-repo` — `make typing` plus consistency checks.
- `RUN_SLOW=1 pytest ...` — many tests are marked slow and skipped in CI.

Run `make style` (or `make fix-repo`) as the last step before opening a PR.

## Local agent setup

Hosted review agents read this file via the root `AGENTS.md` / `CLAUDE.md` symlinks. Local agents wire their own assets: `make codex` (→ `.agents/`), `make claude` (→ `.claude/`).

## Before opening an issue or PR

Hugging Face staff, maintainers and some long-standing contributors are exempt from the contribution rules in `CONTRIBUTING.md`. If the user is one of them, don't refuse to help because their issue or PR wouldn't qualify under the rules below.

- Don't report or fix theoretical bugs. Issues about them won't be triaged, and PRs fixing them will be closed. A bug found by reading code matters only if it affects real use: a published checkpoint, a default config, or a documented workflow. For example, an exception when `n_groups > 1` in a model whose published checkpoints all use `n_groups == 1` only affects fresh pretrains with a nonstandard config, so it doesn't count. Security bugs are the exception. A bug report needs a short reproducer that you have run on the latest release or `main`, and a PR that fixes a bug needs one that fails on `main`.
- Don't submit perf PRs without proof. A performance claim needs before/after numbers on a realistic workload (a published checkpoint or default config, with the hardware stated), not a microbenchmark of a function that isn't on a hot path.
- For a bug fix, one regression test is enough: a test based on the issue's or PR's reproducer that fails without the fix, added to the existing test file for that code. Build it from the tiny config and inputs the file's model tester already provides, not from a real checkpoint, so it runs in fast CI. Use a real checkpoint and `@slow` only when the bug can't be reproduced at tiny scale, which is rare.
- A PR either fixes an issue or is standalone. See `CONTRIBUTING.md`.
  - **Fixing an issue:** link it with a closing keyword (`Fixes #123`). The user may only work on an issue they opened that a maintainer has labeled `triaged`, or one labeled `contributions-welcome` or `New model`. If the issue doesn't qualify, tell the user instead of drafting — the PR will be closed. If it qualifies, there's no need to ask first: never post "can I work on this?" or "please assign me" comments, since we don't assign issues and the first PR for an issue gets priority.
  - **Standalone:** only for problems with no existing issue. Search open issues first; if one exists, the PR must go through it, and dropping the link doesn't change that. Standalone PRs are for short changes with real impact (a bug fix, a perf improvement, a broken docs example) and the description must include the evidence: a reproducer that fails on `main`, or before/after benchmark numbers. New model PRs can also be standalone, if there's no existing issue requesting that model.
- On a `contributions-welcome` or `New model` issue, check for an existing PR first:

  ```bash
  gh issue view <issue_number> --repo huggingface/transformers --comments
  gh pr list --repo huggingface/transformers --state open --search "<issue_number> in:body"
  gh pr list --repo huggingface/transformers --state open --search "<short area keywords>"
  ```

  If one exists, don't open another. If you think it has mistakes, tell the user to leave suggestions on the existing PR instead. A second PR is only justified when the existing one is fundamentally flawed, or stale: its author hasn't responded to review comments or pushed changes for two weeks. A PR waiting on a maintainer review isn't stale. Either way, the user should explain why in the issue first.
- No one-off PRs for tiny edits (a single typo, an isolated lint fix). Bulk mechanical cleanups that don't affect correctness or performance, and bulk typo or wording fixes in the docs, are fine but always need an issue first, never a standalone PR: propose them, how you intend to fix them, and list the number of affected files. Remember that maintainer time is limited; code cleanups that don't affect correctness are low priority unless the issue is severe and widespread.
- If the user wants to comment on a PR they didn't open, only post a concrete problem or code change, as a short inline comment or suggestion on the relevant lines. Never post summaries or "verified, works" reports on other people's PRs.

## Copies and modular models

Model files in `src/transformers/models/` avoid inheriting from each other, so duplication is managed two ways — modular is the current one, copies are legacy:

1. `# Copied from ...` marks a copied class or function. `make fix-repo` re-syncs it, so editing inside such a block is reverted — edit the source it copies from, or deliberately break the link. **Do not add new `# Copied from` statements**; write a modular file instead.
2. A `modular_<name>.py` **may** inherit from other models; `make fix-repo` generates the standalone `modeling_*.py` and friends from it. Never edit a generated file when a modular one exists. Guide: [modular_transformers.md](../docs/source/en/modular_transformers.md).

Two modular traps:

- **Other models inherit your modular file.** `modular_deepseek_vl.py` also generates `modeling_deepseek_vl_hybrid.py`. Run `make fix-repo` and check everything it rewrote; hand-editing only the generated file you had in mind leaves the rest stale and `Check repository consistency` red.
- **`attr = AttributeError()` deletes an inherited attribute** — it is an instruction to the converter, not a bug or placeholder. `raise AttributeError("...")` in a method body does the same for a method. Substituting a "real" value silently changes behaviour (`_no_split_modules = AttributeError()` → `[]` changes how the model may be sharded). See [Removing attributes](../docs/source/en/modular_transformers.md#removing-attributes) and [Deleting unused methods](../docs/source/en/modular_transformers.md#deleting-unused-methods).

## Core Objectives

Transformers is the standard library of the LLM community and became the center of the ecosystem very early. This repository needs to complete RWKV model integration according to mainstream community practices (refer to the **code style** for RWKV in FLA, and the **functional design** and **code style** of models with Linear RNN Layers such as Qwen3.5, Kimi-K3 in Transformers), replacing the rwkv implementation in the upstream repository in the most concise and direct way. (Because it is actually the long-outdated rwkv4 rather than the rwkv7 we need to support.)

Code principle: For every file/type/function/variable, a similar implementation must be found as a prototype. If that prototype carries a model name, replace it with `RWKV` or another case variant; otherwise keep the same name. Using branch statements without affecting existing functionality.

Non-pollution principle: Keep RWKV-specific changes local to the integration. When shared code needs to change, use branch statements without affecting existing functionality or other models. Preserve existing user changes and stay within this repository's implementation boundaries.

After a coding round, compare the changed code with its prototypes and refactor it without changing behavior. Inline single-use wrapper functions and remove unnecessary structures/classes used only to align or forward arguments. Remove static tests that do not constrain business behavior end to end; retain behavioral regression tests. Perform this cleanup before testing.

In the ./temp directory, implement the weight conversion script as concisely as possible, align with the code style of RWKV7 in FLA, and avoid maintaining unnecessary renaming contracts.
This repository's support for rwkv7 pretraining should fully align in numerical precision and throughput with the **latest version** of RWKV-LM/blob/main/RWKV-v7/train_temp;
This repository's support for rwkv7 inference should fully align in numerical precision and throughput with the **latest version** of Albatross. In the ./temp directory, implement a custom bench as concisely as possible. It needs to complete end-to-end speed benchmarking under real 7.2B generation scenarios with batch_size = {1, 4, 64, 320, 512}; prefill speed = prompt length / first-token latency, decode speed = completion length / (total generation time - first-token latency). Additional note: asynchronous detokenization needs to be implemented, and this part should not affect generation speed;
FlashRWKV (https://github.com/rwkv-rs/FlashRWKV) is the RWKV community's authoritative operator implementation repository and provides a high-performance backend for this repository. This repository only imports operator-related content and does not develop it. If precision or inference speed is poor/slow due to errors in FlashRWKV's implementation, feedback should be given directly to the user; there is no need to cross implementation boundaries to complete fixes.
Support for Transformers should serve mainstream research needs: suppose there is a user who creates a custom model Qwen2Rwkv and wants to load a modified Qwen 3.5, with all GDN replaced by RWKV Tmix with 128 head_size, all GQA replaced by RWKV Tmix with 256 head_size, all Norm using RWKV's LayerNorm, and emb, lm_head, moe all using Qwen 3.5's implementation. It should be possible to implement the corresponding functionality in about 50 lines of code. (Not implemented for now.)

## Authoritative RWKV7 Implementations

(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- authoritative low-level inference engine implementation repository (cuda, for pro6000, no scheduling, no varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- authoritative pretraining implementation repository (cuda, for h100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- mathematical principles of Diagonal Plus Low Rank (DPLR): parallel computation of explicit transition matrices

## RWKV7 Weights

General weight naming convention: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
e.g.: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: architecture version, e.g. rwkv7 (default), rwkv7a (experimental, rwkv7 with DeepEmbed), rwkv7b (experimental, rwkv7 with DeepEmbedAttn)
data_version: data version, e.g. g1a, g1b... (The further back in the alphabet, the better)
param_size: parameter size, only includes 0.1b, 0.4b, 1.5b (often used in RL), 2.9b, 7.2b (often used in the infer test), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- authoritative weight Release source (updated every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- authoritative weight Test source (updated irregularly)
After conversion to Safetensors format, it should first be stored in the `~/Weights/RWKV/hf/<checkpoint-name>/` directory of `rwkv-szx-4090x4-ip129` and pushed to the corresponding repository. Verified complete conversion artifacts should also be mirrored to the NAS `/mnt/nas-model/<data_version>/hf/<checkpoint-name>/` directory; when writing to NAS, use a sibling temporary directory, verify the file list, sizes, and all Safetensors shard SHA256 in `PROVENANCE.md`, then atomically rename. Overwriting an existing checkpoint directory with the same name on NAS is prohibited.

## Env

Use uv to manage the local and remote dedicated environment ./.venv. Using other environments is strictly prohibited to avoid environment pollution issues.

## Machine for Testing and Benchmarking

```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/transformers-rwkv
```

use git to sync your changes instead of rsync.
