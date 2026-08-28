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

## 核心目标
Transformers 是 LLM 社区的标准库, 很早成为生态中心, 本仓库需要按照社区主流做法(参考 FLA 中对 RWKV 的**代码风格**, 以及 Transformers 中 Qwen3.5 Kimi-K3 等带有 Linear RNN Layer 的模型的**功能设计**与**代码风格**)完成 RWKV 模型的接入, 通过最简洁直接的方式替代上游仓库中 rwkv 的实现. (因为它实际上是早已过时的 rwkv4 而非我们需要支持的 rwkv7)
代码原则: 每一个文件/类型/函数/变量都需要找到相似实现作为原型, 若该原型带有模型名则将其替换为 `RWKV` 或其它大小写变种, 否则保持同名.
在 ./temp 目录下尽可能精简实现权重转换脚本, 对齐 FLA 中 RWKV7 的代码风格, 避免维护不必要的重命名契约.
本仓库对 rwkv7 预训练的支持数值精度与吞吐应完全对齐 RWKV-LM/blob/main/RWKV-v7/train_temp;
本仓库对 rwkv7 推理的支持数值精度与吞吐应完全对齐 Albatross;
FlashRWKV (https://github.com/rwkv-rs/FlashRWKV) 是 RWKV 社区权威算子实现仓库, 为本仓库提供高性能后端, 本仓库对算子相关内容只做导入不做开发. 如精度与推理速度因为 FlashRWKV 实现错误导致精度差/推理速度慢, 应直接与用户反馈, 无需跨越实现边界完成修复.
为 Transformers 的支持应服务于主流研究需求: 假设存在一个用户, 创建自定义模型 Qwen2Rwkv, 要加载一个魔改的 Qwen 3.5, 所有的 GDN 换成 128 head_size的 RWKV Tmix, 所有 GQA 换成 256 head_size 的 RWKV Tmix, Norm 都用 RWKV 的 LayerNorm, emb lm_head moe 都用 Qwen 3.5 的实现, 应当做到50行代码左右实现对应功能. (暂不实现)

## 权威 RWKV7 实现
(1) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_numpy.py
(2) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/run_rwkv7_qwen35.py
(3) https://github.com/BlinkDL/Albatross -- 权威底层推理引擎实现仓库 (cuda, for pro6000, 无调度, 无varlen)
(4) https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/train_temp -- 权威预训练实现仓库 (cuda, for h100)
(5) https://zhiyuan1i.github.io/posts/dplr-mathematics -- Diagonal Plus Low Rank(DPLR）的数学原理：显式转移矩阵的并行计算

## RWKV7 权重
权重一般命名规范: {arch_version}-{data_version}-{param_size}-{release_date}-{ctx_len}.pth
如: rwkv7-g1h-7.2b-20260710-ctx10240.pth
arch_version: 架构版本, 如 rwkv7(default), rwkv7a(experimental, rwkv7 with DeepEmbed), rwkv7b(experimental, rwkv7 with DeepEmbedAttn)
data_version: 数据版本, 如 g1a, g1b... (The further back in the alphabet, the better)
param_size: 参数规模, 仅有 0.1b, 0.4b, 1.5b(often used in RL), 2.9b, 7.2b(often used in the infer test), 13.3b
(1) https://huggingface.co/BlinkDL/rwkv7-g1/tree/main -- 权威权重 Release 源 (update every month)
(2) https://huggingface.co/BlinkDL/temp-latest-training-models/tree/main -- 权威权重 Test 源 (不定期update)
转换为 safetensor 格式后应固化于 `rwkv-sha-pro6000x8` 的 `~/Weights/RWKV/hf` 目录, 并推送到对应仓库.

## Env
使用 uv 管理本机和远端专属环境 ./.venv, 严禁使用其它环境, 避免环境污染问题。

## Machine for Testing and Benchmarking
```bash
ssh rwkv-sha-pro6000x8
cd ~/Projects/MachineLearning/transformers-rwkv
```
use git to sync your changes instead of rsync.
