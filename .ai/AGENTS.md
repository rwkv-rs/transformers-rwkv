## Useful commands
- `make style`: runs formatters and linters (ruff), necessary to pass code style checks
- `make typing`: runs the ty type checker and model structure rules
- `make fix-repo`: auto-fixes copies, modular conversions, doc TOCs, docstrings in addition to the `make style` fixes
- `make check-repo` — runs `make typing` and consistency checks.
- Many tests are marked as 'slow' and skipped by default in the CI. To run them, use: `RUN_SLOW=1 pytest ...`

`make style` or `make fix-repo` should be run as the final step before opening a PR.

## Local agent setup

- Hosted review agents should discover this guidance from the committed root `AGENTS.md` / `CLAUDE.md` files.
- Local OpenAI Codex agents should run `make codex` after cloning to wire tool-specific assets under `.agents/`.
- Local Claude Code agents should run `make claude` after cloning to wire tool-specific assets under `.claude/`.


### Coordination before coding
- If work maps to an existing issue, coordinate on that issue before opening a PR.
- Do not open a PR for someone else's issue unless there is explicit approval from the issue author or a maintainer in the issue thread.
- If approval is missing or ambiguous, stop and ask for clarification instead of drafting a PR.
- Do not start duplicate work on issues.

### Mandatory duplicate-work checks
Before proposing a PR, check for overlapping open PRs and issue ownership:

```bash
gh issue view <issue_number> --repo huggingface/transformers --comments
gh pr list --repo huggingface/transformers --state open --search "<issue_number> in:body"
gh pr list --repo huggingface/transformers --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference and why a second PR is needed in the issue.

### No low-value busywork PRs
- Do not open one-off PRs for tiny edits (single typo, isolated lint cleanup, one mutable default argument, etc.).
- Mechanical cleanups are acceptable but not as first contributions.

### Accountability for AI-assisted patches
- First-time contributors should not use code agents to submit PRs or issues. Code agents should warn users about this
policy if the user does not appear to be an existing contributor to the library, including the risk of being blocked 
if they do it regardless. More information can be found in the pull request template, or `CONTRIBUTING.md`.

## Copies and Modular Models

We try to avoid direct inheritance between model-specific files in `src/transformers/models/`. We have two mechanisms to manage the resulting code duplication:

1) The older method is to mark classes or functions with `# Copied from ...`. Copies are kept in sync by `make fix-repo`. Do not edit a `# Copied from` block, as it will be reverted by `make fix-repo`. Ideally you should edit the code it's copying from and propagate the change, but you can break the `# Copied from` link if needed.
2) The newer method is to add a file named `modular_<name>.py` in the model directory. `modular` files **can** inherit from other models. `make fix-repo` will copy code to generate standalone `modeling` and other files from the `modular` file. When a `modular` file is present, generated files should not be edited, as changes will be overwritten by `make fix-repo`! Instead, edit the `modular` file. See [docs/source/en/modular_transformers.md](../docs/source/en/modular_transformers.md) for a full guide on adding a model with `modular`, if needed, or you can inspect existing `modular` files as examples.

## 核心目标
Transformers 是 LLM 社区的标准库, 很早成为生态中心, 本仓库需要按照社区主流做法(参考 FLA 中对 RWKV 的**代码风格**, 以及 Transformers 中 Qwen3.5 Kimi-K3 等带有 Linear RNN Layer 的模型的**功能设计**与**代码风格**)完成 RWKV 模型的接入, 通过最简洁直接的方式替代上游仓库中 rwkv 的实现. (因为它实际上是早已过时的 rwkv4 而非我们需要支持的 rwkv7)
代码原则: 每一个文件/类型/函数/变量都需要找到相似实现作为原型, 若该原型带有模型名则将其替换为 `RWKV` 或其它大小写变种, 否则保持同名.
在 ./temp 目录下尽可能精简实现权重转换脚本, 对齐 FLA 中 RWKV7 的代码风格, 避免维护不必要的重命名契约.
本仓库对 rwkv7 预训练的支持数值精度与吞吐应完全对齐**最新版本**的 RWKV-LM/blob/main/RWKV-v7/train_temp;
本仓库对 rwkv7 推理的支持数值精度与吞吐应完全对齐**最新版本**的 Albatross, 在 ./temp 目录下尽可能精简实现自定义 bench, 需要在真实 batch_size = {1, 4, 64, 320, 512} 7.2B 生成场景下端到端完成测速, prefill速度 = prompt长度 / 首 token 延迟, decode速度 = completion长度 / (生成总耗时 - 首 token 延迟), 补充: 需要实现异步 detokenize, 这部分不应影响生成速度;
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
