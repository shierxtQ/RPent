# Codex 评测说明 / Codex evaluation notes (2026-10-19)

**Branch:** `shierxtQ/RPent` `feat/codex-eval`.
- Base: `feat/robocasa365-eval`, which is `sparklesea/RPent` `feat/qwen-gpt-escalation-parallel` at 0d514b1 plus 7 commits.
- Added here: the Codex senior backend and this note.

## 中文

### 1. 分支内容

| 提交 | 内容 |
|---|---|
| d8aa18b | RoboCasa 工具不受 50 s 工具上限限制（Target50 的一次 RLDX 调用约 100 s） |
| 01fe9f4 | Qwen 只回文字时追加 continue note（最多 3 次）；修复工具参数里的前导零数字（`RPENT_COERCE_TOOL_ARGS=1`） |
| 81e0b74 | senior 请求最多带最新 N 张图（`RPENT_ESCALATION_SENIOR_MAX_IMAGES`）；Sorux 超过 64–80 张返回 400 |
| 8e39136 | escalation 可选：关闭 plan、plan 读完整 memory、senior takeover（默认都关） |
| 3937a2b | 流式响应里返回的网关错误也会重试 |
| 4779c47 | Codex planner：Codex 自己重连的流错误不再判为失败；GPT-6 写出的 Windows 风格 memory 路径能读到对应文件 |
| 本分支 | Codex senior：`RPENT_ESCALATION_SENIOR_BACKEND=codex`，Qwen 主控时 GPT 的回合走官方 Codex CLI |

合并到你的最新分支（08d1d81、95aef95）时，只有 `Toolkit.call` 和两个测试文件冲突：你加的 simulator slot 把 50 s 上限那段代码整体缩进了。

### 2. 用官方配置跑 Codex，与 RPent 官方分数对比

Leaderboard 上的 GPT 行都是 `--planner codex`，GPT 每一回合都调用。RoboCasa Target50 一个格子：

```bash
export CODEX_BASE_URL=...  CODEX_API_KEY=...        # Responses 兼容端点；不读 OPENAI_*
export CODEX_HOME=$OUT/codex_home                   # 每个 episode 单独一个
export RPENT_REPO_ROOT=/path/to/RPent-codex         # 单独的 worktree，同一个 commit
export RLDX_MAX_CHUNKS=40 RLDX_SETTLE_PATIENCE=999 RLDX_ACTION_STEPS_PER_CHUNK=8
cd "$RPENT_REPO_ROOT"
rpent --robot robocasa --task-name "$TASK" --split target --seed "$SEED" \
  --vla-endpoint "$VLA" --cuda-device 0 \
  --planner codex --model gpt-6-astra --reasoning-effort high \
  --max-turns 100 --planner-timeout-s 1800 \
  --memory-profile local --memory-dir "$MEMORY/robocasa/GPT_6_astra_high" \
  --output-dir "$OUT"
```

- composite 两个 split 用 `--planner-timeout-s 3600`。打分用 `python -m robots.robocasa.eval.validate_target50 RUNS_ROOT`。
- **effort：** leaderboard 把 RoboCasa GPT-6 59.2% 标成 low。但 HF `RLinf/RPent-memory` PR 15 写的是 "Historical evaluation used `high` reasoning effort"，所以我们用 high。
- **单独的 worktree：** Codex 在 repo root 里有完整 shell 权限。不要让它在其他 episode 正在 import 的 checkout 里运行。
- **CODEX_HOME：** Orin 上的 `~/.codex` 属于这个账号的另一位用户，不要用。
- **安装（Orin，aarch64）：** 在 RPent 的 `.venv` 里装 `openai-codex==0.163.0` 和 `openai-codex-cli-bin==0.163.0`，官方 wheel，不需要 Node。二进制在 `.venv/lib/python3.10/site-packages/codex_cli_bin/bin/codex`。
- **只用真的 Codex CLI，不要伪造 Codex 请求头。** 10-15 Sorux 返回过 403，要求使用标准 Codex 客户端、不要中转。
- LIBERO-PRO（官方 GPT-6 Astra low 92.63%）的命令见 `docs/source-en/rst_source/usage/libero.rst`，加上 `--planner codex --model gpt-6-astra --reasoning-effort low` 即可。

### 3. 我们的方法用 Codex 调 GPT（Codex senior）

在原来的 escalation 环境变量上再加：

```bash
export RPENT_ESCALATION_SENIOR_BACKEND=codex
export RPENT_CODEX_BIN=/path/to/codex_cli_bin/bin/codex
export RPENT_CODEX_TIMEOUT_S=180                    # 每个 Codex 进程，可选
```

- 每个 episode 一个 Codex thread，只追加。每次 GPT 回合只发上一次之后的新消息和新图片，所以前缀能命中缓存。
- GPT 按 JSON schema 回复 `{plan, tool_calls}`，转成普通的 tool call，由同一个 pydantic-ai 循环执行；Qwen 那一侧不变。Codex 自己的工具全部关闭。
- 回复不能用时，追加一次纠正回合；Codex 进程失败时，重试一次。仍然失败就由 Qwen 接这次请求（被否决的放弃保持放弃），等下一个事件再找 GPT。
- 图片转成全分辨率 JPEG，每回合最多 2 MiB、每个 thread 最多 8 MiB，并且总是带上最新两张。
- 日志：`escalation.jsonl` 多了 `codex_thread`、`codex_turn`、`codex_attempts`、`images_attached`；每回合发送的内容在 `codex_turns/`。

### 4. 目前的数据

**Codex senior 冒烟（10-16，LIBERO-PRO；`CodexSenior` 与本分支相同，escalation 路由是旧版本）：**
- 3/3 成功：pair 0 一次，pair 64 两次。
- GPT 输入缓存 60–73%，每回合中位数 25 s。
- 每个 GPT 回合 $0.33–0.40，非 Codex 方式是 $0.73–1.15。
- 本分支的 Codex senior 还没有做线上冒烟，只跑了离线测试。

**官方 Codex planner、GPT-6 high、RoboCasa（10-19 17:08，进行中）：** 13 个差距任务 × seed 1–2 = 26 个格子，已完成 5 个有效格子。

| 格子 | 结果 | 时长 | 回合 | GPT 输入 | 缓存 | 费用 |
|---|---|---:|---:|---:|---:|---:|
| TurnOffStove s1 | ✗ | 1,081 s | 37 | 2.41M | 97.8% | $3.31 |
| CategorizeCondiments s1 | ✗ | 1,744 s | 59 | 4.03M | 95.8% | $6.03 |
| CuttingToolSelection s1 | ✗ | 996 s | 45 | 1.64M | 94.2% | $2.83 |
| GatherTableware s1 | ✗ | 1,292 s | 29 | 2.08M | 95.7% | $3.18 |
| HeatKebabSandwich s1 | ✗ | 1,117 s | 31 | 2.18M | 94.0% | $3.63 |

- 费用按标价 $10 / $1 / $50 每百万 token（未缓存 / 缓存 / 输出）计算，不是账单金额。
- 对比：我们 10-17 的 Qwen+GPT 全量（250 回合）平均每回合 $2.46，GPT 缓存只有 25–48%。这 5 个格子在那次运行里也都失败。
- 这 13 个任务是我们和 RPent 差距最大的任务，不能代表 50 个任务的平均水平。

### 5. 已知问题

- **缓存：** Sorux 只缓存流式请求。非 Codex 客户端没有后端亲和，每个 episode 只有 0–31%；Codex CLI 是 80–99%。
- **计费：** 10-16 实测，Sorux 报告的 input token 是 6,571 + 1.5 × 实际 token，上面的费用就是按这个报告值算的。
- **Windows 路径：** GPT-6 通过 Codex 会写出 `/workspace/path_literals/w_91c7e9e44a4f2e5fc8a78a37/Users/…\…`、`/c/Users/…` 这种路径，虽然 Codex 报告的是 Linux cwd。
  - RoboCasa 的 memory 读取已修复（4779c47）。
  - 其他文件工具仍然会报错，模型通常看到报错后会改正。
  - Codex senior 的 3 次冒烟，每次的 transcript 里都出现过；同期 463 个非 Codex 的 transcript 里一次也没有。
- **基础设施错误：** Codex planner 目前出现过 2 次，都已重跑。

### 6. 测试

- Orin，RPent `.venv`，pytest 9.1.1：`tests/unit_tests` 761 passed、3 skipped，其中 12 个是新的 Codex senior 测试。
- 改动的文件用 ruff 0.15.22 检查，没有问题。

## English (same content)

### 1. What the branch contains

The commits are those in the Chinese table above:
- d8aa18b: no 50 s cap on RoboCasa tools.
- 01fe9f4: continue notes after text-only Qwen replies, and leading-zero JSON repair.
- 81e0b74: senior requests capped at the newest N images.
- 8e39136: optional plan toggle, memory-fed plan, and senior takeover.
- 3937a2b: gateway errors inside streamed responses are retried.
- 4779c47: Codex fixes, so a stream retry is no longer a failure and Windows-style memory paths resolve.
- This branch: the Codex senior.

Merging your newer 08d1d81 and 95aef95 conflicts only in `Toolkit.call` and two test files. The simulator slot re-indents the block that the 50 s cap change edits.

### 2. Official configuration: Codex planner, GPT on every turn

Use the command in section 2 above.
- **Split timeouts:** 1,800 s for atomic, 3,600 s for the composite splits. Score with `validate_target50`.
- **Effort:** use high. The leaderboard says low, but HF PR 15 says historical RoboCasa runs used high.
- **Codex setup:**
  - Run Codex from a separate worktree, because it gets a full shell in the repo root.
  - Give each episode its own `CODEX_HOME`. `~/.codex` on Orin belongs to someone else.
  - Install `openai-codex` and `openai-codex-cli-bin` 0.163.0 in RPent's `.venv`. Both are aarch64 wheels and need no Node.
  - Only use the real Codex CLI; never fake its headers.

### 3. Our method with GPT through Codex

Set `RPENT_ESCALATION_SENIOR_BACKEND=codex` and `RPENT_CODEX_BIN`.
- Each episode keeps one append-only Codex thread, and each GPT turn sends only what is new.
- GPT replies as JSON `{plan, tool_calls}`, which runs in the same tool loop.
- An unusable reply gets one correction turn, and a failed Codex process gets one retry. After that, Qwen takes the request.
- Images are sent as JPEG, within 2 MiB per turn and 8 MiB per thread.

### 4. Results so far

- **Codex senior LIBERO-PRO smoke (10-16):** 3/3 succeeded, 60–73% cached, $0.33–0.40 per GPT turn against $0.73–1.15 without Codex. It has not been smoked live on this branch.
- **Official Codex planner, GPT-6 high, on our 13 largest-gap RoboCasa tasks:**
  - 0 of 5 valid cells solved so far.
  - 94–98% cached, $2.83–6.03 per episode at list prices.
  - The same cells failed in our Qwen+GPT run, which averaged $2.46 per episode with 25–48% cached.

### 5. Known issues

- **Cache:** Sorux caches only streamed requests, and only the Codex CLI gets backend affinity.
- **Billing:** measured on 10-16, Sorux reports input as 6,571 + 1.5× the actual tokens. The costs above use those reported counts.
- **Windows paths:** GPT-6 via Codex writes Windows-style paths.
  - Memory reads are fixed; other file tools still error.
  - All 3 Codex-senior smoke transcripts had one, against 0 of 463 non-Codex transcripts.
- **Infra errors:** 2 so far with the Codex planner, both re-run.

### 6. Tests

761 passed and 3 skipped on Orin, including 12 new Codex senior tests. ruff 0.15.22 is clean on the changed files.
