# RC-ECC on RoboCasa Target50: stage 1 runner

Stage 1 (edge task memory) of the edge-cloud collaborative RSI rule, on the 50 RoboCasa Target50
tasks. Qwen alone explores every task on seed 0 with RPent `--explore`, starting from the cloud
(GPT-6) memory of that task; a solved task's own audit, recipe and task-family note replace the
cloud task files; the result is frozen as a hashed corpus that stages 2 and 3 read.

Files: `stage1.py` (CLI), `common.py` (config, layout, GPU checks, RLDX servers, outcome rules,
scheduler, status; shared with stages 2-3), `explore_cell.sh` (one exploration cell),
`vla_server.sh` (one shared RLDX server), `config.example.toml`.

## What one task does

| Item | Value (config key) |
|---|---|
| Planner | Qwen alone (`RPENT_ESCALATION=0`), the Target50 launcher's Qwen-alone environment: same `RPENT_*` and `RLDX_*` values, CPU rendering, credentials sourced from a file |
| Seed / split | `stage1.seed` = 0, environment split `target` |
| Budget | `stage1.sessions` x `stage1.attempts_per_session` = 5 x 3, `stage1.session_timeout_s` = 2400 s of planner time per session (`session_timeout_s_by_split` overrides), `episode.max_turns` = 100 turns per session |
| Stop | RPent ends the cell at the first solved session; reset and an early unsolved `finish` are refused by the toolkit as usual. A solved task is never relaunched |
| Memory read | the task's own staging corpus: `global/*.md` plus that task's `<Task>_s0.json`, `<Task>_s0_recipe.jsonl`, `<Task>.md` from the cloud corpus (the evaluation boundary). RoboCasa's explore prompt does not name these files; with `stage1.list_memory` (default on) the launcher sets `RPENT_EXPLORE_LIST_MEMORY=1` and the prompt lists them, as evaluation does |
| Memory write | only the cell's inbox in its staging corpus (`--no-auto-merge-memory`); the runner publishes |

**Scenes.** The env server lives for the whole cell, seeded with 0. Each attempt draws the next
scene of seed 0's stream: a new session builds a new toolkit, whose constructor resets the
environment, and an accepted `reset` resets it again (`robots/robocasa/primitives.py`,
`toolkit.py`). Only session 1, attempt 1 is the seed-0 evaluation scene.

**Outcome of a launch** (the Target50 batch runner's rules): RPent's `result.json` decides.
`valid: true` is final (solved, or a counted failure such as `planner_timeout`). A missing or
invalid result is an infrastructure error and the task is relaunched from scratch, at most
`stage1.infra_retries` times, never counted as a task failure. Agentview frames that are per-pixel
noise mark the launch `render_corrupt`; it is stopped and relaunched on its own budget
(`stage1.render_retries`). A non-timeout planner error ends RPent's whole multi-session cell, so a
relaunch restarts at session 1 of seed 0's stream, from a pristine staging corpus (the broken
launch's inbox is kept as `try<N>.inbox`).

**Publishing** uses `robots.robocasa.memory:publish_exploration_cell` (`stage1.publish_impl`):
a solved cell with a non-empty audit and its recipe replaces the task's cloud files (archived in
`_internal/replaced/<Task>_s0/`) with its native pair and its task-family note; any other cell
keeps the cloud files and publishes no task-family note; global drafts merge by RPent's default
rules either way. A task without a valid result publishes nothing. The cloud task notes carry the
legacy layer name `scope: suite`, which RPent's merge rejects; a captured draft that copies it but
names exactly this task (robocasa, `target`, this task id) is rewritten to `scope: task-family`
(listed in `record.json` `inbox.scope_suite_normalized`), so the edge note is not lost.

## Configure

1. Sync this branch to a code directory on the GPU host (never the live RPent tree), and record
   its commit next to it:

   ```bash
   rsync -a --delete --exclude .git ./ HOST:/path/to/ecc-code/
   git rev-parse HEAD | ssh HOST 'cat > /path/to/ecc-code/CODE_COMMIT'
   git status --porcelain | ssh HOST 'cat > /path/to/ecc-code/CODE_DIRTY'
   ```

2. Copy `config.example.toml` outside the repository and fill in the machine values (paths, the
   worker base URL, the credentials file and the names of its variables, the GPUs and ports).
   Pass it with `--config` or `ECC_RC_CONFIG`. The runner only checks that the credentials file
   exists; the cell launcher sources it.

3. Run the runner with the venv's Python (the render-noise check needs numpy and Pillow; `--go`
   refuses without them) and `PYTHONDONTWRITEBYTECODE=1`. Episodes import
   `rpent` and `robots` from `paths.code_dir` (`PYTHONPATH`, `RPENT_REPO_ROOT`); the dry run and
   every launch check that.

## Run

```bash
PY=/path/to/RPent/.venv/bin/python; S=/path/to/ecc-code/scripts/ecc_robocasa/stage1.py
export ECC_RC_CONFIG=/path/to/s1.toml PYTHONDONTWRITEBYTECODE=1

$PY $S plan --run-id s1_full                     # dry run (the default): prints everything, starts nothing
# smoke = the first 2 tasks at low concurrency, inside the real run: start with 2 slots
setsid -f nohup $PY $S run --run-id s1_full --go --concurrency 2 > /tmp/s1_full.out 2>&1 < /dev/null
$PY $S status --run-id s1_full --tasks-table
# when the smoke checks below pass: full speed (read before every launch, capped by scheduler.max_slots
# and by the slots of the endpoints that are up)
echo 50 > <runs_dir>/s1_full/concurrency
```

Smoke checks on the first cells (`<run>/tasks/<Task>/try1/`, `progress.jsonl`):
- `progress.jsonl` has `server_ready` for every server on the claimed GPUs and `launch` rows;
  `endpoints` in `status` are up;
- `run.log` shows `=== turn N/100 ===` lines and `[usage]` lines, `read_text_file` calls on the
  staged cloud files, and no `agent run failed` or repeated `retry` warnings (rate limits);
- `env_server.log` says `MUJOCO_EGL_DEVICE_ID=<n> already set; keeping it` (CPU rendering) and has
  no `Fatal Python error`;
- `staging/_internal/inbox/<Task>_target_s0/wip/` gets notes after a failed attempt;
- a finished cell gets `record.json` with `status`, `sessions_used`, `attempts_used` and `qwen`.

If something is wrong: `stage1.py stop --run-id s1_full --now`, fix, remove `<run>/STOP`, and start
again (`--allow-code-change` if the code changed; the change is recorded in the manifest). The
stopped cells are relaunched and not counted. A separate smoke run (`--run-id s1_smoke --limit 2
--max-slots 1`) also works, but must finish or be stopped first: two runners with one config
compete for the same server ports. The first task, CloseBlenderLid, was not solved by Qwen alone on
any of seeds 1-5 in an earlier Target50 run, so its cell probably uses its whole budget (up to
5 x 2400 s of planner time); a smoke run with `--tasks NavigateKitchen,TurnOnElectricKettle`
(both solved on all five seeds in that run) reaches the solve-and-publish path sooner.

- **Concurrency.** Each `[[endpoints]]` entry is one shared RLDX server (its own port) with `slots`
  cells; several entries may share a GPU (measured full speed: 3 servers x 8-9 cells per GPU,
  64 GB). The config is refused when a GPU's plan (`servers.server_mb` per server +
  `servers.cell_mb` per slot) exceeds `gpus.plan_max_mb`. `scheduler.max_slots` caps the total,
  and a number in `<run>/concurrency` sets the live limit below that cap (read before every
  launch; `--concurrency N` writes it at start). A cell goes to the endpoint with the most free
  slots, and its `RC_GPU` is that server's GPU. A GPU is claimed only when it is in
  `gpus.allowed` and free in nvidia-smi and the platform CSV, and while fewer than
  `servers.max_gpus` GPUs are held; another server on a GPU the run already holds only needs room
  for one more server (`server_mb` + `fit_margin_mb`). A busy GPU's endpoints are skipped and
  checked again every `servers.retry_busy_s` (300 s, at most `servers.max_refusals` times), so a
  GPU freed later is used then (`--force-gpu N` skips the check, for a GPU you already hold).
  Servers get `servers.env` (OMP/MKL/OPENBLAS pools of 4 threads) over the episode environment.
- **RLDX servers.** A server that dies, exits while loading or is not ready in time is started
  again (after 30 s or `retry_busy_s`) if its GPU passes the same check (`servers.max_restarts`,
  default 3); the cells attached to it are stopped and relaunched as infrastructure errors, which
  do not count toward the infrastructure breaker.
- **Stop.** `stage1.py stop --run-id R` writes `<run>/STOP`: no new launches, running cells finish,
  then the runner stops its servers and exits. `--now` also stops the running cells; they are
  relaunched on the next `--go` and not counted. With no runner alive and no cell running, `stop`
  also stops this run's RLDX servers. Remove `STOP` before resuming.
- **Resume.** Run the same `run --run-id R --go` again. Cells run in their own sessions, so a
  killed runner leaves them running; the new runner adopts live cells by their output folder,
  scores the ones that ended, and adopts the running RLDX servers. It refuses to resume if the
  experiment settings, task list, cloud corpus or code changed (`--allow-code-change` records a
  code change instead). Only one runner per run root (`runner.lock`).
- **Runaway guards.** At most `max_slots` cells at once; at most
  `tasks + tasks x infra_retries + tasks` launches, not counting cells you stopped with
  `stop --now` (or `scheduler.max_total_launches`);
  `scheduler.infra_breaker` consecutive invalid launches stop new launches; an optional
  `scheduler.cost_ceiling_usd` over all launches, from the episode logs. Each guard writes `STOP`
  with its reason.
- **Cost.** Qwen tokens per launch are the last cumulative `[usage]` line of each session in
  `run.log` (stage 1 writes no `escalation.jsonl`), priced at `[pricing]` (list price). Every
  launch counts, retries included.

When every task is final the runner freezes the product (`stage1.py freeze` does it by hand,
`stage1.py verify <run>/product` recomputes the hashes). Freezing refuses when the cloud corpus
changed during the run, when a task does not load cleanly, or when a cell wrote into another
task's folder or a shared corpus (`freeze --allow-cross-task-writes` accepts and records it).
`stage1.py assemble` validates the corpus from the tasks final so far (`--out DIR` keeps a copy).

## Run layout

```
<runs_dir>/<run-id>/
  manifest.json          code commit + tree hash, config and experiment hashes, cloud corpus hash, seed, budget, tasks
  config.snapshot.json   the resolved config (no secrets: the credentials file is a path)
  progress.jsonl         every launch, adoption, end, retry, capture, guard and server event
  state/jobs/<Task>.json every launch of a task: folder, endpoint, pid, class, outcome, usage; the final decision
  servers/gpu<N>.{json,log}
  tasks/<Task>/
    staging/             the memory root the cell explored against (global + this task's cloud files + its inbox)
    try<N>/              RPent's --output-dir: run.log, result.json, transcript, sessions/session_00K/, attempts/,
                         <Task>_target_s0.json (Qwen's audit), <Task>_target_s0_recipe.jsonl
    try<N>.console.log
    product/             what is published: run/ (audit + recipe), inbox/ (drafts and wip notes)
    record.json          solved?, sessions/attempts used, first-success attempt, Qwen tokens and $, infra/render
                         retries, publish decision, file-tool paths outside the task's folder
  corpus_check.json      validation of the corpus assembled from the tasks final so far
  product/               the frozen stage-1 product (read-only)
    corpus/              memory corpus for --memory-profile local --memory-dir
    PRODUCT.json         schema rc-ecc-s1-product/1: hashes of every corpus file, corpus tree hash, code, config,
                         cloud corpus, budget, per-task status and publish decision, validation (50 tasks load)
    tasks.json           every task's record.json
    assembly.json        publish results per task, in Target50 order
    FROZEN               the corpus tree hash; written last
```

**What stages 2 and 3 consume.** `product/corpus` as a read-only `--memory-dir` (local profile),
after `stage1.py verify <run>/product` (or recomputing `corpus_tree_sha256` from
`PRODUCT.json`). The corpus is assembled from a fresh copy of the cloud corpus with every task
published in Target50 order, so it depends only on the task outcomes, not on finishing order. All
50 Target50 tasks load with no selection error; a replaced task loads only its native pair
`<Task>_target_s0.*`, its task-family note and `global/*.md`; any other task loads its cloud
files and `global/*.md`. Qwen's global drafts reach every task.

`common.py` gives stages 2-3 the same config loading, layout, manifest and hash helpers, the
`Scheduler` (jobs, endpoints, retries, guards, resume, stop file) and `Stage` interface (`jobs`,
`try_dir`, `prepare_try`, `launch`, `on_final`), the outcome and render rules, the cost from
`escalation.jsonl` or `run.log`, and the status table.

## Concurrency safety and the file-tool boundary

- Each cell writes only into its own staging corpus and output folder; all merging happens in the
  runner, one task at a time in Target50 order, into a fresh copy of the cloud corpus. RPent's own
  merge lock is not relied on.
- A staging corpus is read-only except the cell's inbox, so no cell can change a task's cloud
  memory. While the runner runs, `tasks/` has mode 0311: cells (and you) cannot list it, only open
  paths inside it; the runner sets 0755 again when it exits (after a killed runner, the next
  runner does, or `chmod 755 <run>/tasks`).
- RPent's file tools allow any path outside `<repo_root>/memory` (relative paths resolve against the
  code directory), so a cell could still read, or write into the inbox of, another task's folder,
  the cloud corpus or the runs directory if it guessed the path. Every task's `record.json` lists
  such paths (`file_tool_audit`: `cross_task_reads`, `cross_task_writes`, `code_dir_writes`,
  `outside`); `status` names the tasks; a cross-task write blocks the freeze, and a changed cloud
  corpus is caught by its hash. Residual risk: a cross-task read is detected after the fact, not
  prevented.

## Limits worth knowing before a full run

- The 100 turns and the session timeout apply per session (`rpent/cli/main.py` session loop), so a
  long task gets fewer than 3 attempts in a session; with CPU rendering an RLDX chunk is about
  1.5-2.3x slower than on the reference Jetson host (measured in our setup), which lowers that further.
- A planner error that is not a timeout ends the whole cell, and RoboCasa scores it
  `infrastructure_error`; the relaunch starts at session 1 from a pristine staging corpus, so the
  sessions already explored are discarded (they stay on disk as `try<N>/`).
- Qwen's context grows with every turn of a session; if the provider rejects a long context, that
  request error ends the cell like any other non-timeout error (not verified at $0).

## Tests

`tests/unit_tests/ecc_robocasa/` runs offline with fake cells, a fake RLDX server and a fake
nvidia-smi: scheduling over GPUs, a busy GPU, resume after a killed runner, stop, infrastructure
retries against counted failures, render noise, first-success accounting, the guards, the dry run,
publish, freeze and verify, and the launchers' command and environment. With
`ECC_RC_PORTED_LAUNCHER` (the host's Target50 cell launcher) and `ECC_RC_CONFIG` (a real config)
set, one more test compares the cell launcher's effective environment with that launcher's.

```bash
cd /path/to/ecc-code && PYTHONPATH=$PWD RPENT_REPO_ROOT=$PWD CUDA_VISIBLE_DEVICES= \
  /path/to/RPent/.venv/bin/python -m pytest -q tests/unit_tests/ecc_robocasa
```

# Stage 2: the call-rule ladder (`stage2.py`)

Stage 2 searches, per task, the cheapest rule for when Qwen calls GPT-6. Every task climbs the
ECC rule's ladder on seed 11, one hybrid evaluation episode per rung, from the rule with the least
cloud help to the most; the first rung with a valid success becomes the task's rule, and a task
whose rungs all fail validly gets the strongest rung. Files: `stage2.py` (CLI), `hybrid_episode.py`
(rules, the cell environment, the escalation evidence check and the episode record; shared with
stage 3 and the final test), `hybrid_cell.sh` (one hybrid episode).

| Item | Value (config key) |
|---|---|
| Planner | Qwen worker plus GPT-6 senior at low effort with the 0.2 share cap: the Target50 launcher's base arm (same `RPENT_*`, `RLDX_*`, CPU rendering, credentials sourced from a file) |
| Ladder | `stage2.ladder` (empty: the ECC rule's six rungs (plan, K, N) = (Qwen,12,4), (GPT-6,12,4), (Qwen,6,4), (GPT-6,6,4), (GPT-6,6,3), (GPT-6,6,2), named `c1_qwenplan_K12_N4` ... `c6_gptplan_K6_N2` as in `scripts/ecc/run_ecc_stage2.py`). "GPT-6 plan" is the plain plan turn; Qwen plan is `RPENT_ESCALATION_PLAN=0`; K is `RPENT_ESCALATION_CHECKIN`; N is `RPENT_ESCALATION_ERRORS` |
| Seed / split | `stage2.seed` = 11 (refused if equal to stage 1's seed), environment split `target`; every rung of a task plays the same seed-11 scene |
| Memory | the frozen stage-1 corpus `stage2.s1_product`/corpus, `--memory-profile local`, read-only: evaluation's RoboCasa memory manager refuses writes, the frozen files are 0444, `hybrid_cell.sh` refuses a writable corpus, and the corpus hash is checked after every launch |
| Budget | one valid episode per rung, planner timeout `stage2.timeout_s_by_split` (Target50: 1800 s atomic, 3600 s composite; a timeout is a valid failure), `episode.max_turns` = 100 |
| Invalid episodes | relaunched, at most `stage2.infra_retries` (infrastructure) and `stage2.render_retries` (rendered noise) times per rung; never a rung failure. A rung that exhausts them blocks its task: `unblock --task T` gives that rung a fresh budget, or `freeze --fallback-blocked` gives blocked tasks the fallback rung (recorded as `fallback_infra_blocked`) |
| Fallback | `stage2.fallback_rung` (0: the last rung) when every rung failed validly |

**Escalation evidence.** RPent's escalation wrapper writes a `config` row first in
`escalation.jsonl` (plan, K, N, share cap, corrections length and sha256, models). For every launch
the runner checks that row against the rung (as soon as it appears, so a wrongly launched cell is
stopped at once) and, when the launch ends, replays the wrapper's routing from the logged requests:
a plan turn by request 6 exactly when the plan turn is on, a check-in only after at least K worker
requests and never missing when one was due below the share cap, no takeover, the expected
`corrections_chars` on every senior row, and the worker and senior model names. A mismatch makes
the launch an infrastructure error and writes `STOP` (running cells drain and are checked too).
`stage2.require_config_row` (default true) refuses code without the config row, since it is the
only record of N. Each episode records how often the errors trigger fired.

**Scheduling.** One job per (task, rung) on `common.Scheduler`: all tasks run in parallel, a task's
rungs one after another (rung k+1 only after rung k failed validly; with fewer slots than tasks,
every task's rung k goes before any rung k+1), infrastructure retries first, render retries last.
Slots, the live `concurrency` file, the launch ceiling (tasks x rungs + tasks x infra retries +
tasks, or `scheduler.max_total_launches`), the infrastructure breaker, the cost ceiling, `STOP`,
resume and adoption of live cells work as in stage 1.

```bash
PY=/path/to/RPent/.venv/bin/python; S=/path/to/ecc-code/scripts/ecc_robocasa/stage2.py
export ECC_RC_CONFIG=/path/to/config_s23.toml PYTHONDONTWRITEBYTECODE=1
$PY /path/to/ecc-code/scripts/ecc_robocasa/stage1.py verify /path/to/runs/s1_full/product   # must print "ok": true
$PY $S plan --run-id s2_full                    # dry run: S1 product verified, ladder, first launch; starts nothing
setsid -f nohup $PY $S run --run-id s2_full --go --concurrency 2 > /path/to/s2_full.out 2>&1 < /dev/null
$PY $S status --run-id s2_full --tasks-table    # progress bar, per split: decided/running/queued/blocked, rungs, $
echo 50 > /path/to/runs/s2_full/concurrency     # full speed once the first episodes look right
$PY $S stop --run-id s2_full [--now]            # drain (or stop the running cells: relaunched, not counted)
$PY $S verify /path/to/runs/s2_full/product     # recompute the frozen rules' hashes
$PY $S rules /path/to/runs/s2_full/product      # one line per task, and the summary
```

```
<runs_dir>/<run-id>/
  manifest.json          stage s2: code, config and experiment hashes, the S1 product's corpus and PRODUCT.json hashes,
                         seed, ladder, budget, tasks
  state/jobs/<Task>__r<k>.json   every launch of one rung (class, outcome, usage, evidence); skipped/blocked marks
  tasks/<Task>/
    r<k>/try<n>/         RPent's --output-dir (run.log, result.json, escalation.jsonl, transcript, states)
    r<k>/try<n>.console.log, r<k>/try<n>.infra (a runner-forced infrastructure cause, e.g. an evidence mismatch)
    r<k>/episode.json    the rung's record: valid/success/termination, turns, GPT-6 calls by reason, errors trigger
                         count, vetoes, Qwen and GPT-6 tokens and list-price $, wall time, evidence, file-tool audit
    rule.json            the task's rule once decided; blocked.json when a rung has no valid result
  product/               the frozen stage-2 product (read-only)
    rules/<Task>.json    schema rc-ecc-s2-rule/1: run_ecc_stage2.py's RULES_DIR fields (task, suite, task_id, search_seed,
                         rule {gpt_plan, checkin_K, errors_N}, config, solved_in_search, episodes,
                         gpt_calls_in_search, history[{config, success, gpt_calls, turns, finish_status, last_error}],
                         written_at) plus split, rung, rule_source, launches, GPT-6 calls by reason, errors trigger
                         counts, tokens, $, per-rung details and the S1 corpus hash
    RULES.json           schema rc-ecc-s2-product/1: hashes of every rule file, the rules tree hash, code, config,
                         S1 product, per-task summary, chosen-config counts per split
    episodes.json        every rung's record, for the paper
    FROZEN               the rules tree hash; written last
```

**What stage 3 and the test read.** `product/rules/<Task>.json` (field `rule`) after
`stage2.py verify <run>/product`; in Python, `stage2.load_rules(<run>/product)` verifies and
returns `{task: Rule}`. `gpt_calls` counts GPT-6 calls that returned (`scripts/ecc/run_ecc_stage2.py` also
counted senior rows that ended in an error; those are `gpt_errors` in `episode.json`).

# Stage 3: the cloud correction memory (`stage3.py`)

Stage 3 builds, per task, the correction memory GPT-6 keeps for that task. Every task runs hybrid
episodes on seed 12 under its frozen stage-2 rule, with the frozen stage-1 corpus read-only: up to
`stage3.sessions` x `stage3.attempts_per_session` = 5 x 3 = 15 independent single-shot episodes
(labels `s1_a1` .. `s5_a3`, as in `scripts/ecc/run_ecc_stage3.py`), stopping at the first valid
success. Files: `stage3.py` (CLI), `reflect.py` (digest, prompt, reply parsing and entry
bookkeeping, ported from `scripts/ecc/run_ecc_stage3.py`), `reflect_call.sh` and
`reflect_client.py` (one GPT-6 reflection call).

| Item | Value (config key) |
|---|---|
| Episodes | `hybrid_cell.sh` with the task's rule (`stage3.s2_product`/rules/`<Task>.json`, verified with `stage2.verify` and searched on the same stage-1 corpus), planner timeout `stage3.timeout_s_by_split` (default Target50) |
| Correction memory | before each episode the task's memory is frozen as `corrections_used.md` (0444) and passed as `RPENT_CLOUD_CORRECTIONS` (appended to every senior note, never shown to Qwen); its existence, sha256 and size are checked and logged before every launch, the cell launcher checks the sha256 again, and the escalation log's config row and every senior row must show its length |
| Reflection | after every valid episode, one offline GPT-6 call at the senior's model and effort (`stage3.reflection`); skipped when the episode has no senior turn, or no planner turn in the transcript or `run.log`; a content-filter refusal is recorded as skipped; other failures are retried (`max_calls`), then the run stops |
| Bookkeeping | as `run_ecc_stage3.apply`: an entry counts as used only when listed in `applied`; success verifies; `fails_to_drop` failures without a success drop an entry; a revision resets its failures; at most `max_entries` entries of `field_chars` per field |
| RoboCasa changes | tool results show their outcome first (skill, `ok`/`status`/`error`, `task_progress`, ...) before anything is clipped; an episode cut by the planner timeout is rebuilt from `run.log` with robot results from `states.json`; the prompt describes RoboCasa and names the termination reason |
| Invalid episodes | relaunched under `stage3.infra_retries` / `stage3.render_retries`, never reflected on and never counted; a task whose episode exhausts them stops (`requeue --task T` reruns it) |
| Cost guard | `stage3.cost_ceiling_usd` over episodes and reflections (else `scheduler.cost_ceiling_usd`) |

```bash
S=/path/to/ecc-code/scripts/ecc_robocasa/stage3.py
$PY /path/to/ecc-code/scripts/ecc_robocasa/stage2.py verify /path/to/runs/s2_full/product   # "ok": true
$PY $S plan --run-id s3_full                     # dry run: inputs verified, rules per task, first launch
setsid -f nohup $PY $S run --run-id s3_full --go --concurrency 2 > /path/to/s3_full.out 2>&1 < /dev/null
$PY $S status --run-id s3_full --tasks-table     # progress bar, per split, reflections, entries, $
$PY $S verify /path/to/runs/s3_full/product      # recompute the frozen memories' hashes
$PY $S reflect-only EPISODE_DIR                  # the digest and prompt of one finished episode (no call)
```

The product is `product/final/<Task>.md` (verified entries only, rendered as `run_ecc_stage3.render`;
empty when there are none) and `<Task>.json`, with `PRODUCT.json` (schema `rc-ecc-s3-product/1`).

# The final test (`final_test.py`)

`final_test.py` runs named arms (`[final_test.arms.<name>]`: planner hybrid, qwen or gpt6, memory s1 /
cloud / a path, rule s2 or fixed plan/K/N, corrections none or s3) over task x seed x repeat cells,
with the arms of a cell launched next to each other in a seeded order. Every input product is
verified with its stage's `verify`, checked to be built on the others, and copied read-only into
the run. Episodes start through `hybrid_cell.sh` (`ECC_RC_ESCALATION=0` turns the senior off for
Qwen-alone arms, as the Target50 launcher's `ABL_ESCALATION=0`) and get stage 2's evidence check.
`score` reports task-weighted success per split, GPT-6 calls by reason, how often the errors
trigger fired, cost per episode and per success, and paired tests (McNemar on cells, a sign test
over tasks, a task-clustered sign-flip test, a bootstrap interval). `final_test.py example-config`
prints a `[final_test]` section.

A `gpt6` arm runs the senior's model alone on every turn: `ECC_RC_SENIOR_ALONE=1` makes
`hybrid_cell.sh` start RPent's api planner with `senior.model` at `senior.reasoning_effort`, the
senior's base URL and key, and no escalation wrapper. `max_images` (default 40) becomes
`RPENT_MAX_HISTORY_IMAGES`, since the gateway rejects requests above 64-80 images and the 4 MiB
history budget keeps up to about 100 RoboCasa frames; `reasoning_ids = false` (the default) sets
`RPENT_SEND_REASONING_IDS=0`, as the senior's requests. Its evidence check wants no escalation log,
the model and effort in `result.json` and both settings in `run.log`. Its GPT-6 calls are every
returned request (transcript stats, or `run.log` after a planner timeout), priced at the senior's
list price in the records and the cost guard.

`final_test.share_own_gpus = true` lets the runner claim a GPU that holds only this machine's ECC
processes (started from `paths.venv`) when its free memory covers this run's servers and slots
on it plus `servers.fit_margin_mb`; until then the servers are tried again every
`servers.retry_busy_s` and the runner waits instead of exiting.

# Chaining the stages (`chain.py`)

`chain.py wait-product --stage s1|s2|s3 --run-root RUN` returns once `RUN/product` is frozen and
verifies, and gives up when the run's runner ended without one; `chain.py wait-idle --run-root RUN`
waits until the run's runner and RLDX servers are gone (stages 2 and 3 share ports). `chain.py gate --run-root RUN
--since ISO` follows a stage-2/3 run started at `ISO` at low concurrency: it passes when the first
two jobs to finish are final with a valid result and a matching escalation log (stage 3: and their
reflections are written), and one GPT-6 call (stage 3: and one reflection reply) returned; it fails
on a STOP file, an infrastructure or stopped launch, an evidence mismatch, a GPT-6 call that ended
in an error, a failed reflection, or a runner that exits. If the smoke episodes needed no
GPT-6 call it waits for one (`--senior-wait-s`), then passes with a warning or fails
(`--on-no-senior`). It never changes the run; the caller raises `RUN/concurrency`.
