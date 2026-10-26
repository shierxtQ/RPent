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
