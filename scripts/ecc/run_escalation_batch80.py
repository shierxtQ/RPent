#!/usr/bin/env python3
"""LIBERO-PRO 80-task batch: Qwen-led api planner + GPT escalation, shared VLA/SAM3.

8 suites ({10, goal, object, spatial} x {swap, task}) x tasks 0-9, seed 1, one attempt
per task, PARALLEL tasks at a time.  Each task runs run_escalation_smoke.sh with its own
env_server and the shared perception servers from start_shared_perception.sh.

Resumable: tasks with a result.json are skipped.  A task whose planner never started
(service/infra failure before turn 1) is relaunched up to INFRA_RETRIES times.

RUNNER_SCRIPT=run_explore_qwen.sh (with MEMORY_DIR) runs exploration cells instead.

Usage: python3 scripts/ecc/run_escalation_batch80.py RUN_DIR [--parallel 5] [--only suite:task,...] [--repeats N]
"""

import argparse
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUITES = [
    "libero_10_swap", "libero_10_task", "libero_goal_swap", "libero_goal_task",
    "libero_object_swap", "libero_object_task", "libero_spatial_swap", "libero_spatial_task",
]
TASKS = [(suite, i) for i in range(10) for suite in SUITES]
SEED = 1
INFRA_RETRIES = 2


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def prune_npz(out: Path) -> None:
    """Drop the episode's depth/point-cloud .npz artifacts (~80% of its size); KEEP_NPZ=1 keeps them."""
    if os.environ.get("KEEP_NPZ") == "1":
        return
    for path in out.rglob("*.npz"):
        if path.is_file():
            path.unlink()
    # Codex raw event streams (~10-30 MB each) duplicate run.log/transcript; KEEP_STREAM=1 keeps them.
    if os.environ.get("KEEP_STREAM") != "1":
        for path in out.glob("*.stream.jsonl"):
            path.unlink()
    for path in sorted(out.rglob("*.npz"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
    # Codex keeps a ~25 MB rollout per episode in its home; finished ones are not reused.
    sessions = Path("/mnt/user_dir/hanjinchen/lijinhao/duanyun/codex-home-rpent/sessions")
    cutoff = time.time() - 30 * 60
    for path in sessions.rglob("rollout-*.jsonl") if sessions.is_dir() else ():
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def outcome(out: Path) -> dict:
    log = out / "run.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    transcripts = sorted(out.glob("transcript_*.json"))
    tr = {}
    if transcripts:
        try:
            tr = json.loads(transcripts[-1].read_text())
        except ValueError:
            tr = {}
    stats = tr.get("stats") or {}
    recipe = re.findall(r"\[agent\] recipe: (.+)", text)
    errors = re.findall(r" E \[[^\]]+\] (.+)", text)
    return {
        "started": "=== turn 1/" in text,
        "success": bool(recipe) and "not written" not in recipe[-1],
        "finish_status": (tr.get("finish") or {}).get("status"),
        "turns": stats.get("turns_used"),
        "elapsed_s": stats.get("elapsed_s"),
        "active_elapsed_s": stats.get("active_elapsed_s"),
        "excluded_wait_s": stats.get("excluded_wait_s"),
        "api_retries": text.count("[api_retry]"),
        "tool_timeouts": text.count('"code": "tool_timeout"'),
        "planner_timed_out": "planner timed out" in text,
        "last_error": errors[-1][:300] if errors else None,
    }


def run_one(run_dir: Path, index: int, suite: str, task: int, slots: threading.Semaphore,
            rep: int | None = None, seed: int = SEED) -> None:
    out = run_dir / (f"{index:02d}_{suite}_task{task}_seed{seed}" + (f"_r{rep}" if rep is not None else ""))
    label = f"{index:02d} {suite} t{task} s{seed}" + (f" r{rep}" if rep is not None else "")
    if (out / "result.json").exists():
        return
    with slots:
        for attempt in range(INFRA_RETRIES + 1):
            out.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "SUITE": suite, "TASK": str(task), "SEED": str(seed), "OUTPUT_DIR": str(out),
                   "VLA_ENDPOINT": "http://127.0.0.1:51157", "SAM3_ENDPOINT": "http://127.0.0.1:56897"}
            started = time.time()
            with open(out / f"stdout.attempt{attempt}.log", "w") as log:
                proc = subprocess.Popen([str(ROOT / os.environ.get("RUNNER_SCRIPT", "run_escalation_smoke.sh"))], env=env, stdout=log,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                write_json(out / "launch.json", {"suite": suite, "task": task, "seed": seed, "pid": proc.pid,
                                                 "attempt": attempt, "started_at": datetime.now(timezone.utc)})
                print(f"start {label} attempt={attempt} pid={proc.pid}", flush=True)
                rc = proc.wait()
            res = {"suite": suite, "task": task, "seed": seed, "repeat": rep, "returncode": rc, "attempt": attempt,
                   "wall_s": round(time.time() - started, 1), **outcome(out)}
            if res["started"] or attempt == INFRA_RETRIES:
                break
            print(f"infra-retry {label}: {res['last_error']}", flush=True)
            for stale in out.glob("transcript_*.json"):
                stale.rename(stale.with_suffix(f".attempt{attempt}.json"))
            time.sleep(10)
        write_json(out / "result.json", res)
        prune_npz(out)
        print(f"done  {label} success={res['success']} turns={res['turns']} "
              f"retries={res['api_retries']} rc={rc}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--parallel", type=int, default=5)
    ap.add_argument("--only", default=None, help="comma list of suite:task to (re)run")
    ap.add_argument("--repeats", type=int, default=1, help="independent runs per task (same seed)")
    ap.add_argument("--seeds", default=str(SEED), help="comma list of env seeds (default 1)")
    ap.add_argument("--planner-label", default="api qwen3.6-27b (Infini) + GPT-6-astra escalation (Sorux)")
    args = ap.parse_args()
    only = None
    if args.only:
        only = {(s.split(":")[0], int(s.split(":")[1])) for s in args.only.split(",")}
        unknown = only - set(TASKS)
        if unknown:
            raise SystemExit(f"unknown tasks: {sorted(unknown)}")
    seeds = [int(x) for x in args.seeds.split(",")]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "batch.json", {
        "task_set": "LIBERO-PRO 8 suites x 10 tasks", "suites": SUITES, "seeds": seeds,
        "parallel": args.parallel, "planner": args.planner_label,
        "attempts_per_task": 1, "repeats": args.repeats, "started_at": datetime.now(timezone.utc),
        "only": sorted(only) if only else None,
    })
    slots = threading.Semaphore(args.parallel)
    threads = []
    # Repeat-major order so every task gets its first runs before any fourth.
    for seed, r in [(seed, r) for seed in seeds for r in range(args.repeats)]:
        for index, (suite, task) in enumerate(TASKS, 1):
            if only is not None and (suite, task) not in only:
                continue
            th = threading.Thread(target=run_one, args=(args.run_dir, index, suite, task, slots,
                                                        r if args.repeats > 1 else None, seed))
            th.start()
            threads.append(th)
            time.sleep(0.05)  # keep launch order roughly FIFO through the semaphore
    for th in threads:
        th.join()
    rows = [json.loads(p.read_text()) for p in sorted(args.run_dir.glob("*/result.json"))]
    write_json(args.run_dir / "summary.json", {
        "only": sorted(only) if only else None,
        "finished": len(rows), "success": sum(r["success"] for r in rows),
        "by_suite": {s: f"{sum(r['success'] for r in rows if r['suite'] == s)}/"
                        f"{sum(1 for r in rows if r['suite'] == s)}" for s in SUITES},
        "by_task": {f"{s}:{t}": f"{sum(r['success'] for r in rows if (r['suite'], r['task']) == (s, t))}/"
                                f"{sum(1 for r in rows if (r['suite'], r['task']) == (s, t))}"
                    for s, t in TASKS if any((r['suite'], r['task']) == (s, t) for r in rows)},
    })
    print(json.dumps(json.loads((args.run_dir / "summary.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()
