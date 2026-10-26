#!/usr/bin/env python3
"""ECC stage 2 (ecc_rule.md): per-task search for the cheapest cloud-call rule.

Api Qwen + GPT escalation, one episode per rule config, configs tried from the weakest
(least GPT) to the strongest; a task stops at its first success, or after the strongest
config fails. Task memory is the stage-1 corpus (read-only, evaluation prompts).

  1 Qwen plan  K=12 N=4      4 GPT plan  K=6  N=4
  2 GPT plan   K=12 N=4      5 GPT plan  K=6  N=3
  3 Qwen plan  K=6  N=4      6 GPT plan  K=6  N=2

Each task's choice is written to RULES_DIR/<suite>_t<task>.json (the call-rule memory):
the first successful config, or config 6 when none succeeded.
Resumable: episodes with a result.json are not rerun.

Usage: python3 scripts/ecc/run_ecc_stage2.py RUN_DIR RULES_DIR MEMORY_DIR [--parallel 8] [--seed 2] [--only suite:task,...]
"""

import argparse
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from run_escalation_batch80 import INFRA_RETRIES, ROOT, TASKS, outcome, prune_npz, write_json

CONFIGS = [  # (GPT plan, K, N)
    (False, 12, 4), (True, 12, 4), (False, 6, 4), (True, 6, 4), (True, 6, 3), (True, 6, 2),
]


def config_name(i: int, cfg) -> str:
    plan, k, n = cfg
    return f"c{i + 1}_{'gptplan' if plan else 'qwenplan'}_K{k}_N{n}"


def gpt_calls(out: Path) -> int:
    path = out / "escalation.jsonl"
    if not path.exists():
        return 0
    return sum(json.loads(line).get("role") == "senior" for line in path.open() if line.strip())


def run_episode(out: Path, suite: str, task: int, seed: int, cfg, memory_dir: str, label: str) -> dict:
    if (out / "result.json").exists():
        return json.loads((out / "result.json").read_text())
    plan, k, n = cfg
    for attempt in range(INFRA_RETRIES + 1):
        out.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "SUITE": suite, "TASK": str(task), "SEED": str(seed), "OUTPUT_DIR": str(out),
               "VLA_ENDPOINT": "http://127.0.0.1:51157", "SAM3_ENDPOINT": "http://127.0.0.1:56897",
               "MEMORY_DIR": memory_dir, "RPENT_SHARED_SIM": "1",
               "RPENT_ESCALATION_PLAN": "1" if plan else "0",
               "RPENT_ESCALATION_CHECKIN": str(k), "RPENT_ESCALATION_ERRORS": str(n)}
        env.pop("PLANNER_MODE", None)  # default mode: api Qwen worker + GPT escalation
        started = time.time()
        with open(out / f"stdout.attempt{attempt}.log", "w") as log:
            proc = subprocess.Popen([str(ROOT / "run_escalation_smoke.sh")], env=env, stdout=log,
                                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            write_json(out / "launch.json", {"suite": suite, "task": task, "seed": seed, "pid": proc.pid,
                                             "config": list(cfg), "attempt": attempt,
                                             "started_at": datetime.now(timezone.utc)})
            print(f"start {label} attempt={attempt} pid={proc.pid}", flush=True)
            rc = proc.wait()
        res = {"suite": suite, "task": task, "seed": seed, "config": {"gpt_plan": plan, "K": k, "N": n},
               "returncode": rc, "attempt": attempt, "wall_s": round(time.time() - started, 1), **outcome(out)}
        if res["started"] or attempt == INFRA_RETRIES:
            break
        print(f"infra-retry {label}: {res['last_error']}", flush=True)
        for stale in out.glob("transcript_*.json"):
            stale.rename(stale.with_suffix(f".attempt{attempt}.json"))
        time.sleep(10)
    res["gpt_calls"] = gpt_calls(out)
    write_json(out / "result.json", res)
    prune_npz(out)
    return res


def search_task(run_dir: Path, rules_dir: Path, index: int, suite: str, task: int, seed: int,
                memory_dir: str, slots: threading.Semaphore) -> None:
    tag = f"{suite.replace('libero_', '')}_t{task}"
    rule_path = rules_dir / f"{tag}.json"
    if rule_path.exists():
        return
    task_dir = run_dir / f"{index:02d}_{suite}_task{task}_seed{seed}"
    history = []
    with slots:
        for i, cfg in enumerate(CONFIGS):
            name = config_name(i, cfg)
            res = run_episode(task_dir / name, suite, task, seed, cfg, memory_dir, f"{index:02d} {tag} {name}")
            history.append({"config": name, "success": res["success"], "gpt_calls": res["gpt_calls"],
                            "turns": res.get("turns"), "finish_status": res.get("finish_status"),
                            "last_error": res.get("last_error")})
            print(f"done  {index:02d} {tag} {name} success={res['success']} gpt_calls={res['gpt_calls']}", flush=True)
            if res["success"]:
                break
    chosen = next((h for h in history if h["success"]), None)
    i = [config_name(j, c) for j, c in enumerate(CONFIGS)].index(chosen["config"]) if chosen else len(CONFIGS) - 1
    plan, k, n = CONFIGS[i]
    rule = {"task": tag, "suite": suite, "task_id": task, "search_seed": seed,
            "rule": {"gpt_plan": plan, "checkin_K": k, "errors_N": n}, "config": config_name(i, CONFIGS[i]),
            "solved_in_search": chosen is not None, "episodes": len(history),
            "gpt_calls_in_search": sum(h["gpt_calls"] for h in history), "history": history,
            "written_at": datetime.now(timezone.utc)}
    task_dir.mkdir(parents=True, exist_ok=True)
    write_json(task_dir / "rule.json", rule)
    write_json(rule_path, rule)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("rules_dir", type=Path)
    ap.add_argument("memory_dir")
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma list of suite:task")
    args = ap.parse_args()
    only = None
    if args.only:
        only = {(s.split(":")[0], int(s.split(":")[1])) for s in args.only.split(",")}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.rules_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "batch.json", {
        "stage": "ECC stage 2 call-rule search", "seed": args.seed, "parallel": args.parallel,
        "configs": [config_name(i, c) for i, c in enumerate(CONFIGS)], "memory_dir": args.memory_dir,
        "rules_dir": str(args.rules_dir), "started_at": datetime.now(timezone.utc),
        "only": sorted(only) if only else None})
    slots = threading.Semaphore(args.parallel)
    threads = []
    for index, (suite, task) in enumerate(TASKS, 1):
        if only is not None and (suite, task) not in only:
            continue
        th = threading.Thread(target=search_task, args=(args.run_dir, args.rules_dir, index, suite, task,
                                                         args.seed, args.memory_dir, slots))
        th.start()
        threads.append(th)
        time.sleep(0.05)
    for th in threads:
        th.join()
    rules = [json.loads(p.read_text()) for p in sorted(args.rules_dir.glob("*.json"))]
    by_config = {}
    for r in rules:
        by_config[r["config"]] = by_config.get(r["config"], 0) + 1
    write_json(args.run_dir / "summary.json", {
        "tasks": len(rules), "solved_in_search": sum(r["solved_in_search"] for r in rules),
        "episodes": sum(r["episodes"] for r in rules),
        "gpt_calls_in_search": sum(r["gpt_calls_in_search"] for r in rules),
        "chosen_config_counts": dict(sorted(by_config.items()))})
    print((args.run_dir / "summary.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
