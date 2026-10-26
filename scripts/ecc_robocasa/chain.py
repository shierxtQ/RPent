#!/usr/bin/env python3
# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Chain the RC-ECC stages: wait for a frozen product, then gate a smoke before full speed.

    chain.py wait-product --stage s1|s2|s3 --run-root RUN [--poll-s 60] [--grace-s 180] [--timeout-s 0]
    chain.py wait-idle --run-root RUN [--timeout-s 900]
    chain.py gate --run-root RUN --since ISO [--jobs 2] [--senior-wait-s 2700]
                  [--on-no-senior warn|fail] [--poll-s 30] [--timeout-s 10800]

``wait-product`` returns once ``RUN/product`` is frozen and the stage's own
``verify`` passes (exit 0). It gives up (exit 3) when the run's runner, once
seen, is gone for ``grace_s`` without a frozen product, or when the frozen
product does not verify. ``wait-idle`` waits until no runner holds the run and
none of its RLDX servers runs (the next stage reuses their ports and GPUs).

``gate`` follows a stage-2 or stage-3 run started at ``since`` with a low
concurrency. It passes (exit 0) when the first ``jobs`` jobs to finish since
then are final with a valid result and a matching escalation log (stage 3:
their reflections are written too), and at least one GPT-6 call returned in
the run (stage 3: also one reflection reply), which proves the senior's
credentials and route. If the smoke episodes needed no GPT-6 call, it keeps
waiting for one up to ``senior_wait_s`` and then passes with a warning
(``--on-no-senior warn``) or fails. It fails (exit 4) on a STOP file, a
launch since ``since`` that is an infrastructure error or was stopped, a
final smoke job without a valid result, an evidence mismatch, a GPT-6 call that ended in an
error, a failed reflection, or a runner that exits; exit 5 after
``timeout_s``. It never changes the run: the caller raises the concurrency.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecc_robocasa import common as C  # noqa: E402
from scripts.ecc_robocasa import hybrid_episode as H  # noqa: E402

PASS, GONE, FAIL, TIMEOUT = 0, 3, 4, 5


def say(text: str) -> None:
    print(f"{C.now_iso()} {text}", flush=True)


def verify_product(stage: str, product: Path) -> dict[str, Any]:
    if stage == "s1":
        from scripts.ecc_robocasa import stage1 as S

        return S.verify(product)
    if stage == "s2":
        from scripts.ecc_robocasa import stage2 as S

        return S.verify(product)
    from scripts.ecc_robocasa import stage3 as S

    return S.verify(product)


def _epoch(value: str | None) -> float:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def progress_rows(layout: C.RunLayout, since: float = 0.0) -> list[dict[str, Any]]:
    rows, _ = H.read_rows(layout.progress)
    return [r for r in rows if _epoch(r.get("time")) >= since]


def job_states(layout: C.RunLayout) -> dict[str, dict[str, Any]]:
    states = {}
    if layout.jobs_dir.is_dir():
        for path in sorted(layout.jobs_dir.glob("*.json")):
            state = C.read_json(path)
            if isinstance(state, dict) and state.get("job_id"):
                states[state["job_id"]] = state
    return states


def runner_alive(layout: C.RunLayout) -> bool:
    """Whether the run's runner lives: the pid in ``runner.lock``, alive, naming this run.

    The lock itself is never taken here (``common.lock_holder`` takes it for a
    moment), so a poller can never make a starting runner fail its own lock.
    """
    try:
        text = layout.lock.read_text().strip()
    except OSError:
        return False
    if not text.isdigit():
        return False
    pid = int(text)
    return C.pid_alive(pid) and C.cmdline_has(pid, layout.root.name)


# ---------------------------------------------------------------------------
# wait-product
# ---------------------------------------------------------------------------


def wait_product(
    stage: str,
    run_root: Path,
    *,
    poll_s: float = 60.0,
    grace_s: float = 180.0,
    timeout_s: float = 0.0,
    clock=time.time,
    sleep=time.sleep,
) -> int:
    layout = C.RunLayout(run_root)
    product = layout.product
    started, seen, gone_since, bad_verifies, last_note = clock(), False, None, 0, 0.0
    while True:
        if (product / "FROZEN").exists():
            check = verify_product(stage, product)
            if check.get("ok"):
                say(f"{stage} product frozen and verified: {product}")
                print(json.dumps(check, indent=2, default=str), flush=True)
                return PASS
            bad_verifies += 1
            if bad_verifies >= 3:
                say(
                    f"{stage} product at {product} is frozen but does not verify: {check}"
                )
                return GONE
        elif runner_alive(layout):
            seen, gone_since = True, None
        elif seen or layout.manifest.exists():
            gone_since = gone_since or clock()
            if clock() - gone_since >= grace_s:
                ends = [
                    r
                    for r in progress_rows(layout)
                    if r.get("event") in ("end", "stopped")
                ]
                say(
                    f"{stage} runner at {run_root} ended without a frozen product "
                    f"(last: {json.dumps(ends[-2:], default=str)[:600]})"
                )
                return GONE
        if timeout_s and clock() - started > timeout_s:
            say(f"gave up waiting for the {stage} product after {timeout_s:.0f} s")
            return TIMEOUT
        if clock() - last_note >= 900:
            states = job_states(layout)
            final = sum(1 for s in states.values() if s.get("final"))
            say(
                f"waiting for the {stage} product: runner {'running' if runner_alive(layout) else 'not running'}, "
                f"{final}/{len(states)} jobs final"
            )
            last_note = clock()
        sleep(poll_s)


def servers_alive(layout: C.RunLayout) -> list[str]:
    """This run's RLDX servers that still run (their state files under ``servers/``)."""
    alive = []
    if layout.servers_dir.is_dir():
        for path in sorted(layout.servers_dir.glob("*.json")):
            info = C.read_json(path) or {}
            if (
                info.get("pid")
                and C.pid_alive(int(info["pid"]))
                and not info.get("stopped_at")
            ):
                alive.append(path.stem)
    return alive


def wait_idle(
    run_root: Path,
    *,
    poll_s: float = 10.0,
    timeout_s: float = 900.0,
    clock=time.time,
    sleep=time.sleep,
) -> int:
    """Wait until no runner holds the run and none of its RLDX servers runs (ports and GPUs free)."""
    layout = C.RunLayout(run_root)
    began = clock()
    while True:
        busy = (["runner"] if runner_alive(layout) else []) + servers_alive(layout)
        if not busy:
            say(f"{run_root}: no runner and no RLDX server left")
            return PASS
        if clock() - began > timeout_s:
            say(f"{run_root} still busy after {timeout_s:.0f} s: {busy}")
            return TIMEOUT
        sleep(poll_s)


# ---------------------------------------------------------------------------
# gate
# ---------------------------------------------------------------------------


def senior_rows(try_dir: Path) -> tuple[int, list[str], dict[str, Any] | None]:
    """(GPT-6 calls that returned, error texts, the first GPT-6 row) of one launch."""
    rows, _ = H.read_rows(try_dir / "escalation.jsonl")
    ok, errors, first = 0, [], None
    for row in rows:
        if row.get("role") != "senior":
            continue
        first = first or row
        if "error" in row:
            errors.append(f"request {row.get('request')}: {str(row['error'])[:300]}")
        else:
            ok += 1
    return ok, errors, first


def launch_checks(
    stage: str, state: Mapping[str, Any], entry: Mapping[str, Any]
) -> dict[str, Any]:
    """What the smoke shows for one launch (console line, config row, run log, first GPT-6 row)."""
    try_dir = Path(entry["dir"])
    console = try_dir.with_name(f"{try_dir.name}.console.log")
    try:
        line = next(
            (
                x
                for x in console.read_text(errors="replace").splitlines()
                if x.startswith("hybrid_cell: ")
            ),
            None,
        )
    except OSError:
        line = None
    rows, _ = H.read_rows(try_dir / "escalation.jsonl")
    try:
        log = (try_dir / "run.log").read_text(errors="replace")
    except OSError:
        log = ""
    ok, errors, first = senior_rows(try_dir)
    out = {
        "job": state["job_id"],
        "try": entry.get("n"),
        "class": entry.get("class"),
        "status": entry.get("status"),
        "termination": (entry.get("outcome") or {}).get("termination_reason"),
        "success": (entry.get("outcome") or {}).get("success"),
        "console": line,
        "config_row": (
            rows[0].get("config")
            if rows and isinstance(rows[0].get("config"), dict)
            else None
        ),
        "turns": log.count("=== turn "),
        "usage_lines": log.count("[usage] "),
        "senior_ok": ok,
        "senior_errors": errors[:3],
        "first_senior": {
            k: first.get(k) for k in ("request", "reason", "model", "error")
        }
        if first
        else None,
    }
    if stage == "s2":
        out["evidence_ok"] = (entry.get("evidence") or {}).get("ok")
        out["evidence_problems"] = (entry.get("evidence") or {}).get("problems")
    else:
        evidence = (
            C.read_json(try_dir.parent / f"try{entry.get('n')}.evidence.json") or {}
        )
        out["evidence_ok"] = evidence.get("ok")
        out["evidence_problems"] = evidence.get("problems")
        out["corrections_used"] = (try_dir.parent / "corrections_used.md").is_file()
    cause = Path(f"{try_dir}.infra")
    if cause.exists():
        out["cause"] = cause.read_text(errors="replace").strip()[:300]
    return out


def reflection_view(entry_dir: Path) -> dict[str, Any]:
    record = C.read_json(entry_dir / "reflection.json")
    rstate = C.read_json(entry_dir / "reflection" / "state.json") or {}
    return {
        "status": (record or {}).get("status"),
        "skipped": (record or {}).get("skipped"),
        "call_status": rstate.get("status"),
        "calls": rstate.get("calls"),
        "last_error": rstate.get("last_error"),
        "replied": bool(record) and record.get("status") in ("applied", "unparsed"),
    }


def gate_once(
    layout: C.RunLayout, stage: str, since: float, n_jobs: int
) -> dict[str, Any]:
    """One look at the run: smoke jobs, launches, GPT-6 calls, reflections, problems.

    The smoke jobs are the first ``n_jobs`` jobs that became final after
    ``since`` (the low-concurrency start), so a long first episode does not
    hold the run at low concurrency; every launch since ``since`` must be
    free of infrastructure errors, stops and evidence problems.
    """
    states = job_states(layout)
    rows = progress_rows(layout, since)
    started = [r for r in rows if r.get("event") == "start"]
    launched = {r.get("job") for r in rows if r.get("event") == "launch"}
    smoke: list[str] = []
    for row in rows:
        job = row.get("job")
        if row.get("event") == "final" and job in launched and job not in smoke:
            smoke.append(job)
    smoke = smoke[:n_jobs]
    problems: list[str] = []
    if layout.stop.exists():
        problems.append(
            f"STOP: {json.dumps(C.read_json(layout.stop), default=str)[:400]}"
        )
    for row in rows:
        if row.get("event") == "evidence_mismatch":
            problems.append(
                f"evidence mismatch {row.get('job')}: {row.get('problems')}"
            )
    senior_ok, reflections_replied = 0, 0
    checks, views = [], {}
    for jid, state in states.items():
        for entry in state.get("tries") or []:
            if _epoch(entry.get("launched_at")) < since:
                continue
            ok, errors, _ = senior_rows(Path(entry["dir"]))
            senior_ok += ok
            problems.extend(
                f"GPT-6 call failed in {jid} try {entry.get('n')} ({e})"
                for e in errors[:1]
            )
            if entry.get("status") == "done" and entry.get("class") in (
                "infra",
                "killed",
            ):
                cause = Path(f"{entry['dir']}.infra")
                why = (
                    cause.read_text(errors="replace").strip()[:300]
                    if cause.exists()
                    else (entry.get("outcome") or {}).get("termination_reason")
                )
                problems.append(
                    f"launch {jid} try {entry.get('n')} is {entry['class']} ({why})"
                )
        if stage == "s3" and state.get("tries"):
            view = reflection_view(Path(state["tries"][-1]["dir"]).parent)
            reflections_replied += int(view["replied"])
            if view["call_status"] == "failed":
                problems.append(f"reflection of {jid} failed: {view['last_error']}")
            views[jid] = view
    done = bool(smoke) and len(smoke) == n_jobs
    for jid in smoke:
        state = states.get(jid) or {}
        for entry in state.get("tries") or []:
            check = launch_checks(stage, state, entry)
            checks.append(check)
            if (
                entry.get("status") == "done"
                and check["class"] == "valid"
                and check["evidence_ok"] is not True
            ):
                problems.append(
                    f"smoke launch {jid}: escalation evidence {check['evidence_problems']}"
                )
        final = state.get("final")
        if not final:
            done = False
            continue
        if final.get("class") != "valid":
            problems.append(f"smoke job {jid} ended {final.get('class')}")
        if stage == "s3":
            view = views.get(jid) or {}
            if view.get("status") is None:
                done = False
    return {
        "runner_started": bool(started),
        "smoke": smoke,
        "smoke_done": done,
        "senior_ok": senior_ok,
        "reflections_replied": reflections_replied,
        "problems": problems,
        "checks": checks,
        "reflections": {j: views.get(j) for j in smoke} if stage == "s3" else None,
        "launches": sum(1 for r in rows if r.get("event") == "launch"),
    }


def gate(
    run_root: Path,
    *,
    since: float,
    n_jobs: int = 2,
    senior_wait_s: float = 2700.0,
    on_no_senior: str = "warn",
    poll_s: float = 30.0,
    timeout_s: float = 10800.0,
    start_wait_s: float = 600.0,
    clock=time.time,
    sleep=time.sleep,
) -> int:
    layout = C.RunLayout(run_root)
    began, done_at, last_note = clock(), None, 0.0
    while True:
        manifest = C.read_json(layout.manifest) or {}
        stage = manifest.get("stage")
        view = (
            gate_once(layout, stage, since, n_jobs) if stage in ("s2", "s3") else None
        )
        alive = runner_alive(
            layout
        )  # after the view: a start row implies the lock was taken
        if view and view["problems"]:
            say("SMOKE FAILED: " + " | ".join(view["problems"][:6]))
            print(json.dumps(view, indent=2, default=str), flush=True)
            return FAIL
        if view and view["smoke_done"]:
            done_at = done_at or clock()
            need = ["a GPT-6 call that returned"] if view["senior_ok"] < 1 else []
            if stage == "s3" and view["reflections_replied"] < 1:
                need.append("a reflection reply")
            if not need:
                say(
                    f"SMOKE PASSED: {view['smoke']} valid with matching escalation logs; "
                    f"GPT-6 calls returned: {view['senior_ok']}"
                    + (
                        f"; reflection replies: {view['reflections_replied']}"
                        if stage == "s3"
                        else ""
                    )
                )
                print(json.dumps(view, indent=2, default=str), flush=True)
                return PASS
            if clock() - done_at >= senior_wait_s:
                text = f"smoke episodes passed, but still no {' and no '.join(need)} after {senior_wait_s:.0f} s"
                print(json.dumps(view, indent=2, default=str), flush=True)
                if on_no_senior == "warn":
                    say(f"SMOKE PASSED WITH WARNING: {text}")
                    return PASS
                say(f"SMOKE FAILED: {text}")
                return FAIL
        if not (view and view["runner_started"]):
            if clock() - began > start_wait_s:
                say(f"no runner started at {run_root} within {start_wait_s:.0f} s")
                return FAIL
        elif not alive:
            say("the runner exited before the smoke passed")
            print(json.dumps(view, indent=2, default=str), flush=True)
            return FAIL
        if clock() - began > timeout_s:
            say(f"SMOKE TIMEOUT after {timeout_s:.0f} s")
            if view:
                print(json.dumps(view, indent=2, default=str), flush=True)
            return TIMEOUT
        if clock() - last_note >= 300 and view:
            say(
                f"smoke: jobs {view['smoke']} done={view['smoke_done']} launches={view['launches']} "
                f"GPT-6 returned={view['senior_ok']}"
                + (
                    f" reflection replies={view['reflections_replied']}"
                    if stage == "s3"
                    else ""
                )
            )
            last_note = clock()
        sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("wait-product", help="wait for a frozen, verified product")
    p.add_argument("--stage", choices=("s1", "s2", "s3"), required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--poll-s", type=float, default=60.0)
    p.add_argument("--grace-s", type=float, default=180.0)
    p.add_argument("--timeout-s", type=float, default=0.0)
    p = sub.add_parser(
        "wait-idle", help="wait until a run's runner and RLDX servers are gone"
    )
    p.add_argument("--run-root", required=True)
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--timeout-s", type=float, default=900.0)
    p = sub.add_parser("gate", help="pass or fail the smoke of a stage-2/3 run")
    p.add_argument("--run-root", required=True)
    p.add_argument("--since", required=True, help="ISO time just before the launch")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--senior-wait-s", type=float, default=2700.0)
    p.add_argument("--on-no-senior", choices=("warn", "fail"), default="warn")
    p.add_argument("--poll-s", type=float, default=30.0)
    p.add_argument("--timeout-s", type=float, default=10800.0)
    p.add_argument("--start-wait-s", type=float, default=600.0)
    args = ap.parse_args(argv)
    if args.cmd == "wait-product":
        return wait_product(
            args.stage,
            Path(args.run_root),
            poll_s=args.poll_s,
            grace_s=args.grace_s,
            timeout_s=args.timeout_s,
        )
    if args.cmd == "wait-idle":
        return wait_idle(
            Path(args.run_root), poll_s=args.poll_s, timeout_s=args.timeout_s
        )
    since = _epoch(args.since)
    if not since:
        raise SystemExit(f"--since {args.since!r} is not an ISO time")
    return gate(
        Path(args.run_root),
        since=since,
        n_jobs=args.jobs,
        senior_wait_s=args.senior_wait_s,
        on_no_senior=args.on_no_senior,
        poll_s=args.poll_s,
        timeout_s=args.timeout_s,
        start_wait_s=args.start_wait_s,
    )


if __name__ == "__main__":
    sys.exit(main())
