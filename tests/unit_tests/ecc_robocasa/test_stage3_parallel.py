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

"""RC-ECC stage 3 with parallel sessions (``stage3.parallel_attempts``).

Fake episodes (RPent's real ``EscalationModel`` with scripted models), a fake
reflection call, fake RLDX servers and GPUs, as in ``test_stage3_runner.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import hybrid_episode as H
from scripts.ecc_robocasa import reflect as R
from scripts.ecc_robocasa import stage3 as S3
from tests.unit_tests.ecc_robocasa.test_stage3_runner import STAGE3, S3Env

pytestmark = pytest.mark.timeout(300)
PARALLEL = {"parallel_attempts": True}


def starts(env: S3Env) -> list[dict]:
    return sorted(env.events("start"), key=lambda e: e["t"])


def by_episode(rows: list[dict]) -> dict[tuple[str, str, int], dict]:
    return {(r["task"], r["episode"], r["try"]): r for r in rows}


def snapshot(env: S3Env, task: str, episode: str) -> str:
    return (env.task(task) / episode / "corrections_used.md").read_text()


def edit_config(env: S3Env, **sections) -> None:
    cfg = json.loads(env.config.read_text())
    for name, values in sections.items():
        cfg[name].update(values)
    env.config.write_text(json.dumps(cfg, indent=2))


def popen_runner(env: S3Env) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(STAGE3), "run", "--run-id", "r1", "--go"]
        + ["--config", str(env.config)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )


def kill_runner(env: S3Env, runner: subprocess.Popen) -> None:
    """SIGKILL the runner only: episodes and reflection calls run in their own sessions."""
    os.killpg(runner.pid, signal.SIGKILL)
    runner.wait()
    (env.root / "tasks").chmod(0o755)


def progress_has(env: S3Env, **match) -> bool:
    path = env.root / "progress.jsonl"
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if all(row.get(k) == v for k, v in match.items()):
            return True
    return False


def test_a_session_runs_its_attempts_at_once_and_reflects_on_them_in_order(ecc):
    fail = {"kind": "fail", "requests": 15}
    scenario = {
        "CloseFridge": {  # plan off, K=12: one check-in at request 13 in every episode
            "s1_a1": {**fail, "sleep_s": 2},
            "s1_a2": {**fail, "sleep_s": 3},
            "s1_a3": {**fail, "sleep_s": 1},
            "s2_a1": {"kind": "solve", "requests": 15, "sleep_s": 1},
            "s2_a2": {**fail, "sleep_s": 3},
            "s2_a3": {"kind": "timeout", "requests": 15, "sleep_s": 2},
        }
    }
    memory2 = ["s1_a1.1", "s1_a2.1", "s1_a3.1"]
    reflect = {  # the success's reflection revises an entry the next attempts ran with
        "CloseFridge/s2_a1": {
            "edits": {
                "applied": memory2,
                "add": [
                    {
                        "trigger": "checkin",
                        "situation": "situation seen in CloseFridge/s2_a1",
                        "diagnosis": "the push started above the free edge",
                        "fix": "start 3 cm below the edge",
                    }
                ],
                "revise": [{"id": "s1_a1.1", "fix": "start 5 cm below the edge"}],
                "remove": [],
                "note": "lesson of s2_a1",
            }
        }
    }
    env = S3Env(
        ecc,
        ["CloseFridge"],
        scenario=scenario,
        reflect=reflect,
        stage3={**PARALLEL, "sessions": 3},
    )
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert '"ok": true' in proc.stdout  # frozen

    started = starts(env)
    # Session 1's three attempts start (in any order) before session 2's; the success in session 2
    # stops the task after it: no s3_*.
    assert [e["episode"][:2] for e in started] == ["s1"] * 3 + ["s2"] * 3
    assert sorted((e["episode"], e["try"]) for e in started) == [
        (f"s{s}_a{a}", 1) for s in (1, 2) for a in (1, 2, 3)
    ]
    ends = by_episode(env.events("end"))
    reflect_t = {e["episode"].split("/")[1]: e["t"] for e in env.events("reflect")}
    for s in (1, 2):
        names = [f"s{s}_a{a}" for a in (1, 2, 3)]
        start_t = [e["t"] for e in started if e["episode"] in names]
        end_t = [ends[("CloseFridge", n, 1)]["t"] for n in names]
        # All attempts of the session ran at the same time.
        assert max(start_t) < min(end_t)
        # Every reflection waited for the whole session, then a1, a2, a3 in that order.
        assert min(end_t) <= max(end_t) < reflect_t[names[0]]
        assert reflect_t[names[0]] < reflect_t[names[1]] < reflect_t[names[2]]
    assert (
        min(e["t"] for e in started if e["episode"].startswith("s2_"))
        > reflect_t["s1_a3"]
    )

    # The memory snapshot is per session: the memory as it was when the session started.
    r = {
        n: env.reflection("CloseFridge", n)
        for n in ("s1_a1", "s1_a2", "s1_a3", "s2_a1", "s2_a2", "s2_a3")
    }
    for n in ("s1_a1", "s1_a2", "s1_a3"):
        assert snapshot(env, "CloseFridge", n) == ""
        assert ends[("CloseFridge", n, 1)]["senior_saw"] == ["none"]
    session2 = R.render_file(r["s1_a3"]["entries_after"])
    assert session2.count("\n") == 3
    for n in ("s2_a1", "s2_a2", "s2_a3"):
        assert snapshot(env, "CloseFridge", n) == session2
        assert not os.access(
            env.task("CloseFridge") / n / "corrections_used.md", os.W_OK
        )
        assert set(ends[("CloseFridge", n, 1)]["senior_saw"]) == {"match"}
        assert not ends[("CloseFridge", n, 1)]["worker_saw"]
        evidence = json.loads(
            (env.task("CloseFridge") / n / "try1.evidence.json").read_text()
        )
        assert evidence["ok"]
    assert len({e["corrections_sha256"] for e in started if e["episode"] < "s2"}) == 1
    assert len({e["corrections_sha256"] for e in started if e["episode"] > "s2"}) == 1

    # Each reflection edits the memory the previous one left, and the prompt shows that memory.
    previous: list = []
    shown = {
        e["episode"].split("/")[1]: e["entries_shown"] for e in env.events("reflect")
    }
    for n, rec in r.items():
        assert rec["entries_before"] == previous and rec["status"] == "applied"
        assert shown[n] == [x["id"] for x in previous]
        previous = rec["entries_after"]
    # Only entries the episode was shown, as they still stand, can count as applied in it.
    assert r["s1_a2"]["used"] == [] and r["s1_a2"]["applied_not_shown"] == ["s1_a1.1"]
    assert r["s1_a3"]["applied_not_shown"] == ["s1_a1.1", "s1_a2.1"]
    assert r["s1_a3"]["session_memory"] == [] and r["s2_a2"]["session"] == 2
    for n in ("s2_a1", "s2_a2", "s2_a3"):
        assert r[n]["session_memory"] == memory2
    assert r["s2_a1"]["used"] == memory2 and r["s2_a1"]["applied_not_shown"] == []
    # s2_a1's reflection revised s1_a1.1: s2_a2 and s2_a3 ran with its old text.
    assert r["s2_a2"]["used"] == ["s1_a2.1", "s1_a3.1"]
    assert r["s2_a2"]["applied_not_shown"] == ["s1_a1.1", "s2_a1.1"]
    assert r["s2_a3"]["applied_not_shown"] == ["s1_a1.1", "s2_a1.1", "s2_a2.1"]
    assert r["s2_a3"]["digest_source"] == "run.log"  # the timeout, rebuilt from run.log
    after = {e["id"]: e for e in r["s2_a3"]["entries_after"]}
    revised = after["s1_a1.1"]
    assert revised["fix"] == "start 5 cm below the edge"
    assert revised["used_in"] == ["CloseFridge/s2_a1"] and revised["failures"] == 0
    assert revised["status"] == "verified" and revised["revised_in"] == [
        "CloseFridge/s2_a1"
    ]
    assert after["s1_a2.1"]["used_in"] == [f"CloseFridge/s2_a{a}" for a in (1, 2, 3)]
    assert after["s1_a2.1"]["successes"] == 1 and after["s1_a2.1"]["failures"] == 2
    assert after["s2_a1.1"]["used_in"] == []

    # First success stops the task after its session: all 3 reflected, later sessions not needed.
    record = json.loads((env.task("CloseFridge") / "record.json").read_text())
    assert record["status"] == "solved" and record["first_success"] == "s2_a1"
    assert record["episodes"] == 6 and record["sessions_run"] == 2
    assert record["parallel_attempts"] is True
    assert record["reflections"] == {"applied": 6, "unparsed": 0, "skipped": 0}
    for a in (1, 2, 3):
        state = json.loads(
            (env.root / "state" / "jobs" / f"CloseFridge__s3_a{a}.json").read_text()
        )
        assert state["final"]["class"] == "not_needed" and state["tries"] == []
    history = [
        json.loads(x)
        for x in (env.task("CloseFridge") / "corrections" / "history.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [h["episode"].split("/")[1] for h in history] == list(r)

    product = env.root / "product"
    assert S3.verify(product)["ok"]
    kept = R.verified(r["s2_a3"]["entries_after"])
    assert [e["id"] for e in kept] == memory2 + ["s2_a1.1"]
    assert (product / "final" / "CloseFridge.md").read_text() == R.render_file(kept)
    doc = json.loads((product / "PRODUCT.json").read_text())
    assert doc["budget"]["parallel_attempts"] is True
    assert doc["tasks"]["CloseFridge"]["status"] == "solved"
    manifest = json.loads((env.root / "manifest.json").read_text())
    assert manifest["settings"]["parallel_attempts"] is True

    status = env.run("status", "--run-id", "r1", "--tasks-table")
    assert status.returncode == 0, status.stderr
    assert "[####################] 1/1 tasks final" in status.stdout
    assert "parallel sessions (3 attempts at once)" in status.stdout


def test_a_session_starts_only_when_all_its_attempts_fit_first_sessions_first(ecc):
    tasks = ["CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor"]
    scenario = {t: {"default": {"kind": "fail", "sleep_s": 1}} for t in tasks}
    # 4 slots: one session of 3 at a time; the free 4th slot must stay unused.
    env = S3Env(
        ecc, tasks, scenario=scenario, stage3=PARALLEL, endpoints=((4, 4),), max_slots=4
    )
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    started = starts(env)
    ends = by_episode(env.events("end"))
    sessions = [(e["task"], e["episode"][:2]) for e in started]
    # Sessions launch as blocks of 3, every task's first session before any second one.
    blocks = [sessions[i : i + 3] for i in range(0, len(sessions), 3)]
    assert all(len(set(b)) == 1 for b in blocks)
    assert [b[0] for b in blocks] == [(t, s) for s in ("s1", "s2") for t in tasks]
    for i in range(0, len(started), 3):
        block = started[i : i + 3]
        block_end = sorted(ends[(e["task"], e["episode"], 1)]["t"] for e in block)
        assert max(e["t"] for e in block) < block_end[0]  # together
        if i + 3 < len(started):
            # 4 slots: the next session waited until 3 were free (2 of this session ended).
            assert started[i + 3]["t"] > block_end[1]
    assert len(started) == 18  # 3 tasks x 2 sessions x 3 attempts, all valid failures
    for task in tasks:
        record = json.loads((env.task(task) / "record.json").read_text())
        assert record["status"] == "unsolved" and record["sessions_run"] == 2
    assert S3.verify(env.root / "product")["ok"]


def test_relaunches_stay_inside_the_session_and_an_exhausted_episode_is_requeued(ecc):
    fail = {"kind": "fail", "requests": 15}
    scenario = {
        "CloseFridge": {
            "s1_a1": {**fail, "sleep_s": 0.5},
            "s1_a2": [{"kind": "infra"}, {**fail, "sleep_s": 2}],
            "s1_a3": {**fail, "sleep_s": 0.5},
            "default": {"kind": "solve", "requests": 15},
        },
        "CloseToasterOvenDoor": {
            "s1_a1": {"kind": "fail", "requests": 7},
            "s1_a2": [{"kind": "infra"}, {"kind": "infra"}],
            "s1_a3": {"kind": "fail", "requests": 7, "sleep_s": 1},
            "default": {"kind": "solve", "requests": 7},
        },
    }
    env = S3Env(
        ecc,
        ["CloseFridge", "CloseToasterOvenDoor"],
        scenario=scenario,
        stage3={**PARALLEL, "infra_retries": 1},
        endpoints=((4, 6),),
        max_slots=6,
    )
    env.go()  # CloseToasterOvenDoor stops on infrastructure errors: no freeze
    started = by_episode(env.events("start"))
    ends = by_episode(env.events("end"))
    reflect_t = {e["episode"]: e["t"] for e in env.events("reflect")}

    # The session waited for its relaunch; the invalid launch was never reflected on.
    fridge = env.task("CloseFridge")
    relaunch_end = ends[("CloseFridge", "s1_a2", 2)]["t"]
    assert reflect_t["CloseFridge/s1_a1"] > relaunch_end
    assert env.reflection("CloseFridge", "s1_a2")["try_dir"].endswith("s1_a2/try2")
    assert not (fridge / "s1_a2" / "try1.evidence.json").exists()
    shas = {
        started[("CloseFridge", n, t)]["corrections_sha256"]
        for n, t in (("s1_a1", 1), ("s1_a2", 1), ("s1_a2", 2), ("s1_a3", 1))
    }
    assert len(shas) == 1
    record = json.loads((fridge / "record.json").read_text())
    assert record["status"] == "solved" and record["first_success"] == "s2_a1"
    assert record["infra_retries"] == 1 and record["episodes"] == 6

    # An episode out of relaunches stops its task after the earlier reflections of the session.
    toaster = env.task("CloseToasterOvenDoor")
    record = json.loads((toaster / "record.json").read_text())
    assert record["status"] == "infra_exhausted"
    assert (toaster / "s1_a1" / "reflection.json").exists()
    assert not (toaster / "s1_a2" / "reflection.json").exists()
    assert not (toaster / "s1_a3" / "reflection.json").exists()  # waits for s1_a2's
    assert not (env.root / "product").exists()

    ecc.set_scenario(
        {
            **scenario,
            "CloseToasterOvenDoor": {
                "s1_a2": {"kind": "fail", "requests": 7},
                "default": {"kind": "solve", "requests": 7},
            },
        }
    )
    proc = env.run("requeue", "--run-id", "r1", "--task", "CloseToasterOvenDoor")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert '"ok": true' in proc.stdout
    rerun = [
        e
        for e in env.events("start")
        if e["task"] == "CloseToasterOvenDoor" and e["episode"] == "s1_a2"
    ]
    assert len(rerun) == 3 and len({e["corrections_sha256"] for e in rerun}) == 1
    reflected = [
        e["episode"]
        for e in sorted(env.events("reflect"), key=lambda e: e["t"])
        if e["episode"].startswith("CloseToasterOvenDoor/")
    ]
    assert reflected == [
        f"CloseToasterOvenDoor/{n}"
        for n in ("s1_a1", "s1_a2", "s1_a3", "s2_a1", "s2_a2", "s2_a3")
    ]
    previous: list = []
    for n in ("s1_a1", "s1_a2", "s1_a3"):
        rec = env.reflection("CloseToasterOvenDoor", n)
        assert rec["entries_before"] == previous
        previous = rec["entries_after"]
    record = json.loads((toaster / "record.json").read_text())
    assert record["status"] == "solved" and record["first_success"] == "s2_a1"


def test_a_killed_runner_resumes_mid_session_with_the_same_memory(ecc):
    fail = {"kind": "fail", "requests": 15}
    scenario = {
        "CloseFridge": {
            "s1_a1": {**fail, "sleep_s": 8},
            "s1_a2": {**fail, "sleep_s": 1},
            "s1_a3": {**fail, "sleep_s": 1},
            "default": {"kind": "solve", "requests": 15},
        }
    }
    reflect = {"CloseFridge/s1_a2": {"sleep_s": 5}}
    env = S3Env(
        ecc, ["CloseFridge"], scenario=scenario, reflect=reflect, stage3=PARALLEL
    )
    edit_config(env, scheduler={"launch_stagger_s": 3})

    # 1. Killed right after the session's first launch.
    runner = popen_runner(env)
    ecc.wait_for(
        lambda: progress_has(env, event="launch", job="CloseFridge__s1_a1"), timeout=120
    )
    kill_runner(env, runner)
    assert not progress_has(env, event="launch", job="CloseFridge__s1_a2")

    # 2. A new runner adopts s1_a1 and launches the rest of the session; killed during a2's reflection.
    runner = popen_runner(env)
    ecc.wait_for(
        lambda: progress_has(
            env, event="reflection_start", task="CloseFridge", episode="s1_a2"
        ),
        timeout=180,
    )
    kill_runner(env, runner)

    # 3. A third runner adopts the reflection call and finishes.
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert progress_has(env, event="adopt", job="CloseFridge__s1_a1")
    started = [e for e in env.events("start") if e["episode"].startswith("s1_")]
    assert sorted((e["episode"], e["try"]) for e in started) == [
        ("s1_a1", 1),
        ("s1_a2", 1),
        ("s1_a3", 1),
    ]
    assert len({e["corrections_sha256"] for e in started}) == 1
    calls = [e["episode"] for e in sorted(env.events("reflect"), key=lambda e: e["t"])]
    assert calls == [
        f"CloseFridge/{n}"
        for n in ("s1_a1", "s1_a2", "s1_a3", "s2_a1", "s2_a2", "s2_a3")
    ]
    history = (env.task("CloseFridge") / "corrections" / "history.jsonl").read_text()
    assert [json.loads(x)["episode"] for x in history.splitlines()] == calls
    assert S3.verify(env.root / "product")["ok"]


def test_freeze_checks_the_session_chain_and_resume_refuses_another_mode(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": {"kind": "fail", "requests": 15},
            "s1_a2": {"kind": "fail", "requests": 15},
            "s1_a3": {"kind": "fail", "requests": 15},
            "default": {"kind": "solve", "requests": 15},
        }
    }
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario, stage3=PARALLEL)
    proc = env.go("--no-freeze")
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    args = argparse.Namespace(run_id="r1", run_root=None)
    _, manifest, stage = S3.load_stage(S3.load_config(env.config), args)
    assert stage.parallel and S3.chain_problems(stage, "CloseFridge") == []
    # status, freeze and requeue read the run in its own mode, whatever the config says now.
    edit_config(env, stage3={"parallel_attempts": False})
    sequential_cfg = S3.load_config(env.config)
    assert S3.load_stage(sequential_cfg, args)[2].parallel
    # The sequential chain check rejects these snapshots (s1_a2 ran with the session's memory).
    sequential = S3.Stage3(
        sequential_cfg, stage.layout, stage.tasks, stage.rules, stage.corpus
    )
    assert any(
        "s1_a2: corrections_used.md is not the previous memory" in p
        for p in S3.chain_problems(sequential, "CloseFridge")
    )
    # Resuming the run in the other mode is refused (its experiment hash differs).
    proc = env.run("run", "--run-id", "r1", "--go")
    assert (
        proc.returncode != 0
        and "refusing to resume: experiment settings" in proc.stderr
    )
    edit_config(env, stage3={"parallel_attempts": True})

    snap = env.task("CloseFridge") / "s2_a2" / "corrections_used.md"
    snap.chmod(0o644)
    good = snap.read_text()
    snap.write_text(
        good + "- [x] (verified; trigger: any) situation: s | diagnosis: d | fix: f\n"
    )
    assert S3.chain_problems(stage, "CloseFridge") == [
        "s2_a2: corrections_used.md is not its session's memory"
    ]
    with pytest.raises(SystemExit, match="broken correction-memory chains"):
        S3.freeze(stage, manifest)
    snap.write_text(good)
    result = S3.freeze(stage, manifest)
    assert result["ok"] and S3.verify(env.root / "product")["ok"]


def test_the_parallel_switch_is_opt_in_and_guards_the_concurrency(ecc):
    env = S3Env(ecc, ["CloseFridge"], scenario={})
    sequential = S3.load_config(env.config)
    assert sequential["stage3"]["parallel_attempts"] is False
    edit_config(env, stage3={"parallel_attempts": False})
    explicit = S3.load_config(env.config)
    edit_config(env, stage3={"parallel_attempts": True})
    parallel = S3.load_config(env.config)
    # Off (default or explicit) leaves the experiment settings and hash as they were.
    assert S3.experiment_settings(sequential) == {
        k: sequential["stage3"][k] for k in S3.EXPERIMENT_KEYS
    }
    assert S3.experiment_hash(sequential) == S3.experiment_hash(explicit)
    assert S3.experiment_hash(parallel) != S3.experiment_hash(sequential)
    edit_config(env, stage3={"parallel_attempts": "yes"})
    with pytest.raises(ValueError, match="parallel_attempts"):
        S3.load_config(env.config)

    edit_config(env, stage3={"parallel_attempts": True})
    plan = env.run("plan", "--run-id", "r1")
    assert plan.returncode == 0, plan.stderr
    assert "sessions parallel: the 3 attempts of a session run at once" in plan.stdout
    proc = env.run("run", "--run-id", "r1", "--go", "--concurrency", "2")
    assert proc.returncode != 0
    assert "--concurrency 2 is below the 3 attempts of a session" in proc.stderr
    assert not (ecc.runs / "r1").exists()
    edit_config(env, stage3={"parallel_attempts": False})
    plan = env.run("plan", "--run-id", "r1")
    assert plan.returncode == 0 and "sessions parallel" not in plan.stdout


# ---------------------------------------------------------------------------
# the scheduler's choice, in process
# ---------------------------------------------------------------------------


class Pool:
    """The parts of ServerPool the scheduler's choice reads."""

    def __init__(self, *endpoints: C.Endpoint):
        self.endpoints = list(endpoints)
        self.starting: dict = {}
        self.restart_after: dict = {}


def make_scheduler(
    ecc, tmp_path: Path, *, slots: int, max_slots: int, tasks: list[str]
):
    env = S3Env(ecc, tasks, scenario={}, stage3=PARALLEL, max_slots=max_slots)
    cfg = S3.load_config(env.config)
    layout = C.RunLayout(tmp_path / "run")
    layout.root.mkdir()
    rules = {t: H.Rule(plan=False, checkin=12, errors=4) for t in tasks}
    split_tasks = [("atomic", t) for t in tasks]
    stage = S3.Stage3(cfg, layout, split_tasks, rules, tmp_path / "corpus")
    pool = Pool(C.Endpoint("gpu4", 4, 1, slots, up=True))
    sched = S3.Stage3Scheduler(
        cfg, layout, stage, pool, infra_retries=2, render_retries=6
    )
    states = {
        j.job_id: {"job_id": j.job_id, "task": j.task, "tries": [], "final": None}
        for j in stage.jobs()
    }
    return sched, stage, states


def run(states, jid, *, cls=None):
    """Mark job ``jid`` running (or its last try ended with ``cls``)."""
    entry = {"n": 1, "dir": f"/x/{jid}", "endpoint": "gpu4", "status": "running"}
    if cls:
        entry.update(status="done", **{"class": cls})
    states[jid]["tries"].append(entry)


def test_the_scheduler_launches_whole_sessions(ecc, tmp_path):
    tasks = ["CloseBlenderLid", "CloseFridge"]
    sched, stage, states = make_scheduler(
        ecc, tmp_path, slots=4, max_slots=8, tasks=tasks
    )
    assert sched.next_job(states) == "CloseBlenderLid__s1_a1"
    # A started session goes first, even before a task earlier in the manifest.
    run(states, "CloseFridge__s1_a1")
    assert sched.next_job(states) == "CloseFridge__s1_a2"
    run(states, "CloseFridge__s1_a2")
    run(states, "CloseFridge__s1_a3")
    # 1 endpoint slot left: CloseBlenderLid's session (3 attempts) waits, although max slots allow it.
    assert sched.free_endpoint_slots(states) == 1
    assert sched.next_job(states) is None
    sched.servers.endpoints.append(C.Endpoint("gpu5", 5, 2, 2, up=True))
    assert sched.next_job(states) == "CloseBlenderLid__s1_a1"
    # The attempts of a session being launched go before any relaunch (the session stays one
    # block); relaunches come next, before any new session.
    run(states, "CloseBlenderLid__s1_a1", cls="infra")
    run(states, "CloseBlenderLid__s1_a2")
    fridge = states["CloseFridge__s1_a1"]["tries"][-1]
    fridge.update(status="done", **{"class": "infra"})
    assert sched.next_job(states) == "CloseBlenderLid__s1_a3"
    run(states, "CloseBlenderLid__s1_a3")
    # Then the infra relaunches, in manifest order.
    assert sched.next_job(states) == "CloseBlenderLid__s1_a1"
    run(states, "CloseBlenderLid__s1_a1")
    assert sched.next_job(states) == "CloseFridge__s1_a1"

    # Every task's first session starts before any second session.
    for a in (1, 2, 3):
        jid = f"CloseBlenderLid__s1_a{a}"
        states[jid]["tries"] = []
        run(states, jid, cls="valid")
        states[jid]["final"] = {"class": "valid", "try": 1, "success": False}
        folder = stage.episode_dir("CloseBlenderLid", a - 1)
        folder.mkdir(parents=True)
        (folder / "reflection.json").write_text("{}")
        states[f"CloseFridge__s1_a{a}"]["tries"] = []
    assert sched.free_endpoint_slots(states) == 6
    candidates_ready = [
        j
        for j in stage.jobs()
        if not states[j.job_id]["final"] and stage.ready(j, states)
    ]
    assert {j.job_id for j in candidates_ready} >= {
        "CloseBlenderLid__s2_a1",
        "CloseFridge__s1_a1",
    }
    assert sched.next_job(states) == "CloseFridge__s1_a1"
    # No endpoint up: a job is still named (it cannot launch), so the loop can end on no endpoint.
    for ep in sched.servers.endpoints:
        ep.up = False
    assert sched.free_endpoint_slots(states) == 0
    assert sched.next_job(states) == "CloseFridge__s1_a1"


def test_a_concurrency_below_one_session_starts_nothing_and_says_so(ecc, tmp_path):
    sched, _, states = make_scheduler(
        ecc, tmp_path, slots=12, max_slots=8, tasks=["CloseFridge"]
    )
    sched.layout.concurrency.write_text("2\n")
    assert sched.next_job(states) is None
    assert sched.next_job(states) is None
    rows = [json.loads(x) for x in sched.layout.progress.read_text().splitlines()]
    waits = [r for r in rows if r["event"] == "session_wait"]
    assert (
        len(waits) == 1
        and "concurrency 2 is below the 3 attempts" in waits[0]["reason"]
    )
    # The rest of a started session still launches at a low concurrency.
    run(states, "CloseFridge__s1_a1")
    assert sched.next_job(states) == "CloseFridge__s1_a2"
    sched.layout.concurrency.write_text("6\n")
    states["CloseFridge__s1_a1"]["tries"] = []
    assert sched.next_job(states) == "CloseFridge__s1_a1"


def test_reviewed_counts_only_shown_entries_when_asked():
    def entry(i):
        return {
            "id": i,
            "trigger": "any",
            "situation": "s",
            "diagnosis": "d",
            "fix": "f",
            "status": "candidate",
            "used_in": [],
            "successes": 0,
            "failures": 0,
        }

    reply = json.dumps(
        {"applied": ["a", "b", "zz"], "add": [], "revise": [], "remove": []}
    )
    plain = R.reviewed([entry("a"), entry("b")], reply, "T/s1_a2", True)
    assert set(plain) == {"status", "edits", "used", "entries_after"}
    assert plain["used"] == ["a", "b"]
    limited = R.reviewed([entry("a"), entry("b")], reply, "T/s1_a2", True, shown={"a"})
    assert limited["used"] == ["a"] and limited["applied_not_shown"] == ["b"]
    after = {e["id"]: e for e in limited["entries_after"]}
    assert after["a"]["status"] == "verified" and after["a"]["used_in"] == ["T/s1_a2"]
    assert after["b"]["status"] == "candidate" and after["b"]["used_in"] == []


def test_the_last_reflections_get_a_session_of_calls(ecc, tmp_path):
    """After the scheduler stops, a session's reflections still run one after another."""
    _, stage, _ = make_scheduler(
        ecc, tmp_path, slots=4, max_slots=8, tasks=["CloseFridge"]
    )
    one = float(stage.refl["timeout_s"]) + 300
    assert S3.drain_limit_s(stage) == one * stage.attempts == one * 3
    stage.parallel = False  # the sequential order waits for one reflection, as before
    assert S3.drain_limit_s(stage) == one
