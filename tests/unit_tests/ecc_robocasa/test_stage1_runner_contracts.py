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

"""RC-ECC stage-1 runner with fake cells: dry run, slots over GPUs, retries, guards."""

from __future__ import annotations

import json
import os
import signal
import subprocess

import pytest

from tests.unit_tests.ecc_robocasa.conftest import TASKS

pytestmark = pytest.mark.timeout(300)


def test_dry_run_is_the_default_and_creates_nothing(ecc):
    config = ecc.config(tasks=TASKS[:2])
    for args in (
        ("run", "--config", str(config), "--run-id", "r1"),
        ("plan", "--config", str(config), "--run-id", "r1"),
    ):
        proc = ecc.stage1(*args)
        assert proc.returncode == 0, proc.stderr
        assert "dry run" in proc.stdout
        assert "2 (atomic 2)" in proc.stdout
        assert "5 sessions x 3 attempts, 2400 s per session" in proc.stdout
        assert "ECC_RC_SESSIONS=5" in proc.stdout and "ECC_RC_ATTEMPTS=3" in proc.stdout
    assert not ecc.runs.exists()
    assert ecc.events() == []


def test_cells_spread_over_gpus_within_slots_and_skip_a_busy_gpu(ecc):
    ecc.set_gpus({4: (10, 0), 5: (30000, 90), 6: (10, 0), 7: (5, 0)})
    ecc.set_scenario({task: {"kind": "fail", "sleep_s": 0.4} for task in TASKS[:8]})
    config = ecc.config(
        tasks=TASKS[:8], endpoints=((4, 2), (5, 2), (7, 2)), max_slots=3
    )
    proc = ecc.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr

    starts = [e for e in ecc.events() if e["event"] == "start"]
    ends = {(e["task"], e["try"]): e["t"] for e in ecc.events() if e["event"] == "end"}
    assert sorted(e["task"] for e in starts) == sorted(TASKS[:8])  # one launch per task
    assert {e["gpu"] for e in starts} == {
        "4",
        "7",
    }  # GPU 5 was busy: no server, no cells
    cfg = json.loads(config.read_text())
    port_of = {str(ep["gpu"]): ep["port"] for ep in cfg["endpoints"]}
    for e in starts:
        assert e["endpoint"] == f"http://127.0.0.1:{port_of[e['gpu']]}"
    # At no moment more than 2 cells on one endpoint, nor more than 3 in all.
    for e in starts:
        t = e["t"] + 0.01
        live = [s for s in starts if s["t"] <= t < ends[(s["task"], s["try"])]]
        assert len(live) <= 3
        for gpu in ("4", "7"):
            assert sum(1 for s in live if s["gpu"] == gpu) <= 2
    progress = ecc.progress()
    assert any(
        r["event"] == "server_refused" and r["endpoint"] == "gpu5" for r in progress
    )
    assert any(
        r["event"] == "server_stop" for r in progress
    )  # servers stopped when done
    for e in starts:  # the budget reaches the launcher
        assert e["env"]["ECC_RC_SESSIONS"] == "5" and e["env"]["ECC_RC_ATTEMPTS"] == "3"
        assert (
            e["env"]["ECC_RC_SESSION_TIMEOUT_S"] == "2400"
            and e["env"]["RC_RENDER"] == "cpu"
        )
        assert e["argv"][2] == "0"  # seed 0


def test_first_success_stop_and_attempt_accounting(ecc):
    ecc.set_scenario(
        {
            "CloseBlenderLid": {
                "kind": "solve",
                "solve_session": 2,
                "solve_attempt": 2,
                "global": ["g_lid"],
            },
            "CloseFridge": {"kind": "fail"},
        }
    )
    config = ecc.config(tasks=TASKS[:2])
    ecc.go(config)
    solved = ecc.record("CloseBlenderLid")
    assert solved["status"] == "solved"
    assert solved["sessions_used"] == 2
    assert solved["attempts_used"] == 3 + 2
    assert solved["first_success_attempt"] == 5 and solved["solved_session"] == 2
    assert solved["publish"]["mode"] == "replace"
    assert solved["tries"] == [
        {
            "try": 1,
            "class": "valid",
            "endpoint": "gpu4",
            "termination_reason": "completed",
            "elapsed_s": solved["tries"][0]["elapsed_s"],
            "usd": solved["tries"][0]["usd"],
        }
    ]
    failed = ecc.record("CloseFridge")
    assert (
        failed["status"] == "unsolved"
        and failed["termination_reason"] == "planner_timeout"
    )
    assert failed["sessions_used"] == 5 and failed["attempts_used"] == 15
    assert (
        failed["first_success_attempt"] is None
        and failed["publish"]["mode"] == "keep_cloud"
    )
    # One cell per task: RPent's explore loop is the whole budget, nothing is relaunched after a result.
    assert len([e for e in ecc.events() if e["event"] == "start"]) == 2
    # Tokens come from the per-session cumulative [usage] lines of run.log.
    assert failed["qwen"]["requests"] == 5 * 3 * 2
    assert failed["qwen"]["input_tokens"] == 5 * 1000 * (3 * 2)


def test_infrastructure_errors_are_retried_and_never_counted_as_failures(ecc):
    ecc.set_scenario(
        {
            "CloseBlenderLid": [{"kind": "infra"}, {"kind": "solve"}],
            "CloseFridge": {"kind": "fail"},
            "CloseToasterOvenDoor": [
                {"kind": "infra"},
                {"kind": "crash"},
                {"kind": "infra"},
                {"kind": "solve"},
            ],
            "CoffeeSetupMug": [{"kind": "crash"}, {"kind": "fail"}],
        }
    )
    config = ecc.config(tasks=TASKS[:4], stage1={"infra_retries": 2})
    ecc.go(config)
    a = ecc.record("CloseBlenderLid")
    assert (
        a["status"] == "solved"
        and a["infra_retries"] == 1
        and [t["class"] for t in a["tries"]] == ["infra", "valid"]
    )
    b = ecc.record("CloseFridge")
    assert (
        b["status"] == "unsolved" and len(b["tries"]) == 1
    )  # a counted failure is never retried
    c = ecc.record("CloseToasterOvenDoor")
    assert (
        c["status"] == "infra_exhausted"
        and [t["class"] for t in c["tries"]] == ["infra"] * 3
    )
    assert c["publish"]["mode"] == "none"  # nothing of an invalid cell is published
    d = ecc.record("CoffeeSetupMug")
    assert d["status"] == "unsolved" and [
        t["termination_reason"] for t in d["tries"]
    ] == ["no_result", "planner_timeout"]
    starts = [e for e in ecc.events() if e["event"] == "start"]
    assert len(starts) == 2 + 1 + 3 + 2
    # A retry starts from a pristine staging corpus; the broken try's inbox is kept beside it.
    assert (ecc.run_root / "tasks" / "CloseBlenderLid" / "try1.inbox").is_dir()


def test_render_noise_is_invalid_and_retried_on_its_own_budget(ecc):
    pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    ecc.set_scenario({"OpenDrawer": [{"kind": "noise"}, {"kind": "solve"}]})
    config = ecc.config(tasks=["OpenDrawer"], stage1={"infra_retries": 0})
    ecc.go(config)
    rec = ecc.record("OpenDrawer")
    assert rec["status"] == "solved"
    assert [t["class"] for t in rec["tries"]] == ["render", "valid"]
    assert rec["render_retries"] == 1 and rec["infra_retries"] == 0
    assert (ecc.run_root / "tasks" / "OpenDrawer" / "try1" / "render_corrupt").exists()
    assert any(r["event"] == "render_corrupt" for r in ecc.progress())


def test_launch_ceiling_stops_new_launches(ecc):
    ecc.set_scenario({task: {"kind": "fail"} for task in TASKS[:5]})
    config = ecc.config(
        tasks=TASKS[:5], max_slots=1, scheduler={"max_total_launches": 3}
    )
    proc = ecc.go(config)
    assert "runner finished: stopped" in proc.stdout
    assert len([e for e in ecc.events() if e["event"] == "start"]) == 3
    stop = json.loads((ecc.run_root / "STOP").read_text())
    assert stop["reason"] == "launch ceiling"
    status = ecc.stage1("status", "--config", str(config), "--run-id", "r1")
    assert (
        "[############--------] 3/5" in status.stdout
        and "launch ceiling" in status.stdout
    )


def test_consecutive_infrastructure_errors_trip_the_breaker(ecc):
    ecc.set_scenario({task: {"kind": "infra"} for task in TASKS[:4]})
    config = ecc.config(tasks=TASKS[:4], max_slots=1, scheduler={"infra_breaker": 3})
    proc = ecc.go(config)
    assert "runner finished: stopped" in proc.stdout
    assert len([e for e in ecc.events() if e["event"] == "start"]) == 3
    assert (
        json.loads((ecc.run_root / "STOP").read_text())["reason"]
        == "infrastructure breaker"
    )


def test_cost_ceiling_from_the_episode_logs(ecc):
    # 5 sessions of 6 requests, cumulative 6M input tokens each: about $12.7 per cell at the Qwen list price.
    ecc.set_scenario(
        {task: {"kind": "fail", "tokens": [1_000_000, 10]} for task in TASKS[:3]}
    )
    config = ecc.config(tasks=TASKS[:3], max_slots=1, scheduler={"cost_ceiling_usd": 5})
    proc = ecc.go(config)
    assert "runner finished: stopped" in proc.stdout
    stop = json.loads((ecc.run_root / "STOP").read_text())
    assert stop["reason"] == "cost ceiling" and stop["usd"] > 5
    assert len([e for e in ecc.events() if e["event"] == "start"]) == 1


def test_live_concurrency_file_starts_slow_and_can_be_raised(ecc):
    ecc.set_scenario({task: {"kind": "fail", "sleep_s": 0.3} for task in TASKS[:6]})
    config = ecc.config(tasks=TASKS[:6], endpoints=((4, 3), (7, 3)), max_slots=6)
    runner = subprocess.Popen(
        ecc.cmd(
            "run",
            "--config",
            str(config),
            "--run-id",
            "r1",
            "--go",
            "--concurrency",
            "1",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    ecc.wait_for(
        lambda: len([e for e in ecc.events() if e["event"] == "end"]) >= 2, timeout=120
    )
    starts = [e for e in ecc.events() if e["event"] == "start"]
    ends = [e for e in ecc.events() if e["event"] == "end"]
    first_two = sorted(starts, key=lambda e: e["t"])[:2]
    assert first_two[1]["t"] >= min(e["t"] for e in ends)  # one at a time at first
    (ecc.run_root / "concurrency").write_text("6\n")
    out, _ = runner.communicate(timeout=180)
    assert "runner finished: done" in out
    assert len([e for e in ecc.events() if e["event"] == "start"]) == 6


def test_stage1_seed_other_than_0_is_refused(ecc):
    config = ecc.config(tasks=TASKS[:1], stage1={"seed": 1})
    proc = ecc.stage1("plan", "--config", str(config))
    assert proc.returncode != 0 and "stage1.seed must be 0" in proc.stderr


def test_a_gpu_busy_at_start_is_used_once_it_is_free(ecc):
    ecc.set_gpus({4: (10, 0), 7: (30000, 90)})
    ecc.set_scenario({task: {"kind": "fail", "sleep_s": 1.0} for task in TASKS[:6]})
    config = ecc.config(tasks=TASKS[:6], endpoints=((4, 1), (7, 1)), max_slots=2)
    raw = json.loads(config.read_text())
    raw["servers"]["retry_busy_s"] = 0.5
    config.write_text(json.dumps(raw))
    runner = ecc.popen_go(config)
    ecc.wait_for(
        lambda: any(
            r["event"] == "server_refused" and r["endpoint"] == "gpu7"
            for r in ecc.progress()
        ),
        timeout=120,
    )
    ecc.set_gpus({4: (10, 0), 7: (5, 0)})  # the other user's job ended
    out, _ = runner.communicate(timeout=240)
    assert "runner finished: done" in out
    assert any(
        r["event"] == "server_ready" and r["endpoint"] == "gpu7" for r in ecc.progress()
    )
    gpus = {e["gpu"] for e in ecc.events() if e["event"] == "start"}
    assert gpus == {"4", "7"}


def _servers(config, **settings):
    raw = json.loads(config.read_text())
    raw["servers"].update(settings)
    config.write_text(json.dumps(raw))
    return config


def test_three_servers_share_one_gpu_and_max_gpus_leaves_the_others_free(ecc):
    """The measured full-speed layout: several servers on one claimed GPU, small pools."""
    ecc.set_gpus({4: (10, 0), 5: (10, 0), 7: (5, 0)})
    ecc.set_scenario({task: {"kind": "fail", "sleep_s": 0.4} for task in TASKS[:6]})
    config = _servers(
        ecc.config(
            tasks=TASKS[:6], endpoints=((7, 1), (7, 1), (7, 1), (4, 1)), max_slots=4
        ),
        max_gpus=1,
    )
    ports = [ep["port"] for ep in json.loads(config.read_text())["endpoints"]]
    proc = ecc.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    progress = ecc.progress()
    started = [r["endpoint"] for r in progress if r["event"] == "server_start"]
    assert started == [f"gpu7_{port}" for port in ports[:3]]  # GPU 4 never claimed
    starts = [e for e in ecc.events() if e["event"] == "start"]
    assert sorted(e["task"] for e in starts) == sorted(TASKS[:6])
    assert {e["gpu"] for e in starts} == {"7"}
    assert {e["endpoint"] for e in starts} <= {
        f"http://127.0.0.1:{p}" for p in ports[:3]
    }
    for port in ports[:3]:
        log = (ecc.run_root / "servers" / f"gpu7_{port}.log").read_text()
        assert (
            "RC_GPU=7 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4" in log
        )


def test_a_server_that_dies_on_a_held_gpu_restarts_if_one_more_fits(ecc):
    """Our own servers make the GPU look busy; a restart there checks room, not idleness."""
    tasks = TASKS[:2]
    ecc.set_scenario(
        {t: [{"kind": "hang", "hang_s": 40.0}, {"kind": "solve"}] for t in tasks}
    )
    config = ecc.config(
        tasks=tasks,
        endpoints=((7, 1), (7, 1)),
        max_slots=2,
        scheduler={"infra_breaker": 1},
    )
    ports = [ep["port"] for ep in json.loads(config.read_text())["endpoints"]]
    runner = ecc.popen_go(config)
    ecc.wait_for(
        lambda: len([e for e in ecc.events() if e["event"] == "start"]) == 2,
        timeout=120,
    )
    ecc.set_gpus({7: (40000, 60)})  # what our two servers and cells look like
    server = json.loads(
        (ecc.run_root / "servers" / f"gpu7_{ports[0]}.json").read_text()
    )
    os.killpg(os.getpgid(server["pid"]), signal.SIGKILL)
    out, _ = runner.communicate(timeout=240)
    assert "runner finished: done" in out, out
    progress = ecc.progress()
    restarted = [
        r
        for r in progress
        if r["event"] == "server_start" and r["endpoint"] == f"gpu7_{ports[0]}"
    ]
    assert len(restarted) == 2
    assert not any(r["event"] == "server_refused" for r in progress)
    for task in tasks:
        assert ecc.record(task)["status"] == "solved"
