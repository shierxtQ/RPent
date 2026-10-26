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

"""A killed runner is resumed: live cells are adopted, dead ones scored, nothing launched twice."""

from __future__ import annotations

import json
import os
import signal
import time

import pytest

from scripts.ecc_robocasa.common import pid_alive
from tests.unit_tests.ecc_robocasa.conftest import TASKS

pytestmark = pytest.mark.timeout(300)


def test_resume_after_runner_kill_adopts_live_cells_and_loses_no_result(ecc):
    tasks = TASKS[:5]
    # Three long cells still run when the runner dies; two short ones end while it is down.
    ecc.set_scenario(
        {
            **{t: {"kind": "hang", "hang_s": 6.0} for t in tasks[:3]},
            **{t: {"kind": "solve", "sleep_s": 0.6} for t in tasks[3:]},
        }
    )
    config = ecc.config(tasks=tasks, endpoints=((4, 3), (7, 3)), max_slots=5)
    runner = ecc.popen_go(config)
    ecc.wait_for(
        lambda: len([e for e in ecc.events() if e["event"] == "start"]) == 5,
        timeout=120,
    )
    os.killpg(
        os.getpgid(runner.pid), signal.SIGKILL
    )  # the runner alone: cells have their own sessions
    runner.wait(timeout=30)
    starts = [e for e in ecc.events() if e["event"] == "start"]
    assert all(pid_alive(e["pid"]) for e in starts[:3])
    ecc.wait_for(
        lambda: len([e for e in ecc.events() if e["event"] == "end"]) >= 2, timeout=60
    )

    # The fake RLDX servers outlived the runner; the new runner adopts them as well.
    proc = ecc.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    starts = [e for e in ecc.events() if e["event"] == "start"]
    assert sorted(e["task"] for e in starts) == sorted(tasks)  # no task launched twice
    for task in tasks:
        state = ecc.state(task)
        assert len(state["tries"]) == 1 and state["final"]["class"] == "valid"
        assert ecc.record(task)["status"] == "solved"
    progress = ecc.progress()
    adopted = {r["job"] for r in progress if r["event"] == "adopt"}
    assert adopted and adopted <= set(tasks[:3])
    assert any(r["event"] == "server_adopted" for r in progress)
    assert (ecc.run_root / "product" / "FROZEN").exists()


def test_stop_now_kills_running_cells_which_are_relaunched_without_counting(ecc):
    tasks = TASKS[:2]
    ecc.set_scenario(
        {t: [{"kind": "hang", "hang_s": 30.0}, {"kind": "solve"}] for t in tasks}
    )
    config = ecc.config(tasks=tasks, max_slots=2)
    runner = ecc.popen_go(config)
    ecc.wait_for(
        lambda: len([e for e in ecc.events() if e["event"] == "start"]) == 2,
        timeout=120,
    )
    # While cells run they cannot list the other tasks' folders.
    with pytest.raises(PermissionError):
        list((ecc.run_root / "tasks").iterdir())
    proc = ecc.stage1("stop", "--config", str(config), "--run-id", "r1", "--now")
    assert proc.returncode == 0 and "stopped 2 running cell(s)" in proc.stdout, (
        proc.stdout + proc.stderr
    )
    out, _ = runner.communicate(timeout=120)
    assert "runner finished: stopped" in out
    for task in tasks:
        state = ecc.state(task)
        assert [t["class"] for t in state["tries"]] == ["killed"] and state[
            "final"
        ] is None
    assert json.loads((ecc.run_root / "STOP").read_text())["reason"] == "operator"
    assert (
        len(list((ecc.run_root / "tasks").iterdir())) == 2
    )  # listable again when idle

    (ecc.run_root / "STOP").unlink()
    proc = ecc.go(config)
    assert "runner finished: done" in proc.stdout
    for task in tasks:
        rec = ecc.record(task)
        assert [t["class"] for t in rec["tries"]] == ["killed", "valid"]
        assert rec["infra_retries"] == 0 and rec["status"] == "solved"
    # Operator-stopped tries do not use the launch ceiling.
    status = ecc.stage1("status", "--config", str(config), "--run-id", "r1")
    assert "launches: 2 of ceiling" in status.stdout, status.stdout


def test_second_runner_on_the_same_run_is_refused(ecc):
    ecc.set_scenario({TASKS[0]: {"kind": "hang", "hang_s": 8.0}})
    config = ecc.config(tasks=TASKS[:1])
    runner = ecc.popen_go(config)
    try:
        ecc.wait_for(
            lambda: any(e["event"] == "start" for e in ecc.events()), timeout=120
        )
        second = ecc.stage1("run", "--config", str(config), "--run-id", "r1", "--go")
        assert second.returncode != 0 and "another runner holds" in second.stderr
        status = ecc.stage1("status", "--config", str(config), "--run-id", "r1")
        assert f"runner: pid {runner.pid}" in status.stdout
    finally:
        runner.communicate(timeout=120)
    assert len([e for e in ecc.events() if e["event"] == "start"]) == 1
    time.sleep(0.1)


def test_dead_rldx_server_stops_its_cells_as_infrastructure_and_is_restarted(ecc):
    tasks = TASKS[:2]
    # Try 1 waits long enough for the server to die under it; try 2 solves.
    ecc.set_scenario(
        {t: [{"kind": "hang", "hang_s": 40.0}, {"kind": "solve"}] for t in tasks}
    )
    config = ecc.config(
        tasks=tasks, endpoints=((4, 2),), max_slots=2, scheduler={"infra_breaker": 1}
    )
    runner = ecc.popen_go(config)
    ecc.wait_for(
        lambda: len([e for e in ecc.events() if e["event"] == "start"]) == 2,
        timeout=120,
    )
    server = json.loads((ecc.run_root / "servers" / "gpu4.json").read_text())
    os.killpg(os.getpgid(server["pid"]), signal.SIGKILL)
    out, _ = runner.communicate(timeout=240)
    assert "runner finished: done" in out, out
    events = [r["event"] for r in ecc.progress()]
    assert "server_down" in events and events.count("server_start") == 2
    assert events.count("endpoint_down_stop") == 2
    for task in tasks:
        rec = ecc.record(task)
        assert [t["class"] for t in rec["tries"]] == ["infra", "valid"]
        assert rec["status"] == "solved" and rec["infra_retries"] == 1
        assert ecc.state(task)["tries"][0]["cause"].startswith("endpoint_down gpu4")
    # Endpoint-caused infrastructure errors do not trip the breaker (here set to 1).
    assert not (ecc.run_root / "STOP").exists()


def test_stop_after_a_dead_runner_also_stops_the_servers(ecc):
    ecc.set_scenario({TASKS[0]: {"kind": "hang", "hang_s": 30.0}})
    config = ecc.config(tasks=TASKS[:1])
    runner = ecc.popen_go(config)
    ecc.wait_for(lambda: any(e["event"] == "start" for e in ecc.events()), timeout=120)
    os.killpg(os.getpgid(runner.pid), signal.SIGKILL)
    runner.wait(timeout=30)
    server = json.loads((ecc.run_root / "servers" / "gpu4.json").read_text())
    assert pid_alive(server["pid"])
    proc = ecc.stage1("stop", "--config", str(config), "--run-id", "r1", "--now")
    assert proc.returncode == 0 and "stopped this run's RLDX servers" in proc.stdout, (
        proc.stdout
    )
    ecc.wait_for(lambda: not pid_alive(server["pid"]), timeout=30)
    cell = [e for e in ecc.events() if e["event"] == "start"][0]
    assert not pid_alive(cell["pid"])
