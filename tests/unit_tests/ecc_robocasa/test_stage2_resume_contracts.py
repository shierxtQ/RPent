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

"""Stage 2 survives its runner: live rungs are adopted, the ladder continues, nothing runs twice;
stop --now relaunches without counting; a changed experiment is refused."""

from __future__ import annotations

import json
import os
import signal

import pytest

from scripts.ecc_robocasa.common import pid_alive
from tests.unit_tests.ecc_robocasa.conftest import TASKS
from tests.unit_tests.ecc_robocasa.s2_support import S2Env, build_s1_product

pytestmark = pytest.mark.timeout(400)


@pytest.fixture(scope="module")
def s1(tmp_path_factory):
    env, product = build_s1_product(tmp_path_factory.mktemp("s1"), TASKS[:4])
    yield env, product
    env.cleanup()


@pytest.fixture
def s2(s1, tmp_path):
    env = S2Env(tmp_path, *s1)
    yield env
    env.cleanup()


def _launches(s2, task):
    return [
        e
        for e in s2.progress()
        if e.get("event") == "launch" and e["job"].startswith(f"{task}__r")
    ]


def test_killed_runner_resumes_the_ladder_without_running_a_rung_twice(s2):
    # Two tasks hang in rung 1 when the runner dies; then both fail rung 1 and solve rung 2.
    s2.set_scenario(
        {
            task: {"r1": {"kind": "fail", "sleep_s": 4.0}, "r2": {"kind": "solve"}}
            for task in TASKS[:2]
        }
    )
    s2.config(tasks=TASKS[:2], max_slots=2)
    runner = s2.popen_go()
    s2.wait_for(
        lambda: len([e for e in s2.events() if e["event"] == "start"]) == 2, timeout=120
    )
    os.killpg(
        os.getpgid(runner.pid), signal.SIGKILL
    )  # the runner alone: cells have their own sessions
    runner.wait(timeout=30)
    assert all(pid_alive(e["pid"]) for e in s2.events() if e["event"] == "start")
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    for task in TASKS[:2]:
        assert [e["job"] for e in _launches(s2, task)] == [f"{task}__r1", f"{task}__r2"]
        rule = s2.rule(task)
        assert (rule["rung"], rule["episodes"], rule["launches"]) == (2, 2, 2)
        assert s2.record(task, 1)["evidence"]["ok"] is True
    assert {r["job"] for r in s2.progress() if r["event"] == "adopt"} == {
        f"{t}__r1" for t in TASKS[:2]
    }
    assert (s2.run_root / "product" / "FROZEN").exists()


def test_stop_now_relaunches_the_rung_without_counting_and_resume_checks_the_experiment(
    s2,
):
    s2.set_scenario(
        {TASKS[0]: {"r1": [{"kind": "hang", "hang_s": 30.0}, {"kind": "solve"}]}}
    )
    s2.config(tasks=TASKS[:1], max_slots=1)
    runner = s2.popen_go()
    s2.wait_for(lambda: any(e["event"] == "start" for e in s2.events()), timeout=120)
    proc = s2.cli("stop", "--now")
    assert proc.returncode == 0 and "stopped 1 running cell(s)" in proc.stdout, (
        proc.stdout + proc.stderr
    )
    out, _ = runner.communicate(timeout=120)
    assert "runner finished: stopped" in out
    assert [t["class"] for t in s2.state(TASKS[0], 1)["tries"]] == ["killed"]
    (s2.run_root / "STOP").unlink()

    # A changed ladder is another experiment: refused.
    s2.config(
        tasks=TASKS[:1],
        max_slots=1,
        stage2={"ladder": [{"plan": "gpt", "checkin": 6, "errors": 2}]},
    )
    proc = s2.stage2(
        "run", "--config", str(s2.config_path), "--run-id", s2.run_id, "--go"
    )
    assert proc.returncode != 0 and "experiment settings" in proc.stderr

    s2.config(tasks=TASKS[:1], max_slots=1)
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    record = s2.record(TASKS[0], 1)
    assert (
        record["launches"],
        record["killed"],
        record["infra_retries"],
        record["success"],
    ) == (2, 1, 0, True)
    assert "launches: 1 of ceiling" in s2.cli("status").stdout
    manifest = json.loads((s2.run_root / "manifest.json").read_text())
    assert [h["event"] for h in manifest["history"]] == ["created", "resumed"]
