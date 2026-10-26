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

"""The stage chain: waiting for a frozen product and the smoke gate, on fake stage runs."""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest

from scripts.ecc_robocasa import chain
from scripts.ecc_robocasa import common as C
from tests.unit_tests.ecc_robocasa.conftest import REPO, TASKS
from tests.unit_tests.ecc_robocasa.s2_support import S2Env, build_s1_product

TWO = TASKS[:2]
THREE = TASKS[:3]


@pytest.fixture(scope="module")
def s1(tmp_path_factory):
    env, product = build_s1_product(tmp_path_factory.mktemp("chain_s1"), THREE)
    yield env, product
    env.cleanup()


@pytest.fixture
def s2(tmp_path, s1):
    env = S2Env(tmp_path, *s1)
    yield env
    env.cleanup()


def _no_sleep(_s: float) -> None:
    return None


def test_wait_product_returns_once_the_product_verifies(s1):
    env, product = s1
    assert chain.wait_product("s1", product.parent, sleep=_no_sleep) == chain.PASS


def test_wait_product_gives_up_when_the_runner_ended_without_a_product(tmp_path):
    layout = C.RunLayout(tmp_path / "run")
    C.atomic_write_json(layout.manifest, {"stage": "s2"})
    assert (
        chain.wait_product("s2", layout.root, grace_s=0, sleep=_no_sleep) == chain.GONE
    )
    # A run that does not exist yet is waited for (here until the timeout).
    ticks = iter(range(0, 10_000, 100))
    assert (
        chain.wait_product(
            "s2",
            tmp_path / "later",
            timeout_s=500,
            clock=lambda: float(next(ticks)),
            sleep=_no_sleep,
        )
        == chain.TIMEOUT
    )


def _run(s2: S2Env, scenario: dict) -> float:
    since = time.time() - 1
    s2.set_scenario(scenario)
    s2.config(tasks=TWO, max_slots=2)
    proc = s2.go()
    assert "runner finished" in proc.stdout, proc.stdout + proc.stderr
    return since


def test_gate_passes_valid_smoke_episodes_with_a_returned_gpt6_call(s2, capsys):
    since = _run(
        s2,
        {
            TWO[0]: {
                "kind": "solve",
                "requests": 20,
            },  # check-in at request 13 (K = 12)
            TWO[1]: {"kind": "solve", "requests": 5},
        },
    )
    code = chain.gate(s2.run_root, since=since, poll_s=0, sleep=_no_sleep)
    out = capsys.readouterr().out
    assert code == chain.PASS, out
    assert "SMOKE PASSED:" in out and "WARNING" not in out
    view = json.loads(out[out.index("{") :])
    assert sorted(view["smoke"]) == sorted(f"{t}__r1" for t in TWO)
    first = {c["job"]: c for c in view["checks"]}[f"{TWO[0]}__r1"]
    assert first["first_senior"]["request"] == 13
    assert first["first_senior"]["reason"] == "checkin"
    assert first["config_row"]["checkin"] == 12 and first["config_row"]["errors"] == 4
    assert first["evidence_ok"] is True


def test_runner_alive_reads_the_lock_without_taking_it(tmp_path):
    layout = C.RunLayout(tmp_path / "s2_probe")
    layout.root.mkdir()
    assert not chain.runner_alive(layout)
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time; from scripts.ecc_robocasa import common as C; "
            "h = C.acquire_lock(C.RunLayout(__import__('pathlib').Path(sys.argv[1]))); "
            "print('locked', flush=True); time.sleep(60)",
            str(layout.root),
        ],
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "locked"
        for _ in range(50):  # many polls never make the holder lose its lock
            assert chain.runner_alive(layout)
        assert C.lock_holder(layout) == holder.pid
    finally:
        holder.kill()
        holder.wait()
    assert not chain.runner_alive(layout)  # a dead pid in the lock file


def test_wait_idle_after_the_runner_and_its_servers_are_gone(s2):
    _run(s2, {TWO[0]: {"kind": "solve"}, TWO[1]: {"kind": "solve"}})
    assert chain.servers_alive(C.RunLayout(s2.run_root)) == []
    assert chain.wait_idle(s2.run_root, sleep=_no_sleep) == chain.PASS


def test_gate_smoke_is_the_first_jobs_to_finish(s2, capsys):
    """A long first episode does not hold the run at low concurrency."""
    since = time.time() - 1
    s2.set_scenario(
        {
            THREE[0]: {"kind": "solve", "requests": 20, "sleep_s": 4},
            THREE[1]: {"kind": "solve", "requests": 20},
            THREE[2]: {"kind": "solve", "requests": 20},
        }
    )
    s2.config(tasks=THREE, max_slots=2)
    assert "runner finished" in s2.go().stdout
    assert chain.gate(s2.run_root, since=since, poll_s=0, sleep=_no_sleep) == chain.PASS
    out = capsys.readouterr().out
    view = json.loads(out[out.index("{") :])
    assert view["smoke"] == [f"{THREE[1]}__r1", f"{THREE[2]}__r1"]


def test_gate_waits_for_a_gpt6_call_then_warns_or_fails(s2, capsys):
    since = _run(
        s2,
        {
            TWO[0]: {"kind": "solve", "requests": 5},
            TWO[1]: {"kind": "solve", "requests": 5},
        },
    )
    code = chain.gate(
        s2.run_root, since=since, senior_wait_s=0, poll_s=0, sleep=_no_sleep
    )
    assert code == chain.PASS
    assert "SMOKE PASSED WITH WARNING" in capsys.readouterr().out
    code = chain.gate(
        s2.run_root,
        since=since,
        senior_wait_s=0,
        on_no_senior="fail",
        poll_s=0,
        sleep=_no_sleep,
    )
    assert code == chain.FAIL


def test_gate_fails_on_an_infrastructure_smoke_launch_or_a_stop_file(s2, capsys):
    since = _run(
        s2,
        {
            TWO[0]: {"r1": [{"kind": "infra"}, {"kind": "solve", "requests": 20}]},
            TWO[1]: {"kind": "solve", "requests": 20},
        },
    )
    code = chain.gate(s2.run_root, since=since, poll_s=0, sleep=_no_sleep)
    assert code == chain.FAIL
    assert "is infra" in capsys.readouterr().out
    s2.run_id = "s2_stop"
    clean = _run(
        s2,
        {
            TWO[0]: {"kind": "solve", "requests": 20},
            TWO[1]: {"kind": "solve", "requests": 20},
        },
    )
    C.write_stop(C.RunLayout(s2.run_root), "operator")
    code = chain.gate(s2.run_root, since=clean, poll_s=0, sleep=_no_sleep)
    assert code == chain.FAIL and "STOP" in capsys.readouterr().out


def test_gate_fails_on_a_gpt6_call_that_ended_in_an_error(s2, capsys):
    since = _run(
        s2,
        {
            TWO[0]: {"kind": "solve", "requests": 20},
            TWO[1]: {"kind": "solve", "requests": 5},
        },
    )
    states = chain.job_states(C.RunLayout(s2.run_root))
    try_dir = states[f"{TWO[0]}__r1"]["tries"][-1]["dir"]
    with open(f"{try_dir}/escalation.jsonl", "a") as log:
        log.write(
            json.dumps(
                {
                    "request": 30,
                    "role": "senior",
                    "reason": "checkin",
                    "error": "HTTP 403",
                }
            )
            + "\n"
        )
    assert chain.gate(s2.run_root, since=since, poll_s=0, sleep=_no_sleep) == chain.FAIL
    assert "GPT-6 call failed" in capsys.readouterr().out
