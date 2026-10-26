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

"""RC-ECC stage-2 ladder with fake hybrid cells: rung order, first success, fallback,
infrastructure and render retries, evidence mismatches, blocked tasks, freeze and verify."""

from __future__ import annotations

import json
import stat

import pytest

from tests.unit_tests.ecc_robocasa.conftest import TASKS
from tests.unit_tests.ecc_robocasa.s2_support import S2Env, build_s1_product

pytestmark = pytest.mark.timeout(400)

LADDER = [
    "c1_qwenplan_K12_N4",
    "c2_gptplan_K12_N4",
    "c3_qwenplan_K6_N4",
    "c4_gptplan_K6_N4",
    "c5_gptplan_K6_N3",
    "c6_gptplan_K6_N2",
]
#: (plan, K, N) per rung, the ECC rule's table.
RUNGS = [
    (False, 12, 4),
    (True, 12, 4),
    (False, 6, 4),
    (True, 6, 4),
    (True, 6, 3),
    (True, 6, 2),
]


@pytest.fixture(scope="module")
def s1(tmp_path_factory):
    env, product = build_s1_product(tmp_path_factory.mktemp("s1"), TASKS[:6])
    yield env, product
    env.cleanup()


@pytest.fixture
def s2(s1, tmp_path):
    env = S2Env(tmp_path, *s1)
    yield env
    env.cleanup()


def test_dry_run_prints_the_ladder_and_creates_nothing(s2):
    s2.config(tasks=TASKS[:2])
    proc = s2.cli("plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "dry run" in out and "2 (atomic 2); seed 11" in out
    assert "  ".join(f"{i}:{name}" for i, name in enumerate(LADDER, 1)) in out
    assert "all fail -> rung 6" in out
    assert "stage-1 product" in out and "verified" in out and "50 tasks load" in out
    assert "config row required; this code writes it: yes" in out
    for line in (
        "ECC_RC_PLAN=0",
        "ECC_RC_CHECKIN=12",
        "ECC_RC_ERRORS=4",
        "ECC_RC_TIMEOUT_S=1800",
    ):
        assert f"    {line}" in out
    assert f"{s2.product / 'corpus'}" in out  # the memory dir of the first launch
    assert not s2.runs.exists() and s2.events() == []


def test_config_refuses_stage1_seed_and_bad_ladders(s2):
    for stage2, fragment in (
        ({"seed": 0}, "must differ from stage 1"),
        ({"ladder": [{"plan": "gpt", "checkin": 6, "errors": 2}] * 2}, "repeat"),
        ({"ladder": [{"plan": "maybe", "checkin": 6, "errors": 2}]}, "plan must be"),
        ({"fallback_rung": 7}, "fallback_rung"),
    ):
        s2.config(tasks=TASKS[:1], stage2=stage2)
        proc = s2.cli("plan")
        assert proc.returncode != 0 and fragment in proc.stderr, (
            stage2,
            proc.stderr[-500:],
        )


def test_ladder_stops_at_first_valid_success_and_retries_never_count(s2):
    s2.set_scenario(
        {
            # r1 solves: rule c1, one episode, rungs 2-6 never run.
            "CloseBlenderLid": {"r1": {"kind": "solve", "requests": 15}},
            # valid failure, planner timeout (valid), then success at r3.
            "CloseFridge": {
                "r1": {"kind": "fail"},
                "r2": {"kind": "timeout"},
                "r3": {"kind": "solve"},
            },
            # every rung fails validly: the fallback is rung 6.
            "CloseToasterOvenDoor": {"default": {"kind": "fail", "requests": 25}},
            # two invalid launches at r1 (infra, crash) do not count; r1 then fails, r2 solves.
            "CoffeeSetupMug": {
                "r1": [{"kind": "infra"}, {"kind": "crash"}, {"kind": "fail"}],
                # four tool errors in a row after the plan turn (request 6) fire the N=4 trigger once.
                "r2": {"kind": "solve", "error_results": [8, 9, 10, 11]},
            },
            # rendered noise is stopped and relaunched on its own budget.
            "NavigateKitchen": {"r1": [{"kind": "noise"}, {"kind": "solve"}]},
        }
    )
    tasks = TASKS[:3] + ["CoffeeSetupMug", "NavigateKitchen"]
    s2.config(tasks=tasks, max_slots=3, endpoints=((4, 2), (7, 2)))
    proc = s2.go(timeout=300)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr

    expected = {
        "CloseBlenderLid": (1, "first_success", 1),
        "CloseFridge": (3, "first_success", 3),
        "CloseToasterOvenDoor": (6, "fallback_all_failed", 6),
        "CoffeeSetupMug": (2, "first_success", 2),
        "NavigateKitchen": (1, "first_success", 1),
    }
    for task, (rung, source, episodes) in expected.items():
        rule = s2.rule(task)
        assert (rule["rung"], rule["rule_source"], rule["episodes"]) == (
            rung,
            source,
            episodes,
        ), task
        plan, k, n = RUNGS[rung - 1]
        assert rule["rule"] == {"gpt_plan": plan, "checkin_K": k, "errors_N": n}
        assert rule["config"] == LADDER[rung - 1] and rule["ladder"] == LADDER
        assert (
            rule["search_seed"] == 11
            and rule["suite"] == "robocasa"
            and rule["task_id"] == task
        )
        assert rule["solved_in_search"] is (source == "first_success")
        assert [h["config"] for h in rule["history"]] == LADDER[:episodes]
        assert rule["s1_product"]["corpus_tree_sha256"]
    assert [h["termination_reason"] for h in s2.rule("CloseFridge")["history"]] == [
        "completed",
        "planner_timeout",
        "completed",
    ]
    assert s2.rule("CloseFridge")["history"][1]["last_error"] == "planner_timeout"
    # Skipped rungs are final without a launch; nothing ran after the deciding rung.
    for k in range(2, 7):
        assert s2.state("CloseBlenderLid", k)["final"]["class"] == "skipped"
        assert s2.state("CloseBlenderLid", k)["tries"] == []

    # Invalid launches are relaunched and recorded, never a rung failure.
    coffee = s2.record("CoffeeSetupMug", 1)
    assert (
        coffee["launches"],
        coffee["infra_retries"],
        coffee["valid"],
        coffee["success"],
    ) == (3, 2, True, False)
    assert s2.rule("CoffeeSetupMug")["launches"] == 4
    nav = s2.record("NavigateKitchen", 1)
    assert (nav["launches"], nav["render_retries"], nav["success"]) == (2, 1, True)
    assert (
        s2.run_root / "tasks" / "NavigateKitchen" / "r1" / "try1" / "render_corrupt"
    ).exists()

    # Each episode ran with its rung's rule, seed 11, the frozen corpus and the split's timeout.
    starts = [e for e in s2.events() if e["event"] == "start"]
    for e in starts:
        plan, k, n = RUNGS[e["rung"] - 1]
        assert (e["plan"], e["checkin"], e["errors"]) == (plan, k, n), e
        assert (e["seed"], e["memory"], e["timeout"]) == (
            11,
            str(s2.product / "corpus"),
            "1800",
        )
        assert e["corrections"] == ""
    # The fake cells drove the escalation wrapper of the code under test.
    from tests.unit_tests.ecc_robocasa.conftest import REPO

    origins = {
        e.get("rpent") for e in s2.events() if e["event"] == "end" and e.get("rpent")
    }
    assert origins == {str((REPO / "rpent" / "planner" / "escalation.py").resolve())}, (
        origins
    )
    # Rungs of one task never overlap and run in ladder order (the runner's own log).
    for task in tasks:
        mine = [
            (e["event"], int(e["job"].rsplit("__r", 1)[1]))
            for e in s2.progress()
            if e.get("event") in ("launch", "done")
            and e.get("job", "").startswith(f"{task}__r")
        ]
        assert [kind for kind, _ in mine] == ["launch", "done"] * (len(mine) // 2), (
            task,
            mine,
        )
        rungs = [rung for kind, rung in mine if kind == "launch"]
        assert rungs == sorted(rungs), (task, rungs)

    # Per-rung records: GPT-6 calls by reason, the errors trigger, tokens and cost.
    rec = s2.record("CoffeeSetupMug", 2)
    assert rec["config"] == "c2_gptplan_K12_N4" and rec["evidence"]["ok"] is True
    assert rec["gpt_calls_by_reason"]["plan"] == 1
    assert rec["errors_trigger"] == 1 and rec["gpt_calls_by_reason"]["errors"] == 1
    assert rec["gpt6"]["calls"] == rec["gpt_calls"] > 0 and rec["qwen"]["calls"] > 0
    assert rec["usd"] > 0 and rec["usage_source"] == "escalation.jsonl"
    assert rec["evidence"]["config_row"]["errors"] == 4
    r1 = s2.record("CloseBlenderLid", 1)
    assert (
        r1["gpt_calls_by_reason"]["plan"] == 0
        and r1["gpt_calls_by_reason"]["checkin"] == 1
    )
    assert r1["evidence"]["checkin_bounds"] == [12, 12]
    assert s2.rule("CloseToasterOvenDoor")["gpt_calls_in_search"] == sum(
        h["gpt_calls"] for h in s2.rule("CloseToasterOvenDoor")["history"]
    )

    # The frozen product: one rule per task, hashed, read-only, loadable.
    from scripts.ecc_robocasa import stage2

    product = s2.run_root / "product"
    assert stage2.verify(product)["ok"]
    doc = json.loads((product / "RULES.json").read_text())
    assert doc["schema"] == "rc-ecc-s2-product/1" and doc["seed"] == 11
    assert doc["summary"]["tasks"] == 5 and doc["summary"]["solved_in_search"] == 4
    assert doc["summary"]["chosen_config_counts"] == {
        "c1_qwenplan_K12_N4": 2,
        "c2_gptplan_K12_N4": 1,
        "c3_qwenplan_K6_N4": 1,
        "c6_gptplan_K6_N2": 1,
    }
    assert (
        doc["s1_product"]["corpus_tree_sha256"]
        == (s2.product / "FROZEN").read_text().strip()
    )
    rules = stage2.load_rules(product)
    assert rules["CloseFridge"].as_json() == {
        "gpt_plan": False,
        "checkin_K": 6,
        "errors_N": 4,
    }
    assert not (product / "rules" / "CloseFridge.json").stat().st_mode & stat.S_IWUSR
    episodes = json.loads((product / "episodes.json").read_text())
    assert [r["rung"] for r in episodes["CloseToasterOvenDoor"]] == [1, 2, 3, 4, 5, 6]
    (product / "rules").chmod(0o755)
    path = product / "rules" / "CloseFridge.json"
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"checkin_K": 6', '"checkin_K": 12'))
    assert not stage2.verify(product)["ok"]

    status = s2.cli("status", "--tasks-table")
    assert status.returncode == 0, status.stderr
    assert "[####################] 5/5 tasks with a rule" in status.stdout
    assert "CloseToasterOvenDoor" in status.stdout and "FFFFFF" in status.stdout
    assert "c6_gptplan_K6_N2 (fallback_all_failed)" in status.stdout
    line = next(
        row for row in status.stdout.splitlines() if row.strip().startswith("atomic")
    )
    # decided 5/5, then r1..r6 = 2 1 1 0 0 0 and one fallback.
    assert line.split()[1:2] == ["5/5"] and line.split()[5:12] == [
        "2",
        "1",
        "1",
        "0",
        "0",
        "0",
        "1",
    ]


def test_rung_one_of_every_task_runs_before_any_rung_two(s2):
    s2.set_scenario(
        {task: {"default": {"kind": "fail", "requests": 3}} for task in TASKS[:4]}
    )
    s2.config(
        tasks=TASKS[:4],
        max_slots=2,
        stage2={
            "ladder": [
                {"plan": "qwen", "checkin": 12, "errors": 4},
                {"plan": "gpt", "checkin": 6, "errors": 2},
            ]
        },
    )
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    launches = [e["job"] for e in s2.progress() if e["event"] == "launch"]
    rungs = [int(job.rsplit("__r", 1)[1]) for job in launches]
    assert rungs == [1, 1, 1, 1, 2, 2, 2, 2], launches
    for task in TASKS[:4]:
        assert (
            s2.rule(task)["rule_source"] == "fallback_all_failed"
            and s2.rule(task)["rung"] == 2
        )


def test_evidence_mismatch_is_infra_stops_the_run_and_resumes_after_a_fix(s2):
    # A broken launcher: the wrapper checks in every 5 requests although the rung says K=12.
    s2.set_scenario(
        {
            "CloseBlenderLid": {
                "r1": {
                    "kind": "fail",
                    "requests": 20,
                    "bug": {"checkin": 5},
                    "sleep_s": 0.3,
                }
            },
            "CloseFridge": {"r1": {"kind": "solve", "sleep_s": 1.5}},
        }
    )
    s2.config(tasks=TASKS[:2], max_slots=2)
    proc = s2.go()
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    stop = json.loads((s2.run_root / "STOP").read_text())
    assert (
        stop["reason"] == "escalation evidence mismatch"
        and stop["job"] == "CloseBlenderLid__r1"
    )
    assert any("checkin=5" in p for p in stop["problems"])
    state = s2.state("CloseBlenderLid", 1)
    assert state["tries"][0]["class"] == "infra" and state["final"] is None
    assert "evidence_mismatch" in state["tries"][0]["cause"]
    assert any(e["event"] == "evidence_mismatch" for e in s2.progress())
    # The other running cell drained and was scored normally.
    assert s2.state("CloseFridge", 1)["final"]["success"] is True
    status = s2.cli("status")
    assert (
        "EVIDENCE MISMATCH in 1 launch(es)" in status.stdout
        and "escalation evidence mismatch" in status.stdout
    )

    # Fixed launcher, STOP removed: the rung is relaunched and counts normally.
    s2.set_scenario({"CloseBlenderLid": {"r1": {"kind": "solve"}}})
    (s2.run_root / "STOP").unlink()
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    rule = s2.rule("CloseBlenderLid")
    assert (rule["rung"], rule["episodes"], rule["launches"]) == (1, 1, 2)


def test_a_valid_episode_without_an_escalation_log_is_a_mismatch(s2):
    s2.set_scenario({"CloseBlenderLid": {"r1": {"kind": "solve", "no_log": True}}})
    s2.config(tasks=TASKS[:1], max_slots=1)
    proc = s2.go()
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    assert any(
        "senior was off" in p
        for p in json.loads((s2.run_root / "STOP").read_text())["problems"]
    )


def test_a_rung_without_valid_result_blocks_the_task_until_unblocked(s2):
    s2.set_scenario(
        {
            "CloseBlenderLid": {"r1": {"kind": "fail"}, "r2": {"kind": "infra"}},
            "CloseFridge": {"r1": {"kind": "solve"}},
        }
    )
    s2.config(
        tasks=TASKS[:2],
        max_slots=2,
        stage2={"infra_retries": 1},
        scheduler={"infra_breaker": 0},
    )
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert "not frozen: blocked tasks ['CloseBlenderLid']" in proc.stdout
    assert s2.state("CloseBlenderLid", 2)["final"]["class"] == "infra_exhausted"
    assert all(
        s2.state("CloseBlenderLid", k)["final"]["class"] == "blocked"
        for k in range(3, 7)
    )
    assert not (s2.run_root / "tasks" / "CloseBlenderLid" / "rule.json").exists()
    freeze = s2.cli("freeze")
    assert freeze.returncode != 0 and "unblock" in freeze.stderr
    assert (
        "blocked (a rung without a valid result): CloseBlenderLid"
        in s2.cli("status").stdout
    )

    s2.set_scenario(
        {"CloseBlenderLid": {"r1": {"kind": "fail"}, "r2": {"kind": "solve"}}}
    )
    unblock = s2.cli("unblock", "--task", "CloseBlenderLid")
    assert unblock.returncode == 0, unblock.stderr
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    rule = s2.rule("CloseBlenderLid")
    assert (rule["rung"], rule["rule_source"], rule["launches"]) == (
        2,
        "first_success",
        4,
    )
    assert (s2.run_root / "product" / "FROZEN").exists()


def test_fallback_for_blocked_tasks_is_explicit_and_recorded(s2):
    s2.set_scenario({"CloseBlenderLid": {"r1": {"kind": "infra"}}})
    s2.config(
        tasks=TASKS[:1],
        max_slots=1,
        stage2={"infra_retries": 0},
        scheduler={"infra_breaker": 0},
    )
    s2.go()
    proc = s2.cli("freeze", "--fallback-blocked")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((s2.run_root / "product" / "RULES.json").read_text())
    assert doc["blocked_fallback"] == ["CloseBlenderLid"]
    assert doc["tasks"]["CloseBlenderLid"]["rule_source"] == "fallback_infra_blocked"
    assert doc["tasks"]["CloseBlenderLid"]["rung"] == 6


def test_dry_run_reports_a_stage1_product_that_is_not_frozen_yet(s2, tmp_path):
    pending = tmp_path / "s1_pending" / "product"
    pending.mkdir(parents=True)
    s2.config(tasks=TASKS[:1], stage2={"s1_product": str(pending)})
    proc = s2.cli("plan")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "stage-1 product NOT READY" in proc.stdout and "first launch:" in proc.stdout
    proc = s2.go()
    assert proc.returncode != 0 and "does not verify" in proc.stderr
    assert not s2.run_root.exists()


def test_file_tool_writes_outside_the_episode_are_audited(s2):
    corpus_note = str(s2.product / "corpus" / "global" / "QWEN_NOTE.md")
    other = str(
        s2.run_root / "tasks" / "CloseBlenderLid" / "r1" / "note.md"
    )  # another task's folder
    own = str(
        s2.run_root / "tasks" / "CloseFridge" / "r1" / "try1" / "note.md"
    )  # its own folder: fine
    s2.set_scenario(
        {
            "CloseBlenderLid": {
                "r1": {
                    "kind": "solve",
                    "file_calls": [["write_text_file", corpus_note]],
                }
            },
            "CloseFridge": {
                "r1": {
                    "kind": "solve",
                    "file_calls": [
                        ["write_text_file", other],
                        ["write_text_file", own],
                    ],
                }
            },
        }
    )
    s2.config(tasks=TASKS[:2], max_slots=2)
    proc = s2.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    # A write into another rung's folder blocks the freeze; an attempt on the read-only corpus does not.
    assert "refusing to freeze" in proc.stderr
    listed = json.loads(proc.stderr.split("--allow-cross-writes): ", 1)[1].strip())
    assert listed == {"CloseFridge": [other]}
    assert s2.record("CloseBlenderLid", 1)["file_tool_audit"]["memory_writes"] == [
        corpus_note
    ]
    assert s2.record("CloseFridge", 1)["file_tool_audit"]["cross_writes"] == [other]
    assert "CloseFridge r1 (cross_writes)" in s2.cli("status").stdout
    proc = s2.cli("freeze", "--allow-cross-writes")
    assert proc.returncode == 0, proc.stderr
    doc = json.loads((s2.run_root / "product" / "RULES.json").read_text())
    assert doc["cross_writes_accepted"] == {"CloseFridge": [other]}
