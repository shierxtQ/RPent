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

"""RC-ECC stage 3 v2: session seeds, scene-free entries, cross-scene verification, stop rules.

Fake episodes (RPent's real ``EscalationModel`` with scripted models), a fake
reflection call that also answers rewrite requests, fake RLDX servers and
GPUs, as in ``test_stage3_runner.py``. In the fake episodes CloseFridge's rule
(Qwen plan, K = 12) calls GPT-6 only from request 13: an episode of 9
requests has no senior turn, one of 15 has one check-in.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import reflect as R
from scripts.ecc_robocasa import stage3 as S3
from tests.unit_tests.ecc_robocasa.test_stage3_parallel import (
    edit_config,
    kill_runner,
    popen_runner,
    progress_has,
)
from tests.unit_tests.ecc_robocasa.test_stage3_runner import STAGE3, S3Env

pytestmark = pytest.mark.timeout(400)
SEEDS = [12, 13, 14]
V2 = {
    "sessions": 3,
    "session_seeds": SEEDS,
    "cross_scene_verify": True,
    "stop_rule": "cross_scene",
}
PARALLEL_V2 = {**V2, "parallel_attempts": True}
SOLVE = {"kind": "solve", "requests": 15}  # CloseFridge: one GPT-6 check-in
FAIL = {"kind": "fail", "requests": 15}
QWEN_SOLVE = {"kind": "solve", "requests": 9}  # CloseFridge: no GPT-6 turn


def seeds_started(env: S3Env, task: str) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for e in env.events("start"):
        if e["task"] == task:
            out.setdefault(e["episode"], set()).add(e["seed"])
    return out


def record(env: S3Env, task: str) -> dict:
    return json.loads((env.task(task) / "record.json").read_text())


def final_entries(env: S3Env, task: str) -> list[dict]:
    return json.loads((env.root / "product" / "final" / f"{task}.json").read_text())[
        "entries"
    ]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_the_v2_switches_are_opt_in_and_validated(ecc):
    env = S3Env(ecc, ["CloseFridge"], scenario={})
    off = S3.load_config(env.config)
    edit_config(env, stage3=dict(S3.V2_DEFAULTS))
    explicit = S3.load_config(env.config)
    # Off (absent or explicit defaults): the experiment settings and hash are stage 3's.
    assert S3.experiment_settings(off) == {
        k: off["stage3"][k] for k in S3.EXPERIMENT_KEYS
    }
    assert S3.experiment_hash(off) == S3.experiment_hash(explicit)
    assert "session_seeds" not in off["stage3"]  # never merged into the config
    edit_config(env, stage3={**V2, "scene_free": True})
    on = S3.load_config(env.config)
    assert S3.experiment_settings(on) == {
        **{k: on["stage3"][k] for k in S3.EXPERIMENT_KEYS},
        "session_seeds": SEEDS,
        "scene_free": True,
        "cross_scene_verify": True,
        "stop_rule": "cross_scene",
    }
    assert S3.experiment_hash(on) != S3.experiment_hash(off)
    plan = env.run("plan", "--run-id", "r1")
    assert plan.returncode == 0, plan.stderr
    assert (
        "stage 3 v2: session seeds [12, 13, 14] (a session's attempts share its seed)"
        in plan.stdout
    )
    assert "stop rule cross_scene" in plan.stdout

    bad = [
        ({"session_seeds": [12, 3, 14]}, "Target50 cells"),
        ({"session_seeds": [12, 11, 14]}, "stage 2's seed"),
        ({"session_seeds": [12, 13]}, "2 seeds for 3 sessions"),
        ({"session_seeds": [12, 13, 13]}, "distinct"),
        ({"session_seeds": [13, 14, 15]}, "must equal stage3.seed"),
        ({"session_seeds": [12, "13", 14]}, "list of nonnegative integers"),
        (
            {
                "session_seeds": [],
                "cross_scene_verify": True,
                "stop_rule": "first_success",
            },
            "needs stage3.session_seeds",
        ),
        ({"cross_scene_verify": False}, 'stop_rule = "cross_scene" needs'),
        ({"stop_rule": "never"}, "stop_rule must be one of"),
        ({"scene_free": "yes"}, "scene_free must be true or false"),
        ({"scene_free_allowed_numbers": [0.95]}, "numbers as strings"),
        ({"unneeded_sessions": 4}, "unneeded_sessions"),
    ]
    for change, message in bad:
        edit_config(env, stage3={**S3.V2_DEFAULTS, **V2, **change})
        with pytest.raises(ValueError, match=message):
            S3.load_config(env.config)


# ---------------------------------------------------------------------------
# cross-scene verification and the stop rule, parallel sessions
# ---------------------------------------------------------------------------


def test_sessions_run_on_their_seeds_and_only_another_scene_verifies(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": SOLVE,  # a success on seed 12: under stage 3 the task would stop here
            "s1_a2": FAIL,
            "s1_a3": FAIL,
            "s2_a1": SOLVE,  # seed 13: applies the three seed-12 entries
            "s2_a2": FAIL,
            "s2_a3": FAIL,
        }
    }
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario, stage3=PARALLEL_V2)
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert '"ok": true' in proc.stdout  # frozen, verified

    assert seeds_started(env, "CloseFridge") == {
        **{f"s1_a{a}": {12} for a in (1, 2, 3)},
        **{f"s2_a{a}": {13} for a in (1, 2, 3)},
    }
    for name, seed in (("s1_a1", 12), ("s2_a3", 13)):
        assert (
            json.loads((env.task("CloseFridge") / name / "episode.json").read_text())[
                "seed"
            ]
            == seed
        )
        state = json.loads(
            (env.root / "state" / "jobs" / f"CloseFridge__{name}.json").read_text()
        )
        assert state["seed"] == seed
    # Session 1: written from a success, nothing is verified, and the success does not stop the task.
    s1 = env.reflection("CloseFridge", "s1_a3")["entries_after"]
    assert [x["id"] for x in s1] == ["s1_a1.1", "s1_a2.1", "s1_a3.1"]
    assert {x["status"] for x in s1} == {"candidate"}
    assert {(x["written_seed"], x["text_seed"]) for x in s1} == {(12, 12)}
    # Session 2 (seed 13): the success applied the three entries it ran with: verified.
    a1 = env.reflection("CloseFridge", "s2_a1")
    assert a1["seed"] == 13 and a1["used"] == ["s1_a1.1", "s1_a2.1", "s1_a3.1"]
    assert a1["verified_now"] == ["s1_a1.1", "s1_a2.1", "s1_a3.1"]
    rec = record(env, "CloseFridge")
    assert rec["stop_reason"] == "verified" and rec["status"] == "solved"
    assert rec["sessions_run"] == 2 and rec["episodes"] == 6
    assert rec["first_success"] == "s1_a1" and rec["successes_by_seed"] == {
        "12": 1,
        "13": 1,
    }
    assert rec["session_seeds"] == SEEDS and rec["seeds_run"] == [12, 13]
    for a in (1, 2, 3):
        state = json.loads(
            (env.root / "state" / "jobs" / f"CloseFridge__s3_a{a}.json").read_text()
        )
        assert state["final"]["class"] == "not_needed" and state["seed"] == 14

    kept = final_entries(env, "CloseFridge")
    assert [x["id"] for x in kept] == ["s1_a1.1", "s1_a2.1", "s1_a3.1"]
    for x in kept:
        assert x["status"] == "verified" and x["text_seed"] == 12
        assert x["verified_by"] == {"episode": "CloseFridge/s2_a1", "seed": 13}
        assert {"episode": "CloseFridge/s2_a1", "seed": 13, "success": True} in x[
            "applied_in"
        ]
    # The rendered memory the final test reads is unchanged in format.
    product = env.root / "product"
    assert (product / "final" / "CloseFridge.md").read_text() == R.render_file(kept)
    doc = json.loads((product / "PRODUCT.json").read_text())
    assert (
        doc["v2"]["session_seeds"] == SEEDS and doc["v2"]["stop_rule"] == "cross_scene"
    )
    assert doc["tasks"]["CloseFridge"]["stop_reason"] == "verified"
    assert doc["summary"]["stop_reasons"]["verified"] == 1
    manifest = json.loads((env.root / "manifest.json").read_text())
    assert manifest["budget"]["session_seeds"] == SEEDS
    assert "cross-scene-verified" in manifest["budget"]["stop"]
    status = env.run("status", "--run-id", "r1", "--tasks-table")
    assert status.returncode == 0, status.stderr
    assert "stage 3 v2: session seeds [12, 13, 14]" in status.stdout
    assert "stopped: verified 1" in status.stdout and "stop verified" in status.stdout


def test_no_correction_needed_budget_and_infra_stops(ecc):
    scenario = {
        "CloseFridge": {"default": QWEN_SOLVE},  # Qwen alone every time
        "CloseToasterOvenDoor": {
            "default": {"kind": "fail", "requests": 7}
        },  # never solved
        "CloseBlenderLid": {
            "s1_a1": {"kind": "fail", "requests": 7},
            "s1_a2": [{"kind": "infra"}, {"kind": "infra"}],
            "s1_a3": {"kind": "fail", "requests": 7},
        },
    }
    env = S3Env(
        ecc,
        ["CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor"],
        scenario=scenario,
        stage3={**PARALLEL_V2, "infra_retries": 1},
        endpoints=((4, 6),),
        max_slots=6,
    )
    env.go()  # CloseBlenderLid ran out of relaunches: not frozen
    fridge = record(env, "CloseFridge")
    assert (
        fridge["stop_reason"] == "no_correction_needed" and fridge["sessions_run"] == 2
    )
    assert fridge["status"] == "solved" and fridge["entries_kept"] == 0
    assert fridge["reflections"] == {"applied": 0, "unparsed": 0, "skipped": 6}
    toaster = record(env, "CloseToasterOvenDoor")
    assert toaster["stop_reason"] == "budget" and toaster["status"] == "unsolved"
    assert toaster["sessions_run"] == 3 and toaster["episodes"] == 9
    assert seeds_started(env, "CloseToasterOvenDoor")["s3_a2"] == {14}
    blender = record(env, "CloseBlenderLid")
    assert (
        blender["stop_reason"] == "infra_exhausted"
        and blender["status"] == "infra_exhausted"
    )
    assert not (env.root / "product").exists()


def test_a_session_that_needed_gpt6_resets_the_no_correction_count(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": QWEN_SOLVE,
            "s1_a2": QWEN_SOLVE,
            "s1_a3": QWEN_SOLVE,
            "s2_a1": QWEN_SOLVE,
            "s2_a2": SOLVE,  # a GPT-6 check-in: this session needed the senior
            "s2_a3": QWEN_SOLVE,
            "default": QWEN_SOLVE,
        }
    }
    reflect = {
        "default": {"edits": {"applied": [], "add": [], "revise": [], "remove": []}}
    }
    env = S3Env(
        ecc, ["CloseFridge"], scenario=scenario, reflect=reflect, stage3=PARALLEL_V2
    )
    env.go()
    rec = record(env, "CloseFridge")
    # Sessions 2 and 3 are not two GPT-6-free sessions in a row; the budget ends the task.
    assert rec["stop_reason"] == "budget" and rec["sessions_run"] == 3
    # One GPT-6-free session is enough with unneeded_sessions = 1 (a new run).
    edit_config(env, stage3={**PARALLEL_V2, "unneeded_sessions": 1})
    proc = env.run("run", "--run-id", "r2", "--go")
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    rec = json.loads(
        (ecc.runs / "r2" / "tasks" / "CloseFridge" / "record.json").read_text()
    )
    assert rec["stop_reason"] == "no_correction_needed" and rec["sessions_run"] == 1


# ---------------------------------------------------------------------------
# sequential order: the same scene never verifies
# ---------------------------------------------------------------------------


def test_sequential_same_scene_successes_do_not_verify(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": FAIL,
            "s1_a2": SOLVE,  # seed 12, applies s1_a1.1 (written on seed 12)
            "s1_a3": SOLVE,
            "s2_a1": FAIL,
            "s2_a2": SOLVE,  # seed 13: verifies what it applied
            "s2_a3": FAIL,
        }
    }
    reflect = {
        # s2_a1 revises s1_a2.1: its text is then from seed 13 and s2_a2 cannot verify it.
        "CloseFridge/s2_a1": {
            "edits": {
                "applied": ["s1_a1.1", "s1_a2.1", "s1_a3.1"],
                "add": [],
                "revise": [{"id": "s1_a2.1", "fix": "start below the edge"}],
                "remove": [],
            }
        }
    }
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario, reflect=reflect, stage3=V2)
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    a2 = {x["id"]: x for x in env.reflection("CloseFridge", "s1_a2")["entries_after"]}
    assert a2["s1_a1.1"]["status"] == "candidate" and a2["s1_a1.1"]["successes"] == 1
    assert a2["s1_a1.1"]["applied_in"] == [
        {"episode": "CloseFridge/s1_a2", "seed": 12, "success": True}
    ]
    assert a2["s1_a2.1"]["status"] == "candidate"  # written from a success
    s1 = env.reflection("CloseFridge", "s1_a3")["entries_after"]
    assert {x["status"] for x in s1} == {"candidate"}
    after = {
        x["id"]: x for x in env.reflection("CloseFridge", "s2_a2")["entries_after"]
    }
    assert (
        after["s1_a1.1"]["status"] == "verified"
        and after["s1_a3.1"]["status"] == "verified"
    )
    assert (
        after["s1_a2.1"]["status"] == "candidate"
        and after["s1_a2.1"]["text_seed"] == 13
    )
    rec = record(env, "CloseFridge")
    assert rec["stop_reason"] == "verified" and rec["sessions_run"] == 2
    assert seeds_started(env, "CloseFridge")["s2_a3"] == {
        13
    }  # the session is completed
    kept = final_entries(env, "CloseFridge")
    assert sorted(x["id"] for x in kept) == ["s1_a1.1", "s1_a3.1"]
    assert all(x["verified_by"]["episode"] == "CloseFridge/s2_a2" for x in kept)


# ---------------------------------------------------------------------------
# scene-free entries: the check and the rewrite request
# ---------------------------------------------------------------------------

LANG = "Close the toaster oven door."  # the fake episodes' task language


def add(fix, situation="the worker pushed the door from above"):
    return {
        "trigger": "checkin",
        "situation": situation,
        "diagnosis": "wrong contact point",
        "fix": fix,
    }


def test_scene_bound_entries_are_rewritten_once_or_dropped(ecc):
    scenario = {
        "CloseFridge": {"s1_a1": FAIL, "s1_a2": FAIL, "s1_a3": FAIL, "default": SOLVE}
    }
    reflect = {
        "CloseFridge/s1_a1": {
            "edits": {
                "applied": [],
                "add": [
                    add("push below the free edge, then re-check task_progress"),
                    add("push at pixels [[120, 44], [122, 47]]"),
                    add(f'call rldx_arm({{"prompt": "{LANG}"}})'),
                ],
                "revise": [],
                "remove": [],
            },
            # add 2 comes back scene-free; add 3 still names a pixel row: dropped
            "fix": {
                "rewrites": {
                    "add": [
                        {
                            "n": 2,
                            "fix": "push just below the handle seen in the current image",
                        },
                        {
                            "n": 3,
                            "fix": "call rldx_arm with the task language, aiming at row 5",
                        },
                    ]
                }
            },
        },
        "CloseFridge/s1_a2": {
            "edits": {
                "applied": [],
                "add": [add("open from step 3 on")],
                "revise": [],
                "remove": [],
            },
            "fix": {"status": "refused"},
        },
        "CloseFridge/s1_a3": {
            "edits": {
                "applied": [],
                "add": [],
                "revise": [{"id": "s1_a1.1", "fix": "push at joint_p 0.875"}],
                "remove": [],
            },
            # the first rewrite call fails; the retry rewrites it
            "fix": [{"status": "error"}, {}],
        },
        "default": {
            "edits": {
                "applied": [],
                "add": [add("re-localize the door, then retry")],
                "revise": [],
                "remove": [],
            }
        },
    }
    env = S3Env(
        ecc,
        ["CloseFridge"],
        scenario=scenario,
        reflect=reflect,
        stage3={**PARALLEL_V2, "scene_free": True},
    )
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    calls = [
        e
        for e in sorted(env.events("reflect"), key=lambda e: e["t"])
        if e["episode"].startswith("CloseFridge/s1")
    ]
    assert [(e["episode"].split("/")[1], e["kind"]) for e in calls] == [
        ("s1_a1", "main"),
        ("s1_a1", "fix"),
        ("s1_a2", "main"),
        ("s1_a2", "fix"),
        ("s1_a3", "main"),
        ("s1_a3", "fix"),
        ("s1_a3", "fix"),
    ]
    a1 = env.reflection("CloseFridge", "s1_a1")
    sf = a1["scene_free"]
    assert [r["item"] for r in sf["rejected"]] == ["add 2", "add 3"]
    assert sf["rejected"][0]["problems"][0]["rule"] == "a list of coordinates or pixels"
    assert sf["rejected"][1]["problems"][0]["rule"] == "this scene's task language"
    assert sf["rewrite"]["status"] == "ok"
    assert sf["rewrite"]["kept"] == ["add 2"] and sf["rewrite"]["dropped"] == ["add 3"]
    assert sf["rewrite"]["still_rejected"][0]["problems"][0]["match"] == "row 5"
    assert len(a1["edits_proposed"]["add"]) == 3 and len(a1["edits"]["add"]) == 2
    assert [x["fix"] for x in a1["entries_after"]] == [
        "push below the free edge, then re-check task_progress",
        "push just below the handle seen in the current image",
    ]
    one_call = env.reflection("CloseFridge", "s2_a1")["usd"]  # a clean reply: one call
    assert a1["calls"] == 2 and a1["usd"] == pytest.approx(2 * one_call)
    folder = env.task("CloseFridge") / "s1_a1" / "reflection"
    fix_prompt = (folder / "fix_prompt.txt").read_text()
    assert fix_prompt.startswith((folder / "prompt.txt").read_text())
    assert R.SCENE_FREE_RULE in (folder / "prompt.txt").read_text()
    assert (
        '- add 2: fix has a list of coordinates or pixels ("[120, 44]")' in fix_prompt
    )
    request = json.loads((folder / "fix_request.json").read_text())
    assert request["kind"] == "fix" and request["attempt"] == 1
    assert request["prompt_sha256"] == R.prompt_sha256(fix_prompt)

    a2 = env.reflection("CloseFridge", "s1_a2")
    assert a2["scene_free"]["rewrite"]["status"] == "refused"
    assert a2["scene_free"]["rewrite"]["dropped"] == ["add 1"]
    assert a2["entries_after"] == a2["entries_before"]
    a3 = env.reflection("CloseFridge", "s1_a3")
    assert (
        json.loads(
            (
                env.task("CloseFridge") / "s1_a3" / "reflection" / "fix_request.json"
            ).read_text()
        )["attempt"]
        == 2
    )
    assert a3["scene_free"]["rewrite"]["kept"] == ["revise s1_a1.1"]
    revised = {x["id"]: x for x in a3["entries_after"]}["s1_a1.1"]
    assert (
        revised["fix"]
        == "re-localize the target in the current image, then retry the skill once"
    )
    rstate = json.loads(
        (env.task("CloseFridge") / "s1_a3" / "reflection" / "state.json").read_text()
    )
    assert [h.get("phase") for h in rstate["history"]] == [None, "fix", "fix"]
    assert [h["status"] for h in rstate["history"]] == ["ok", "error", "ok"]
    clean = env.reflection("CloseFridge", "s2_a1")
    assert clean["scene_free"] == {"rejected": [], "rewrite": None}
    assert progress_has(
        env, event="scene_rejected", task="CloseFridge", episode="s1_a1"
    )
    # Nothing scene-bound reached the memory; the run froze and verifies.
    for name in ("s1_a1", "s1_a2", "s1_a3", "s2_a1", "s2_a2", "s2_a3"):
        for x in env.reflection("CloseFridge", name)["entries_after"]:
            text = " ".join(x[k] for k in R.ENTRY_FIELDS)
            assert R.scene_problems(text, language=LANG) == [], (name, x)
    rec = record(env, "CloseFridge")
    assert rec["scene_free"] == {
        "on": True,
        "reflections_with_rejections": 3,
        "rejected_items": 4,
        "rewritten_kept": 2,
        "dropped": 2,
    }
    assert S3.verify(env.root / "product")["ok"]


# ---------------------------------------------------------------------------
# resume after a kill: mid-session and during a rewrite request
# ---------------------------------------------------------------------------


def test_a_killed_runner_resumes_mid_session_and_mid_rewrite(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": {**FAIL, "sleep_s": 8},
            "s1_a2": {**FAIL, "sleep_s": 1},
            "s1_a3": {**FAIL, "sleep_s": 1},
            "default": SOLVE,
        }
    }
    reflect = {
        "CloseFridge/s1_a2": {
            "edits": {
                "applied": [],
                "add": [add("push at row 120")],
                "revise": [],
                "remove": [],
            },
            "fix": {"sleep_s": 6},
        }
    }
    env = S3Env(
        ecc,
        ["CloseFridge"],
        scenario=scenario,
        reflect=reflect,
        stage3={**PARALLEL_V2, "scene_free": True},
    )
    edit_config(env, scheduler={"launch_stagger_s": 3})
    runner = popen_runner(env)
    ecc.wait_for(
        lambda: progress_has(env, event="launch", job="CloseFridge__s1_a1"), timeout=120
    )
    kill_runner(env, runner)
    runner = popen_runner(env)
    ecc.wait_for(
        lambda: progress_has(
            env,
            event="reflection_start",
            task="CloseFridge",
            episode="s1_a2",
            phase="fix",
        ),
        timeout=180,
    )
    kill_runner(env, runner)
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert progress_has(env, event="adopt", job="CloseFridge__s1_a1")
    started = [
        (e["episode"], e["try"], e["seed"])
        for e in env.events("start")
        if e["episode"].startswith("s1_")
    ]
    assert sorted(started) == [("s1_a1", 1, 12), ("s1_a2", 1, 12), ("s1_a3", 1, 12)]
    fixes = [
        e
        for e in env.events("reflect")
        if e["episode"] == "CloseFridge/s1_a2" and e["kind"] == "fix"
    ]
    assert len(fixes) == 1  # the rewrite call was adopted, not sent again
    a2 = env.reflection("CloseFridge", "s1_a2")
    assert a2["scene_free"]["rewrite"]["kept"] == ["add 1"]
    order = [
        e["episode"]
        for e in sorted(env.events("reflect"), key=lambda e: e["t"])
        if e["kind"] == "main"
    ]
    assert order[:3] == [f"CloseFridge/s1_a{a}" for a in (1, 2, 3)]
    rec = record(env, "CloseFridge")
    assert rec["stop_reason"] == "verified" and rec["sessions_run"] == 2
    assert S3.verify(env.root / "product")["ok"]


# ---------------------------------------------------------------------------
# freeze and verify with the provenance; the run's own mode
# ---------------------------------------------------------------------------


def test_freeze_and_verify_check_the_provenance_and_the_run_mode(ecc):
    scenario = {
        "CloseFridge": {"s1_a1": FAIL, "s1_a2": FAIL, "s1_a3": FAIL, "default": SOLVE}
    }
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario, stage3=PARALLEL_V2)
    proc = env.go("--no-freeze")
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    args = argparse.Namespace(run_id="r1", run_root=None)
    # status, freeze and requeue read the run in its own mode, whatever the config says now.
    edit_config(
        env,
        stage3={
            "session_seeds": [],
            "cross_scene_verify": False,
            "stop_rule": "first_success",
            "parallel_attempts": False,
        },
    )
    _, manifest, stage = S3.load_stage(S3.load_config(env.config), args)
    assert (
        stage.v2
        and stage.parallel
        and stage.session_seeds == SEEDS
        and stage.stop_rule == "cross_scene"
    )
    assert S3.chain_problems(stage, "CloseFridge") == []
    proc = env.run("run", "--run-id", "r1", "--go")
    assert (
        proc.returncode != 0
        and "refusing to resume: experiment settings" in proc.stderr
    )
    edit_config(env, stage3=PARALLEL_V2)

    # An entry marked verified without a success on another scene breaks the chain.
    path = env.task("CloseFridge") / "s2_a1" / "reflection.json"
    good = path.read_text()
    doc = json.loads(good)
    (x,) = [x for x in doc["entries_after"] if x["status"] == "verified"][:1]
    x["text_seed"] = 13
    path.chmod(0o644)
    path.write_text(json.dumps(doc))
    problems = S3.chain_problems(stage, "CloseFridge")
    assert any(
        f"entry {x['id']} is verified without a success on another scene" in p
        for p in problems
    )
    path.write_text(good)
    episode = env.task("CloseFridge") / "s2_a2" / "episode.json"
    ep = json.loads(episode.read_text())
    episode.write_text(json.dumps({**ep, "seed": 12}))
    assert "s2_a2: not recorded on its session's seed 13" in S3.chain_problems(
        stage, "CloseFridge"
    )
    with pytest.raises(SystemExit, match="broken correction-memory chains"):
        S3.freeze(stage, manifest)
    episode.write_text(json.dumps(ep))
    result = S3.freeze(stage, manifest)
    assert result["ok"]

    product = env.root / "product"
    check = S3.verify(product)
    assert check["ok"], check
    doc = json.loads((product / "PRODUCT.json").read_text())
    copy = ecc.root / "final_copy"
    shutil.copytree(product / "final", copy)
    for p in copy.iterdir():
        p.chmod(0o644)
    assert S3.v2_product_problems(copy, doc) == []
    data = json.loads((copy / "CloseFridge.json").read_text())
    data["entries"][0]["verified_by"]["seed"] = data["entries"][0]["text_seed"]
    (copy / "CloseFridge.json").write_text(json.dumps(data))
    (copy / "CloseFridge.md").write_text(
        "- [x] (verified; trigger: any) situation: s | diagnosis: d | fix: f\n"
    )
    problems = S3.v2_product_problems(copy, doc)
    assert any("has no cross-scene verification" in p for p in problems)
    assert "CloseFridge.md does not render CloseFridge.json's entries" in problems
    # final_test reads <Task>.md; a v2 product passes its stage-3 verify.
    assert S3.verify(product)["ok"]


class ChainStage:
    """The parts of a parallel v2 ``Stage3`` that :func:`S3.v2_chain_problems` reads."""

    cross_scene = True
    scene_free = True
    allowed_numbers: list[str] = []

    def __init__(self, root: Path):
        self.root = root

    def name_of(self, e: int) -> str:
        return R.episode_name(e)

    def seed_of(self, e: int) -> int:
        return SEEDS[e // 3]

    def episode_dir(self, task: str, e: int) -> Path:
        return self.root / task / self.name_of(e)


def chain_entry(i, text_seed=12, **fields):
    return {
        "id": i,
        "trigger": "checkin",
        "situation": "the worker repeated a capped skill call",
        "diagnosis": "the skill made no progress from this pose",
        "fix": "re-localize the target in the current image, then retry the skill once",
        "status": "candidate",
        "written_in": "T/s1_a1",
        "used_in": [],
        "successes": 0,
        "failures": 0,
        "written_seed": 12,
        "text_seed": text_seed,
        "text_from": "T/s1_a1",
        "applied_in": [],
        "verified_by": None,
        **fields,
    }


def test_the_freeze_check_proves_verification_and_matches_the_runner_scene_check(
    tmp_path,
):
    """Freeze re-checks each reflection as the runner did: an entry verified here was in the
    episode's corrections file as it stood; the scene check runs per changed field against
    the episode's own language (a joined text or another scene's language never fails it)."""
    stage = ChainStage(tmp_path)
    folder = stage.episode_dir("T", 4)  # s2_a2, seed 13
    folder.mkdir(parents=True)
    (folder / "episode.json").write_text(json.dumps({"seed": 13}))
    try_dir = folder / "try1"
    try_dir.mkdir()
    (try_dir / "states.json").write_text(
        json.dumps({"steps": [{"extras": {"task_language": "Open the left drawer."}}]})
    )
    x = chain_entry("s1_a1.1")
    verified = {
        **x,
        "status": "verified",
        "successes": 1,
        "used_in": ["T/s2_a2"],
        "applied_in": [{"episode": "T/s2_a2", "seed": 13, "success": True}],
        "verified_by": {"episode": "T/s2_a2", "seed": 13},
    }
    # Written on seed 12 with "the right drawer" (that scene's language was other); revised
    # here in its fix only; "...step" + "3 tries" would join into a step number.
    y = chain_entry(
        "s1_a2.1",
        situation="the gripper slipped when trying to open the left drawer",
        diagnosis="the grasp was too shallow at the start of the step",
    )
    y_after = {**y, "fix": "3 tries at most, then re-localize the handle"}
    record = {
        "episode": "T/s2_a2",
        "seed": 13,
        "success": True,
        "status": "applied",
        "used": ["s1_a1.1"],
        "try_dir": str(try_dir),
        "entries_before": [x, y],
        "entries_after": [verified, y_after],
    }
    assert S3.v2_chain_problems(stage, "T", 4, record, [x, y]) == []
    # The episode ran with another text of the entry (a1's reflection revised it after the
    # session began): a success that never saw this text cannot verify it.
    shown_other = {**x, "fix": "an older fix"}
    (problem,) = S3.v2_chain_problems(stage, "T", 4, record, [shown_other, y])
    assert "entry s1_a1.1 is verified without a success on another scene" in problem
    # No applied_in row of this episode, or a text from this scene: refused as well.
    for bad in (
        {**verified, "applied_in": []},
        {**verified, "text_seed": 13},
    ):
        broken = {**record, "entries_after": [bad, y_after]}
        assert S3.v2_chain_problems(stage, "T", 4, broken, [x, y]) != []
    # A changed field with a scene detail is caught, with the field and the rule named.
    leaky = {**record, "entries_after": [verified, {**y, "fix": "push at row 120"}]}
    (problem,) = S3.v2_chain_problems(stage, "T", 4, leaky, [x, y])
    assert (
        "entry s1_a2.1 is not scene-free: fix a pixel position ('row 120')" in problem
    )
    # A new entry's trigger is checked too.
    new = chain_entry("s2_a2.1", text_seed=13, trigger="checkin at step 19")
    added = {**record, "entries_after": [verified, y_after, new]}
    (problem,) = S3.v2_chain_problems(stage, "T", 4, added, [x, y])
    assert "entry s2_a2.1 is not scene-free: trigger" in problem


def test_reflect_only_prints_the_v2_prompt_with_a_v2_config(ecc):
    """The offline prompt check (no call) shows the instructions of the config's mode."""
    smoke = Path(__file__).resolve().parent / "data" / "smoke_episode"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env.pop("ECC_RC_CONFIG", None)
    plain = subprocess.run(
        [sys.executable, str(STAGE3), "reflect-only", str(smoke)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert plain.returncode == 0, plain.stderr
    assert R.CONCRETE_RULE in plain.stdout and R.SCENE_FREE_RULE not in plain.stdout
    s3 = S3Env(ecc, ["CloseFridge"], scenario={}, stage3={**V2, "scene_free": True})
    v2 = s3.run("reflect-only", str(smoke))
    assert v2.returncode == 0, v2.stderr
    assert R.SCENE_FREE_RULE in v2.stdout and R.FROZEN_RULE in v2.stdout
    assert R.CONCRETE_RULE not in v2.stdout
