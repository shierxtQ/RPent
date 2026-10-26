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

"""RC-ECC stage-3 reflection: digest of RoboCasa episodes, prompt, reply and bookkeeping.

``data/smoke_episode`` is a real RoboCasa evaluation episode (CloseToasterOvenDoor,
seed 1, a hybrid K=5 smoke) with paths and the memory note's text replaced.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from scripts.ecc_robocasa import reflect as R

SMOKE = Path(__file__).resolve().parent / "data" / "smoke_episode"


def smoke_messages() -> list[dict]:
    return json.loads(
        (SMOKE / "transcript_CloseToasterOvenDoor_target_s1.json").read_text()
    )["messages"]


def tool_result(name: str) -> str:
    return next(
        m["content"]
        for m in smoke_messages()
        if m.get("role") == "tool" and m.get("name") == name
    )


def failed_rldx() -> str:
    """The real rldx_arm result with the outcome of a capped, unsuccessful call."""
    data = json.loads(tool_result("rldx_arm"))
    data["success"] = data["robocasa_terminated"] = False
    data["task_progress"] = {"door_closed": False}
    data["log"]["result"].update(
        ok=False, status="cap", error="no progress in 40 chunks"
    )
    return json.dumps(data, indent=2)


# -- tool results: outcome first ---------------------------------------------


def test_the_real_rldx_outcome_lies_beyond_his_clip_and_comes_first_here():
    content = tool_result("rldx_arm")
    collapsed = re.sub(r"\s+", " ", content)
    # The LIBERO digest showed the first 160-300 characters: no outcome in them.
    assert collapsed.index('"ok": true') > 300
    assert '"ok"' not in R._short(content, 300)
    shown = R.show_result(content, 160)
    assert shown.startswith('[skill=rldx_arm ok=true status="success" ')
    for field in (
        "chunks=29",
        "task_progress={}",
        "success=true",
        "terminated=true",
        "step=1",
        "eef=[3.170,-0.317,1.227]",
        "grip=0.040",
    ):
        assert field in shown
    outcome, rest = R.tool_outcome(content)
    assert (
        "Close the toaster oven door." not in outcome
    )  # the task language is not repeated
    assert len(R._short(rest, 160)) <= 163


def test_a_failed_motion_is_visible_whatever_the_clip():
    shown = R.show_result(failed_rldx(), 0)
    assert shown.startswith(
        '[skill=rldx_arm ok=false status="cap" error="no progress in 40 chunks"'
    )
    assert 'task_progress={"door_closed": false}' in shown and "success=false" in shown


def test_primitive_and_tool_level_failures():
    move = {
        "step": 4,
        "task_progress": {},
        "state": {"robot0_eef_pos": [1, 2, 3]},
        "success": False,
        "log": {
            "command": {"action": "navigate_to", "xy": [1, 2]},
            "result": {
                "ok": False,
                "steps": 600,
                "final_dist": 0.41,
                "moved": 0.02,
                "stuck": True,
            },
        },
    }
    shown = R.show_result(json.dumps(move), 50)
    assert shown.startswith(
        "[skill=navigate_to ok=false stuck=true final_dist=0.41 moved=0.02 steps=600"
    )
    refused = R.show_result(
        json.dumps({"error": "finish refused", "reason": "2 of 3 attempts left"}), 50
    )
    assert refused == '[error="finish refused" reason="2 of 3 attempts left"]'
    initial = R.show_result(tool_result("view_env_state"), 100)
    assert "skill=" not in initial and initial.startswith(
        "[task_progress={} success=false"
    )
    read = R.show_result(tool_result("read_text_file"), 100)
    assert (
        "path=" in read and "placeholder" not in read
    )  # the file's content is not dumped
    projected = R.show_result(tool_result("back_project_batch"), 60)
    assert projected.startswith('[step=0 summary={"valid_count": 6')
    assert R.show_result("not json at all", 5) == "not j..."


# -- the digest ----------------------------------------------------------------


def test_digest_of_the_smoke_episode_shows_the_plan_turn_with_outcomes():
    d = R.digest(SMOKE)
    assert (d["interventions"], d["shown"], d["turns"], d["source"]) == (
        1,
        1,
        6,
        "transcript",
    )
    assert d["by_reason"] == {"plan": 1}
    text = d["text"]
    assert text.startswith("### Intervention at turn 3 (trigger: plan)")
    before, after = text.split("Your turn:")
    assert "worker turn 1:" in before and "worker turn 2:" in before
    assert "call back_project_batch(" in after.split("After:")[0]
    assert (
        '-> [skill=rldx_arm ok=true status="success"' in after
    )  # turn 4, after the plan
    assert d["finish"]["status"] == "success"
    assert R.decide(d) is None
    assert R.task_language(SMOKE) == "Close the toaster oven door."


def timed_out(tmp_path: Path) -> Path:
    """The smoke episode as a planner timeout leaves it: transcript without turns."""
    out = tmp_path / "ep"
    shutil.copytree(SMOKE, out)
    path = out / "transcript_CloseToasterOvenDoor_target_s1.json"
    record = json.loads(path.read_text())
    record.update(messages=record["messages"][:1], finish=None, stats={})
    path.write_text(json.dumps(record))
    return out


def test_a_timed_out_episode_is_rebuilt_from_run_log_with_full_outcomes(tmp_path):
    out = timed_out(tmp_path)
    assert R.transcript_turns(out)[0] == []
    d = R.digest(out)
    assert (d["source"], d["turns"], d["shown"]) == ("run.log", 6, 1)
    # The logged rldx_arm result is clipped at 350 characters; states.json step 1 restores it.
    logged = next(
        line
        for line in (out / "run.log").read_text().splitlines()
        if "[tool<] rldx_arm" in line
    )
    assert '"ok"' not in logged
    assert '-> [skill=rldx_arm ok=true status="success" chunks=29' in d["text"]
    assert "says: The task is complete." in d["text"]
    turns, _ = R.run_log_turns(out)
    assert (
        "Now I'll write the audit file and finish." in turns[4].texts[0]
    )  # a continuation line
    assert [name for name, _, _ in turns[2].calls] == ["back_project_batch"]
    assert d["finish"]["status"] == "success"  # from the FINISH line
    prompt = R.build_prompt(
        d,
        language="x",
        rule={},
        success=False,
        termination="planner_timeout",
        entries=[],
    )
    assert "rebuilt from the run log" in prompt and "(planner_timeout)" in prompt


def test_no_planner_turn_anywhere_is_skipped(tmp_path):
    out = timed_out(tmp_path)
    (out / "run.log").write_text(
        "2026-10-26 12:22:53 I [agent] physical agent cmd: rpent ...\n"
    )
    d = R.digest(out)
    assert (d["interventions"], d["shown"], d["source"]) == (1, 0, "none")
    assert R.decide(d).startswith("no planner turn")
    (out / "escalation.jsonl").write_text('{"config": {"plan": true}}\n')
    assert R.decide(R.digest(out)) == "no senior intervention"


# -- prompt --------------------------------------------------------------------


def test_prompt_is_robocasa_and_keeps_his_reply_schema():
    d = R.digest(SMOKE)
    entries = [
        {
            "id": "s1_a1.1",
            "status": "candidate",
            "trigger": "checkin",
            "situation": "a",
            "diagnosis": "b",
            "fix": "c",
            "used_in": [],
            "successes": 0,
            "failures": 1,
        }
    ]
    prompt = R.build_prompt(
        d,
        language=R.task_language(SMOKE),
        rule={"gpt_plan": True, "checkin_K": 6, "errors_N": 2},
        success=True,
        termination="completed",
        entries=entries,
    )
    assert (
        "LIBERO" not in prompt and "RoboCasa kitchen" in prompt and "rldx_arm" in prompt
    )
    assert "Task: Close the toaster oven door." in prompt
    assert (
        'Call rule of this task: {"gpt_plan": true, "checkin_K": 6, "errors_N": 2}'
        in prompt
    )
    assert "Episode outcome: SUCCESS after 6 turns (completed)" in prompt
    shown = json.loads(
        prompt.split("Current correction memory (JSON):\n")[1].split(
            "\n\nEdit the memory"
        )[0]
    )
    assert shown == [
        {k: entries[0][k] for k in R.SHOWN_KEYS}
    ]  # bookkeeping fields stay hidden
    assert (
        '{"applied": ["id", ...],' in prompt
        and '"note": "one sentence on what this episode taught"}' in prompt
    )
    assert "rebuilt from the run log" not in prompt


# -- reply and bookkeeping (run_ecc_stage3.apply) --------------------------------


def entry(eid: str, **extra) -> dict:
    return {
        "id": eid,
        "trigger": "checkin",
        "situation": "s",
        "diagnosis": "d",
        "fix": "f",
        "status": "candidate",
        "written_in": "T/s1_a1",
        "used_in": [],
        "successes": 0,
        "failures": 0,
        **extra,
    }


def test_parse_edits_as_his_and_drops_malformed_items():
    text = (
        'Sure.\n{"applied": ["a"], "add": [{"situation": "x", "diagnosis": "y", "fix": "z"}, "bad"], '
        '"revise": "oops", "remove": [1, ["x"]], "note": "n"}\nthanks'
    )
    edits = R.parse_edits(text)
    assert (
        edits["applied"] == ["a"] and edits["remove"] == ["1"] and edits["revise"] == []
    )
    assert edits["add"] == [{"situation": "x", "diagnosis": "y", "fix": "z"}]
    assert edits["malformed"] == 3 and edits["note"] == "n"
    with pytest.raises(ValueError):
        R.parse_edits("no json here")
    done = R.reviewed([entry("a")], "no json", "T/s1_a2", False)
    assert (
        done["status"] == "unparsed"
        and done["entries_after"] == [entry("a")]
        and done["used"] == []
    )


def test_apply_verifies_on_success_and_drops_after_two_failures():
    s = R.Settings()
    edits = dict(
        R.NO_EDITS,
        add=[
            {"trigger": "errors", "situation": "x" * 900, "diagnosis": "y", "fix": "z"}
        ],
    )
    after = R.apply([entry("s1_a1.1")], edits, "T/s1_a2", False, {"s1_a1.1"}, s)
    assert after[0]["failures"] == 1 and after[0]["status"] == "candidate"
    assert (
        after[1]["id"] == "s1_a2.1"
        and after[1]["status"] == "candidate"
        and len(after[1]["situation"]) == 500
    )
    after = R.apply(after, dict(R.NO_EDITS), "T/s1_a3", False, {"s1_a1.1"}, s)
    assert [e["id"] for e in after] == [
        "s1_a2.1"
    ]  # applied in 2 failures, never in a success
    revised = R.apply(
        [entry("a", failures=1)],
        dict(R.NO_EDITS, revise=[{"id": "a", "fix": "new"}]),
        "T/s2_a1",
        False,
        {"a"},
        s,
    )
    assert (
        revised[0]["fix"] == "new" and revised[0]["failures"] == 0
    )  # a revision restarts the count
    won = R.apply(
        [entry("a"), entry("b")],
        dict(R.NO_EDITS, add=[{"situation": "x", "diagnosis": "y", "fix": "z"}]),
        "T/s2_a2",
        True,
        {"a"},
        s,
    )
    assert [(e["id"], e["status"]) for e in won] == [
        ("a", "verified"),
        ("b", "candidate"),
        ("s2_a2.1", "verified"),
    ]
    assert R.verified(won) == [won[0], won[2]]
    full = [entry(str(i)) for i in range(8)]
    assert (
        len(
            R.apply(
                full,
                dict(
                    R.NO_EDITS, add=[{"situation": "x", "diagnosis": "y", "fix": "z"}]
                ),
                "T/s3_a1",
                True,
                set(),
                s,
            )
        )
        == 8
    )


def test_the_interventions_cap_cuts_at_a_block_boundary():
    blocks = [
        f"### Intervention at turn {i} (trigger: checkin)\n" + "x" * 300
        for i in range(1, 6)
    ]
    text = "\n\n".join(blocks)
    clipped = R.clip_blocks(text, 1000)  # each block is 346 characters: two fit
    assert clipped == "\n\n".join(blocks[:2]) + (
        "\n\n(3 later intervention(s) left out: the prompt keeps the first 1000 characters)"
    )
    assert R.clip_blocks(text, len(text)) == text
    assert (
        R.clip_blocks("y" * 50, 10)
        == "y" * 10
        + "...\n\n(0 later intervention(s) left out: "
        "the prompt keeps the first 10 characters)"
    )


def test_render_and_episode_names_are_as_in_run_ecc_stage3():
    e = entry("s2_a1.3", status="verified")
    assert (
        R.render([e])
        == "- [s2_a1.3] (verified; trigger: checkin) situation: s | diagnosis: d | fix: f"
    )
    assert R.render_file([]) == "" and R.render_file([e]).endswith("\n")
    names = [R.episode_name(n) for n in range(15)]
    assert names[:4] == ["s1_a1", "s1_a2", "s1_a3", "s2_a1"] and names[-1] == "s5_a3"
    assert len(set(names)) == 15
