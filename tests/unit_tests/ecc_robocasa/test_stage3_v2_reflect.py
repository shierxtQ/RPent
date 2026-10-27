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

"""Stage 3 v2 reflection bookkeeping: scene-free entries and cross-scene verification.

The positive validator cases are fix texts GPT-6 wrote in stage 3 (seed 12)
on LoadDishwasher, OpenDrawer and others.
"""

from __future__ import annotations

import json

import pytest

from scripts.ecc_robocasa import reflect as R

S = R.Settings()


def entry(i, *, status="candidate", text_seed=12, successes=0, failures=0, fix="f"):
    return {
        "id": i,
        "trigger": "checkin",
        "situation": f"situation of {i}",
        "diagnosis": "d",
        "fix": fix,
        "status": status,
        "written_in": "T/s1_a1",
        "used_in": [],
        "successes": successes,
        "failures": failures,
        "written_seed": 12,
        "text_seed": text_seed,
        "text_from": "T/s1_a1",
        "applied_in": [],
        "verified_by": None,
    }


def edits(applied=(), add=(), revise=(), remove=()):
    return {
        "applied": list(applied),
        "add": list(add),
        "revise": list(revise),
        "remove": list(remove),
        "note": "n",
    }


ADD = {"trigger": "checkin", "situation": "s", "diagnosis": "d", "fix": "f"}


# ---------------------------------------------------------------------------
# the scene check
# ---------------------------------------------------------------------------

SCENE_BOUND = [
    (
        "At step 0, use back_project_batch with bowl pixels [[164,12],[166,16],[168,11]] "
        "and cup pixels [[161,36],[165,36],[168,39]], camera='agentview'.",
        "",
        {
            "a list of coordinates or pixels",
            "a step, turn or seed number of this episode",
        },
    ),
    (
        'Initially call rldx_skill({"prompt":"Open the right drawer."}); this single call '
        "completed with success=true and joint_p=1.0653.",
        "Open the right drawer.",
        {
            "this scene's task language",
            "a measured value (a number with two or more decimals)",
        },
    ),
    (
        "the visible right drawer handle is near image row 204, column 135",
        "",
        {"a pixel position"},
    ),
    (
        "the next call detected a dish0 grasp; Dish1 slipped",
        "",
        {"an object instance id"},
    ),
    ("Robot spawned at base ~[3.44,-0.95]", "", {"a list of coordinates or pixels"}),
    (
        "Treat grip≈0.04 as open in this episode",
        "",
        {"a measured value (a number with two or more decimals)"},
    ),
    (
        'back_project_batch({"pixels": [[204, 131]], "step": 19}); on seed 12 at turn 7',
        "",
        {
            "a list of coordinates or pixels",
            "a step, turn or seed number of this episode",
        },
    ),
    (
        "call rldx_skill with prompt='Pick the pizza cutter from the drawer'",
        "Pick the pizza cutter from the drawer and place it on the counter.",
        {"this scene's task language"},
    ),
]
SCENE_FREE = [
    (
        "Call rldx_skill with the task language verbatim; after two consecutive calls "
        "without progress, re-localize the handle in the current image first.",
        "Open the right drawer.",
    ),
    (
        "If the gripper closes on nothing, release, back off 5 cm and re-grasp at the "
        "handle's centre.",
        "",
    ),
    ("Check in every K=6 requests; turn the base about 0.3 rad (steps=20).", ""),
    ("Qwen3 and GPT-6 agree; rldx_arm keeps the base clamped.", ""),
    (
        "The worker's grasp didn't hold, so it should re-localize.",
        "Close the fridge door.",
    ),
]


@pytest.mark.parametrize(("text", "language", "rules"), SCENE_BOUND)
def test_the_scene_check_flags_what_stage_3_notes_carried(text, language, rules):
    found = R.scene_problems(text, language=language)
    assert {rule for rule, _ in found} >= rules
    assert all(
        match and match.lower() in text.lower() or rule.startswith("this scene")
        for rule, match in found
    )


@pytest.mark.parametrize(("text", "language"), SCENE_FREE)
def test_the_scene_check_passes_scene_free_rules(text, language):
    assert R.scene_problems(text, language=language) == []


def test_allowed_numbers_are_exempt_from_the_decimals_rule_only():
    text = "joint_p above 0.95 means the drawer is open"
    assert [r for r, _ in R.scene_problems(text)] == [
        "a measured value (a number with two or more decimals)"
    ]
    assert R.scene_problems(text, allowed=["0.95"]) == []
    assert R.scene_problems("at [0.95, 0.10]", allowed=["0.95", "0.10"]) != []


def test_rejections_check_only_items_apply_would_use():
    before = [entry("a"), entry("v", status="verified")]
    bad = {**ADD, "fix": "push at row 120"}
    found = R.scene_rejections(
        edits(
            add=[
                ADD,
                bad,
                {"trigger": "any", "situation": "x [1,2]", "diagnosis": "", "fix": ""},
            ],
            revise=[
                {"id": "a", "fix": "use [[1,2]]"},
                {
                    "id": "v",
                    "fix": "use [[1,2]]",
                },  # frozen: its revision is ignored anyway
                {"id": "zz", "fix": "use [[1,2]]"},  # no such entry
            ],
        ),
        before,
        frozen={"v"},
    )
    assert [r["item"] for r in found] == ["add 2", "revise a"]
    assert found[0]["problems"] == [
        {"field": "fix", "rule": "a pixel position", "match": "row 120"}
    ]


def test_an_added_trigger_and_the_stored_cut_of_a_field_are_checked_too():
    """``apply`` stores an add's trigger as written and cuts fields to ``field_chars``: neither
    may carry a scene detail past the check (a revision's trigger is never applied)."""
    before = [entry("a")]
    found = R.scene_rejections(
        edits(
            add=[{**ADD, "trigger": "checkin at step 19"}],
            revise=[{"id": "a", "trigger": "step 3", "fix": "f2"}],
        ),
        before,
    )
    assert [(r["item"], [p["field"] for p in r["problems"]]) for r in found] == [
        ("add 1", ["trigger"])
    ]
    # Whole, "1.23.4" is no measured value; cut to 15 characters it ends in "1.23".
    cut = {**ADD, "fix": "aaaaaaaaaa 1.23.4"}
    assert R.scene_rejections(edits(add=[cut]), []) == []
    (row,) = R.scene_rejections(edits(add=[cut]), [], chars=15)
    assert row["problems"] == [
        {
            "field": "fix",
            "rule": "a measured value (a number with two or more decimals)",
            "match": "1.23",
        }
    ]
    # A rewrite is held to the same check: a scene-bound trigger drops it.
    proposed = edits(add=[{**ADD, "fix": "row 5"}])
    rejections = R.scene_rejections(proposed, [])
    merged, outcome = R.merge_rewrites(
        proposed,
        rejections,
        {"add": {1: {"trigger": "seed 12", "fix": "push below the handle"}}},
    )
    assert merged["add"] == [] and outcome["dropped"] == ["add 1"]
    assert outcome["still_rejected"][0]["problems"][0]["field"] == "trigger"


# ---------------------------------------------------------------------------
# the rewrite request
# ---------------------------------------------------------------------------


def test_the_rewrite_request_names_every_problem_and_parses_the_reply():
    reply = json.dumps(edits(add=[{**ADD, "fix": "push at row 120"}]))
    rejections = R.scene_rejections(json.loads(reply), [], language="")
    prompt = R.build_fix_prompt("THE PROMPT", reply, rejections)
    assert prompt.startswith("THE PROMPT\n\nYour reply was:\n" + reply)
    assert '- add 1: fix has a pixel position ("row 120")' in prompt
    assert prompt.rstrip().endswith('"fix": "..."}]}')
    rewrites = R.parse_rewrites(
        'ok: {"add": [{"n": "1", "fix": "push below the handle"}, {"n": 1, "fix": "dup"}], '
        '"revise": [{"id": 7, "fix": "x"}, "junk"]}'
    )
    assert rewrites == {
        "add": {1: {"n": "1", "fix": "push below the handle"}},
        "revise": {"7": {"id": 7, "fix": "x"}},
    }
    assert R.parse_rewrites("no json") == {"add": {}, "revise": {}}


def test_rewrites_replace_rejected_items_or_drop_them():
    proposed = edits(
        applied=["a"],
        add=[
            ADD,
            {**ADD, "situation": "dish0 fell", "fix": "grasp at [[1,2]]"},
            {**ADD, "fix": "row 5"},
        ],
        revise=[{"id": "a", "fix": "go to [3.44,-0.95]"}, {"id": "b", "fix": "row 9"}],
    )
    before = [entry("a"), entry("b")]
    rejections = R.scene_rejections(proposed, before)
    assert [r["item"] for r in rejections] == ["add 2", "add 3", "revise a", "revise b"]
    rewrites = {
        # add 2: clean rewrite of the fix only; its situation still names dish0 -> dropped
        # add 3: clean rewrite -> kept, trigger and the other fields from the original
        "add": {
            2: {"fix": "grasp the dish at its rim"},
            3: {"fix": "push below the handle"},
        },
        # revise a: clean -> kept; revise b: left out -> dropped
        "revise": {"a": {"fix": "go back to the counter you started at"}},
    }
    merged, outcome = R.merge_rewrites(proposed, rejections, rewrites)
    assert merged["add"] == [ADD, {**ADD, "fix": "push below the handle"}]
    assert merged["revise"] == [
        {"id": "a", "fix": "go back to the counter you started at"}
    ]
    assert merged["applied"] == ["a"] and merged["note"] == "n"
    assert outcome["kept"] == ["add 3", "revise a"]
    assert outcome["dropped"] == ["add 2", "revise b"]
    assert [x["item"] for x in outcome["still_rejected"]] == ["add 2"]
    assert outcome["still_rejected"][0]["problems"][0]["match"] == "dish0"
    # A refused rewrite request (no rewrites) drops every rejected item.
    merged, outcome = R.merge_rewrites(proposed, rejections, None)
    assert merged["add"] == [ADD] and merged["revise"] == []
    assert outcome["kept"] == [] and len(outcome["dropped"]) == 4


def test_the_prompt_variants_and_the_default_prompt():
    d = {
        "interventions": 1,
        "by_reason": {"checkin": 1},
        "shown": 1,
        "turns": 3,
        "source": "transcript",
        "finish": {"summary": "x"},
        "text": "### Intervention at turn 1 (trigger: checkin)",
    }
    kw = {
        "language": "Open the drawer.",
        "rule": {},
        "success": True,
        "termination": "completed",
        "entries": [],
    }
    plain = R.build_prompt(d, **kw)
    assert plain == R.PROMPT.format(
        language="Open the drawer.",
        rule="{}",
        outcome="SUCCESS",
        turns=3,
        termination="completed",
        summary="x",
        source_note="",
        interventions=d["text"],
        entries="[]",
        max_entries=8,
    )
    assert R.CONCRETE_RULE in plain and R.SCENE_FREE_RULE not in plain
    free = R.build_prompt(d, **kw, scene_free=True)
    assert R.SCENE_FREE_RULE in free and R.CONCRETE_RULE not in free
    assert R.FROZEN_RULE not in free
    both = R.build_prompt(d, **kw, scene_free=True, frozen_verified=True)
    assert R.FROZEN_RULE + R.LIMITS in both
    # Only the instructions change: the reply schema and the episode are the same.
    assert (
        plain.split("Reply with JSON only:")[1]
        == both.split("Reply with JSON only:")[1]
    )
    assert plain.split("Edit the memory")[0] == both.split("Edit the memory")[0]


# ---------------------------------------------------------------------------
# cross-scene verification
# ---------------------------------------------------------------------------


def cross(entries, e, episode, success, used, seed):
    return R.apply(
        entries, e, episode, success, set(used), S, seed=seed, cross_scene=True
    )


def test_written_or_revised_from_a_success_is_not_verified():
    after = cross(
        [entry("a")],
        edits(add=[ADD], revise=[{"id": "a", "fix": "g"}]),
        "T/s2_a1",
        True,
        [],
        13,
    )
    by = {x["id"]: x for x in after}
    assert by["s2_a1.1"]["status"] == "candidate"
    assert by["s2_a1.1"]["written_seed"] == 13 and by["s2_a1.1"]["text_seed"] == 13
    assert by["s2_a1.1"]["applied_in"] == [] and by["s2_a1.1"]["verified_by"] is None
    assert by["a"]["status"] == "candidate" and by["a"]["text_seed"] == 13
    assert by["a"]["text_from"] == "T/s2_a1" and by["a"]["fix"] == "g"
    # run_ecc_stage3's bookkeeping verifies both.
    old = R.apply(
        [entry("a")],
        edits(add=[ADD], revise=[{"id": "a", "fix": "g"}]),
        "T/s2_a1",
        True,
        set(),
        S,
    )
    assert {x["status"] for x in old} == {"verified"}


def test_a_success_on_the_same_scene_does_not_verify():
    after = cross(
        [entry("a", text_seed=12)], edits(applied=["a"]), "T/s1_a2", True, ["a"], 12
    )
    (a,) = after
    assert a["status"] == "candidate" and a["verified_by"] is None
    assert a["successes"] == 1 and a["used_in"] == ["T/s1_a2"]
    assert a["applied_in"] == [{"episode": "T/s1_a2", "seed": 12, "success": True}]


def test_a_success_on_another_scene_verifies_and_freezes_the_text():
    before = [entry("a", text_seed=12, fix="the verified fix")]
    reply = json.dumps(edits(applied=["a"], revise=[{"id": "a", "fix": "a new fix"}]))
    done = R.reviewed(before, reply, "T/s2_a1", True, S, seed=13, cross_scene=True)
    (a,) = done["entries_after"]
    assert a["status"] == "verified"
    assert a["verified_by"] == {"episode": "T/s2_a1", "seed": 13}
    assert (
        a["fix"] == "the verified fix" and a["text_seed"] == 12
    )  # the revision is ignored
    assert done["revise_ignored_verified"] == ["a"] and done["verified_now"] == ["a"]
    # A verified entry is never revised later, but may still be removed.
    later = R.reviewed(
        done["entries_after"],
        json.dumps(edits(revise=[{"id": "a", "fix": "other"}])),
        "T/s2_a2",
        False,
        S,
        seed=13,
        cross_scene=True,
    )
    assert later["entries_after"][0]["fix"] == "the verified fix"
    assert later["revise_ignored_verified"] == ["a"] and later["verified_now"] == []
    gone = R.reviewed(
        done["entries_after"],
        json.dumps(edits(remove=["a"])),
        "T/s3_a1",
        False,
        S,
        seed=14,
        cross_scene=True,
    )
    assert gone["entries_after"] == []


def test_failures_drop_an_entry_and_a_revision_restarts_its_scene():
    a = entry("a", text_seed=12, failures=1)
    # Applied in a second failure without any success: dropped (run_ecc_stage3's rule).
    assert cross([a], edits(applied=["a"]), "T/s2_a1", False, ["a"], 13) == []
    # Revised in that failure: kept, failures reset, its text now from seed 13 ...
    (b,) = cross(
        [a],
        edits(applied=["a"], revise=[{"id": "a", "fix": "g"}]),
        "T/s2_a1",
        False,
        ["a"],
        13,
    )
    assert b["failures"] == 0 and b["text_seed"] == 13 and b["status"] == "candidate"
    assert b["applied_in"] == [{"episode": "T/s2_a1", "seed": 13, "success": False}]
    # ... so a success on seed 13 does not verify it, one on seed 14 does.
    assert (
        cross([b], edits(applied=["a"]), "T/s2_a2", True, ["a"], 13)[0]["status"]
        == "candidate"
    )
    assert (
        cross([b], edits(applied=["a"]), "T/s3_a1", True, ["a"], 14)[0]["status"]
        == "verified"
    )
    # A same-scene success still counts as a success for the drop rule (run_ecc_stage3's counters).
    c = cross(
        [entry("c", failures=1)], edits(applied=["c"]), "T/s1_a2", True, ["c"], 12
    )[0]
    assert (
        cross([c], edits(applied=["c"]), "T/s2_a1", False, ["c"], 13)[0]["failures"]
        == 2
    )


def test_an_entry_without_a_text_scene_is_never_verified_and_unseen_ids_do_not_count():
    old = {
        k: v for k, v in entry("a").items() if k not in ("text_seed", "written_seed")
    }
    assert (
        cross([old], edits(applied=["a"]), "T/s2_a1", True, ["a"], 13)[0]["status"]
        == "candidate"
    )
    done = R.reviewed(
        [entry("a")],
        json.dumps(edits(applied=["a", "zz"])),
        "T/s2_a2",
        True,
        S,
        shown=set(),
        seed=13,
        cross_scene=True,
    )
    assert done["used"] == [] and done["applied_not_shown"] == ["a"]
    assert done["entries_after"][0]["status"] == "candidate"


def test_reviewed_without_the_switches_is_unchanged():
    before = [entry("a")]
    reply = json.dumps(edits(applied=["a"], add=[ADD]))
    plain = R.reviewed(before, reply, "T/s1_a2", True, S)
    assert set(plain) == {"status", "edits", "used", "entries_after"}
    assert plain == R.reviewed(
        before, reply, "T/s1_a2", True, S, cross_scene=False, seed=None
    )
    assert {x["status"] for x in plain["entries_after"]} == {"verified"}
    # Given edits are applied instead of the reply's.
    given = R.reviewed(
        before, "unparseable", "T/s1_a2", True, S, edits=edits(applied=["a"])
    )
    assert given["status"] == "applied" and given["used"] == ["a"]
