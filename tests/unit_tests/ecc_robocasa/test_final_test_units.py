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

"""The final-test runner's pure parts: arms, rules, corrections, cell order, statistics, error runs."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics

import pytest

from scripts.ecc_robocasa import final_test as FT
from scripts.ecc_robocasa import hybrid_episode as HE

# -- arms --------------------------------------------------------------------


@pytest.mark.parametrize(
    "arm",
    [
        {"planner": "hybrid", "memory": "s1", "rule": "s2"},
        {"planner": "hybrid", "memory": "s1", "rule": "s2", "corrections": "s3"},
        {
            "planner": "hybrid",
            "memory": "cloud",
            "rule": "fixed",
            "plan": True,
            "checkin": 5,
            "errors": 2,
        },
        {"planner": "qwen", "memory": "cloud", "note": "Target50 Qwen-only"},
        {
            "planner": "hybrid",
            "memory": "/abs/corpus",
            "rule": "fixed",
            "plan": False,
            "checkin": 12,
            "errors": 4,
        },
    ],
)
def test_valid_arms(arm):
    FT.validate_arm("arm_1", arm)


@pytest.mark.parametrize(
    "arm, message",
    [
        ({"planner": "qwen", "memory": "cloud", "rule": "s2"}, "no senior"),
        ({"planner": "qwen", "memory": "cloud", "corrections": "s3"}, "no senior"),
        ({"planner": "hybrid", "memory": "s1"}, "needs rule"),
        ({"planner": "hybrid", "memory": "s1", "rule": "s2", "checkin": 6}, "drop"),
        (
            {
                "planner": "hybrid",
                "memory": "s1",
                "rule": "fixed",
                "plan": True,
                "checkin": 6,
            },
            "needs plan, checkin and errors",
        ),
        (
            {
                "planner": "hybrid",
                "memory": "s1",
                "rule": "fixed",
                "plan": 1,
                "checkin": 6,
                "errors": 2,
            },
            "plan must be",
        ),
        (
            {
                "planner": "hybrid",
                "memory": "s1",
                "rule": "fixed",
                "plan": True,
                "checkin": 0,
                "errors": 2,
            },
            "positive",
        ),
        (
            {"planner": "hybrid", "memory": "relative/path", "rule": "s2"},
            "memory must be",
        ),
        (
            {"planner": "hybrid", "memory": "s1", "rule": "s2", "takeover": 6},
            "unknown keys",
        ),
        ({"planner": "codex", "memory": "s1"}, "planner must be"),
    ],
)
def test_invalid_arms(arm, message):
    with pytest.raises(ValueError, match=message):
        FT.validate_arm("arm_1", arm)


def test_arm_names_cannot_break_job_ids():
    for name in ("a__b", "", "-x", "x" * 41, "a/b"):
        with pytest.raises(ValueError, match="arm name"):
            FT.validate_arm(name, {"planner": "qwen", "memory": "cloud"})


# -- stage-2 rules -------------------------------------------------------------


def _rule_doc(rung: int, **extra):
    rule = HE.LEADER_LADDER[rung - 1]
    return {
        "task": "CloseFridge",
        "rule": rule.as_json(),
        "config": rule.config_name(rung),
        **extra,
    }


@pytest.mark.parametrize("rung", range(1, 7))
def test_rule_files_in_the_stage2_schema_map_to_their_rung(rung):
    parsed = FT.parse_rule(_rule_doc(rung, solved_in_search=True))
    rule = HE.LEADER_LADDER[rung - 1]
    assert parsed == {
        "plan": rule.plan,
        "checkin": rule.checkin,
        "errors": rule.errors,
        "rung": rung,
    }


def test_rule_files_that_disagree_or_leave_the_ladder_are_refused():
    doc = _rule_doc(2)
    doc["config"] = "c6_gptplan_K6_N2"
    with pytest.raises(ValueError, match="disagrees"):
        FT.parse_rule(doc)
    with pytest.raises(ValueError, match="disagrees"):
        FT.parse_rule({**_rule_doc(3), "rung": 4})
    off = {"rule": {"gpt_plan": True, "checkin_K": 5, "errors_N": 2}}
    with pytest.raises(ValueError, match="not a rung"):
        FT.parse_rule(off)
    assert FT.parse_rule(off, allow_off_ladder=True)["rung"] is None
    with pytest.raises(ValueError, match="no"):
        FT.parse_rule({"rule": {"gpt_plan": True, "checkin_K": 6}})
    with pytest.raises(ValueError, match="rule"):
        FT.parse_rule({"gpt_plan": True, "checkin_K": 6, "errors_N": 2})
    with pytest.raises(ValueError, match="plan must be"):
        FT.parse_rule({"rule": {"gpt_plan": "yes", "checkin_K": 6, "errors_N": 2}})


# -- corrections -------------------------------------------------------------------


def test_corrections_info_is_what_rpent_logs():
    """Length and hash of the stripped text, as the escalation config row records them."""
    raw = "\n  - [c1] (verified; trigger: checkin) situation: lid | diagnosis: x | fix: y \n\n".encode()
    info = FT.corrections_info(raw)
    text = raw.decode().strip()
    assert info["chars"] == len(text) and info["bytes"] == len(raw)
    assert info["sha256"] == hashlib.sha256(raw).hexdigest()
    assert info["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    cfg = {
        "senior": {"max_share": 0.2, "model": "openai:gpt-6-astra"},
        "worker": {"model": "openai-chat:qwen3.6-27b"},
    }
    routing = HE.expected_routing(
        cfg, HE.Rule(True, 6, 2), corrections_text=raw.decode()
    )
    assert (routing["corrections_chars"], routing["corrections_sha256"]) == (
        info["chars"],
        info["text_sha256"],
    )
    empty = FT.corrections_info(b" \n")
    assert (empty["chars"], empty["text_sha256"]) == (0, None)


# -- cells and their arm order ---------------------------------------------------------

TASKS = [
    ("atomic", "CloseFridge"),
    ("atomic", "OpenDrawer"),
    ("composite_seen", "PreSoakPan"),
]


def _cfg(**ft):
    return {"final_test": {**FT.FT_DEFAULTS, **ft}}


def test_jobs_cover_every_cell_and_arm_with_a_cells_arms_adjacent():
    arms = ["s2_rules", "s3_rules_corr", "base_hybrid"]
    cfg = _cfg(seeds=[1, 2, 3], repeats=2)
    jobs = FT.build_jobs(cfg, arms, TASKS)
    assert len(jobs) == len(TASKS) * 3 * 2 * len(arms)
    assert len({j.job_id for j in jobs}) == len(jobs)
    assert [j.order for j in jobs] == list(range(len(jobs)))
    for start in range(0, len(jobs), len(arms)):
        block = jobs[start : start + len(arms)]
        assert len({j.meta["cell"] for j in block}) == 1
        assert sorted(j.meta["arm"] for j in block) == sorted(arms)
        assert [j.meta["arm_position"] for j in block] == list(range(len(arms)))
    # seed-major: every task of seed 1 (repeat 1) before seed 2
    assert [j.seed for j in jobs[: len(TASKS) * len(arms)]] == [1] * len(TASKS) * len(
        arms
    )
    assert jobs[0].job_id.split("__")[1] == "CloseFridge_s1_r1"


def test_arm_order_is_seeded_reproducible_and_mixed():
    arms = ["s2_rules", "s3_rules_corr"]
    a = FT.build_jobs(_cfg(seeds=list(range(1, 41))), arms, TASKS)
    b = FT.build_jobs(_cfg(seeds=list(range(1, 41))), arms, TASKS)
    c = FT.build_jobs(_cfg(seeds=list(range(1, 41)), order_seed=7), arms, TASKS)
    assert [j.job_id for j in a] == [j.job_id for j in b]
    assert [j.job_id for j in a] != [j.job_id for j in c]
    first = [j.meta["arm"] for j in a if j.meta["arm_position"] == 0]
    assert (
        0.3 < first.count("s2_rules") / len(first) < 0.7
    )  # 120 cells, each arm first about half the time
    assert FT.arm_order(20261027, arms, "CloseFridge", 1, 1) == FT.arm_order(
        20261027, list(reversed(arms)), "CloseFridge", 1, 1
    )


def test_shuffled_cell_order_is_a_permutation():
    arms = ["a", "b"]
    seed_major = FT.build_jobs(_cfg(seeds=[1, 2]), arms, TASKS)
    shuffled = FT.build_jobs(_cfg(seeds=[1, 2], cell_order="shuffled"), arms, TASKS)
    assert sorted(j.job_id for j in seed_major) == sorted(j.job_id for j in shuffled)
    assert [j.meta["cell"] for j in seed_major] != [j.meta["cell"] for j in shuffled]


# -- statistics ------------------------------------------------------------------------------


def test_exact_mcnemar_values():
    assert FT.binom_two_sided(0, 0) == 1.0
    assert FT.binom_two_sided(0, 5) == pytest.approx(2 / 32)
    assert FT.binom_two_sided(1, 8) == pytest.approx(2 * (1 + 9) / 512)
    assert FT.binom_two_sided(8, 1) == FT.binom_two_sided(1, 8)
    assert FT.binom_two_sided(5, 5) == 1.0


def _brute_force(diffs):
    stat = abs(statistics.fmean(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        hits += (
            abs(statistics.fmean(s * d for s, d in zip(signs, diffs))) >= stat - 1e-12
        )
    return hits / total


def test_sign_flip_test_is_exact_for_few_tasks_and_close_by_monte_carlo():
    diffs = [1.0, 0.5, 0.0, 1.0, -0.5, 1.0, 0.0, 0.5]
    exact = FT.sign_flip_test(diffs, permutations=1000, seed=1)
    assert exact["method"] == "exact" and exact["tasks_differing"] == 6
    assert exact["p"] == pytest.approx(_brute_force(diffs))
    assert exact["stat"] == pytest.approx(statistics.fmean(diffs))
    many = [1.0] * 14 + [-1.0] * 6 + [0.0] * 10
    mc = FT.sign_flip_test(many, permutations=20000, seed=3)
    assert mc["method"] == "monte_carlo_20000"
    # the null distribution of sum of 20 +-1 signs: P(|sum| >= 8) = 2 * P(Binom(20, .5) >= 14)
    want = 2 * sum(math.comb(20, k) for k in range(14, 21)) / 2**20
    assert mc["p"] == pytest.approx(want, abs=0.01)
    assert FT.sign_flip_test(many, permutations=20000, seed=3) == mc
    assert FT.sign_flip_test([0.0, 0.0], permutations=10, seed=1)["p"] == 1.0
    assert FT.sign_flip_test([], permutations=10, seed=1)["p"] is None


def test_task_weighted_rates_follow_target50():
    outcomes = {
        ("atomic", "A1"): [True, True, False],  # 2/3
        ("atomic", "A2"): [True],  # 1
        ("composite_seen", "S1"): [False, False],  # 0
        ("composite_unseen", "U1"): [True, False],  # 1/2
    }
    tw = FT.task_weighted(outcomes)
    assert tw["atomic"] == pytest.approx(100 * (2 / 3 + 1) / 2)
    assert tw["composite_seen"] == 0 and tw["composite_unseen"] == 50
    assert tw["overall"] == pytest.approx(
        100 * (2 / 3 + 1 + 0 + 0.5) / 4
    )  # mean over tasks, not over splits
    assert (tw["tasks"], tw["atomic_tasks"]) == (4, 2)


def _record(arm, task, seed, success, split="atomic", **extra):
    return {
        "arm": arm,
        "task": task,
        "split": split,
        "seed": seed,
        "rep": 1,
        "job_id": f"{arm}__{task}_s{seed}_r1",
        "valid": True,
        "evidence_ok": True,
        "success": success,
        **extra,
    }


def test_paired_comparison_on_constructed_cells():
    records = []
    # task T1: A wins both seeds; T2: B wins seed 1, tie seed 2; T3: tie; T4 composite: A wins
    for seed, (a, b) in enumerate([(True, False), (True, False)], 1):
        records += [_record("A", "T1", seed, a), _record("B", "T1", seed, b)]
    records += [_record("A", "T2", 1, False), _record("B", "T2", 1, True)]
    records += [_record("A", "T2", 2, True), _record("B", "T2", 2, True)]
    records += [_record("A", "T3", 1, False), _record("B", "T3", 1, False)]
    records += [
        _record("A", "T4", 1, True, "composite_seen"),
        _record("B", "T4", 1, False, "composite_seen"),
    ]
    # not scored: invalid in B, and an evidence mismatch in A
    records += [
        _record("A", "T5", 1, True),
        {**_record("B", "T5", 1, False), "valid": False},
    ]
    records += [
        {**_record("A", "T6", 1, True), "evidence_ok": False},
        _record("B", "T6", 1, False),
    ]
    p = FT.paired(records, "A", "B", permutations=1000, bootstrap=2000, seed=1)
    assert p["cells"] == 6
    q = p["all"]
    assert (q["a_only"], q["b_only"]) == (3, 1)
    assert q["mcnemar_p"] == pytest.approx(FT.binom_two_sided(3, 1))
    assert (q["tasks"], q["tasks_a_better"], q["tasks_b_better"]) == (4, 2, 1)
    diffs = [1.0, -0.5, 0.0, 1.0]
    assert q["task_weighted_diff_pct"] == pytest.approx(100 * statistics.fmean(diffs))
    assert q["permutation"]["p"] == pytest.approx(_brute_force(diffs))
    low, high = q["bootstrap_ci_pct"]
    assert low <= q["task_weighted_diff_pct"] <= high
    assert p["composite_seen"]["cells"] == 1 and p["atomic"]["cells"] == 5
    included = FT.paired(
        records, "A", "B", permutations=10, bootstrap=0, seed=1, include_mismatch=True
    )
    assert included["cells"] == 7


def test_arm_summary_counts_gpt6_calls_errors_and_cost():
    def ep(reasons, errors, qwen, usd_q, usd_g):
        return {
            "gpt_calls": sum(reasons.values()),
            "gpt_calls_by_reason": reasons,
            "errors_trigger": errors,
            "qwen": {"calls": qwen, "usd": usd_q},
            "gpt6": {"usd": usd_g},
        }

    records = [
        _record(
            "A",
            "T1",
            1,
            True,
            episode=ep({"plan": 1, "checkin": 2}, 0, 20, 0.1, 1.0),
            usd=1.1,
            usd_all_tries=1.1,
            error_runs={"errored_results": 2, "max_run": 2},
        ),
        _record(
            "A",
            "T2",
            1,
            False,
            episode=ep({"plan": 1, "checkin": 4, "errors": 1}, 1, 40, 0.2, 2.0),
            usd=2.2,
            usd_all_tries=3.0,
            error_runs={"errored_results": 3, "max_run": 3},
        ),
        {**_record("A", "T3", 1, False), "valid": False, "usd_all_tries": 0.5},
    ]
    s = FT.arm_summary(records, "A")
    assert (s["scored"], s["successes"], s["invalid"]) == (2, 1, 1)
    assert s["gpt6_calls_per_episode"] == pytest.approx(4.5)
    assert s["gpt6_calls_per_episode_by_reason"] == {
        "checkin": 3.0,
        "errors": 0.5,
        "plan": 1.0,
    }
    assert s["qwen_calls_per_episode"] == 30
    assert s["usd_per_episode"] == pytest.approx(1.65) and s[
        "usd_per_success"
    ] == pytest.approx(3.3)
    assert s["usd_all_launches"] == pytest.approx(4.6)
    assert s["errors_trigger"]["turns"] == 1 and s["errors_trigger"]["episodes"] == 1
    assert s["errors_trigger"]["episodes_with_errored_run_ge"] == {
        "2": 2,
        "3": 1,
        "4": 0,
    }
    assert (
        s["task_weighted_pct"]["overall"] == 50.0
    )  # T3 (invalid) is not a scored cell
    strict = FT.arm_summary(records, "A", invalid_as_failure=True)
    assert (
        strict["task_weighted_pct"]["overall"] == pytest.approx(100 / 3)
        and strict["scored"] == 2
    )
    text = FT.format_score(
        FT.score(
            records + [_record("B", "T1", 1, False)],
            ["A", "B"],
            [("A", "B")],
            permutations=100,
            bootstrap=100,
            seed=1,
        )
    )
    assert "A vs B: 1 cells scored in both" in text and "McNemar p = " in text


# -- transcripts: error runs ---------------------------------------------------------------


def test_tool_categories_match_how_the_trigger_reads_results():
    assert FT.tool_category(json.dumps({"step": 3, "error": "bad arguments"})) == "err"
    assert (
        FT.tool_category(json.dumps({"step": 3, "log": {"result": {"ok": False}}}))
        == "ok"
    )
    retry = [{"type": "json_invalid", "loc": [], "msg": "Invalid JSON", "input": "{"}]
    assert FT.tool_category(json.dumps(retry)) == "retry"
    assert (
        FT.tool_category("Unknown tool name: ''. Available tools: 'finish'")
        == "unknown"
    )
    assert FT.tool_category(json.dumps([{"error": "x"}])) == "err"


def test_error_runs_skip_validation_retries_and_need_turns(tmp_path):
    def tool(content):
        return {
            "role": "tool",
            "name": "scripted_grasp",
            "content": json.dumps(content),
        }

    retry = [{"type": "json_invalid", "loc": [], "msg": "Invalid JSON", "input": "{"}]
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": []},
        tool({"step": 1, "error": "bad"}),
        tool(retry),
        tool({"step": 2, "error": "bad"}),
        tool({"step": 3}),
        tool({"step": 4, "error": "bad"}),
    ]
    (tmp_path / "transcript_X_target_s1.json").write_text(
        json.dumps({"messages": messages})
    )
    assert FT.error_runs(tmp_path) == {"errored_results": 3, "max_run": 2}
    (tmp_path / "transcript_X_target_s1.json").write_text(
        json.dumps({"messages": messages[:1]})
    )
    assert FT.error_runs(tmp_path) is None


def test_a_qwen_alone_launch_must_not_leave_an_escalation_log(tmp_path):
    assert FT.qwen_alone_evidence(tmp_path)["ok"]
    (tmp_path / "escalation.jsonl").write_text('{"config": {}}\n')
    result = FT.qwen_alone_evidence(tmp_path)
    assert not result["ok"] and "Qwen alone" in result["problems"][0]


def test_the_example_config_section_is_valid():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = pytest.importorskip("tomli")
    section = tomllib.loads(FT.EXAMPLE_CONFIG)["final_test"]
    cfg = {"final_test": FT.C._deep_merge(FT.FT_DEFAULTS, section)}
    FT.validate_final_test(cfg)
    assert set(section["arms"]) == {
        "s2_rules",
        "s3_rules_corr",
        "base_hybrid",
        "qwen_only",
        "gpt6_plan_k12",
    }
    assert set(section["products"]) <= set(FT.FT_DEFAULTS["products"])
    assert set(section["scheduler"]) <= set(FT.C.DEFAULTS["scheduler"])
