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

"""The escalation evidence check against logs written by RPent's real escalation wrapper.

A genuine log must always pass (a false mismatch would stop a paid run); a log
written with another plan switch, K, N or corrections must fail whenever the
episode was long enough to show the difference.
"""

from __future__ import annotations

import importlib.util
import json
import random

import pytest

from scripts.ecc_robocasa import hybrid_episode as H
from tests.unit_tests.ecc_robocasa.s2_support import FAKE_HYBRID


def _fake():
    spec = importlib.util.spec_from_file_location("fake_hybrid_cell", FAKE_HYBRID)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE = _fake()


def _log(
    tmp_path,
    *,
    plan,
    checkin,
    errors,
    requests=40,
    corrections="",
    giveup_at=(),
    error_results=(),
    senior_text_at=(),
    name="e",
):
    out = tmp_path / name
    out.mkdir()
    FAKE.drive(
        out,
        {
            "requests": requests,
            "giveup_at": list(giveup_at),
            "error_results": list(error_results),
            "senior_text_at": list(senior_text_at),
        },
        {
            "plan": plan,
            "checkin": checkin,
            "errors": errors,
            "max_share": 0.2,
            "corrections": corrections,
            "worker": "qwen3.6-27b",
            "senior": "gpt-6-astra",
        },
    )
    return out / "escalation.jsonl"


def _expected(plan, checkin, errors, corrections=""):
    cfg = {
        "senior": {"max_share": 0.2, "model": "openai:gpt-6-astra"},
        "worker": {"model": "openai-chat:qwen3.6-27b"},
    }
    return H.expected_routing(
        cfg, H.Rule(plan, checkin, errors), corrections_text=corrections
    )


@pytest.mark.parametrize(
    "plan,checkin,errors", [(False, 12, 4), (True, 12, 4), (False, 6, 4), (True, 6, 2)]
)
def test_genuine_logs_pass_and_show_the_rule(tmp_path, plan, checkin, errors):
    log = _log(tmp_path, plan=plan, checkin=checkin, errors=errors)
    check = H.check_escalation(log, _expected(plan, checkin, errors))
    assert check["ok"], check["problems"]
    ev, counts = check["evidence"], check["counts"]
    assert (
        ev["config_row"]["checkin"] == checkin and ev["config_row"]["errors"] == errors
    )
    assert (ev["plan_turn_at"] is not None) is plan
    if plan:
        assert ev["plan_turn_at"] <= 6 and counts["senior_by_reason"]["plan"] == 1
    low, high = ev["checkin_bounds"]
    assert low <= checkin == high  # a check-in at exactly K requests was seen
    assert counts["requests"] == 40 and counts["senior_by_reason"]["checkin"] >= 2


def test_each_wrong_switch_is_a_mismatch(tmp_path):
    cases = [
        # launched with                       what the rule expected          fragment
        (
            {"plan": False, "checkin": 12, "errors": 4},
            (True, 12, 4),
            "no plan turn by request 6",
        ),
        (
            {"plan": True, "checkin": 12, "errors": 4},
            (False, 12, 4),
            "plan turn at request",
        ),
        (
            {"plan": False, "checkin": 6, "errors": 4},
            (False, 12, 4),
            "check-in at request 7 after 6 worker requests (K=12)",
        ),
        (
            {"plan": False, "checkin": 12, "errors": 4},
            (False, 6, 4),
            "but no check-in (K=6)",
        ),
        (
            {"plan": False, "checkin": 5, "errors": 4},
            (False, 6, 4),
            "after 5 worker requests (K=6)",
        ),
        (
            {"plan": True, "checkin": 6, "errors": 2},
            (True, 6, 3),
            "config row errors=2, expected 3",
        ),
    ]
    for i, (launched, rule, fragment) in enumerate(cases):
        log = _log(tmp_path, name=f"c{i}", **launched)
        check = H.check_escalation(log, _expected(*rule))
        assert not check["ok"] and any(fragment in p for p in check["problems"]), (
            launched,
            rule,
            check["problems"],
        )
        # Without the config row, the replay alone still catches plan and K (not N).
        rows = [line for line in log.read_text().splitlines() if '"config"' not in line]
        bare = tmp_path / f"bare{i}.jsonl"
        bare.write_text("\n".join(rows) + "\n")
        replay = H.check_escalation(bare, _expected(*rule), require_config_row=False)
        assert replay["ok"] is (rule[2] != launched["errors"]), replay["problems"]


def test_corrections_must_match_on_every_senior_row(tmp_path):
    text = "- [s1_a1.1] (checkin) situation: x; fix: y\n"
    log = _log(tmp_path, plan=True, checkin=6, errors=2, corrections=text)
    assert H.check_escalation(log, _expected(True, 6, 2, corrections=text))["ok"]
    problems = H.check_escalation(log, _expected(True, 6, 2))["problems"]
    assert any("corrections_chars" in p and "expected 0" in p for p in problems)
    other = H.check_escalation(log, _expected(True, 6, 2, corrections=text + "more"))[
        "problems"
    ]
    assert any("corrections_sha256" in p for p in other)
    plain = _log(tmp_path, plan=True, checkin=6, errors=2, name="plain")
    assert any(
        "corrections_chars 0, expected" in p
        for p in H.check_escalation(plain, _expected(True, 6, 2, text))["problems"]
    )


def test_errors_trigger_count_and_share_cap_replay(tmp_path):
    # Errors after the plan turn fire N=2 twice; give-up reviews push the senior to its share cap.
    log = _log(
        tmp_path,
        plan=True,
        checkin=6,
        errors=2,
        requests=60,
        error_results=[8, 9, 20, 21],
        giveup_at=[10, 11, 12, 30, 31],
        senior_text_at=[6, 21],
    )
    check = H.check_escalation(log, _expected(True, 6, 2))
    assert check["ok"], check["problems"]
    counts = check["counts"]
    assert counts["errors_trigger"] == counts["senior_by_reason"]["errors"] >= 1
    assert counts["senior_by_reason"]["giveup"] >= 3 and counts["vetoes"] >= 3
    assert (
        counts["after_guidance_calls"] == 2
    )  # the text-only plan and check-in replies
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    reasons = [r["reason"] for r in rows if r.get("role") == "senior"]
    assert reasons.count("errors") == counts["errors_trigger"]


def test_randomized_genuine_logs_never_mismatch(tmp_path):
    rng = random.Random(20261027)
    for i in range(60):
        plan, checkin, errors = (
            rng.choice(H.LEADER_LADDER).plan,
            rng.choice([5, 6, 12]),
            rng.choice([2, 3, 4]),
        )
        requests = rng.randint(1, 70)
        log = _log(
            tmp_path,
            name=f"r{i}",
            plan=plan,
            checkin=checkin,
            errors=errors,
            requests=requests,
            giveup_at=rng.sample(range(1, 71), rng.randint(0, 6)),
            error_results=rng.sample(range(1, 71), rng.randint(0, 25)),
            senior_text_at=rng.sample(range(1, 71), rng.randint(0, 10)),
            corrections="fix" if rng.random() < 0.3 else "",
        )
        corrections = (
            json.loads(log.read_text().splitlines()[0])["config"]["corrections_chars"]
            and "fix"
        )
        check = H.check_escalation(
            log, _expected(plan, checkin, errors, corrections or "")
        )
        assert check["ok"], (i, plan, checkin, errors, requests, check["problems"])


def test_missing_or_partial_logs(tmp_path):
    missing = tmp_path / "none.jsonl"
    assert not H.check_escalation(missing, _expected(True, 6, 2))["ok"]
    assert H.check_escalation(missing, _expected(True, 6, 2), result_valid=False)["ok"]
    log = _log(tmp_path, plan=True, checkin=6, errors=2, requests=30)
    lines = log.read_text().splitlines()
    cut = tmp_path / "cut.jsonl"
    cut.write_text(
        "\n".join(lines[:-3]) + '\n{"request": 3'
    )  # a timeout cut the last rows mid-line
    check = H.check_escalation(cut, _expected(True, 6, 2))
    assert check["ok"] and check["evidence"]["unparsable_lines"] == 1, check["problems"]
    no_config = tmp_path / "noconfig.jsonl"
    no_config.write_text("\n".join(lines[1:]) + "\n")
    assert (
        "no config row"
        in H.check_escalation(no_config, _expected(True, 6, 2))["problems"][0]
    )


def test_rule_names_and_ladder_match_the_ecc_stage2_runner():
    assert [r.config_name(i) for i, r in enumerate(H.LEADER_LADDER, 1)] == [
        "c1_qwenplan_K12_N4",
        "c2_gptplan_K12_N4",
        "c3_qwenplan_K6_N4",
        "c4_gptplan_K6_N4",
        "c5_gptplan_K6_N3",
        "c6_gptplan_K6_N2",
    ]
    assert H.parse_ladder([{"plan": "qwen", "checkin": 12, "errors": 4}]) == (
        H.Rule(False, 12, 4),
    )
    rule = H.Rule(True, 6, 3)
    assert H.Rule.from_json(rule.as_json()) == rule
    from rpent.planner import escalation

    assert H.PLAN_FALLBACK_REQUEST == escalation.PLAN_FALLBACK_REQUEST
    assert set(H.SENIOR_REASONS) == {"plan", "stall", "errors", "checkin", "giveup"}
