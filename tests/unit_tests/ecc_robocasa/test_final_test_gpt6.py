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

"""The final test's GPT-6-alone arm (``planner = "gpt6"``) and shared GPUs, offline.

Fake episodes (``fakes/fake_gpt6_cell.py``) leave what RPent's api planner with
GPT-6 alone writes; a fake RLDX server and fake nvidia-smi stand in for the GPUs.
No model request.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import final_test as FT
from tests.unit_tests.ecc_robocasa.conftest import FAKES
from tests.unit_tests.ecc_robocasa.test_final_test_runner import (
    ARMS,
    THREE,
    FinalEnv,
    _stage,
)

FAKE_GPT6 = FAKES / "fake_gpt6_cell.py"
GPT6 = {"planner": "gpt6", "memory": "cloud"}
ALL_ARMS = {**ARMS, "gpt6_only": GPT6}
PRICING = C.DEFAULTS["pricing"]
#: The knob keys of the hybrid and Qwen-alone arms (unchanged by the GPT-6-alone arm).
OLD_KNOBS = {
    "arm",
    "planner",
    "escalation",
    "memory",
    "memory_dir",
    "timeout_s",
    "plan",
    "checkin",
    "errors",
    "rung",
    "rule_source",
    "corrections",
}


def senior_usd(tokens_in: int, cached: int, tokens_out: int) -> float:
    return (
        (tokens_in - cached) * PRICING["senior_input_per_mtok"]
        + cached * PRICING["senior_cached_input_per_mtok"]
        + tokens_out * PRICING["senior_output_per_mtok"]
    ) / 1e6


@pytest.fixture
def gfe(ecc, tmp_path):
    """FinalEnv whose configs define the GPT-6-alone arm and launch the fake GPT-6 cell."""
    fe = FinalEnv(ecc, tmp_path)

    def config(*, arms=("gpt6_only", "qwen_only"), **ft):
        path = fe.config(arms=arms, cell=[sys.executable, str(FAKE_GPT6)], **ft)
        raw = json.loads(path.read_text())
        # the run arms' definitions, the GPT-6-alone arm included
        raw["final_test"]["arms"] = ALL_ARMS
        path.write_text(json.dumps(raw, indent=2))
        return path

    fe.gpt6_config = config
    return fe


# -- configuration and knobs ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("arm", "fragment"),
    [
        (
            {"planner": "gpt6", "memory": "cloud", "rule": "fixed"},
            "GPT-6-alone arm has no senior",
        ),
        (
            {"planner": "gpt6", "memory": "cloud", "corrections": "s3"},
            "GPT-6-alone arm has no senior",
        ),
        (
            {"planner": "gpt6", "memory": "cloud", "checkin": 5},
            "GPT-6-alone arm has no senior",
        ),
        (
            {"planner": "gpt6", "memory": "cloud", "max_images": 0},
            "max_images must be a positive integer",
        ),
        (
            {"planner": "gpt6", "memory": "cloud", "max_images": True},
            "max_images must be a positive integer",
        ),
        (
            {"planner": "gpt6", "memory": "cloud", "reasoning_ids": "no"},
            "reasoning_ids must be true or false",
        ),
        (
            {"planner": "qwen", "memory": "cloud", "max_images": 40},
            "apply to a GPT-6-alone arm only",
        ),
        (
            {**ARMS["base_hybrid"], "reasoning_ids": False},
            "apply to a GPT-6-alone arm only",
        ),
        ({"planner": "codex", "memory": "cloud"}, "planner must be"),
    ],
)
def test_arm_validation(arm, fragment):
    with pytest.raises(ValueError, match=fragment):
        FT.validate_arm("a1", arm)


def test_gpt6_arm_defaults_and_valid_variants():
    for arm in (
        GPT6,
        {**GPT6, "max_images": 12, "reasoning_ids": True},
        {"planner": "gpt6", "memory": "s1", "note": "x"},
    ):
        FT.validate_arm("gpt6_only", arm)


def test_knobs_and_launch_environment(gfe):
    stage = _stage(
        gfe,
        gfe.gpt6_config(
            arms=("gpt6_only", "qwen_only", "s3_rules_corr", "base_hybrid"), seeds=(1,)
        ),
    )
    endpoint = C.Endpoint("gpu4", 4, 29999, 2)
    seen = set()
    for job in stage.jobs():
        knobs = stage.knobs[job.job_id]
        env = FT.cell_env(
            stage.cfg, job, 1, endpoint, knobs, parent={"PATH": "/usr/bin:/bin"}
        )
        arm = job.meta["arm"]
        seen.add(arm)
        if arm == "gpt6_only":
            assert set(knobs) == OLD_KNOBS | {
                "model",
                "reasoning_effort",
                "max_images",
                "reasoning_ids",
            }
            assert (knobs["planner"], knobs["escalation"], knobs["memory"]) == (
                "gpt6",
                False,
                "cloud",
            )
            assert (knobs["model"], knobs["reasoning_effort"]) == (
                "openai:gpt-6-astra",
                "low",
            )
            assert (
                knobs["max_images"],
                knobs["reasoning_ids"],
                knobs["corrections"],
            ) == (40, False, None)
            assert (env["ECC_RC_ESCALATION"], env["ECC_RC_SENIOR_ALONE"]) == ("0", "1")
            assert (env["ECC_RC_MAX_IMAGES"], env["ECC_RC_REASONING_IDS"]) == (
                "40",
                "0",
            )
            assert (
                env["ECC_RC_SENIOR_MODEL"] == "openai:gpt-6-astra"
                and env["ECC_RC_CORRECTIONS"] == ""
            )
            assert stage.expected(job) is None
        else:
            assert set(knobs) == OLD_KNOBS  # the existing arms' knobs are unchanged
            for key in (
                "ECC_RC_SENIOR_ALONE",
                "ECC_RC_MAX_IMAGES",
                "ECC_RC_REASONING_IDS",
            ):
                assert key not in env
    assert seen == {"gpt6_only", "qwen_only", "s3_rules_corr", "base_hybrid"}


def test_experiment_of_a_config_without_gpt6_arms_is_unchanged(gfe):
    """The new final_test keys are machine values: a run created before keeps its experiment hash."""
    cfg = FT.load_config(gfe.config(arms=("qwen_only", "base_hybrid")))
    assert set(FT.experiment(cfg)["final_test"]) == set(FT.FT_EXPERIMENT_KEYS)
    for key in (
        "share_own_gpus",
        "gpu_apps_cmd",
        "gpu_uuid_cmd",
        "own_process_prefixes",
    ):
        assert key not in FT.FT_EXPERIMENT_KEYS


# -- usage and evidence --------------------------------------------------------------------


def _episode(
    tmp_path: Path,
    *,
    stats: dict | None,
    model="openai:gpt-6-astra",
    log_lines=(),
    valid=True,
) -> Path:
    out = tmp_path / "try1"
    out.mkdir(parents=True)
    lines = list(log_lines) + [
        "x I [api_loop] [usage] in=10000 out=100 cache_read=2000 cache_write=0 requests=1",
        "x I [api_loop] [usage] in=30000 out=250 cache_read=6000 cache_write=0 requests=2",
    ]
    (out / "run.log").write_text("\n".join(lines) + "\n")
    (out / "transcript_T_target_s1.json").write_text(
        json.dumps(
            {
                "stats": stats if stats is not None else {"elapsed_s": 3.0},
                "messages": [],
            }
        )
    )
    (out / "result.json").write_text(
        json.dumps(
            {
                "valid": valid,
                "success": False,
                "termination_reason": "completed" if valid else "infrastructure_error",
                "planner": {
                    "backend": "api",
                    "model": model,
                    "reasoning_effort": "low",
                    "max_turns": 100,
                },
            }
        )
    )
    return out


KNOBS = {
    "model": "openai:gpt-6-astra",
    "reasoning_effort": "low",
    "max_images": 40,
    "reasoning_ids": False,
}
GOOD_LOG = (
    "x I [api_loop] history images: at most 40 per request (RPENT_MAX_HISTORY_IMAGES)",
    "x I [api_loop] earlier response item ids are not sent back (RPENT_SEND_REASONING_IDS=0)",
)


def test_usage_from_the_transcript_and_after_a_timeout_from_run_log(tmp_path):
    stats = {
        "total_input_tokens": 40000,
        "cache_read_tokens": 9000,
        "total_output_tokens": 400,
        "requests": 3,
    }
    u = FT.senior_alone_usage(_episode(tmp_path / "a", stats=stats), PRICING)
    assert (u["source"], u["planner"], u["senior_calls"], u["worker_calls"]) == (
        "transcript",
        "gpt6",
        3,
        0,
    )
    assert (u["senior_in"], u["senior_cached"], u["senior_out"]) == (40000, 9000, 400)
    assert (
        u["usd"] == pytest.approx(senior_usd(40000, 9000, 400))
        and u["worker_usd"] == 0.0
    )
    t = FT.senior_alone_usage(
        _episode(tmp_path / "b", stats=None), PRICING
    )  # a planner timeout
    assert (
        t["source"],
        t["senior_calls"],
        t["senior_in"],
        t["senior_cached"],
        t["senior_out"],
    ) == (
        "run.log",
        2,
        30000,
        6000,
        250,
    )
    assert t["usd"] == pytest.approx(senior_usd(30000, 6000, 250))
    # the generic usage would have priced the same tokens at the worker's price
    generic = C.episode_usage(tmp_path / "b" / "try1", PRICING)
    assert generic["source"] == "run.log" and generic["usd"] < t["usd"] / 10


def test_evidence_accepts_a_gpt6_alone_episode_and_counts_every_request(tmp_path):
    stats = {
        "total_input_tokens": 1,
        "cache_read_tokens": 0,
        "total_output_tokens": 1,
        "requests": 7,
    }
    check = FT.senior_alone_evidence(
        _episode(tmp_path, stats=stats, log_lines=GOOD_LOG), KNOBS, PRICING
    )
    assert check["ok"], check["problems"]
    counts = check["counts"]
    assert (
        counts["requests"],
        counts["senior_calls"],
        counts["senior_by_reason"]["planner"],
    ) == (7, 7, 7)
    assert (
        counts["senior_by_reason"]["checkin"] == 0
        and check["evidence"]["planner"]["model"] == KNOBS["model"]
    )


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"model": "openai-chat:qwen3.6-27b"}, "result.json planner model"),
        ({"log_lines": GOOD_LOG[1:]}, "40-image cap"),
        ({"log_lines": GOOD_LOG[:1]}, "response item ids are off"),
        ({"escalation": True}, "escalation.jsonl exists"),
    ],
)
def test_evidence_mismatches(tmp_path, change, fragment):
    kwargs = {"stats": {"requests": 2}, "log_lines": GOOD_LOG, **change}
    escalation = kwargs.pop("escalation", False)
    out = _episode(tmp_path, **kwargs)
    if escalation:
        (out / "escalation.jsonl").write_text("{}\n")
    check = FT.senior_alone_evidence(out, KNOBS, PRICING)
    assert not check["ok"] and any(fragment in p for p in check["problems"]), check[
        "problems"
    ]


def test_evidence_of_an_invalid_launch_checks_only_the_escalation_log(tmp_path):
    out = _episode(tmp_path, stats=None, log_lines=(), valid=False)
    assert FT.senior_alone_evidence(out, KNOBS, PRICING)["ok"]
    check = FT.senior_alone_evidence(out, {**KNOBS, "reasoning_ids": True}, PRICING)
    assert check["ok"] and check["counts"]["senior_calls"] == 2


# -- the runner end to end -----------------------------------------------------------------


def test_run_records_gpt6_calls_tokens_and_cost(gfe):
    gfe.ecc.set_scenario(
        {
            "gpt6_only": {"kind": "solve", "requests": 6, "tokens": [20000, 5000, 300]},
            "gpt6_only:CloseFridge": [
                {"kind": "infra"},
                {"kind": "timeout", "requests": 4},
            ],
            "qwen_only": {"kind": "fail", "requests": 10},
        }
    )
    config = gfe.gpt6_config(seeds=(1, 2))
    proc = gfe.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    records = {}
    for task in THREE:
        for seed in (1, 2):
            for arm in ("gpt6_only", "qwen_only"):
                job = FT.job_name(arm, task, seed, 1)
                records[job] = gfe.record(job)
    for job, rec in records.items():
        ep = rec["episode"]
        assert rec["valid"] and rec["evidence_ok"], (job, rec["episode"]["evidence"])
        if rec["arm"] == "qwen_only":
            assert (ep["gpt_calls"], ep["qwen"]["calls"], ep["usage_source"]) == (
                0,
                10,
                "run.log",
            )
            continue
        timeout = rec["task"] == "CloseFridge"
        requests = 4 if timeout else 7  # the transcript also counts the last reply
        assert (ep["gpt_calls"], ep["gpt6"]["calls"], ep["qwen"]["calls"]) == (
            requests,
            requests,
            0,
        )
        assert ep["gpt_calls_by_reason"]["planner"] == requests
        assert ep["usage_source"] == ("run.log" if timeout else "transcript")
        want = senior_usd(20000 * requests, 5000 * requests, 300 * requests)
        assert (
            rec["usd"] == pytest.approx(want, rel=1e-6)
            and rec["usage"]["planner"] == "gpt6"
        )
        assert ep["gpt6"]["usd"] == pytest.approx(want, rel=1e-4)
        assert rec["success"] is (not timeout)
        state = gfe.state(job)
        assert all(
            t["usage"]["planner"] == "gpt6" for t in state["tries"] if t.get("usage")
        )
        if timeout:  # the infra launch was relaunched, never scored; its cost counts
            assert [t["class"] for t in state["tries"]] == ["infra", "valid"]
            assert rec["usd_all_tries"] == pytest.approx(
                want + senior_usd(20000 * 9, 5000 * 9, 300 * 9), rel=1e-6
            )
    starts = [e for e in gfe.ecc.events() if e.get("event") == "start"]
    gpt6_starts = [e["env"] for e in starts if e["arm"] == "gpt6_only"]
    assert len(gpt6_starts) == 8  # 6 episodes + 2 infra relaunches
    for env in gpt6_starts:
        assert (
            env["ECC_RC_ESCALATION"],
            env["ECC_RC_SENIOR_ALONE"],
            env["ECC_RC_MAX_IMAGES"],
        ) == ("0", "1", "40")
        assert (env["ECC_RC_REASONING_IDS"], env["ECC_RC_SENIOR_MODEL"]) == (
            "0",
            "openai:gpt-6-astra",
        )
        assert env["ECC_RC_SENIOR_EFFORT"] == "low" and env["ECC_RC_CORRECTIONS"] == ""
    qwen_env = [e["env"] for e in starts if e["arm"] == "qwen_only"]
    assert qwen_env and all(e["ECC_RC_ESCALATION"] == "0" for e in qwen_env)
    score = gfe.cli(
        "score",
        "--config",
        str(config),
        "--run-id",
        "ft1",
        "--json",
        str(gfe.root / "score.json"),
    )
    assert score.returncode == 0, score.stderr
    doc = json.loads((gfe.root / "score.json").read_text())
    g, q = doc["arms"]["gpt6_only"], doc["arms"]["qwen_only"]
    assert g["scored"] == 6 and g["qwen_calls_per_episode"] == 0
    assert g["gpt6_calls_per_episode"] == pytest.approx((7 * 4 + 4 * 2) / 6)
    assert g["gpt6_calls_per_episode_by_reason"] == {
        "planner": pytest.approx((7 * 4 + 4 * 2) / 6)
    }
    assert (
        g["usd_gpt6"] == pytest.approx(g["usd_scored"], rel=1e-3) and g["usd_qwen"] == 0
    )
    assert q["gpt6_calls_per_episode"] == 0 and q["qwen_calls_per_episode"] == 10
    assert "gpt6_only" in score.stdout and "planner" in score.stdout


def test_cost_guard_prices_gpt6_alone_launches_as_the_senior(gfe):
    """One episode costs $1.29 at GPT-6's list price: the $0.5 ceiling stops the run after the
    first launch. At the worker's price ($0.055 each) all 6 episodes would have run."""
    gfe.ecc.set_scenario(
        {"default": {"kind": "fail", "requests": 5, "tokens": [20000, 0, 300]}}
    )
    per_episode = senior_usd(20000 * 6, 0, 300 * 6)  # 6 requests with the last reply
    assert per_episode == pytest.approx(1.29)
    config = gfe.gpt6_config(
        arms=("gpt6_only",), seeds=(1, 2), scheduler={"cost_ceiling_usd": 0.5}
    )
    proc = gfe.go(config, "--max-slots", "1")
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    stop = json.loads((gfe.run_root() / "STOP").read_text())
    # the guard may already trip on the running launch's senior-priced snapshot
    assert stop["reason"] == "cost ceiling" and stop["usd"] >= 0.5
    launched = [e for e in gfe.ecc.events() if e.get("event") == "start"]
    assert len(launched) == 1
    state = gfe.state(FT.job_name("gpt6_only", "CloseBlenderLid", 1, 1))
    assert state["tries"][0]["usage"]["usd"] == pytest.approx(per_episode)


def test_running_gpt6_launches_are_priced_as_the_senior(gfe, tmp_path):
    stage = _stage(gfe, gfe.gpt6_config(arms=("gpt6_only",), seeds=(1,)))
    cfg = stage.cfg
    layout = C.RunLayout(tmp_path / "run")
    stage.layout = layout
    pool = FT.FinalServerPool(cfg, layout, {}, lambda row: None)
    sched = FT.FinalScheduler(
        cfg, layout, stage, pool, infra_retries=2, render_retries=6
    )
    job = stage.jobs()[0]
    out = _episode(
        tmp_path / "ep", stats=None
    )  # still running: no transcript stats yet
    state = sched.state(job.job_id)
    state["tries"].append({"n": 1, "dir": str(out), "status": "running", "pid": 999999})
    sched.save(state)
    assert C.total_cost(sched.all_states(), cfg["pricing"]) < 0.02  # the worker's price
    sched._price_running_alone()
    usage = sched.state(job.job_id)["tries"][0]["usage"]
    assert usage["partial"] and usage["planner"] == "gpt6"
    want = senior_usd(30000, 6000, 250)
    assert C.total_cost(sched.all_states(), cfg["pricing"]) == pytest.approx(want)
    assert stage.total_cost(sched.all_states()) == pytest.approx(want)
    # when the launch ends, finish_try's generic usage is replaced by the senior-priced one
    (out / "transcript_T_target_s1.json").write_text(
        json.dumps(
            {
                "stats": {
                    "total_input_tokens": 50000,
                    "cache_read_tokens": 0,
                    "total_output_tokens": 500,
                    "requests": 3,
                }
            }
        )
    )
    state = sched.state(job.job_id)
    state["tries"][0].update(status="done", usage=C.episode_usage(out, cfg["pricing"]))
    state["tries"][0]["class"] = "valid"
    sched.save(state)
    assert sched.state(job.job_id)["tries"][0]["usage"]["usd"] == pytest.approx(
        senior_usd(50000, 0, 500)
    )


def test_evidence_mismatch_stops_the_run(gfe):
    gfe.ecc.set_scenario(
        {
            "default": {"kind": "fail", "requests": 3},
            f"{FT.job_name('gpt6_only', 'CloseBlenderLid', 1, 1)}": {
                "kind": "fail",
                "bug": ["no_cap_log"],
            },
        }
    )
    config = gfe.gpt6_config(arms=("gpt6_only",), seeds=(1,))
    proc = gfe.go(config, "--max-slots", "1")
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    rows = [r for r in gfe.progress() if r.get("event") == "evidence_mismatch"]
    assert rows and "image cap" in " ".join(rows[0]["problems"])
    state = gfe.state(FT.job_name("gpt6_only", "CloseBlenderLid", 1, 1))
    assert state["tries"][0]["class"] == "infra" and not state["final"]


def test_go_refuses_gpt6_arms_with_a_launcher_without_the_switch(gfe, tmp_path):
    old = tmp_path / "old_cell.sh"
    old.write_text("#!/bin/bash\n# honours ECC_RC_ESCALATION only\nexit 1\n")
    config = gfe.gpt6_config(
        arms=("gpt6_only",), seeds=(1,), launcher=["bash", str(old)]
    )
    plan = gfe.cli("plan", "--config", str(config), "--run-id", "ft1")
    assert (
        plan.returncode == 0
        and "lacks the ECC_RC_ESCALATION / ECC_RC_SENIOR_ALONE switch" in plan.stdout
    )
    proc = gfe.cli("run", "--config", str(config), "--run-id", "ft1", "--go")
    assert proc.returncode != 0 and "lacks the" in (proc.stdout + proc.stderr)


def test_plan_shows_the_gpt6_arm(gfe):
    config = gfe.gpt6_config(arms=("gpt6_only",), seeds=(1,))
    proc = gfe.cli("plan", "--config", str(config), "--run-id", "ft1")
    assert proc.returncode == 0, proc.stderr
    assert (
        "GPT-6 alone (openai:gpt-6-astra, effort low, at most 40 images per request, reasoning ids off)"
        in proc.stdout
    )
    assert (
        "ECC_RC_SENIOR_ALONE=1" in proc.stdout and "ECC_RC_MAX_IMAGES=40" in proc.stdout
    )
    assert "config row required" not in proc.stdout  # no hybrid arm
    assert not gfe.run_root().exists()


# -- GPUs shared with this machine's other runs --------------------------------------------


def _gpu_files(
    root: Path, apps: list[tuple[int, int, int]]
) -> tuple[list[str], list[str]]:
    """Fake nvidia-smi app and uuid queries: ``apps`` = [(gpu, pid, MiB), ...]."""
    uuids = root / "uuids.txt"
    uuids.write_text("".join(f"{g}, GPU-fake-{g}\n" for g in range(8)))
    listing = root / "apps.txt"
    listing.write_text("".join(f"GPU-fake-{g}, {pid}, {mb}\n" for g, pid, mb in apps))
    reader = "import sys; print(open(sys.argv[1]).read(), end='')"
    return [sys.executable, "-c", reader, str(listing)], [
        sys.executable,
        "-c",
        reader,
        str(uuids),
    ]


@pytest.fixture
def procs():
    started = []

    def spawn(argv):
        p = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started.append(p)
        time.sleep(0.2)
        return p.pid

    yield spawn
    for p in started:
        p.kill()
        p.wait()


def _share_cfg(gfe, tmp_path, apps, *, used, util=60, share=True):
    apps_cmd, uuid_cmd = _gpu_files(tmp_path, apps)
    gfe.ecc.set_gpus({4: (used, util), 5: (10, 0), 6: (20000, 60), 7: (5, 0)})
    config = gfe.gpt6_config(
        arms=("gpt6_only",),
        seeds=(1,),
        share_own_gpus=share,
        gpu_apps_cmd=apps_cmd,
        gpu_uuid_cmd=uuid_cmd,
        own_process_prefixes=[str(Path(sys.executable).parent)],
    )
    return FT.stage_cfg(FT.load_config(config))


@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="reads process command lines from /proc"
)
def test_shared_gpu_room_rule(gfe, tmp_path, procs):
    ours = procs([sys.executable, "-c", "import time; time.sleep(30)"])
    other = procs(["sleep", "30"])
    cfg = _share_cfg(gfe, tmp_path, [(4, ours, 50000)], used=50600)
    usage = C.gpu_usage(cfg)
    gpu_procs = FT.gpu_processes(cfg)
    assert gpu_procs[4] == [(ours, 50000)]
    assert FT.shared_gpu_room(cfg, 4, 25000, usage, gpu_procs)[0]
    ok, why = FT.shared_gpu_room(cfg, 4, 40000, usage, gpu_procs)
    assert not ok and "40000 MiB needed" in why
    ok, why = FT.shared_gpu_room(
        cfg, 4, 1000, usage, {4: [(ours, 50000), (other, 300)]}
    )
    assert not ok and "not started from" in why
    ok, why = FT.shared_gpu_room(
        cfg, 4, 1000, usage, {4: [(ours, 40000)]}
    )  # 10.6 GB nobody here accounts for
    assert not ok and "not visible here" in why
    assert not FT.shared_gpu_room(cfg, 1, 1000, usage, gpu_procs)[
        0
    ]  # never a GPU outside gpus.allowed


@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="reads process command lines from /proc"
)
def test_server_pool_starts_on_a_gpu_shared_with_our_runs_only_when_it_fits(
    gfe, tmp_path, procs
):
    ours = procs([sys.executable, "-c", "import time; time.sleep(30)"])
    for share, used, started in (
        (False, 50600, False),
        (True, 50600, True),
        (True, 70600, False),
    ):
        cfg = _share_cfg(gfe, tmp_path, [(4, ours, used - 600)], used=used, share=share)
        cfg["endpoints"] = [
            cfg["endpoints"][0]
        ]  # one endpoint on GPU 4 (1 server, 4 slots)
        assert cfg["endpoints"][0]["gpu"] == 4
        layout = C.RunLayout(tmp_path / f"run_{share}_{used}")
        rows = []
        pool = FT.FinalServerPool(cfg, layout, {}, rows.append)
        try:
            up = pool.ensure(wait=True)
        finally:
            pool.stop_all()
        events = [r["event"] for r in rows]
        assert bool(up) is started, rows
        if started:
            assert "server_shared_gpu" in events and "server_ready" in events
        else:
            assert "server_refused" in events and "server_start" not in events
        if (
            share and not started
        ):  # 81559 - 70600 MiB free < 17500 + 4 x 600 + 2048 needed
            assert any("21948 MiB needed" in str(r.get("reason")) for r in rows), rows


def _busy_config(gfe, tmp_path, *, share: bool, ours: int) -> Path:
    apps_cmd, uuid_cmd = _gpu_files(tmp_path, [(4, ours, 70000), (5, ours, 70000)])
    gfe.ecc.set_gpus({4: (70600, 60), 5: (70600, 60), 6: (20000, 60), 7: (5, 0)})
    config = gfe.gpt6_config(
        arms=("gpt6_only",),
        tasks=THREE[:1],
        seeds=(1,),
        share_own_gpus=share,
        gpu_apps_cmd=apps_cmd,
        gpu_uuid_cmd=uuid_cmd,
        own_process_prefixes=[str(Path(sys.executable).parent)],
    )
    raw = json.loads(config.read_text())
    raw["servers"].update(retry_busy_s=1, max_refusals=60)
    config.write_text(json.dumps(raw, indent=2))
    return config


@pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="reads process command lines from /proc"
)
def test_with_shared_gpus_the_runner_waits_for_room(gfe, tmp_path, procs):
    ours = procs([sys.executable, "-c", "import time; time.sleep(60)"])
    gfe.ecc.set_scenario({"default": {"kind": "solve", "requests": 2}})
    config = _busy_config(gfe, tmp_path, share=True, ours=ours)
    runner = subprocess.Popen(
        [sys.executable, str(FT.REPO_ROOT / "scripts" / "ecc_robocasa" / "final_test.py"), "run",
         "--config", str(config), "--run-id", "ft1", "--go"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    try:
        gfe.ecc.wait_for(
            lambda: (
                (gfe.run_root() / "progress.jsonl").exists()
                and sum(r.get("event") == "server_refused" for r in gfe.progress()) >= 4
            ),
            timeout=60,
        )
        assert not any(r.get("event") == "launch" for r in gfe.progress())
        assert runner.poll() is None  # still waiting
        gfe.ecc.set_gpus(
            {4: (10, 0), 5: (10, 0), 6: (20000, 60), 7: (5, 0)}
        )  # the other run ended
        out, _ = runner.communicate(timeout=120)
    finally:
        if runner.poll() is None:
            runner.kill()
    assert "waiting for GPU room" in out and "runner finished: done" in out, out
    reasons = [
        r["reason"] for r in gfe.progress() if r.get("event") == "server_refused"
    ]
    assert any("21948 MiB needed" in r or "MiB needed" in r for r in reasons), reasons
    record = gfe.record(FT.job_name("gpt6_only", THREE[0], 1, 1))
    assert record["valid"] and record["success"] and record["evidence_ok"]


def test_without_shared_gpus_the_runner_still_exits_when_no_server_starts(
    gfe, tmp_path
):
    config = _busy_config(gfe, tmp_path, share=False, ours=1)
    proc = gfe.cli(
        "run", "--config", str(config), "--run-id", "ft1", "--go", timeout=120
    )
    assert "runner finished: no_endpoint" in proc.stdout, proc.stdout + proc.stderr
    assert "waiting for GPU room" not in proc.stdout
