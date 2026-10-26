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

"""The final-test runner end to end, offline: fake stage products, fake episodes that drive
RPent's real escalation wrapper, a fake RLDX server and a fake nvidia-smi. No model request."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import final_test as FT
from scripts.ecc_robocasa import hybrid_episode as HE
from scripts.ecc_robocasa import stage1 as S1
from tests.unit_tests.ecc_robocasa.conftest import REPO, TASKS

pytest.importorskip("pydantic_ai")  # the fake episodes run RPent's escalation wrapper

FINAL = REPO / "scripts" / "ecc_robocasa" / "final_test.py"
FAKE_CELL = Path(__file__).resolve().parent / "test_final_test_fake_cell.py"
HYBRID_CELL = REPO / "scripts" / "ecc_robocasa" / "hybrid_cell.sh"
THREE = TASKS[:3]  # CloseBlenderLid, CloseFridge, CloseToasterOvenDoor (atomic)
#: Rungs given to the fake stage-2 rules, by task index.
RUNGS = [2, 6, 1, 3, 5, 4, 2, 6, 1, 3]
ARMS = {
    "s2_rules": {
        "planner": "hybrid",
        "memory": "s1",
        "rule": "s2",
        "corrections": "none",
    },
    "s3_rules_corr": {
        "planner": "hybrid",
        "memory": "s1",
        "rule": "s2",
        "corrections": "s3",
    },
    "base_hybrid": {
        "planner": "hybrid",
        "memory": "cloud",
        "rule": "fixed",
        "plan": True,
        "checkin": 5,
        "errors": 2,
    },
    "qwen_only": {"planner": "qwen", "memory": "cloud"},
}


class FinalEnv:
    """Frozen stage products (S1 corpus, S2 rules, S3 corrections) next to the S1 test machine.

    Their layout and documents are the ones ``stage1/2/3.verify`` accept.
    """

    def __init__(self, ecc, root: Path):
        self.ecc = ecc
        ecc.run_id = "ft1"  # the S1 fixture's cleanup then covers this run's servers
        self.root = root
        self.s1 = root / "products" / "s1" / "product"
        self._s1_product()
        self.s1_hash = json.loads((self.s1 / "PRODUCT.json").read_text())[
            "corpus_tree_sha256"
        ]
        self.s2 = self.s2_product(root / "products" / "s2" / "product", self.s1_hash)
        self.rules = self.s2 / "rules"
        self.s3 = self.s3_product(root / "products" / "s3" / "product")
        self.final = self.s3 / "final"

    def _s1_product(self) -> None:
        corpus = self.s1 / "corpus"
        shutil.copytree(self.ecc.cloud, corpus)
        (self.s1 / "tasks.json").write_text("{}\n")
        files = C.tree_files(corpus)
        digest = C.tree_hash(files)
        C.atomic_write_json(
            self.s1 / "PRODUCT.json",
            {
                "schema": S1.PRODUCT_SCHEMA,
                "corpus": "corpus",
                "corpus_tree_sha256": digest,
                "files": files,
                "tasks_json_sha256": C.sha256_file(self.s1 / "tasks.json"),
            },
        )
        (self.s1 / "FROZEN").write_text(digest + "\n")
        S1._make_read_only(corpus)
        assert S1.verify(self.s1)["ok"]

    def s2_product(self, product: Path, s1_hash: str) -> Path:
        from scripts.ecc_robocasa import stage2 as S2

        (product / "rules").mkdir(parents=True)
        tasks = {}
        for i, task in enumerate(TASKS):
            rung = RUNGS[i]
            rule = HE.LEADER_LADDER[rung - 1]
            doc = {
                "schema": S2.RULE_SCHEMA,
                "task": task,
                "rule": rule.as_json(),
                "config": rule.config_name(rung),
                "rung": rung,
                "rule_source": "first_success" if rung < 6 else "fallback",
                "solved_in_search": rung < 6,
            }
            (product / "rules" / f"{task}.json").write_text(json.dumps(doc))
            tasks[task] = {"rung": rung, "config": doc["config"], "rule": doc["rule"]}
        (product / "episodes.json").write_text("{}\n")
        files = C.tree_files(product / "rules")
        digest = C.tree_hash(files)
        C.atomic_write_json(
            product / "RULES.json",
            {
                "schema": S2.PRODUCT_SCHEMA,
                "rules": "rules",
                "rules_tree_sha256": digest,
                "files": files,
                "episodes_json_sha256": C.sha256_file(product / "episodes.json"),
                "tasks": tasks,
                "s1_product": {"corpus_tree_sha256": s1_hash},
            },
        )
        (product / "FROZEN").write_text(digest + "\n")
        S1._make_read_only(product / "rules")
        assert S2.verify(product)["ok"]
        return product

    def s3_product(self, product: Path) -> Path:
        from scripts.ecc_robocasa import stage3 as S3

        final = product / "final"
        final.mkdir(parents=True)
        for i, task in enumerate(TASKS):
            text = (
                ""
                if i % 3 == 2
                else f"- [c{i}] (verified; trigger: checkin) situation: {task} lid | fix: push left\n"
            )
            (final / f"{task}.md").write_text(text)
            (final / f"{task}.json").write_text(
                json.dumps({"task": task, "entries": []})
            )
        for name in ("tasks.json", "summary.json"):
            (product / name).write_text("{}\n")
        files = C.tree_files(final)
        digest = C.tree_hash(files)
        rule_files = {
            task: C.sha256_file(self.rules / f"{task}.json") for task in TASKS
        }
        C.atomic_write_json(
            product / "PRODUCT.json",
            {
                "schema": S3.PRODUCT_SCHEMA,
                "final": "final",
                "final_tree_sha256": digest,
                "files": files,
                "tasks_json_sha256": C.sha256_file(product / "tasks.json"),
                "summary_json_sha256": C.sha256_file(product / "summary.json"),
                "tasks": {task: {} for task in TASKS},
                "inputs": {
                    "s1": {"corpus_tree_sha256": self.s1_hash},
                    "rules": {
                        "folder": str(self.rules),
                        "name": "{task}.json",
                        "files": rule_files,
                    },
                },
            },
        )
        (product / "FROZEN").write_text(digest + "\n")
        S1._make_read_only(final)
        assert S3.verify(product)["ok"]
        return product

    def config(
        self,
        *,
        arms=("s2_rules", "s3_rules_corr"),
        tasks=THREE,
        seeds=(1, 2),
        cell=None,
        products=None,
        **ft,
    ) -> Path:
        raw = json.loads(
            self.ecc.config(
                tasks=list(tasks), endpoints=((4, 4), (5, 4)), max_slots=6
            ).read_text()
        )
        raw["final_test"] = {
            "seeds": list(seeds),
            "tasks": list(tasks),
            "run_arms": list(arms),
            "launcher": cell or [sys.executable, str(FAKE_CELL)],
            "products": products
            or {"s1": str(self.s1), "s2": str(self.s2), "s3": str(self.s3)},
            "arms": ARMS,
            "permutations": 2000,
            "bootstrap": 500,
            **ft,
        }
        path = self.root / "ft_config.json"
        path.write_text(json.dumps(raw, indent=2))
        return path

    def cli(self, *args: str, timeout: float = 240) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(FINAL), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def go(
        self, config: Path, *extra: str, run_id: str = "ft1"
    ) -> subprocess.CompletedProcess:
        proc = self.cli(
            "run", "--config", str(config), "--run-id", run_id, "--go", *extra
        )
        assert proc.returncode in (0, 1), proc.stdout + proc.stderr
        return proc

    def run_root(self, run_id: str = "ft1") -> Path:
        return self.ecc.runs / run_id

    def state(self, job: str, run_id: str = "ft1") -> dict:
        return json.loads(
            (self.run_root(run_id) / "state" / "jobs" / f"{job}.json").read_text()
        )

    def record(self, job: str, run_id: str = "ft1") -> dict:
        return json.loads(
            (self.run_root(run_id) / "cells" / job / "episode.json").read_text()
        )

    def progress(self, run_id: str = "ft1") -> list[dict]:
        return [
            json.loads(line)
            for line in (self.run_root(run_id) / "progress.jsonl")
            .read_text()
            .splitlines()
        ]


@pytest.fixture
def fe(ecc, tmp_path):
    return FinalEnv(ecc, tmp_path)


def _stage(fe, config: Path, args: str = "") -> FT.FinalTest:
    cfg = FT.load_config(config)
    ns = FT.argparse.Namespace(arms=None, tasks=None, seeds=None, repeats=0, limit=0)
    run_cfg = FT.stage_cfg(FT.apply_cli(cfg, ns))
    tasks = FT.run_tasks(run_cfg)
    layout = C.RunLayout(fe.root / "probe_run")
    inputs = FT.snapshot_inputs(
        layout, FT.read_inputs(run_cfg, run_cfg["final_test"]["run_arms"], tasks), tasks
    )
    return FT.FinalTest(
        run_cfg,
        layout,
        FT.build_jobs(run_cfg, run_cfg["final_test"]["run_arms"], tasks),
        inputs,
    )


# -- the launch ----------------------------------------------------------------------------


def test_plan_lists_inputs_and_creates_nothing(fe):
    config = fe.config(arms=("s2_rules", "s3_rules_corr", "base_hybrid"))
    proc = fe.cli("plan", "--config", str(config), "--run-id", "ft1")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert (
        "3 tasks (atomic 3) x seeds [1, 2] x 1 repeat(s) = 6 cells; 3 arms; 18 episodes"
        in out
    )
    assert "(product): 3 tasks; per rung 1: 1, 2: 1, 6: 1" in out
    assert "s3 corrections" in out and "1 empty" in out and "WARNING" not in out
    assert "memory s1:" in out and "memory cloud:" in out
    assert "ECC_RC_ESCALATION=" in out and "ECC_RC_CORRECTIONS" in out
    assert not fe.run_root().exists()


def test_knobs_follow_each_tasks_rule_and_the_launch_environment(fe):
    stage = _stage(
        fe,
        fe.config(
            arms=("s2_rules", "s3_rules_corr", "base_hybrid", "qwen_only"), seeds=(1,)
        ),
    )
    by = {(j.meta["arm"], j.task): j for j in stage.jobs()}
    for i, task in enumerate(THREE):
        rule = HE.LEADER_LADDER[RUNGS[i] - 1]
        for arm in ("s2_rules", "s3_rules_corr"):
            k = stage.knobs[by[(arm, task)].job_id]
            assert (k["plan"], k["checkin"], k["errors"], k["rung"]) == (
                rule.plan,
                rule.checkin,
                rule.errors,
                RUNGS[i],
            )
            assert k["memory"] == "s1" and k["timeout_s"] == 1800
        corr = stage.knobs[by[("s3_rules_corr", task)].job_id]["corrections"]
        text = (fe.final / f"{task}.md").read_text()
        assert (
            corr["chars"] == len(text.strip())
            and Path(corr["file"]).read_text() == text
        )
        assert stage.knobs[by[("s2_rules", task)].job_id]["corrections"] is None
    endpoint = C.Endpoint("gpu4", 4, 29999, 2)
    job = by[("s3_rules_corr", "CloseFridge")]
    env = FT.cell_env(
        stage.cfg,
        job,
        1,
        endpoint,
        stage.knobs[job.job_id],
        parent={"PATH": "/usr/bin:/bin"},
    )
    rule = HE.LEADER_LADDER[RUNGS[1] - 1]
    assert (env["ECC_RC_PLAN"], env["ECC_RC_CHECKIN"], env["ECC_RC_ERRORS"]) == (
        "1" if rule.plan else "0",
        str(rule.checkin),
        str(rule.errors),
    )
    assert env["ECC_RC_ESCALATION"] == "1" and env["ECC_RC_TIMEOUT_S"] == "1800"
    assert env["ECC_RC_CORRECTIONS"].endswith("inputs/corrections/CloseFridge.md")
    assert (
        env["ECC_RC_CORRECTIONS_SHA256"]
        == stage.knobs[job.job_id]["corrections"]["sha256"]
    )
    qwen = by[("qwen_only", "CloseFridge")]
    env = FT.cell_env(
        stage.cfg,
        qwen,
        1,
        endpoint,
        stage.knobs[qwen.job_id],
        parent={"PATH": "/usr/bin:/bin"},
    )
    assert env["ECC_RC_ESCALATION"] == "0" and env["ECC_RC_CORRECTIONS"] == ""
    base = by[("base_hybrid", "CloseFridge")]
    assert stage.knobs[base.job_id]["memory"] == "cloud"
    assert (stage.knobs[base.job_id]["checkin"], stage.knobs[base.job_id]["rung"]) == (
        5,
        None,
    )
    # the snapshot is read-only and loads every task
    assert not os.access(
        Path(stage.knobs[job.job_id]["memory_dir"]) / "task-specific", os.W_OK
    )
    assert FT.verify_snapshot(stage.layout, stage.inputs) == []


def _dumper(path: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "json.dump({'argv': sys.argv, 'env': dict(os.environ)}, open(os.environ['FAKE_DUMP'], 'w'))\n"
    )
    path.chmod(0o755)
    return path


@pytest.mark.skipif(
    not HYBRID_CELL.is_file(), reason="hybrid_cell.sh is not in this checkout"
)
def test_the_hybrid_launcher_receives_each_arms_routing(fe, tmp_path):
    """The real hybrid_cell.sh, with rpent replaced by a dumper: what each arm's episode gets."""
    venv = tmp_path / "dumpvenv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "bin" / "python").chmod(0o755)
    _dumper(venv / "bin" / "rpent")
    llvmpipe = tmp_path / "llvmpipe.sh"
    llvmpipe.write_text("echo 8\n")
    config = fe.config(
        arms=("s3_rules_corr", "qwen_only"), seeds=(1,), cell=["bash", str(HYBRID_CELL)]
    )
    raw = json.loads(config.read_text())
    raw["paths"].update(venv=str(venv), llvmpipe_index_script=str(llvmpipe))
    config.write_text(json.dumps(raw))
    stage = _stage(fe, config)
    endpoint = C.endpoints_from_config(stage.cfg)[0]
    for job in stage.jobs():
        if job.task != "CloseFridge":
            continue
        argv, env = stage.launch(job, 1, endpoint)
        env = dict(
            env,
            PATH="/usr/bin:/bin",
            HOME=str(tmp_path),
            FAKE_DUMP=str(tmp_path / "dump.json"),
        )
        proc = subprocess.run(
            argv, env=env, capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, proc.stderr
        dump = json.loads((tmp_path / "dump.json").read_text())
        e, a = dump["env"], dump["argv"]
        k = stage.knobs[job.job_id]
        assert a[a.index("--memory-dir") + 1] == k["memory_dir"]
        assert a[a.index("--planner-timeout-s") + 1] == "1800"
        assert a[a.index("--output-dir") + 1] == str(stage.try_dir(job, 1))
        if job.meta["arm"] == "s3_rules_corr":
            from rpent.cli.main import _build_argparser
            from rpent.robots import get_robot_spec

            parser = _build_argparser()
            spec = get_robot_spec("robocasa")
            spec.add_cli_args(parser, use_dashboard=False)
            parsed = parser.parse_args(
                a[1:]
            )  # RPent's own parser and RoboCasa config read the command
            assert (parsed.seed, parsed.max_turns, parsed.planner_timeout_s) == (
                1,
                100,
                1800,
            )
            assert parsed.memory_profile == "local" and not parsed.explore
            assert Path(spec.parse_config(parsed).output_dir) == stage.try_dir(job, 1)
            assert e["RPENT_ESCALATION"] == "1"
            assert e["RPENT_ESCALATION_CHECKIN"] == str(k["checkin"]) and e[
                "RPENT_ESCALATION_ERRORS"
            ] == str(k["errors"])
            assert e.get("RPENT_ESCALATION_PLAN") == (None if k["plan"] else "0")
            assert e["RPENT_CLOUD_CORRECTIONS"] == k["corrections"]["file"]
        elif FT.launcher_runs_qwen_alone(stage.cfg):
            assert e["RPENT_ESCALATION"] == "0" and "RPENT_CLOUD_CORRECTIONS" not in e
        assert not [name for name in e if name.startswith("ECC_RC_")]


def test_qwen_arms_need_the_launcher_switch(fe):
    config = fe.config(
        arms=("qwen_only",),
        seeds=(1,),
        tasks=THREE[:1],
        cell=["bash", str(HYBRID_CELL)],
    )
    plan = fe.cli("plan", "--config", str(config))
    assert plan.returncode == 0, plan.stderr
    if HYBRID_CELL.is_file() and FT.QWEN_SWITCH in HYBRID_CELL.read_text():
        assert "WARNING" not in plan.stdout
        return
    assert "has no ECC_RC_ESCALATION switch" in plan.stdout
    proc = fe.cli("run", "--config", str(config), "--run-id", "ft1", "--go")
    assert proc.returncode != 0 and "has no ECC_RC_ESCALATION switch" in proc.stderr
    assert not fe.run_root().exists()


# -- runs ----------------------------------------------------------------------------------------


def test_two_arms_run_interleaved_and_are_scored_paired(fe):
    # 16 requests: a rung-1 rule (no plan, K = 12) reaches its first check-in at request 13
    fe.ecc.set_scenario(
        {
            "s3_rules_corr": {"kind": "solve", "requests": 16},
            "s2_rules:CloseFridge": {"kind": "solve", "requests": 16},
            "default": {"kind": "fail", "requests": 16},
        }
    )
    config = fe.config()
    proc = fe.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    stage = _stage(fe, config)
    launches = [r["job"] for r in fe.progress() if r.get("event") == "launch"]
    assert launches == [
        j.job_id for j in stage.jobs()
    ]  # cell by cell, arms in the seeded order
    for job in stage.jobs():
        record = fe.record(job.job_id)
        assert record["valid"] and record["evidence_ok"], record
        assert record["success"] == (
            job.meta["arm"] == "s3_rules_corr" or job.task == "CloseFridge"
        )
        evidence = json.loads(
            (fe.run_root() / "cells" / job.job_id / "try1.evidence.json").read_text()
        )
        assert evidence["ok"] and evidence["counts"]["senior_calls"] >= 1
        rows = [
            json.loads(x)
            for x in (Path(record["final_try_dir"]) / "escalation.jsonl")
            .read_text()
            .splitlines()
        ]
        want = (record["knobs"]["corrections"] or {}).get("chars", 0)
        seniors = [r for r in rows if r.get("role") == "senior"]
        assert seniors and all(r.get("corrections_chars", 0) == want for r in seniors)
        plan_rows = [r for r in seniors if r.get("reason") == "plan"]
        assert bool(plan_rows) == record["knobs"]["plan"]
    starts = [
        json.loads(x)
        for x in fe.ecc.fake_log.read_text().splitlines()
        if '"start"' in x
    ]
    assert all(not s["memory_writable"] for s in starts)
    assert {
        bool(s["env"]["ECC_RC_CORRECTIONS"]) for s in starts if s["arm"] == "s2_rules"
    } == {False}
    assert all(
        s["env"]["RC_GPU"] in ("4", "5") and s["env"]["RC_RENDER"] == "cpu"
        for s in starts
    )
    score = fe.cli(
        "score",
        "--config",
        str(config),
        "--run-id",
        "ft1",
        "--json",
        str(fe.root / "score.json"),
    )
    assert score.returncode == 0, score.stderr
    doc = json.loads((fe.root / "score.json").read_text())
    s3, s2 = doc["arms"]["s3_rules_corr"], doc["arms"]["s2_rules"]
    assert (s3["scored"], s3["successes"], s2["successes"]) == (6, 6, 2)
    assert s3["task_weighted_pct"]["atomic"] == 100 and s2["task_weighted_pct"][
        "overall"
    ] == pytest.approx(100 / 3)
    assert s3["gpt6_calls_per_episode"] > 0 and s3["usd_per_episode"] > 0
    (pair,) = doc["pairs"]
    assert (pair["a"], pair["b"], pair["cells"]) == ("s2_rules", "s3_rules_corr", 6)
    assert (pair["all"]["a_only"], pair["all"]["b_only"]) == (0, 4)
    assert pair["all"]["mcnemar_p"] == pytest.approx(2 / 16)
    assert pair["all"]["task_weighted_diff_pct"] == pytest.approx(-200 / 3)
    assert "s2_rules vs s3_rules_corr: 6 cells scored in both" in score.stdout
    flipped = fe.cli(
        "score",
        "--config",
        str(config),
        "--run-id",
        "ft1",
        "--pair",
        "s3_rules_corr:s2_rules",
        "--invalid-as-failure",
        "--json",
        str(fe.root / "score2.json"),
    )
    assert flipped.returncode == 0, flipped.stderr
    doc2 = json.loads((fe.root / "score2.json").read_text())
    assert doc2["invalid_as_failure"] and doc2["pairs"][0]["all"]["a_only"] == 4
    assert doc2["pairs"][0]["all"]["permutation"]["stat"] == pytest.approx(2 / 3)
    bad = fe.cli(
        "score",
        "--config",
        str(config),
        "--run-id",
        "ft1",
        "--pair",
        "s2_rules:base_hybrid",
    )
    assert bad.returncode != 0 and "two different run arms" in bad.stderr
    status = fe.cli("status", "--config", str(config), "--run-id", "ft1")
    assert status.returncode == 0, status.stderr
    assert "[####################] 12/12 episodes final" in status.stdout
    assert "s3_rules_corr" in status.stdout and "stop file: absent" in status.stdout
    assert oct((fe.run_root() / "cells").stat().st_mode & 0o777) == "0o755"


def test_a_routing_mismatch_stops_the_run_and_its_launch_is_retried(fe):
    job = "s2_rules__CloseFridge_s1_r1"
    fe.ecc.set_scenario(
        {
            job: [
                {"kind": "solve", "requests": 16, "bug": {"checkin": 9}},
                {"kind": "solve", "requests": 16},
            ]
        }
    )
    config = fe.config(arms=("s2_rules",), tasks=THREE[:2], seeds=(1,))
    proc = fe.go(config)
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    stop = json.loads((fe.run_root() / "STOP").read_text())
    assert stop["reason"] == "escalation evidence mismatch" and stop["job"] == job
    state = fe.state(job)
    assert state["final"] is None
    assert state["tries"][0]["class"] == "infra" and state["tries"][0][
        "cause"
    ].startswith("evidence_mismatch")
    evidence = json.loads(
        (fe.run_root() / "cells" / job / "try1.evidence.json").read_text()
    )
    assert not evidence["ok"] and any(
        "checkin" in p or "check-in" in p for p in evidence["problems"]
    )
    assert any(r.get("event") == "evidence_mismatch" for r in fe.progress())
    (fe.run_root() / "STOP").unlink()
    proc = fe.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    state = fe.state(job)
    assert [t["class"] for t in state["tries"]] == ["infra", "valid"] and state[
        "final"
    ]["success"]
    record = fe.record(job)
    assert record["evidence_ok"] and record["tries"][0]["cause"].startswith(
        "evidence_mismatch"
    )
    manifest = json.loads((fe.run_root() / "manifest.json").read_text())
    assert [h["event"] for h in manifest["history"]] == ["created", "resumed"]


def test_infrastructure_and_render_failures_are_retried_not_scored(fe):
    fe.ecc.set_scenario(
        {
            "s2_rules__CloseBlenderLid_s1_r1": [{"kind": "infra"}, {"kind": "solve"}],
            "s2_rules__CloseFridge_s1_r1": [{"kind": "noise"}, {"kind": "fail"}],
            "s2_rules__CloseToasterOvenDoor_s1_r1": {"kind": "timeout"},
        }
    )
    config = fe.config(arms=("s2_rules",), seeds=(1,))
    proc = fe.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    a = fe.state("s2_rules__CloseBlenderLid_s1_r1")
    assert [t["class"] for t in a["tries"]] == ["infra", "valid"] and a["final"][
        "success"
    ]
    b = fe.state("s2_rules__CloseFridge_s1_r1")
    assert [t["class"] for t in b["tries"]] == ["render", "valid"] and not b["final"][
        "success"
    ]
    timed_out = fe.record("s2_rules__CloseToasterOvenDoor_s1_r1")
    assert (
        timed_out["valid"]
        and not timed_out["success"]
        and timed_out["episode"]["termination_reason"] == "planner_timeout"
    )
    assert (
        timed_out["error_runs"] is None
    )  # the transcript of a timed-out episode keeps no turn
    doc = FT.score(
        [
            fe.record(j)
            for j in (
                "s2_rules__CloseBlenderLid_s1_r1",
                "s2_rules__CloseFridge_s1_r1",
                "s2_rules__CloseToasterOvenDoor_s1_r1",
            )
        ],
        ["s2_rules"],
        [],
        permutations=10,
        bootstrap=0,
        seed=1,
    )
    assert (
        doc["arms"]["s2_rules"]["scored"],
        doc["arms"]["s2_rules"]["successes"],
    ) == (3, 1)


def test_qwen_alone_arm_leaves_no_escalation_log(fe):
    fe.ecc.set_scenario({"default": {"kind": "solve"}})
    config = fe.config(arms=("qwen_only", "base_hybrid"), tasks=THREE[:2], seeds=(1,))
    proc = fe.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    qwen = fe.record("qwen_only__CloseFridge_s1_r1")
    assert (
        qwen["evidence_ok"]
        and not (Path(qwen["final_try_dir"]) / "escalation.jsonl").exists()
    )
    assert qwen["episode"]["gpt_calls"] == 0 and qwen["episode"]["qwen"]["calls"] == 12
    base = fe.record("base_hybrid__CloseFridge_s1_r1")
    assert (
        base["evidence_ok"]
        and base["knobs"]["memory"] == "cloud"
        and base["episode"]["gpt_calls"] >= 2
    )
    fe.ecc.set_scenario({"qwen_only": {"kind": "solve", "keep_senior": True}})
    proc = fe.go(
        fe.config(arms=("qwen_only",), tasks=THREE[:1], seeds=(1,)), run_id="ft2"
    )
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    evidence = json.loads(
        (
            fe.run_root("ft2")
            / "cells"
            / "qwen_only__CloseBlenderLid_s1_r1"
            / "try1.evidence.json"
        ).read_text()
    )
    assert not evidence["ok"] and "Qwen alone" in evidence["problems"][0]


def test_resume_refuses_changed_inputs_or_settings(fe):
    fe.ecc.set_scenario({"default": {"kind": "fail"}})
    config = fe.config(arms=("s3_rules_corr",), tasks=THREE[:1], seeds=(1,))
    assert "runner finished: done" in fe.go(config).stdout
    raw = json.loads(config.read_text())
    raw["worker"]["model"] = "openai-chat:other-model"
    changed = fe.root / "changed.json"
    changed.write_text(json.dumps(raw))
    proc = fe.cli("run", "--config", str(changed), "--run-id", "ft1", "--go")
    assert (
        proc.returncode != 0 and "worker/senior/episode settings differ" in proc.stderr
    )
    proc = fe.cli(
        "run", "--config", str(config), "--run-id", "ft1", "--go", "--seeds", "1,2"
    )
    assert proc.returncode != 0 and "final_test seeds differ" in proc.stderr
    # the read-only commands read the run's own settings from its manifest
    status = fe.cli("status", "--config", str(changed), "--run-id", "ft1")
    assert status.returncode == 0 and "1/1 episodes final" in status.stdout
    snapshot = fe.run_root() / "inputs" / "corrections" / "CloseBlenderLid.md"
    snapshot.parent.chmod(0o755)
    snapshot.chmod(0o644)
    snapshot.write_text("- tampered\n")
    proc = fe.cli("run", "--config", str(config), "--run-id", "ft1", "--go")
    assert proc.returncode != 0 and "inputs/ changed" in proc.stderr
    stop = fe.cli("stop", "--config", str(config), "--run-id", "ft1")
    assert stop.returncode == 0 and "stopped this run's RLDX servers" in stop.stdout
    assert json.loads((fe.run_root() / "STOP").read_text())["reason"] == "operator"


def test_missing_corrections_are_refused_unless_allowed(fe):
    bare = (
        fe.root / "bare_final"
    )  # a bare folder (products.s3_corrections) instead of a frozen product
    shutil.copytree(fe.final, bare)
    S1._make_writable(bare)
    (bare / "CloseFridge.md").unlink()
    products = {"s1": str(fe.s1), "s2": str(fe.s2), "s3_corrections": str(bare)}
    config = fe.config(
        arms=("s3_rules_corr",), tasks=THREE[:2], seeds=(1,), products=products
    )
    proc = fe.cli("plan", "--config", str(config))
    assert (
        proc.returncode != 0
        and "stage-3 corrections missing for 1 task(s)" in proc.stderr
    )
    fe.ecc.set_scenario({"default": {"kind": "solve"}})
    config = fe.config(
        arms=("s3_rules_corr",),
        tasks=THREE[:2],
        seeds=(1,),
        products=products,
        allow_missing_corrections=True,
    )
    proc = fe.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    record = fe.record("s3_rules_corr__CloseFridge_s1_r1")
    assert record["evidence_ok"] and record["knobs"]["corrections"]["chars"] == 0
    inputs = json.loads((fe.run_root() / "inputs.json").read_text())
    assert inputs["corrections"]["CloseFridge"]["source_missing"] is True


def test_stage_products_must_chain(fe):
    """S2 rules searched on another S1 corpus, or S3 corrections built under other rules, are refused."""
    other = fe.s2_product(fe.root / "products" / "s2_other" / "product", "0" * 64)
    products = {"s1": str(fe.s1), "s2": str(other), "s3": str(fe.s3)}
    config = fe.config(
        arms=("s3_rules_corr",), tasks=THREE[:1], seeds=(1,), products=products
    )
    proc = fe.cli("plan", "--config", str(config))
    assert proc.returncode != 0 and "do not chain" in proc.stderr
    assert "searched on S1 corpus 000000000000" in proc.stderr
    config = fe.config(
        arms=("s2_rules",),
        tasks=THREE[:1],
        seeds=(1,),
        products=products,
        allow_product_mismatch=True,
    )
    proc = fe.cli("plan", "--config", str(config))
    assert (
        proc.returncode == 0
        and "WARNING  the S2 rules were searched on S1 corpus" in proc.stdout
    )
    broken = fe.root / "products" / "s3_broken"
    shutil.copytree(fe.s3, broken)
    S1._make_writable(broken)
    (broken / "final" / "CloseBlenderLid.md").write_text("- edited after the freeze\n")
    products = {"s1": str(fe.s1), "s2": str(fe.s2), "s3": str(broken)}
    proc = fe.cli(
        "plan",
        "--config",
        str(
            fe.config(
                arms=("s3_rules_corr",), tasks=THREE[:1], seeds=(1,), products=products
            )
        ),
    )
    assert proc.returncode != 0 and "does not verify" in proc.stderr
