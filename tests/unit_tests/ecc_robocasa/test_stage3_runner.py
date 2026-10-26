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

"""RC-ECC stage-3 runner with fake episodes and a fake reflection call.

The fake episodes route every request through RPent's real ``EscalationModel``,
so the escalation logs the runner checks are the wrapper's own.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import reflect as R
from scripts.ecc_robocasa import stage1 as S1
from scripts.ecc_robocasa import stage3 as S3
from tests.unit_tests.ecc_robocasa.conftest import FAKES, REPO

pytestmark = pytest.mark.timeout(300)
STAGE3 = REPO / "scripts" / "ecc_robocasa" / "stage3.py"
RULES = {  # (gpt_plan, K, N): rungs 6, 1 and 2 of the ladder
    "CloseBlenderLid": (True, 6, 2),
    "CloseFridge": (False, 12, 4),
    "CloseToasterOvenDoor": (True, 12, 4),
}


def make_s1_product(ecc) -> Path:
    """A frozen stage-1 product that ``stage1.verify`` accepts (the synthetic cloud corpus)."""
    product = ecc.root / "s1" / "product"
    shutil.copytree(ecc.cloud, product / "corpus")
    files = C.tree_files(product / "corpus")
    (product / "tasks.json").write_text("{}\n")
    doc = {
        "schema": S1.PRODUCT_SCHEMA,
        "run_id": "s1_fake",
        "corpus": "corpus",
        "corpus_tree_sha256": C.tree_hash(files),
        "files": files,
        "tasks_json_sha256": C.sha256_file(product / "tasks.json"),
    }
    (product / "PRODUCT.json").write_text(json.dumps(doc))
    (product / "FROZEN").write_text(doc["corpus_tree_sha256"] + "\n")
    S1._make_read_only(product / "corpus")
    assert S1.verify(product)["ok"]
    return product


def make_rules(ecc, rules=RULES) -> Path:
    folder = ecc.root / "s2_rules"
    folder.mkdir()
    for task, (plan, k, n) in rules.items():
        (folder / f"{task}.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "rule": {"gpt_plan": plan, "checkin_K": k, "errors_N": n},
                }
            )
        )
    return folder


class S3Env:
    def __init__(
        self,
        ecc,
        tasks,
        *,
        scenario,
        reflect=None,
        stage3=None,
        endpoints=((4, 3),),
        max_slots=3,
    ):
        self.ecc = ecc
        ecc.set_scenario(scenario)
        self.reflect = ecc.root / "reflect.json"
        self.reflect.write_text(json.dumps(reflect or {}))
        config = ecc.config(tasks=[], endpoints=endpoints, max_slots=max_slots)
        cfg = json.loads(config.read_text())
        cfg["episode"]["env"]["ECC_RC_FAKE_REFLECT"] = str(self.reflect)
        cfg["stage3"] = {
            "tasks": list(tasks),
            "s1_product": str(make_s1_product(ecc)),
            "s2_rules": str(make_rules(ecc, {t: RULES[t] for t in tasks})),
            "sessions": 2,
            "attempts_per_session": 3,
            "launcher": [sys.executable, str(FAKES / "fake_s3_cell.py")],
            "reflection": {
                "cmd": [sys.executable, str(FAKES / "fake_reflect.py")],
                "retry_delay_s": 0,
                "max_calls": 2,
            },
            **(stage3 or {}),
        }
        cfg["senior"] = {"model": "openai:gpt-6-astra", "reasoning_effort": "low"}
        config.write_text(json.dumps(cfg, indent=2))
        self.config = config

    def run(self, *args: str, timeout: float = 240) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(STAGE3), *args, "--config", str(self.config)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def go(self, *extra: str) -> subprocess.CompletedProcess:
        proc = self.run("run", "--run-id", "r1", "--go", *extra)
        assert proc.returncode in (0, 1), proc.stdout + proc.stderr
        return proc

    @property
    def root(self) -> Path:
        return self.ecc.runs / "r1"

    def task(self, task: str) -> Path:
        return self.root / "tasks" / task

    def reflection(self, task: str, episode: str) -> dict:
        return json.loads((self.task(task) / episode / "reflection.json").read_text())

    def events(self, name: str) -> list[dict]:
        return [e for e in self.ecc.events() if e["event"] == name]


def test_dry_run_prints_the_plan_and_creates_nothing(ecc):
    env = S3Env(ecc, ["CloseBlenderLid"], scenario={})
    proc = env.run("plan", "--run-id", "r1")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "dry run" in out and "stage-1 product" in out and "verified" in out
    assert "2 sessions x 3 attempts = 6 single-shot episodes per task" in out
    assert "c6_gptplan_K6_N2" in out and "seed 12" in out
    assert (
        "ECC_RC_PLAN=1" in out
        and "ECC_RC_CHECKIN=6" in out
        and "ECC_RC_ERRORS=2" in out
    )
    assert "corrections_used.md" in out
    assert not ecc.runs.exists() and ecc.events() == []


def test_stage3_runs_episodes_until_the_first_success_and_freezes_verified_entries(ecc):
    scenario = {
        "CloseBlenderLid": {"s1_a1": {"kind": "solve"}},
        # Plan off, K=12: the first senior turn is the check-in at request 13.
        "CloseFridge": {
            "s1_a1": {"kind": "fail", "requests": 15},
            "s1_a2": [{"kind": "infra"}, {"kind": "fail", "requests": 15}],
            "s1_a3": {"kind": "timeout", "requests": 15},
            "s2_a1": {"kind": "solve", "requests": 15},
        },
        "CloseToasterOvenDoor": {"default": {"kind": "fail", "requests": 7}},
    }
    reflect = {"CloseToasterOvenDoor/s1_a2": {"status": "refused"}}
    env = S3Env(ecc, list(RULES), scenario=scenario, reflect=reflect)
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert '"ok": true' in proc.stdout  # frozen

    starts = env.events("start")
    by_task = {}
    for e in starts:
        by_task.setdefault(e["task"], []).append((e["episode"], e["try"]))
    assert by_task["CloseBlenderLid"] == [("s1_a1", 1)]
    assert by_task["CloseFridge"] == [
        ("s1_a1", 1),
        ("s1_a2", 1),
        ("s1_a2", 2),
        ("s1_a3", 1),
        ("s2_a1", 1),
    ]
    assert [ep for ep, _ in by_task["CloseToasterOvenDoor"]] == [
        R.episode_name(n) for n in range(6)
    ]
    assert all(
        e["seed"] == 12 and e["memory"].endswith("s1/product/corpus") for e in starts
    )
    assert {(e["task"], e["plan"], e["checkin"], e["errors"]) for e in starts} == {
        (t, "1" if p else "0", str(k), str(n)) for t, (p, k, n) in RULES.items()
    }

    # The memory chain: each episode ran with the previous reflection's entries, shown to the senior only.
    ends = {(e["task"], e["episode"], e["try"]): e for e in env.events("end")}
    fridge = env.task("CloseFridge")
    previous: list = []
    for name in ("s1_a1", "s1_a2", "s1_a3", "s2_a1"):
        snapshot = fridge / name / "corrections_used.md"
        assert snapshot.read_text() == R.render_file(previous)
        assert not os.access(snapshot, os.W_OK)
        record = env.reflection("CloseFridge", name)
        assert record["entries_before"] == previous
        end = ends[("CloseFridge", name, 2 if name == "s1_a2" else 1)]
        text = snapshot.read_text().strip()
        assert end["corrections"] == str(snapshot) and not end["worker_saw"]
        assert end["senior_saw"] and set(end["senior_saw"]) == {
            "match" if text else "none"
        }
        previous = record["entries_after"]
    # The invalid launch was relaunched, never reflected on or counted.
    assert not (fridge / "s1_a2" / "try1.evidence.json").exists()
    assert (fridge / "s1_a2" / "try2.evidence.json").exists()
    assert env.reflection("CloseFridge", "s1_a2")["try_dir"].endswith("s1_a2/try2")
    # The timed-out episode was reviewed from run.log.
    assert env.reflection("CloseFridge", "s1_a3")["digest_source"] == "run.log"
    calls = {e["episode"]: e for e in env.events("reflect")}
    assert (
        calls["CloseFridge/s1_a3"]["run_log_note"]
        and not calls["CloseFridge/s1_a1"]["run_log_note"]
    )
    assert all(
        c["sha_ok"] and c["outcome_first"] and c["language"] for c in calls.values()
    )
    assert all(
        c["cred_names"] == ["FAKE_SENIOR_URL", "FAKE_SENIOR_KEY"]
        for c in calls.values()
    )
    assert all(
        (c["model"], c["effort"]) == ("openai:gpt-6-astra", "low")
        for c in calls.values()
    )

    # run_ecc_stage3's bookkeeping: s1_a1's entry was applied in 2 failures (s1_a2, s1_a3) and dropped; the rest was
    # verified by the success in s2_a1.
    last = env.reflection("CloseFridge", "s2_a1")
    assert [e["id"] for e in last["entries_after"]] == ["s1_a2.1", "s1_a3.1", "s2_a1.1"]
    assert all(e["status"] == "verified" for e in last["entries_after"])
    # A refusal is a skipped reflection: the memory is unchanged.
    refused = env.reflection("CloseToasterOvenDoor", "s1_a2")
    assert (
        refused["status"] == "skipped"
        and refused["skipped"] == "content filter refusal"
    )
    assert refused["entries_after"] == refused["entries_before"]

    product = env.root / "product"
    assert S3.verify(product)["ok"]
    final = product / "final"
    assert sorted(p.name for p in final.iterdir()) == sorted(
        f"{t}{s}" for t in RULES for s in (".md", ".json")
    )
    assert (final / "CloseFridge.md").read_text() == R.render_file(
        last["entries_after"]
    )
    blender = env.reflection("CloseBlenderLid", "s1_a1")
    assert (
        (final / "CloseBlenderLid.md").read_text()
        == R.render_file(blender["entries_after"])
        != ""
    )
    assert (
        final / "CloseToasterOvenDoor.md"
    ).read_text() == ""  # unsolved: nothing verified
    doc = json.loads((product / "PRODUCT.json").read_text())
    assert doc["schema"] == S3.PRODUCT_SCHEMA and doc["seed"] == 12
    assert (
        doc["tasks"]["CloseFridge"]["status"] == "solved"
        and doc["tasks"]["CloseFridge"]["episodes"] == 4
    )
    assert doc["tasks"]["CloseToasterOvenDoor"]["status"] == "unsolved"
    assert doc["summary"]["reflections_skipped"] == 1 and doc["summary"]["solved"] == 2
    history = [
        json.loads(x)
        for x in (fridge / "corrections" / "history.jsonl").read_text().splitlines()
    ]
    assert [h["episode"] for h in history] == [
        f"CloseFridge/{n}" for n in ("s1_a1", "s1_a2", "s1_a3", "s2_a1")
    ]
    assert all(h["prompt_sha256"] and h["reply"] for h in history)
    record = json.loads((fridge / "record.json").read_text())
    assert record["infra_retries"] == 1 and record["first_success"] == "s2_a1"
    assert record["usd"]["reflections"] > 0
    jobs = json.loads(
        (env.root / "state" / "jobs" / "CloseFridge__s2_a2.json").read_text()
    )
    assert jobs["final"]["class"] == "not_needed" and jobs["tries"] == []

    status = env.run("status", "--run-id", "r1", "--tasks-table")
    assert status.returncode == 0, status.stderr
    assert "[####################] 3/3 tasks final" in status.stdout
    assert "reflections: 11 done (10 applied, 0 unparsed, 1 skipped)" in status.stdout
    assert "reflections $" in status.stdout and "CloseFridge" in status.stdout


@pytest.mark.parametrize("sleep_s", [0.05, 30])
def test_an_escalation_mismatch_is_an_infrastructure_error_that_stops_the_run(
    ecc, sleep_s
):
    # The launcher ran the wrapper with K=3 instead of the rule's 12.
    scenario = {
        "CloseFridge": {
            "s1_a1": {"kind": "fail", "bug": {"checkin": 3}, "sleep_s": sleep_s}
        }
    }
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario)
    started = time.time()
    proc = env.go()
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    stop = json.loads((env.root / "STOP").read_text())
    assert stop["reason"] == "escalation evidence mismatch"
    state = json.loads(
        (env.root / "state" / "jobs" / "CloseFridge__s1_a1.json").read_text()
    )
    assert (
        state["tries"][0]["class"] == "infra"
        and "evidence_mismatch" in state["tries"][0]["cause"]
    )
    rows = [r for r in env.ecc.progress() if r["event"] == "evidence_mismatch"]
    assert rows and any("checkin=3" in p for p in rows[0]["problems"])
    if sleep_s > 1:  # still running when its config row appeared: stopped at once
        assert rows[0]["when"] == "config row of a running episode"
        assert time.time() - started < 25
    assert len(env.events("start")) == 1 and env.events("reflect") == []


def test_episode_folders_are_unlistable_and_senior_memory_reads_are_flagged(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": {"kind": "fail", "requests": 15},
            "s1_a2": {"kind": "solve", "requests": 15, "peek": True},
        }
    }
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario)
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    assert [e["episode_dir_listable"] for e in env.events("start")] == [False, False]
    assert os.listdir(
        env.task("CloseFridge") / "s1_a1"
    )  # listable again once the runner exited
    record = json.loads((env.task("CloseFridge") / "record.json").read_text())
    access = [h["file_access"] for h in record["history"]]
    assert access[0]["senior_memory_reads"] == []
    assert [Path(p).name for p in access[1]["senior_memory_reads"]] == [
        "corrections_used.md"
    ]
    status = env.run("status", "--run-id", "r1")
    assert (
        "file-tool access outside the episode folder: CloseFridge (cross_reads, senior_memory_reads)"
        in status.stdout
    )


def test_skipped_reflections_and_a_failing_call_stop_the_run(ecc):
    scenario = {
        "CloseBlenderLid": {"s1_a1": {"kind": "no_turns"}, "s1_a2": {"kind": "solve"}},
        "CloseFridge": {"default": {"kind": "fail", "requests": 15, "sleep_s": 8}},
    }
    reflect = {"CloseFridge/s1_a1": {"status": "error"}}
    env = S3Env(
        ecc, ["CloseBlenderLid", "CloseFridge"], scenario=scenario, reflect=reflect
    )
    proc = env.go()
    assert "runner finished: stopped" in proc.stdout, proc.stdout + proc.stderr
    first = env.reflection("CloseBlenderLid", "s1_a1")
    assert first["status"] == "skipped" and first["skipped"] == "no senior intervention"
    assert (
        json.loads((env.task("CloseBlenderLid") / "record.json").read_text())["status"]
        == "solved"
    )
    calls = [e for e in env.events("reflect") if e["episode"] == "CloseFridge/s1_a1"]
    assert [c["call"] for c in calls] == [1, 2]  # max_calls
    stop = json.loads((env.root / "STOP").read_text())
    assert stop["reason"] == "reflection failed" and stop["task"] == "CloseFridge"
    assert not (
        env.task("CloseFridge") / "s1_a2"
    ).exists()  # the next episode waits for the memory
    assert not (env.root / "product").exists()

    # The provider is back: the operator removes STOP and starts the runner again.
    env.reflect.write_text("{}")
    ecc.set_scenario(
        {**scenario, "CloseFridge": {"s1_a2": {"kind": "solve", "requests": 15}}}
    )
    (env.root / "STOP").unlink()
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    calls = [
        e["call"] for e in env.events("reflect") if e["episode"] == "CloseFridge/s1_a1"
    ]
    assert calls == [1, 2, 3]  # a fresh call budget for the failed reflection
    assert env.reflection("CloseFridge", "s1_a1")["status"] == "applied"
    assert (
        json.loads((env.task("CloseFridge") / "record.json").read_text())["status"]
        == "solved"
    )
    assert S3.verify(env.root / "product")["ok"]


def test_a_reflection_survives_a_killed_runner_and_is_applied_once(ecc):
    scenario = {
        "CloseFridge": {
            "s1_a1": {"kind": "fail", "requests": 15},
            "s1_a2": {"kind": "solve", "requests": 15},
        }
    }
    reflect = {"CloseFridge/s1_a1": {"sleep_s": 4}}
    env = S3Env(ecc, ["CloseFridge"], scenario=scenario, reflect=reflect)
    runner = subprocess.Popen(
        [
            sys.executable,
            str(STAGE3),
            "run",
            "--run-id",
            "r1",
            "--go",
            "--config",
            str(env.config),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    progress = env.root / "progress.jsonl"
    ecc.wait_for(
        lambda: progress.exists() and '"reflection_start"' in progress.read_text(),
        timeout=120,
    )
    os.killpg(runner.pid, signal.SIGKILL)
    runner.wait()
    (env.root / "tasks").chmod(0o755)
    proc = env.go()
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    calls = [e for e in env.events("reflect") if e["episode"] == "CloseFridge/s1_a1"]
    assert len(calls) == 1  # the earlier runner's call was adopted, not repeated
    history = (
        (env.task("CloseFridge") / "corrections" / "history.jsonl")
        .read_text()
        .splitlines()
    )
    assert [json.loads(x)["episode"] for x in history] == [
        "CloseFridge/s1_a1",
        "CloseFridge/s1_a2",
    ]
    assert S3.verify(env.root / "product")["ok"]


def test_rules_come_from_a_verified_stage2_product_on_the_same_stage1_corpus(
    ecc, tmp_path
):
    env = S3Env(ecc, ["CloseFridge"], scenario={})
    cfg = json.loads(env.config.read_text())
    s1 = S1.verify(Path(cfg["stage3"]["s1_product"]))
    product = tmp_path / "s2" / "product"
    (product / "rules").mkdir(parents=True)
    (product / "rules" / "CloseFridge.json").write_text(
        json.dumps(
            {
                "schema": "rc-ecc-s2-rule/1",
                "task": "CloseFridge",
                "rung": 3,
                "config": "c3_qwenplan_K6_N4",
                "rule": {"gpt_plan": False, "checkin_K": 6, "errors_N": 4},
            }
        )
    )
    (product / "episodes.json").write_text("{}\n")

    def freeze_s2(corpus_sha: str) -> None:
        files = C.tree_files(product / "rules")
        doc = {
            "schema": "rc-ecc-s2-product/1",
            "rules": "rules",
            "rules_tree_sha256": C.tree_hash(files),
            "files": files,
            "episodes_json_sha256": C.sha256_file(product / "episodes.json"),
            "tasks": ["CloseFridge"],
            "seed": 11,
            "s1_product": {"corpus_tree_sha256": corpus_sha},
        }
        for name in ("RULES.json", "FROZEN"):
            if (product / name).exists():
                (product / name).chmod(0o644)
        (product / "RULES.json").write_text(json.dumps(doc))
        (product / "FROZEN").write_text(doc["rules_tree_sha256"] + "\n")
        (product / "rules" / "CloseFridge.json").chmod(0o444)

    freeze_s2(s1["corpus_tree_sha256"])
    cfg["stage3"].update(s2_product=str(product), s2_rules="")
    env.config.write_text(json.dumps(cfg))
    loaded = S3.load_config(env.config)
    args = type("A", (), {"tasks": None, "limit": 0})()
    stage, manifest, inputs = S3.build(loaded, args, C.RunLayout(tmp_path / "run"))
    assert stage.rules["CloseFridge"].as_json() == {
        "gpt_plan": False,
        "checkin_K": 6,
        "errors_N": 4,
    }
    assert inputs["rules"]["kind"] == "s2_product" and manifest["tasks"][0]["rung"] == 3
    freeze_s2("0" * 64)
    with pytest.raises(SystemExit, match="another stage-1 corpus"):
        S3.build(loaded, args, C.RunLayout(tmp_path / "run"))
    (product / "rules" / "CloseFridge.json").chmod(0o644)
    with pytest.raises(SystemExit, match="does not verify"):
        S3.build(S3.load_config(env.config), args, C.RunLayout(tmp_path / "run"))


def test_config_refuses_reserved_seeds(ecc):
    env = S3Env(ecc, ["CloseFridge"], scenario={})
    cfg = json.loads(env.config.read_text())
    for seed in (0, 3, 11):
        cfg["stage3"]["seed"] = seed
        env.config.write_text(json.dumps(cfg))
        with pytest.raises(ValueError, match="stage3.seed"):
            S3.load_config(env.config)
