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

"""hybrid_cell.sh's GPT-6-alone mode (``ECC_RC_SENIOR_ALONE=1``): the command and environment.

``rpent`` is replaced by a dumper, so nothing runs. The launch environment is the
final-test runner's own (``final_test.cell_env``). The GPT-6-alone launch must
differ from the hybrid and Qwen-alone launches of the same cell only in the
documented variables, and those launches must not change at all; with
``ECC_RC_OLD_HYBRID_CELL`` (the launcher before this mode) they are compared
byte for byte with the old launcher's.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import final_test as FT
from scripts.ecc_robocasa import hybrid_episode as H
from tests.unit_tests.ecc_robocasa.conftest import REPO

HYBRID = REPO / "scripts" / "ecc_robocasa" / "hybrid_cell.sh"
#: What a GPT-6-alone launch changes against the hybrid launch of the same cell.
GPT6_VS_HYBRID = {
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "RPENT_ESCALATION",
    "RPENT_MAX_HISTORY_IMAGES",
    "RPENT_SEND_REASONING_IDS",
}
GPT6_VS_QWEN = GPT6_VS_HYBRID - {"RPENT_ESCALATION"}


def _dumper(path: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "json.dump({'argv': sys.argv, 'env': dict(os.environ), 'cwd': os.getcwd()}, open(os.environ['FAKE_DUMP'], 'w'))\n"
    )
    path.chmod(0o755)
    return path


def _knobs(planner: str, **extra) -> dict:
    """Knobs as final_test.resolve_knobs makes them (the hybrid rule = the unused routing values
    the runner passes for the alone arms: plan on, K = senior.checkin (5), N = 2)."""
    knobs = {
        "planner": planner,
        "escalation": planner == "hybrid",
        "timeout_s": 1800,
        "corrections": None,
        "plan": True if planner == "hybrid" else None,
        "checkin": 5 if planner == "hybrid" else None,
        "errors": 2 if planner == "hybrid" else None,
    }
    if planner == "gpt6":
        knobs.update(
            model="openai:gpt-6-astra",
            reasoning_effort="low",
            max_images=40,
            reasoning_ids=False,
        )
    knobs.update(extra)
    return knobs


@pytest.fixture
def launch(ecc, tmp_path):
    venv = tmp_path / "fakevenv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    _dumper(venv / "bin" / "rpent")
    llvmpipe = tmp_path / "llvmpipe.sh"
    llvmpipe.write_text("echo 8\n")
    config = ecc.config(tasks=["CloseFridge"], endpoints=((4, 2),))
    raw = json.loads(config.read_text())
    raw["paths"].update(venv=str(venv), llvmpipe_index_script=str(llvmpipe))
    raw["episode"]["env"] = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    config.write_text(json.dumps(raw))
    cfg = C.load_config(config)
    corpus = tmp_path / "corpus"
    (corpus / "global").mkdir(parents=True)
    (corpus / "task-specific").mkdir()
    (corpus / "global" / "GLOBAL_MEMORY.md").write_text("# global\n")
    for path in (corpus / "global", corpus / "task-specific", corpus):
        path.chmod(0o555)
    job = C.Job(
        job_id="a__CloseFridge_s1_r1",
        task="CloseFridge",
        split="atomic",
        seed=1,
        order=0,
    )

    def run(knobs, *, overrides=None, script=HYBRID, corrections=None, stage_rule=None):
        endpoint = C.endpoints_from_config(cfg)[0]
        parent = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
        if stage_rule is not None:
            # A stage-2 rung or stage-3 episode: hybrid_env alone (no ECC_RC_ESCALATION), as
            # stage2.py and stage3.py launch them; stage 3 adds its corrections file and sha256.
            env = H.hybrid_env(
                cfg,
                split="atomic",
                try_n=1,
                endpoint=endpoint,
                rule=stage_rule,
                timeout_s=1800,
                corrections=corrections,
                corrections_sha256=C.sha256_file(corrections) if corrections else "",
                parent=parent,
            )
        else:
            if corrections:
                knobs = dict(
                    knobs, corrections={"file": str(corrections), "sha256": ""}
                )
            env = FT.cell_env(cfg, job, 1, endpoint, knobs, parent=parent)
        env.update(FAKE_DUMP=str(tmp_path / "dump.json"), **(overrides or {}))
        (tmp_path / "dump.json").unlink(missing_ok=True)
        out = tmp_path / "out"
        proc = subprocess.run(
            ["bash", str(script), "atomic", "CloseFridge", "1", str(out), str(corpus)],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        dump = (
            json.loads((tmp_path / "dump.json").read_text())
            if proc.returncode == 0
            else None
        )
        return proc, dump

    run.cfg, run.venv, run.corpus = cfg, venv, corpus
    yield run
    for path in (corpus, corpus / "global", corpus / "task-specific"):
        path.chmod(0o755)


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_gpt6_alone_command_and_environment(launch):
    proc, dump = launch(_knobs("gpt6"))
    assert proc.returncode == 0, proc.stderr
    assert (
        "senior=off planner=openai:gpt-6-astra effort=low images=40 reasoning_ids=0"
        in proc.stdout
    )
    argv, env = dump["argv"], dump["env"]
    assert (
        _flag(argv, "--planner"),
        _flag(argv, "--model"),
        _flag(argv, "--reasoning-effort"),
    ) == (
        "api",
        "openai:gpt-6-astra",
        "low",
    )
    assert (
        "--base-url" not in argv
    )  # the route comes from OPENAI_BASE_URL, as the worker's
    assert (env["OPENAI_API_KEY"], env["OPENAI_BASE_URL"]) == (
        "fake",
        "https://senior.invalid",
    )
    assert env["RPENT_ESCALATION"] == "0"
    assert (
        env["RPENT_MAX_HISTORY_IMAGES"] == "40"
        and env["RPENT_SEND_REASONING_IDS"] == "0"
    )
    # the Target50 episode environment otherwise: CPU rendering, RLDX values, retries, notes
    for key, value in {
        "RPENT_COERCE_TOOL_ARGS": "1",
        "RPENT_TOOL_TIMEOUT_S": "0",
        "RPENT_CONTINUE_NOTES": "3",
        "RLDX_MAX_CHUNKS": "40",
        "CUDA_VISIBLE_DEVICES": "",
        "MUJOCO_EGL_DEVICE_ID": "8",
        "RC_RENDER": "cpu",
        "PYTHONPATH": str(REPO),
    }.items():
        assert env.get(key) == value, key
    assert "RPENT_CLOUD_CORRECTIONS" not in env
    assert not [k for k in env if k.startswith("ECC_RC_")]


def test_gpt6_alone_differs_from_the_hybrid_and_qwen_launches_only_as_documented(
    launch,
):
    _, gpt6 = launch(_knobs("gpt6"))
    for planner, allowed in (("hybrid", GPT6_VS_HYBRID), ("qwen", GPT6_VS_QWEN)):
        proc, other = launch(_knobs(planner))
        assert proc.returncode == 0, proc.stderr
        ge, oe = gpt6["env"], other["env"]
        differing = {k for k in set(ge) | set(oe) if ge.get(k) != oe.get(k)}
        assert differing == allowed, (
            planner,
            {k: (ge.get(k), oe.get(k)) for k in differing},
        )
        ga, oa = list(gpt6["argv"]), list(other["argv"])
        for flag in ("--model", "--reasoning-effort"):
            ga[ga.index(flag) + 1] = oa[oa.index(flag) + 1]
        assert ga == oa


def test_reasoning_ids_on_leaves_the_switch_unset(launch):
    proc, dump = launch(_knobs("gpt6", reasoning_ids=True, max_images=12))
    assert proc.returncode == 0, proc.stderr
    assert "RPENT_SEND_REASONING_IDS" not in dump["env"]
    assert dump["env"]["RPENT_MAX_HISTORY_IMAGES"] == "12"
    assert "images=12 reasoning_ids=1" in proc.stdout


def test_hybrid_and_qwen_launches_carry_no_gpt6_switch(launch):
    for planner in ("hybrid", "qwen"):
        proc, dump = launch(_knobs(planner))
        assert proc.returncode == 0, proc.stderr
        for key in ("RPENT_MAX_HISTORY_IMAGES", "RPENT_SEND_REASONING_IDS"):
            assert key not in dump["env"]
        assert _flag(dump["argv"], "--model") == "openai-chat:qwen3.6-27b"
        assert "planner=" not in proc.stdout


def test_gpt6_alone_refusals(launch, tmp_path):
    corrections = tmp_path / "corrections_used.md"
    corrections.write_text("- entry\n")
    for overrides, fragment in (
        ({"ECC_RC_ESCALATION": "1"}, "needs ECC_RC_ESCALATION=0"),
        ({"ECC_RC_SENIOR_ALONE": "yes"}, "ECC_RC_SENIOR_ALONE must be 0 or 1"),
        ({"ECC_RC_MAX_IMAGES": ""}, "ECC_RC_MAX_IMAGES must be a positive integer"),
        ({"ECC_RC_MAX_IMAGES": "0"}, "ECC_RC_MAX_IMAGES must be a positive integer"),
        ({"ECC_RC_MAX_IMAGES": "4O"}, "ECC_RC_MAX_IMAGES must be a positive integer"),
        ({"ECC_RC_REASONING_IDS": "2"}, "ECC_RC_REASONING_IDS must be 0 or 1"),
    ):
        proc, _ = launch(_knobs("gpt6"), overrides=overrides)
        assert proc.returncode == 2 and fragment in proc.stderr, (
            overrides,
            proc.stderr,
        )
    proc, _ = launch(_knobs("gpt6"), corrections=corrections)
    assert proc.returncode == 2 and "need the senior" in proc.stderr


@pytest.mark.skipif(
    not os.environ.get("ECC_RC_OLD_HYBRID_CELL"),
    reason="needs the launcher before the GPT-6-alone mode (ECC_RC_OLD_HYBRID_CELL)",
)
def test_hybrid_and_qwen_launches_are_unchanged_from_the_old_launcher(launch):
    old = Path(os.environ["ECC_RC_OLD_HYBRID_CELL"])
    assert "ECC_RC_SENIOR_ALONE" not in old.read_text()
    for knobs in (
        _knobs("hybrid"),
        _knobs("hybrid", plan=False, checkin=6, errors=4),
        _knobs("qwen"),
    ):
        new_proc, new = launch(knobs)
        old_proc, before = launch(knobs, script=old)
        assert new_proc.returncode == old_proc.returncode == 0, (
            new_proc.stderr,
            old_proc.stderr,
        )
        assert new_proc.stdout == old_proc.stdout
        assert new["argv"] == before["argv"] and new["cwd"] == before["cwd"]
        assert new["env"] == before["env"]


@pytest.mark.skipif(
    not os.environ.get("ECC_RC_OLD_HYBRID_CELL"),
    reason="needs the launcher before the GPT-6-alone mode (ECC_RC_OLD_HYBRID_CELL)",
)
def test_stage_launches_are_unchanged_from_the_old_launcher(launch, tmp_path):
    """Stage-2 rungs and stage-3 episodes (with their corrections file) launch as before."""
    old = Path(os.environ["ECC_RC_OLD_HYBRID_CELL"])
    corrections = tmp_path / "corrections_used.md"
    corrections.write_text(
        "- [s1_a1.1] (candidate; trigger: checkin) situation: a | diagnosis: b | fix: c\n"
    )
    for rule, corr in (
        (H.Rule(plan=False, checkin=12, errors=4), None),
        (H.Rule(plan=True, checkin=6, errors=2), None),
        (H.Rule(plan=True, checkin=5, errors=2), corrections),
    ):
        new_proc, new = launch(None, stage_rule=rule, corrections=corr)
        old_proc, before = launch(None, stage_rule=rule, corrections=corr, script=old)
        assert new_proc.returncode == old_proc.returncode == 0, (
            new_proc.stderr,
            old_proc.stderr,
        )
        assert new_proc.stdout == old_proc.stdout
        assert new["argv"] == before["argv"] and new["cwd"] == before["cwd"]
        assert new["env"] == before["env"]
        assert (new["env"].get("RPENT_CLOUD_CORRECTIONS") == str(corrections)) == bool(
            corr
        )
        assert "RPENT_MAX_HISTORY_IMAGES" not in new["env"]
