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

"""The hybrid cell launcher: the command and environment it builds for one rung.

``rpent`` is replaced by a dumper, so nothing runs. The last test compares the
launcher with the Target50 launcher's base arm it reproduces; it needs that
launcher and a real config (``ECC_RC_PORTED_LAUNCHER``, ``ECC_RC_CONFIG``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import hybrid_episode as H
from tests.unit_tests.ecc_robocasa.conftest import REPO

HYBRID = REPO / "scripts" / "ecc_robocasa" / "hybrid_cell.sh"


def _dumper(path: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "json.dump({'argv': sys.argv, 'env': dict(os.environ), 'cwd': os.getcwd()}, open(os.environ['FAKE_DUMP'], 'w'))\n"
    )
    path.chmod(0o755)
    return path


def _venv(root: Path) -> Path:
    (root / "bin").mkdir(parents=True)
    python = root / "bin" / "python"
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(0o755)
    _dumper(root / "bin" / "rpent")
    return root


def _frozen_corpus(root: Path) -> Path:
    (root / "global").mkdir(parents=True)
    (root / "task-specific").mkdir()
    (root / "global" / "GLOBAL_MEMORY.md").write_text("# global\n")
    for path in (root / "global", root / "task-specific", root):
        path.chmod(0o555)
    return root


@pytest.fixture
def launch(ecc, tmp_path):
    """Run hybrid_cell.sh with the runner's own environment builder, as stage2.py launches it."""
    venv = _venv(tmp_path / "fakevenv")
    llvmpipe = tmp_path / "llvmpipe.sh"
    llvmpipe.write_text("echo 8\n")
    config = ecc.config(tasks=["CloseFridge"], endpoints=((4, 2),))
    raw = json.loads(config.read_text())
    raw["paths"].update(venv=str(venv), llvmpipe_index_script=str(llvmpipe))
    raw["episode"]["env"] = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    config.write_text(json.dumps(raw))
    cfg = C.load_config(config)
    corpus = _frozen_corpus(tmp_path / "corpus")

    def run(
        rule=H.Rule(False, 12, 4),
        split="atomic",
        task="CloseFridge",
        *,
        overrides=None,
        corrections=None,
        sha="",
    ):
        endpoint = C.endpoints_from_config(cfg)[0]
        env = H.hybrid_env(
            cfg,
            split=split,
            try_n=1,
            endpoint=endpoint,
            rule=rule,
            timeout_s=H.TARGET50_TIMEOUT_S[split],
            corrections=corrections,
            corrections_sha256=sha,
            parent={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        env.update(FAKE_DUMP=str(tmp_path / "dump.json"), **(overrides or {}))
        (tmp_path / "dump.json").unlink(missing_ok=True)
        out = tmp_path / "out"
        proc = subprocess.run(
            ["bash", str(HYBRID), split, task, "11", str(out), str(corpus)],
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
        return proc, dump, out

    run.cfg, run.venv, run.corpus = cfg, venv, corpus
    yield run
    for path in (corpus, corpus / "global", corpus / "task-specific"):
        path.chmod(0o755)


def test_shell_syntax():
    subprocess.run(["bash", "-n", str(HYBRID)], check=True)


def test_rung_command_and_environment(launch):
    proc, dump, out = launch()
    assert proc.returncode == 0, proc.stderr
    assert "plan=0 checkin=12 errors=4 corrections=none timeout=1800s" in proc.stdout
    assert dump["argv"][0] == str(launch.venv / "bin" / "rpent")
    assert dump["argv"][1:] == [
        "--robot", "robocasa", "--task-name", "CloseFridge", "--split", "target", "--seed", "11",
        "--vla-endpoint", C.endpoints_from_config(launch.cfg)[0].url, "--cuda-device", "4",
        "--planner", "api", "--model", "openai-chat:qwen3.6-27b", "--reasoning-effort", "none",
        "--max-turns", "100", "--planner-timeout-s", "1800",
        "--memory-profile", "local", "--memory-dir", str(launch.corpus), "--output-dir", str(out),
    ]  # fmt: skip
    env = dump["env"]
    expected = {
        "OPENAI_API_KEY": "fake-worker-key",
        "OPENAI_BASE_URL": "https://worker.invalid/v1",
        "RPENT_ESCALATION": "1",
        "RPENT_ESCALATION_MODEL": "openai:gpt-6-astra",
        "RPENT_ESCALATION_BASE_URL": "https://senior.invalid",
        "RPENT_ESCALATION_KEY_ENV": "FAKE_SENIOR_KEY",
        "FAKE_SENIOR_KEY": "fake",
        "RPENT_ESCALATION_REASONING_EFFORT": "low",
        "RPENT_ESCALATION_CHECKIN": "12",
        "RPENT_ESCALATION_MAX_SHARE": "0.2",
        "RPENT_ESCALATION_PLAN": "0",
        "RPENT_ESCALATION_ERRORS": "4",
        "RPENT_COERCE_TOOL_ARGS": "1",
        "RPENT_TOOL_TIMEOUT_S": "0",
        "RPENT_CONTINUE_NOTES": "3",
        "RPENT_ESCALATION_SENIOR_MAX_IMAGES": "40",
        "RLDX_MAX_CHUNKS": "40",
        "RLDX_SETTLE_PATIENCE": "999",
        "RLDX_ACTION_STEPS_PER_CHUNK": "8",
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "",
        "MUJOCO_EGL_DEVICE_ID": "8",
        "LP_NUM_THREADS": "8",
        "RC_RENDER": "cpu",
        "RC_GPU": "4",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(REPO),
        "RPENT_REPO_ROOT": str(REPO),
    }
    assert {k: env.get(k) for k in expected} == expected
    for absent in (
        "RPENT_CLOUD_CORRECTIONS",
        "RPENT_EXPLORE_LIST_MEMORY",
        "RPENT_ESCALATION_TAKEOVER",
        "RPENT_ESCALATION_PLAN_FILES",
        "RLDX_RESET_SEED",
    ):
        assert absent not in env
    assert not [k for k in env if k.startswith("ECC_RC_")]
    assert dump["cwd"] == str(REPO)


def test_plan_on_leaves_the_switch_unset_and_rules_reach_rpent(launch):
    proc, dump, _ = launch(
        H.Rule(True, 6, 2), split="composite_seen", task="PreSoakPan"
    )
    assert proc.returncode == 0, proc.stderr
    env = dump["env"]
    assert (
        "RPENT_ESCALATION_PLAN" not in env
    )  # on, as in the Target50 launcher without ABL_PLAN=0
    assert (env["RPENT_ESCALATION_CHECKIN"], env["RPENT_ESCALATION_ERRORS"]) == (
        "6",
        "2",
    )
    assert dump["argv"][dump["argv"].index("--planner-timeout-s") + 1] == "3600"


def test_rpent_cli_parses_the_launch_as_an_evaluation_with_read_only_memory(launch):
    from rpent.cli.main import _build_argparser
    from rpent.robots import get_robot_spec

    proc, dump, out = launch()
    assert proc.returncode == 0, proc.stderr
    parser = _build_argparser()
    spec = get_robot_spec("robocasa")
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(dump["argv"][1:])
    assert not args.explore and args.seed == 11 and args.memory_profile == "local"
    assert (args.planner_timeout_s, args.max_turns) == (1800, 100)
    config = spec.parse_config(args)
    assert config.recipe_tag == "CloseFridge_target_s11"
    assert Path(config.output_dir) == out
    # Evaluation builds RoboCasa's task-bounded memory manager, which is read-only.
    from robots.robocasa.memory import RoboCasaMemoryManager, TaskMemory

    selection = TaskMemory.load(
        launch.corpus, "CloseFridge", profile="local", split="target"
    )
    manager = RoboCasaMemoryManager(selection)
    assert manager._memory_access == "read_only"
    write = manager.get_common_tool_bindings()["write_text_file"][1]
    with pytest.raises(PermissionError, match="writing to memory is denied"):
        write(path=str(launch.corpus / "global" / "NOTE.md"), content="x")


def test_corrections_file_must_exist_and_match(launch, tmp_path):
    corrections = tmp_path / "corrections_used.md"
    corrections.write_text("- entry\n")
    sha = hashlib.sha256(corrections.read_bytes()).hexdigest()
    proc, dump, _ = launch(H.Rule(True, 6, 2), corrections=corrections, sha=sha)
    assert proc.returncode == 0, proc.stderr
    assert dump["env"]["RPENT_CLOUD_CORRECTIONS"] == str(corrections)
    proc, _, _ = launch(H.Rule(True, 6, 2), corrections=corrections, sha="0" * 64)
    assert proc.returncode == 2 and "sha256" in proc.stderr
    proc, _, _ = launch(H.Rule(True, 6, 2), corrections=tmp_path / "missing.md")
    assert proc.returncode == 2 and "not found" in proc.stderr


def test_qwen_alone_switch_turns_the_senior_off(launch, tmp_path):
    """ECC_RC_ESCALATION=0 (final-test Qwen-alone arms) is the launcher's ABL_ESCALATION=0."""
    proc, dump, _ = launch(overrides={"ECC_RC_ESCALATION": "0"})
    assert proc.returncode == 0, proc.stderr
    assert dump["env"]["RPENT_ESCALATION"] == "0"
    assert "RPENT_CLOUD_CORRECTIONS" not in dump["env"]
    assert "senior=off" in proc.stdout
    proc, dump, _ = launch(overrides={"ECC_RC_ESCALATION": "1"})
    assert proc.returncode == 0 and dump["env"]["RPENT_ESCALATION"] == "1"
    assert "senior=off" not in proc.stdout
    corrections = tmp_path / "corrections_used.md"
    corrections.write_text("- entry\n")
    proc, _, _ = launch(corrections=corrections, overrides={"ECC_RC_ESCALATION": "0"})
    assert proc.returncode == 2 and "need the senior" in proc.stderr


def test_refusals(launch, tmp_path):
    for overrides, fragment in (
        ({"RC_RENDER": "gpu"}, "RC_RENDER=cpu"),
        ({"RC_GPU": "1"}, "not one of"),
        ({"ECC_RC_PLAN": "yes"}, "ECC_RC_PLAN must be 0 or 1"),
        ({"ECC_RC_ESCALATION": "off"}, "ECC_RC_ESCALATION must be 0 or 1"),
        ({"ECC_RC_CHECKIN": "0"}, "ECC_RC_CHECKIN must be a positive integer"),
        ({"ECC_RC_ERRORS": ""}, "ECC_RC_ERRORS is not set"),
        (
            {"ECC_RC_CRED_FILE": str(tmp_path / "missing.env")},
            "credentials file not found",
        ),
    ):
        proc, _, _ = launch(overrides=overrides)
        assert proc.returncode == 2 and fragment in proc.stderr, (
            overrides,
            proc.stderr,
        )
    empty = tmp_path / "empty_code"
    empty.mkdir()
    proc, _, _ = launch(overrides={"ECC_RC_CODE_DIR": str(empty)})
    assert proc.returncode == 3 and "do not resolve" in proc.stderr
    launch.corpus.chmod(0o755)
    proc, _, _ = launch()
    assert proc.returncode == 2 and "is writable" in proc.stderr


@pytest.mark.skipif(
    not (os.environ.get("ECC_RC_PORTED_LAUNCHER") and os.environ.get("ECC_RC_CONFIG")),
    reason="needs the host's Target50 launcher and a real config",
)
@pytest.mark.parametrize(
    ("rule", "qwen_alone"),
    [
        (H.Rule(False, 12, 4), False),
        (H.Rule(True, 12, 4), False),
        (H.Rule(True, 6, 2), False),
        (H.Rule(True, 5, 2), True),
    ],
)
def test_effective_environment_matches_the_target50_base_arm(
    tmp_path, rule, qwen_alone
):
    """Same parent environment and fake credentials: only the documented differences remain.

    The Qwen-alone case (``ECC_RC_ESCALATION=0``) is compared with the launcher's
    ``ABL_ESCALATION=0`` arm.
    """
    ported = Path(os.environ["ECC_RC_PORTED_LAUNCHER"])
    cfg = C.load_config(os.environ["ECC_RC_CONFIG"])
    text = ported.read_text()
    sourced = re.search(r'\. "\$HOME/([^"]+)"', text).group(1)
    home = tmp_path / "home"
    cred = home / sourced
    cred.parent.mkdir(parents=True)
    names = cfg["credentials"]
    cred.write_text(
        f"{names['worker_key_var']}=fake-worker\n{names['senior_base_url_var']}=https://senior.invalid\n"
        f"{names['senior_key_var']}=fake-senior\n"
    )
    dumper = _dumper(tmp_path / "dump.py")
    patched = tmp_path / "ported.sh"
    assert "exec .venv/bin/rpent " in text
    patched.write_text(text.replace("exec .venv/bin/rpent ", f'exec "{dumper}" '))
    mine = tmp_path / "mine.sh"
    mine.write_text(
        HYBRID.read_text().replace('exec "$VENV/bin/rpent" ', f'exec "{dumper}" ')
    )
    endpoint = C.endpoints_from_config(cfg)[0]
    minimal = {"PATH": "/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8"}
    env_mine = H.hybrid_env(
        cfg,
        split="atomic",
        try_n=1,
        endpoint=endpoint,
        rule=rule,
        timeout_s=1800,
        parent=minimal,
    )
    env_mine.update(ECC_RC_CRED_FILE=str(cred), FAKE_DUMP=str(tmp_path / "mine.json"))
    if qwen_alone:
        env_mine["ECC_RC_ESCALATION"] = "0"
    env_ported = {k: v for k, v in env_mine.items() if not k.startswith("ECC_RC_")}
    env_ported.update(
        FAKE_DUMP=str(tmp_path / "ported.json"),
        **({} if rule.plan else {"ABL_PLAN": "0"}),
        **({"ABL_ESCALATION": "0"} if qwen_alone else {}),
    )
    a = subprocess.run(
        ["bash", str(patched), "atomic", "CloseFridge", "11", str(tmp_path / "ported_out"), str(rule.checkin)],
        env=env_ported, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    assert a.returncode == 0, a.stderr
    corpus = _frozen_corpus(tmp_path / "corpus")
    b = subprocess.run(
        ["bash", str(mine), "atomic", "CloseFridge", "11", str(tmp_path / "mine_out"), str(corpus)],
        env=env_mine, capture_output=True, text=True, timeout=120,
    )  # fmt: skip
    for path in (corpus, corpus / "global", corpus / "task-specific"):
        path.chmod(0o755)
    assert b.returncode == 0, b.stderr
    pd, md = (
        json.loads((tmp_path / f"{n}.json").read_text()) for n in ("ported", "mine")
    )
    pe, me = pd["env"], md["env"]
    differing = {k for k in set(pe) | set(me) if pe.get(k) != me.get(k)}
    assert differing <= {
        "PYTHONPATH",
        "RPENT_REPO_ROOT",
        "RPENT_ESCALATION_ERRORS",
        "ABL_PLAN",
        "ABL_ESCALATION",
        "PWD",
        "OLDPWD",
        "_",
        "FAKE_DUMP",
    }, {k: (pe.get(k), me.get(k)) for k in differing}
    assert (
        pe["RPENT_ESCALATION"] == me["RPENT_ESCALATION"] == ("0" if qwen_alone else "1")
    )
    assert (
        me["RPENT_ESCALATION_ERRORS"] == str(rule.errors)
        and "RPENT_ESCALATION_ERRORS" not in pe
    )
    assert me["PYTHONPATH"] == me["RPENT_REPO_ROOT"] == cfg["paths"]["code_dir"]
    pa, ma = pd["argv"], md["argv"]
    expected = list(pa)
    for flag, value in {
        "--memory-dir": str(corpus),
        "--output-dir": str(tmp_path / "mine_out"),
    }.items():
        expected[expected.index(flag) + 1] = value
    assert ma[1:] == expected[1:]
