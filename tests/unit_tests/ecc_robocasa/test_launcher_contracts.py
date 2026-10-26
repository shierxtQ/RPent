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

"""The exploration cell and RLDX server launchers: the environment and command they build.

``rpent`` and the server are replaced by a dumper, so nothing runs. The last test
compares the cell launcher with the Target50 launcher it reproduces; it needs that
launcher and a real config (``ECC_RC_PORTED_LAUNCHER``, ``ECC_RC_CONFIG``).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import stage1
from tests.unit_tests.ecc_robocasa.conftest import REPO

EXPLORE = REPO / "scripts" / "ecc_robocasa" / "explore_cell.sh"
SERVER = REPO / "scripts" / "ecc_robocasa" / "vla_server.sh"


def _dumper(path: Path) -> Path:
    path.write_text(
        f"#!{sys.executable}\nimport json, os, sys\n"
        "json.dump({'argv': sys.argv, 'env': dict(os.environ), 'cwd': os.getcwd()}, open(os.environ['FAKE_DUMP'], 'w'))\n"
    )
    path.chmod(0o755)
    return path


def _venv(root: Path, *, python_dumps: bool) -> Path:
    (root / "bin").mkdir(parents=True)
    python = root / "bin" / "python"
    if python_dumps:
        _dumper(python)
    else:
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)
    _dumper(root / "bin" / "rpent")
    return root


def _staging(root: Path) -> Path:
    (root / "global").mkdir(parents=True)
    (root / "global" / "GLOBAL_MEMORY.md").write_text("# global\n")
    return root


@pytest.fixture
def launch(ecc, tmp_path):
    """Build a cell's environment with the runner's own code, as stage1.py launches it."""
    venv = _venv(tmp_path / "fakevenv", python_dumps=False)
    llvmpipe = tmp_path / "llvmpipe.sh"
    llvmpipe.write_text("echo 8\n")
    config = ecc.config(tasks=["CloseFridge"], endpoints=((4, 2),))
    raw = json.loads(config.read_text())
    raw["paths"].update(venv=str(venv), llvmpipe_index_script=str(llvmpipe))
    raw["episode"]["env"] = {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    config.write_text(json.dumps(raw))
    cfg = C.load_config(config)

    def run(
        split="atomic", task="CloseFridge", *, cfg=cfg, overrides=None, script=EXPLORE
    ):
        job = C.Job(job_id=task, task=task, split=split, seed=0, order=0)
        endpoint = C.endpoints_from_config(cfg)[0]
        env = stage1.episode_env(
            cfg,
            job,
            1,
            endpoint,
            parent={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        )
        env.update(FAKE_DUMP=str(tmp_path / "dump.json"), **(overrides or {}))
        out, staging = (
            tmp_path / "out",
            _staging(tmp_path / "staging")
            if not (tmp_path / "staging").exists()
            else tmp_path / "staging",
        )
        proc = subprocess.run(
            ["bash", str(script), split, task, "0", str(out), str(staging)],
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
        return proc, dump, out, staging

    run.cfg = cfg
    run.venv = venv
    return run


def test_shell_syntax():
    for script in (EXPLORE, SERVER):
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_exploration_cell_command_and_environment(launch):
    proc, dump, out, staging = launch()
    assert proc.returncode == 0, proc.stderr
    assert dump["argv"][1:] == [
        "--robot",
        "robocasa",
        "--task-name",
        "CloseFridge",
        "--split",
        "target",
        "--seed",
        "0",
        "--vla-endpoint",
        C.endpoints_from_config(launch.cfg)[0].url,
        "--cuda-device",
        "4",
        "--planner",
        "api",
        "--model",
        "openai-chat:qwen3.6-27b",
        "--reasoning-effort",
        "none",
        "--max-turns",
        "100",
        "--planner-timeout-s",
        "2400",
        "--memory-profile",
        "local",
        "--memory-dir",
        str(staging),
        "--output-dir",
        str(out),
        "--explore",
        "--explore-sessions",
        "5",
        "--explore-attempts-per-session",
        "3",
        "--no-auto-merge-memory",
    ]
    assert dump["argv"][0] == str(launch.venv / "bin" / "rpent")
    env = dump["env"]
    expected = {
        "OPENAI_API_KEY": "fake-worker-key",
        "OPENAI_BASE_URL": "https://worker.invalid/v1",
        "RPENT_ESCALATION": "0",
        "RPENT_EXPLORE_LIST_MEMORY": "1",
        "RPENT_COERCE_TOOL_ARGS": "1",
        "RPENT_TOOL_TIMEOUT_S": "0",
        "RPENT_CONTINUE_NOTES": "3",
        "RPENT_ESCALATION_SENIOR_MAX_IMAGES": "40",
        "RPENT_ESCALATION_KEY_ENV": "FAKE_SENIOR_KEY",
        "RPENT_ESCALATION_BASE_URL": "https://senior.invalid",
        "RLDX_MAX_CHUNKS": "40",
        "RLDX_SETTLE_PATIENCE": "999",
        "RLDX_ACTION_STEPS_PER_CHUNK": "8",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "CUDA_VISIBLE_DEVICES": "",
        "MUJOCO_EGL_DEVICE_ID": "8",
        "LP_NUM_THREADS": "8",
        "RC_RENDER": "cpu",
        "RC_GPU": "4",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
        "HF_HOME": "/nonexistent/hf",
        "HF_HUB_CACHE": "/nonexistent/hf/hub",
        "ROBOCASA_ASSETS_PATH": "/nonexistent/assets",
        "PYTHONPATH": str(REPO),
        "RPENT_REPO_ROOT": str(REPO),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TORCH_CUDA_ARCH_LIST": "9.0",
    }
    assert {k: env.get(k) for k in expected} == expected
    assert "RLDX_RESET_SEED" not in env
    assert not [
        k for k in env if k.startswith("ECC_RC_")
    ]  # inputs are not passed on to rpent
    assert dump["cwd"] == str(REPO)


def test_rpent_cli_accepts_the_launch_command(launch):
    """RPent's own parser and RoboCasa config read the launcher's arguments as intended."""
    from rpent.cli.main import _build_argparser
    from rpent.robots import get_robot_spec

    proc, dump, out, staging = launch()
    assert proc.returncode == 0, proc.stderr
    parser = _build_argparser()
    spec = get_robot_spec("robocasa")
    spec.add_cli_args(parser, use_dashboard=False)
    args = parser.parse_args(dump["argv"][1:])
    assert args.robot_name == "robocasa" and args.explore and args.seed == 0
    assert (args.explore_sessions, args.explore_attempts_per_session) == (5, 3)
    assert args.auto_merge_memory is False and args.memory_profile == "local"
    assert (args.planner_timeout_s, args.max_turns) == (2400, 100)
    config = spec.parse_config(args)
    assert config.recipe_tag == "CloseFridge_target_s0"
    assert config.prompt_vars["memory_inbox"] == str(
        staging.resolve() / "_internal" / "inbox" / "CloseFridge_target_s0"
    )
    assert Path(config.output_dir) == out


def test_session_timeout_per_split_and_refusals(launch, tmp_path):
    cfg = {
        **launch.cfg,
        "stage1": {
            **launch.cfg["stage1"],
            "session_timeout_s_by_split": {"composite_seen": 3600},
        },
    }
    proc, dump, _, _ = launch("composite_seen", "PreSoakPan", cfg=cfg)
    assert proc.returncode == 0, proc.stderr
    assert dump["argv"][dump["argv"].index("--planner-timeout-s") + 1] == "3600"

    proc, _, _, _ = launch(overrides={"RC_RENDER": "gpu"})
    assert proc.returncode == 2 and "RC_RENDER=cpu" in proc.stderr
    proc, _, _, _ = launch(overrides={"RC_GPU": "1"})
    assert proc.returncode == 2 and "not one of" in proc.stderr
    empty = tmp_path / "empty_code"
    empty.mkdir()
    proc, _, _, _ = launch(overrides={"ECC_RC_CODE_DIR": str(empty)})
    assert proc.returncode == 3 and "do not resolve" in proc.stderr
    proc, _, _, _ = launch(
        overrides={"ECC_RC_CRED_FILE": str(tmp_path / "missing.env")}
    )
    assert proc.returncode == 2 and "credentials file not found" in proc.stderr


def test_rldx_server_command_and_environment(ecc, tmp_path):
    venv = _venv(tmp_path / "servervenv", python_dumps=True)
    config = ecc.config(tasks=["CloseFridge"], endpoints=((7, 2),))
    raw = json.loads(config.read_text())
    raw["paths"]["venv"] = str(venv)
    config.write_text(json.dumps(raw))
    cfg = C.load_config(config)
    env = dict(
        stage1.base_env(cfg, parent={"PATH": "/usr/bin:/bin"}),
        RC_GPU="7",
        FAKE_DUMP=str(tmp_path / "dump.json"),
    )
    log = tmp_path / "server.log"
    proc = subprocess.run(
        ["bash", str(SERVER), "43999", str(log)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    dump = json.loads((tmp_path / "dump.json").read_text())
    assert dump["argv"][1:] == [
        "robots/robocasa/vla_server.py",
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        "43999",
        "--cuda-device",
        "7",
        "--model-path",
        "/nonexistent/ckpt",
    ]
    assert dump["cwd"] == str(REPO)
    e = dump["env"]
    assert (
        e["RLDX_MAX_CHUNKS"],
        e["RLDX_SETTLE_PATIENCE"],
        e["RLDX_ACTION_STEPS_PER_CHUNK"],
    ) == ("40", "999", "8")
    assert e["HF_HUB_CACHE"] == "/nonexistent/hf/hub" and e["PYTHONPATH"] == str(REPO)
    assert not [k for k in e if k.startswith("ECC_RC_")]


@pytest.mark.skipif(
    not (os.environ.get("ECC_RC_PORTED_LAUNCHER") and os.environ.get("ECC_RC_CONFIG")),
    reason="needs the host's Target50 launcher and a real config",
)
def test_effective_environment_matches_the_target50_launcher(tmp_path):
    """Same parent environment and fake credentials: only the documented differences remain."""
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
    patched_ported = tmp_path / "ported.sh"
    assert "exec .venv/bin/rpent " in text
    patched_ported.write_text(
        text.replace("exec .venv/bin/rpent ", f'exec "{dumper}" ')
    )
    mine = tmp_path / "mine.sh"
    mine.write_text(
        EXPLORE.read_text().replace('exec "$VENV/bin/rpent" ', f'exec "{dumper}" ')
    )
    gpu = cfg["endpoints"][0]["gpu"]
    endpoint = [ep for ep in C.endpoints_from_config(cfg) if ep.gpu == gpu][0]
    job = C.Job(
        job_id="CloseFridge", task="CloseFridge", split="atomic", seed=0, order=0
    )
    minimal = {"PATH": "/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8"}
    env_mine = stage1.episode_env(cfg, job, 1, endpoint, parent=minimal)
    env_mine.update(ECC_RC_CRED_FILE=str(cred), FAKE_DUMP=str(tmp_path / "mine.json"))
    env_ported = {k: v for k, v in env_mine.items() if not k.startswith("ECC_RC_")}
    env_ported.update(ABL_ESCALATION="0", FAKE_DUMP=str(tmp_path / "ported.json"))
    out_root = tmp_path / "ported_out"
    a = subprocess.run(
        ["bash", str(patched_ported), "atomic", "CloseFridge", "0", str(out_root), "5"],
        env=env_ported,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert a.returncode == 0, a.stderr
    staging = _staging(tmp_path / "staging")
    b = subprocess.run(
        [
            "bash",
            str(mine),
            "atomic",
            "CloseFridge",
            "0",
            str(tmp_path / "mine_out"),
            str(staging),
        ],
        env=env_mine,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert b.returncode == 0, b.stderr
    ported_dump = json.loads((tmp_path / "ported.json").read_text())
    mine_dump = json.loads((tmp_path / "mine.json").read_text())
    pe, me = ported_dump["env"], mine_dump["env"]
    differing = {k for k in set(pe) | set(me) if pe.get(k) != me.get(k)}
    assert differing <= {
        "PYTHONPATH",
        "RPENT_REPO_ROOT",
        "RPENT_EXPLORE_LIST_MEMORY",
        "ABL_ESCALATION",
        "PWD",
        "OLDPWD",
        "_",
        "FAKE_DUMP",
    }, {k: (pe.get(k), me.get(k)) for k in differing}
    assert me["PYTHONPATH"] == me["RPENT_REPO_ROOT"] == cfg["paths"]["code_dir"]
    pa, ma = ported_dump["argv"], mine_dump["argv"]
    tail = [
        "--explore",
        "--explore-sessions",
        str(cfg["stage1"]["sessions"]),
        "--explore-attempts-per-session",
        str(cfg["stage1"]["attempts_per_session"]),
        "--no-auto-merge-memory",
    ]
    assert ma[len(pa) :] == tail
    swap = {
        "--planner-timeout-s": str(cfg["stage1"]["session_timeout_s"]),
        "--memory-dir": str(staging),
        "--output-dir": str(tmp_path / "mine_out"),
    }
    expected = list(pa)
    for flag, value in swap.items():
        expected[expected.index(flag) + 1] = value
    assert ma[: len(pa)] == expected
