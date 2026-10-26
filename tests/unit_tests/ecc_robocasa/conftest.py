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

"""Offline fixtures for the RC-ECC stage-1 runner: a small cloud corpus, fake cells,
a fake RLDX server and a fake nvidia-smi. No model request, simulator or GPU."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
FAKES = Path(__file__).resolve().parent / "fakes"
STAGE1 = REPO / "scripts" / "ecc_robocasa" / "stage1.py"
# Atomic Target50 tasks, in manifest order.
TASKS = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class EccEnv:
    """One test's machine: corpus, fake venv/credentials/GPUs and the runner command."""

    def __init__(self, root: Path):
        self.root = root
        self.cloud = root / "cloud" / "GPT_6_astra_high"
        self.runs = root / "runs"
        self.venv = root / "venv"
        self.cred = root / "cred.env"
        self.gpus = root / "gpus.json"
        self.scenario = root / "scenario.json"
        self.fake_log = root / "fake_log.jsonl"
        self._make_corpus()
        (self.venv / "bin").mkdir(parents=True)
        python = self.venv / "bin" / "python"
        python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        python.chmod(0o755)
        self.cred.write_text(
            "FAKE_WORKER_KEY=fake-worker-key\nFAKE_SENIOR_URL=https://senior.invalid\nFAKE_SENIOR_KEY=fake\n"
        )
        self.cred.chmod(0o600)
        self.set_gpus({4: (10, 0), 5: (10, 0), 6: (20000, 60), 7: (5, 0)})
        self.scenario.write_text("{}")
        self.run_id = "r1"

    def _make_corpus(self) -> None:
        (self.cloud / "global").mkdir(parents=True)
        (self.cloud / "task-specific").mkdir()
        (self.cloud / "README.md").write_text("# cloud corpus (synthetic)\n")
        (self.cloud / "global" / "GLOBAL_MEMORY.md").write_text(
            "# Global memory (synthetic)\nRe-ground every scene.\n"
        )
        for task in TASKS:
            (self.cloud / "task-specific" / f"{task}_s0.json").write_text(
                json.dumps({"task": task, "seed": 0}) + "\n"
            )
            (self.cloud / "task-specific" / f"{task}_s0_recipe.jsonl").write_text(
                '{"action": "rldx_arm"}\n'
            )
            if task != "OpenCabinet":
                (self.cloud / "task-specific" / f"{task}.md").write_text(
                    f"---\nscope: suite\ntask_id: {task}\n---\n# {task}\n"
                )

    def set_gpus(self, gpus: dict[int, tuple[int, int]]) -> None:
        self.gpus.write_text(json.dumps({str(k): list(v) for k, v in gpus.items()}))

    def set_scenario(self, scenario: dict) -> None:
        self.scenario.write_text(json.dumps(scenario))

    def config(
        self,
        *,
        tasks,
        endpoints=((4, 2),),
        max_slots=4,
        stage1=None,
        scheduler=None,
        **extra,
    ) -> Path:
        cfg = {
            "paths": {
                "code_dir": str(REPO),
                "venv": str(self.venv),
                "cloud_corpus": str(self.cloud),
                "runs_dir": str(self.runs),
                "assets": "/nonexistent/assets",
                "hf_home": "/nonexistent/hf",
                "rldx_checkpoint": "/nonexistent/ckpt",
                "llvmpipe_index_script": "/nonexistent/llvmpipe.sh",
            },
            "credentials": {
                "file": str(self.cred),
                "worker_key_var": "FAKE_WORKER_KEY",
                "senior_base_url_var": "FAKE_SENIOR_URL",
                "senior_key_var": "FAKE_SENIOR_KEY",
            },
            "worker": {"base_url": "https://worker.invalid/v1"},
            "episode": {
                "env": {
                    "ECC_RC_FAKE_SCENARIO": str(self.scenario),
                    "ECC_RC_FAKE_LOG": str(self.fake_log),
                }
            },
            "stage1": {"tasks": list(tasks), **(stage1 or {})},
            "gpus": {
                "allowed": [4, 5, 6, 7],
                "query_cmd": [
                    sys.executable,
                    str(FAKES / "fake_smi.py"),
                    str(self.gpus),
                ],
            },
            "endpoints": [
                {"gpu": gpu, "port": free_port(), "slots": slots}
                for gpu, slots in endpoints
            ],
            "servers": {
                "mode": "managed",
                "cmd": [sys.executable, str(FAKES / "fake_vla_server.py")],
                "ready_timeout_s": 30,
            },
            "launch": {"cell": [sys.executable, str(FAKES / "fake_cell.py")]},
            "scheduler": {
                "max_slots": max_slots,
                "launch_stagger_s": 0,
                "poll_s": 0.1,
                "min_available_mb": 0,
                "kill_grace_s": 2,
                "cost_check_s": 0,
                **(scheduler or {}),
            },
        }
        for key, value in extra.items():
            cfg[key] = value
        path = self.root / "config.json"
        path.write_text(json.dumps(cfg, indent=2))
        return path

    # -- running the CLI -----------------------------------------------------

    def cmd(self, *args: str) -> list[str]:
        return [sys.executable, str(STAGE1), *args]

    def stage1(self, *args: str, timeout: float = 180) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.cmd(*args),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def go(
        self, config: Path, *extra: str, timeout: float = 180
    ) -> subprocess.CompletedProcess:
        proc = self.stage1(
            "run",
            "--config",
            str(config),
            "--run-id",
            self.run_id,
            "--go",
            *extra,
            timeout=timeout,
        )
        assert proc.returncode in (0, 1), proc.stdout + proc.stderr
        return proc

    def popen_go(self, config: Path) -> subprocess.Popen:
        return subprocess.Popen(
            self.cmd("run", "--config", str(config), "--run-id", self.run_id, "--go"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            start_new_session=True,
        )

    # -- reading the run -----------------------------------------------------

    @property
    def run_root(self) -> Path:
        return self.runs / self.run_id

    def state(self, task: str) -> dict:
        return json.loads(
            (self.run_root / "state" / "jobs" / f"{task}.json").read_text()
        )

    def record(self, task: str) -> dict:
        return json.loads((self.run_root / "tasks" / task / "record.json").read_text())

    def events(self) -> list[dict]:
        if not self.fake_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.fake_log.read_text().splitlines()
            if line.strip()
        ]

    def progress(self) -> list[dict]:
        path = self.run_root / "progress.jsonl"
        return (
            [json.loads(line) for line in path.read_text().splitlines()]
            if path.exists()
            else []
        )

    def wait_for(self, predicate, timeout: float = 60.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.1)
        raise AssertionError("condition not reached in time")

    def cleanup(self) -> None:
        """Stop fake servers and cells this test left behind (only processes of this test)."""
        from scripts.ecc_robocasa.common import cmdline_has

        pids = [e["pid"] for e in self.events()]
        servers = self.run_root / "servers"
        for path in servers.glob("gpu*.json") if servers.is_dir() else []:
            pids.append(json.loads(path.read_text()).get("pid"))
        for pid in {p for p in pids if p}:
            if cmdline_has(pid, "fake_") and cmdline_has(pid, str(self.root)):
                try:
                    os.killpg(os.getpgid(pid), 9)
                except OSError:
                    pass


@pytest.fixture
def ecc(tmp_path):
    env = EccEnv(tmp_path)
    yield env
    env.cleanup()
