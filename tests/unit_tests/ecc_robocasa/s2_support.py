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

"""Offline machine for the RC-ECC stage-2 runner: a frozen stage-1 product built by the
real stage-1 runner with fake cells, fake hybrid cells, fake RLDX servers and GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from tests.unit_tests.ecc_robocasa.conftest import FAKES, REPO, EccEnv, free_port

STAGE2 = REPO / "scripts" / "ecc_robocasa" / "stage2.py"
FAKE_HYBRID = FAKES / "fake_hybrid_cell.py"


def build_s1_product(root: Path, tasks: list[str]) -> tuple[EccEnv, Path]:
    """Run stage 1 with fake cells (every task solved) and return its frozen product."""
    env = EccEnv(root)
    env.set_scenario({task: {"kind": "solve", "sleep_s": 0.01} for task in tasks})
    config = env.config(tasks=tasks, max_slots=4, endpoints=((4, 4),))
    proc = env.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    product = env.run_root / "product"
    assert (product / "FROZEN").exists()
    return env, product


class S2Env:
    """One stage-2 test's machine on top of a frozen stage-1 product."""

    def __init__(self, root: Path, s1: EccEnv, product: Path):
        self.root = root
        self.s1 = s1
        self.product = product
        self.runs = root / "s2runs"
        self.scenario = root / "s2_scenario.json"
        self.fake_log = root / "s2_fake_log.jsonl"
        self.gpus = root / "gpus.json"
        self.gpus.write_text(
            json.dumps({"4": [10, 0], "5": [10, 0], "6": [20000, 60], "7": [5, 0]})
        )
        self.scenario.write_text("{}")
        self.run_id = "s2"
        self.config_path = root / "config_s2.json"

    def set_scenario(self, scenario: dict) -> None:
        self.scenario.write_text(json.dumps(scenario))

    def config(
        self,
        *,
        tasks,
        endpoints=((4, 4),),
        max_slots=4,
        stage2=None,
        scheduler=None,
        **extra,
    ) -> Path:
        cfg = {
            "paths": {
                "code_dir": str(REPO),
                "venv": str(self.s1.venv),
                "cloud_corpus": str(self.s1.cloud),
                "runs_dir": str(self.runs),
                "assets": "/nonexistent/assets",
                "hf_home": "/nonexistent/hf",
                "rldx_checkpoint": "/nonexistent/ckpt",
                "llvmpipe_index_script": "/nonexistent/llvmpipe.sh",
            },
            "credentials": {
                "file": str(self.s1.cred),
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
            "stage2": {
                "s1_product": str(self.product),
                "tasks": list(tasks),
                "launcher": [sys.executable, str(FAKE_HYBRID)],
                **(stage2 or {}),
            },
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
        cfg.update(extra)
        self.config_path.write_text(json.dumps(cfg, indent=2))
        return self.config_path

    def stage2(self, *args: str, timeout: float = 240) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(STAGE2), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        )

    def go(self, *extra: str, timeout: float = 240) -> subprocess.CompletedProcess:
        proc = self.stage2(
            "run",
            "--config",
            str(self.config_path),
            "--run-id",
            self.run_id,
            "--go",
            *extra,
            timeout=timeout,
        )
        assert proc.returncode in (0, 1), proc.stdout + proc.stderr
        return proc

    def popen_go(self) -> subprocess.Popen:
        return subprocess.Popen(
            [
                sys.executable,
                str(STAGE2),
                "run",
                "--config",
                str(self.config_path),
                "--run-id",
                self.run_id,
                "--go",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            start_new_session=True,
        )

    def cli(self, cmd: str, *extra: str) -> subprocess.CompletedProcess:
        return self.stage2(
            cmd, "--config", str(self.config_path), "--run-id", self.run_id, *extra
        )

    @property
    def run_root(self) -> Path:
        return self.runs / self.run_id

    def state(self, task: str, rung: int) -> dict:
        return json.loads(
            (self.run_root / "state" / "jobs" / f"{task}__r{rung}.json").read_text()
        )

    def rule(self, task: str) -> dict:
        return json.loads((self.run_root / "tasks" / task / "rule.json").read_text())

    def record(self, task: str, rung: int) -> dict:
        return json.loads(
            (self.run_root / "tasks" / task / f"r{rung}" / "episode.json").read_text()
        )

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
