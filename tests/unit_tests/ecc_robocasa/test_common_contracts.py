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

"""Shared RC-ECC helpers: config checks, outcome rules, usage parsing, GPU checks."""

from __future__ import annotations

import json
import sys

import pytest

from scripts.ecc_robocasa import common as C
from scripts.ecc_robocasa import stage1
from tests.unit_tests.ecc_robocasa.conftest import FAKES, REPO


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_config_refuses_gpus_outside_the_allowed_set_and_an_overfull_gpu(tmp_path):
    base = {"paths": {"code_dir": str(REPO)}, "gpus": {"allowed": [4, 5, 6, 7]}}
    for endpoints, message in (
        ([{"gpu": 0, "port": 1}], "not in gpus.allowed"),
        ([{"gpu": 4, "port": 1}, {"gpu": 5, "port": 1}], "missing or repeated"),
        # 4 servers + 16 cells measured 77.0 of 80 GB: refused by the memory plan.
        (
            [{"gpu": 4, "port": p, "slots": 4} for p in (1, 2, 3, 4)],
            "exceeds gpus.plan_max_mb",
        ),
    ):
        path = _write(tmp_path / "c.json", json.dumps({**base, "endpoints": endpoints}))
        with pytest.raises(ValueError, match=message):
            C.load_config(path)


def test_three_servers_share_a_gpu_with_distinct_names(tmp_path):
    """The measured full-speed layout: 3 servers per GPU, 25 cells per GPU (64 GB)."""
    endpoints = [
        {"gpu": gpu, "port": port, "slots": slots}
        for gpu, ports in ((4, (11, 12, 13)), (7, (21, 22, 23)))
        for port, slots in zip(ports, (9, 8, 8))
    ] + [{"gpu": 5, "port": 31, "slots": 10}]
    path = _write(
        tmp_path / "c.json",
        json.dumps(
            {
                "paths": {"code_dir": str(REPO)},
                "gpus": {"allowed": [4, 5, 6, 7]},
                "endpoints": endpoints,
            }
        ),
    )
    cfg = C.load_config(path)
    names = [ep.name for ep in C.endpoints_from_config(cfg)]
    assert names == [
        "gpu4_11",
        "gpu4_12",
        "gpu4_13",
        "gpu7_21",
        "gpu7_22",
        "gpu7_23",
        "gpu5",
    ]
    assert cfg["servers"]["env"]["OMP_NUM_THREADS"] == "4"


def test_gpu_fits_checks_room_not_utilization(tmp_path):
    gpus = _write(
        tmp_path / "g.json",
        json.dumps({"4": [40000, 90], "5": [62100, 10], "6": [10, 0, 0]}),
    )
    cfg = C._deep_merge(
        C.DEFAULTS,
        {
            "gpus": {
                "allowed": [4, 5, 6, 7],
                "query_cmd": [sys.executable, str(FAKES / "fake_smi.py"), str(gpus)],
            }
        },
    )
    assert C.gpu_usage(cfg)[4]["total_mb"] == 81559
    assert C.gpu_fits(cfg, 4, 19548)[0]  # busy, but a server fits
    assert not C.gpu_fits(cfg, 5, 19548)[0]  # 19459 MiB left: not enough
    assert not C.gpu_fits(cfg, 6, 100)[0]  # total not reported
    assert not C.gpu_fits(cfg, 0, 100)[0]  # never outside the allowed set


def test_example_config_parses_and_has_no_machine_values():
    cfg = C.load_config(REPO / "scripts" / "ecc_robocasa" / "config.example.toml")
    assert cfg["stage1"]["sessions"] == 5 and cfg["stage1"]["attempts_per_session"] == 3
    assert (
        cfg["stage1"]["session_timeout_s"] == 2400
        and cfg["episode"]["max_turns"] == 100
    )
    assert all(
        str(v).startswith("/path/to/") or not v
        for k, v in cfg["paths"].items()
        if k not in ("code_dir", "mesa_egl_vendor_json", "cuda_lib")
    )


def test_render_mark_wins_and_missing_result_is_infrastructure(tmp_path):
    assert C.read_outcome(tmp_path)["termination_reason"] == "no_result"
    _write(
        tmp_path / "result.json",
        json.dumps({"valid": True, "success": True, "termination_reason": "completed"}),
    )
    assert C.classify(C.read_outcome(tmp_path), killed=False) == "valid"
    (tmp_path / C.RENDER_MARK).write_text("x")
    assert C.classify(C.read_outcome(tmp_path), killed=False) == "render"
    (tmp_path / "result.json").unlink()
    assert C.read_outcome(tmp_path)["termination_reason"] == "render_corrupt"
    assert (
        C.classify(
            {"valid": False, "termination_reason": "infrastructure_error"}, killed=False
        )
        == "infra"
    )
    assert C.classify({"valid": True}, killed=True) == "killed"


def test_usage_sums_the_last_cumulative_line_of_every_session(tmp_path):
    log = _write(
        tmp_path / "run.log",
        "\n".join(
            [
                "x I [api_loop] [usage] in=100 out=5 cache_read=0 cache_write=0 requests=1",
                "x I [api_loop] [usage] in=300 out=9 cache_read=0 cache_write=0 requests=2",
                "x I [agent] === handing off to agent 2/5 ===",
                "x I [api_loop] [usage] in=50 out=1 cache_read=0 cache_write=0 requests=1",
                "x I [agent] usage: in=50 out=1 tool_calls=1",
            ]
        )
        + "\n",
    )
    usage = C.usage_from_run_log(log)
    assert (usage["input_tokens"], usage["output_tokens"], usage["requests"]) == (
        350,
        10,
        3,
    )
    cost = C.episode_usage(tmp_path, C.DEFAULTS["pricing"])
    assert cost["source"] == "run.log" and cost["usd"] == pytest.approx(
        (350 * 3 / 7.1 + 10 * 18 / 7.1) / 1e6
    )


def test_explore_stats_count_resets_and_the_first_success(tmp_path):
    def session(n, steps):
        _write(
            tmp_path / "sessions" / f"session_{n:03d}" / "states.json",
            json.dumps({"steps": steps}),
        )

    ok = {"extras": {"success": True}}
    no = {"extras": {"success": False}}
    reset = {
        "command": {"action": "reset"},
        "result": {"ok": True, "attempt": 2},
        "extras": {"success": False},
    }
    refused = {
        "command": {"action": "reset"},
        "result": {"error": "reset refused"},
        "extras": {"success": False},
    }
    session(1, [no, no, reset, no, reset, no, refused])
    session(2, [no, reset, ok])
    stats = stage1.explore_stats(tmp_path)
    assert stats["sessions_used"] == 2 and stats["attempts_used"] == 3 + 2
    assert stats["solved_session"] == 2 and stats["first_success_attempt"] == 5


def test_gpu_free_uses_nvidia_smi_and_a_fresh_platform_csv(tmp_path):
    gpus = _write(
        tmp_path / "g.json", json.dumps({"4": [10, 0], "5": [30000, 99], "7": [5, 0]})
    )
    csv = _write(
        tmp_path / "latest.csv",
        "timestamp, index, uuid, name, pstate, utilization.gpu [%], "
        "utilization.memory [%], memory.used [MiB], memory.total [MiB]\n"
        "t, 4, u, H100, P0, 0 %, 0 %, 9 MiB, 81559 MiB\n"
        "t, 7, u, H100, P0, 97 %, 10 %, 40000 MiB, 81559 MiB\n",
    )
    cfg = C._deep_merge(
        C.DEFAULTS,
        {
            "gpus": {
                "allowed": [4, 5, 6, 7],
                "query_cmd": [sys.executable, str(FAKES / "fake_smi.py"), str(gpus)],
            },
            "paths": {"gpu_status_csv": str(csv)},
        },
    )
    assert C.gpu_free(cfg, 4)[0]
    assert not C.gpu_free(cfg, 5)[0]  # busy per nvidia-smi
    free, reason = C.gpu_free(cfg, 7)
    assert not free and "platform CSV" in reason
    assert not C.gpu_free(cfg, 0)[0]  # never outside the allowed set
    assert not C.gpu_free(cfg, 6)[0]  # not reported


def test_tree_hash_changes_with_any_file(tmp_path):
    _write(tmp_path / "a" / "x.md", "one")
    first = C.tree_hash(C.tree_files(tmp_path))
    _write(tmp_path / "a" / "x.md", "two")
    assert C.tree_hash(C.tree_files(tmp_path)) != first
