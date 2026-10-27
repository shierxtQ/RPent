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

"""Stage 3 with the v2 switches off equals the stage 3 of an earlier checkout.

``ECC_RC_OLD_CODE`` names a checkout of the code before stage 3 v2 (skipped
without it). The same fake scenarios (sequential and parallel sessions) run
once with that checkout's runner, fakes and RPent and once with this tree's;
every file the runs write under ``tasks/``, ``state/`` and ``product/`` must
be the same apart from times, process ids, ports and the run's own paths.
The reflection functions are compared on random inputs, and the config and
experiment hashes of the same config must be equal.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.ecc_robocasa import reflect as R
from scripts.ecc_robocasa import stage3 as S3
from tests.unit_tests.ecc_robocasa.conftest import REPO
from tests.unit_tests.ecc_robocasa.test_stage3_runner import S3Env

OLD = Path(os.environ.get("ECC_RC_OLD_CODE") or "/nonexistent")
pytestmark = [
    pytest.mark.timeout(600),
    pytest.mark.skipif(
        not (OLD / "scripts" / "ecc_robocasa" / "stage3.py").is_file(),
        reason="set ECC_RC_OLD_CODE to a checkout of the code before stage 3 v2",
    ),
]
#: Keys whose values are times, process ids, ports or hashes over files that hold times.
VOLATILE = {
    "at",
    "t",
    "time",
    "created_at",
    "updated_at",
    "finalized_at",
    "started_at",
    "ended_at",
    "done_at",
    "launched_at",
    "finished_at",
    "started_ts",
    "ended_ts",
    "next_at",
    "pid",
    "endpoint",
    "url",
    "port",
    "latency_s",
    "elapsed_s",
    "reflection_json_sha256",
    "tasks_json_sha256",
    "summary_json_sha256",
    "final_tree_sha256",
    "files",
    "code",
    "config_sha256",
    "run_id",
}
SCENARIOS = {
    "sequential": (
        {},
        {
            "CloseBlenderLid": {"s1_a1": {"kind": "solve"}},
            "CloseFridge": {
                "s1_a1": {"kind": "fail", "requests": 15},
                "s1_a2": [{"kind": "infra"}, {"kind": "fail", "requests": 15}],
                "s1_a3": {"kind": "timeout", "requests": 15},
                "s2_a1": {"kind": "solve", "requests": 15},
            },
            "CloseToasterOvenDoor": {"default": {"kind": "fail", "requests": 7}},
        },
        {"CloseToasterOvenDoor/s1_a2": {"status": "refused"}},
    ),
    "parallel": (
        {"parallel_attempts": True},
        {
            "CloseBlenderLid": {"default": {"kind": "fail", "requests": 7}},
            "CloseFridge": {
                "s1_a1": {"kind": "fail", "requests": 15},
                "s1_a2": {"kind": "fail", "requests": 9},  # no senior turn: skipped
                "s1_a3": {"kind": "timeout", "requests": 15},
                "s2_a1": {"kind": "solve", "requests": 15},
                "s2_a2": {"kind": "fail", "requests": 15},
                "s2_a3": {"kind": "fail", "requests": 15},
            },
            "CloseToasterOvenDoor": {
                "s1_a1": {"kind": "solve", "requests": 7},
                "default": {"kind": "fail", "requests": 7},
            },
        },
        {
            "CloseFridge/s2_a1": {
                "edits": {
                    "applied": ["s1_a1.1", "s1_a3.1"],
                    "add": [],
                    "revise": [{"id": "s1_a1.1", "fix": "start 5 cm below the edge"}],
                    "remove": ["s1_a3.1"],
                }
            }
        },
    ),
}


def normalize(value, subs):
    if isinstance(value, dict):
        return {k: normalize(v, subs) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [normalize(v, subs) for v in value]
    if isinstance(value, str):
        for old, new in subs:
            value = value.replace(old, new)
        return value
    return value


def deep_diff(a, b, path="") -> list[str]:
    """Where two normalized values differ (for the failure message)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += deep_diff(
                a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}"
            )
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [
            d
            for i, (x, y) in enumerate(zip(a, b))
            for d in deep_diff(x, y, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {str(a)[:200]!r} != {str(b)[:200]!r}"]


def collect(root: Path, subs) -> dict[str, object]:
    """Every JSON, JSONL, Markdown and text file of a run, normalized."""
    out = {}
    for top in ("tasks", "state", "product"):
        for path in sorted((root / top).rglob("*")):
            if (
                not path.is_file()
                or path.name in ("FROZEN",)
                or path.suffix in (".png", ".npz", ".mp4")
            ):
                continue
            rel = path.relative_to(root).as_posix()
            if "/try" in rel and not rel.endswith((".evidence.json",)):
                continue  # the fake episodes' own files (they log times)
            text = path.read_text(errors="replace")
            if path.suffix == ".json":
                out[rel] = normalize(json.loads(text), subs)
            elif path.suffix == ".jsonl":
                out[rel] = [
                    normalize(json.loads(x), subs)
                    for x in text.splitlines()
                    if x.strip()
                ]
            elif path.suffix == ".log":
                continue  # call logs: the client's own output
            else:
                for old, new in subs:
                    text = text.replace(old, new)
                out[rel] = text
    return out


def run_with(env: S3Env, code: Path, run_id: str) -> subprocess.CompletedProcess:
    """Run stage 3 from ``code`` (its runner, fakes and RPent) under ``run_id``."""
    cfg = json.loads(env.config.read_text())
    fakes = code / "tests" / "unit_tests" / "ecc_robocasa" / "fakes"
    cfg["paths"]["code_dir"] = str(code)
    cfg["stage3"]["launcher"] = [sys.executable, str(fakes / "fake_s3_cell.py")]
    cfg["stage3"]["reflection"]["cmd"] = [
        sys.executable,
        str(fakes / "fake_reflect.py"),
    ]
    cfg["servers"]["cmd"] = [sys.executable, str(fakes / "fake_vla_server.py")]
    cfg["gpus"]["query_cmd"][1] = str(fakes / "fake_smi.py")
    path = env.ecc.root / f"config_{run_id}.json"
    path.write_text(json.dumps(cfg, indent=2))
    proc = subprocess.run(
        [
            sys.executable,
            str(code / "scripts" / "ecc_robocasa" / "stage3.py"),
            "run",
            "--run-id",
            run_id,
            "--go",
            "--config",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=420,
        cwd=str(code),
        env=dict(
            os.environ,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(code),
            RPENT_REPO_ROOT=str(code),
        ),
    )
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    return proc


@pytest.mark.parametrize("mode", sorted(SCENARIOS))
def test_runs_with_the_v2_switches_off_equal_the_old_runner(ecc, mode):
    stage3, scenario, reflect = SCENARIOS[mode]
    env = S3Env(
        ecc,
        ["CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor"],
        scenario=scenario,
        reflect=reflect,
        stage3=stage3,
        endpoints=((4, 6),),
        max_slots=6,
    )
    runs = {}
    for name, code in (("old", OLD), ("new", REPO)):
        run_with(env, code, f"r_{name}")
        root = ecc.runs / f"r_{name}"
        subs = [
            (str(root), "<RUN>"),
            (str(code.resolve()), "<CODE>"),
            (str(code), "<CODE>"),
            (f"r_{name}", "<RUN_ID>"),
        ]
        runs[name] = collect(root, subs)
    old, new = runs["old"], runs["new"]
    assert sorted(old) == sorted(new)
    assert len(old) > 50
    differ = {k: deep_diff(old[k], new[k]) for k in old if old[k] != new[k]}
    assert differ == {}, json.dumps(dict(list(differ.items())[:3]), indent=1)[:3000]
    # The prompts the fake reflection call received are byte-identical too.
    prompts = [p for p in old if p.endswith("prompt.txt")]
    assert prompts and all(old[p] == new[p] for p in prompts)
    # The two runs are one experiment: same experiment hash in the manifests.
    manifests = [
        json.loads((ecc.runs / f"r_{n}" / "manifest.json").read_text())
        for n in ("old", "new")
    ]
    assert manifests[0]["experiment_sha256"] == manifests[1]["experiment_sha256"]
    assert manifests[0]["settings"] == manifests[1]["settings"]
    assert manifests[0]["budget"] == manifests[1]["budget"]


def load_old(module: str):
    spec = importlib.util.spec_from_file_location(
        f"old_{module}", OLD / "scripts" / "ecc_robocasa" / f"{module}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_the_reflection_functions_equal_the_old_ones_on_random_inputs():
    old = load_old("reflect")
    assert old.PROMPT == R.PROMPT
    rnd = random.Random(20261027)

    def entry(i):
        return {
            "id": f"s{rnd.randint(1, 5)}_a{rnd.randint(1, 3)}.{i}",
            "trigger": rnd.choice(R.TRIGGERS),
            "situation": f"s{i}",
            "diagnosis": "d",
            "fix": "f",
            "status": rnd.choice(["candidate", "verified"]),
            "written_in": "T/s1_a1",
            "used_in": [],
            "successes": rnd.randint(0, 1),
            "failures": rnd.randint(0, 2),
        }

    for _ in range(2000):
        entries = [entry(i) for i in range(rnd.randint(0, 8))]
        ids = [e["id"] for e in entries] + ["zz"]
        edits = {
            "applied": rnd.sample(ids, rnd.randint(0, len(ids))),
            "add": [
                {
                    "trigger": "checkin",
                    "situation": "a",
                    "diagnosis": "b" * rnd.randint(0, 2),
                    "fix": "c" * rnd.randint(0, 600),
                }
                for _ in range(rnd.randint(0, 4))
            ],
            "revise": [
                {"id": rnd.choice(ids), "fix": "new", "diagnosis": ""}
                for _ in range(rnd.randint(0, 3))
            ],
            "remove": rnd.sample(ids, rnd.randint(0, min(2, len(ids)))),
            "note": "n",
        }
        reply = (
            json.dumps(edits)
            if rnd.random() > 0.1
            else rnd.choice(["no json", "[1, 2]", "{bad"])
        )
        success = rnd.random() > 0.5
        shown = (
            None
            if rnd.random() > 0.5
            else set(rnd.sample(ids, rnd.randint(0, len(ids))))
        )
        settings = R.Settings(
            max_entries=rnd.randint(1, 8), fails_to_drop=rnd.randint(1, 3)
        )
        a = old.reviewed(
            entries,
            reply,
            "T/s2_a1",
            success,
            old.Settings(**settings.__dict__),
            shown=shown,
        )
        b = R.reviewed(entries, reply, "T/s2_a1", success, settings, shown=shown)
        assert json.dumps(a) == json.dumps(b)
    d = {
        "interventions": 2,
        "by_reason": {"plan": 1, "checkin": 1},
        "shown": 2,
        "turns": 9,
        "source": "run.log",
        "finish": {"summary": "x" * 900},
        "text": "### Intervention at turn 1 (trigger: plan)\nabc\n\n### Intervention at turn 5 (trigger: checkin)\n"
        + "y" * 50000,
    }
    for success in (True, False):
        kw = {
            "language": "Open the left drawer.",
            "rule": {"gpt_plan": True},
            "success": success,
            "termination": "completed",
            "entries": [entry(1)],
        }
        assert old.build_prompt(d, **kw) == R.build_prompt(d, **kw)


def test_the_config_and_experiment_hashes_equal_the_old_ones(ecc):
    env = S3Env(ecc, ["CloseFridge"], scenario={})
    script = (
        "import json, sys; from scripts.ecc_robocasa import stage3 as S, common as C; "
        "cfg = S.load_config(sys.argv[1]); "
        "print(json.dumps([S.experiment_hash(cfg), C.config_hash(cfg), S.experiment_settings(cfg)]))"
    )
    for extra in ({}, {"parallel_attempts": True}):
        cfg = json.loads(env.config.read_text())
        cfg["stage3"].update(extra)
        path = ecc.root / "hash_config.json"
        path.write_text(json.dumps(cfg))
        out = {}
        for name, code in (("old", OLD), ("new", REPO)):
            proc = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                capture_output=True,
                text=True,
                cwd=str(code),
                env=dict(os.environ, PYTHONPATH=str(code), PYTHONDONTWRITEBYTECODE="1"),
            )
            assert proc.returncode == 0, proc.stderr
            out[name] = json.loads(proc.stdout)
        assert out["old"] == out["new"]
    assert re.fullmatch(r"[0-9a-f]{64}", out["new"][0])
    # In process too: this tree's stage 3 with the v2 switches set to their defaults.
    cfg = S3.load_config(path)
    assert S3.experiment_hash(cfg) == out["new"][0]
