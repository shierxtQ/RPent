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

"""Fake final-test episode launcher with GPT-6-alone episodes (no model, simulator or network).

Called like ``hybrid_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR``. Without
``ECC_RC_SENIOR_ALONE=1`` it runs the final test's fake cell
(``test_final_test_fake_cell.py``) unchanged. With it, it writes what RPent's api
planner with GPT-6 alone leaves: ``run.log`` with the planner command, the image
cap and reasoning-id lines RPent logs, a ``[usage]`` line per turn (with cached
input), a transcript with the run's stats (none after a planner timeout) and
``result.json`` naming the model and effort; no ``escalation.jsonl``.

The scenario (``ECC_RC_FAKE_SCENARIO``) is read as by the fake cell: per job id,
``<arm>:<Task>``, ``<arm>`` or ``default``, a spec or a list of specs (one per
try) with ``kind`` (``solve`` | ``fail`` | ``timeout`` | ``infra`` | ``crash``),
``requests``, ``tokens`` (``[in, cached, out]`` per request), ``sleep_s`` and
``bug`` (``escalation_log``, ``model``, ``no_cap_log``, ``no_ids_log``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_CELL = HERE.parent / "test_final_test_fake_cell.py"


def main() -> int:
    if os.environ.get("ECC_RC_SENIOR_ALONE") != "1":
        os.execv(sys.executable, [sys.executable, str(FAKE_CELL), *sys.argv[1:]])
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(FAKE_CELL.parent))
    from fake_cell import png, write_result
    from test_final_test_fake_cell import spec_for

    split, task, seed, out = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        Path(sys.argv[4]),
    )
    job = out.parent.name
    arm = job.split("__", 1)[0]
    n_try = int(os.environ.get("ECC_RC_TRY", "1"))
    if os.environ.get("ECC_RC_ESCALATION") != "0":
        print(
            "fake gpt6 cell: ECC_RC_SENIOR_ALONE=1 without ECC_RC_ESCALATION=0",
            file=sys.stderr,
        )
        return 2
    model, effort = (
        os.environ["ECC_RC_SENIOR_MODEL"],
        os.environ["ECC_RC_SENIOR_EFFORT"],
    )
    images, ids = (
        int(os.environ["ECC_RC_MAX_IMAGES"]),
        os.environ["ECC_RC_REASONING_IDS"],
    )
    scenario = json.loads(Path(os.environ["ECC_RC_FAKE_SCENARIO"]).read_text())
    spec = spec_for(scenario, job, arm, task, n_try)
    kind, bug = spec.get("kind", "fail"), set(spec.get("bug", []))
    if fake_log := os.environ.get("ECC_RC_FAKE_LOG"):
        with open(fake_log, "a") as handle:
            env = {
                k: os.environ.get(k)
                for k in (
                    "ECC_RC_ESCALATION",
                    "ECC_RC_SENIOR_ALONE",
                    "ECC_RC_MAX_IMAGES",
                    "ECC_RC_REASONING_IDS",
                    "ECC_RC_SENIOR_MODEL",
                    "ECC_RC_SENIOR_EFFORT",
                    "ECC_RC_CORRECTIONS",
                    "ECC_RC_TIMEOUT_S",
                    "RC_GPU",
                    "VLA_ENDPOINT",
                )
            }
            row = {"event": "start", "job": job, "arm": arm, "task": task, "seed": seed, "split": split,
                   "try": n_try, "pid": os.getpid(), "t": time.time(), "env": env}  # fmt: skip
            handle.write(json.dumps(row) + "\n")
    out.mkdir(parents=True, exist_ok=True)
    if kind == "crash":
        return 1
    started = time.time()
    requests = int(spec.get("requests", 8))
    tokens_in, cached, tokens_out = spec.get("tokens", [20000, 5000, 300])
    model_logged = "openai:other-model" if "model" in bug else model
    with (out / "run.log").open("w") as log:
        log.write(
            f"2026-10-27 00:00:00 I [agent] physical agent cmd: rpent --robot robocasa --task-name {task} "
            f"--planner api --model {model} --reasoning-effort {effort}\n"
        )
        if "no_cap_log" not in bug:
            log.write(
                f"2026-10-27 00:00:00 I [api_loop] history images: at most {images} per request "
                "(RPENT_MAX_HISTORY_IMAGES)\n"
            )
        if ids == "0" and "no_ids_log" not in bug:
            log.write(
                "2026-10-27 00:00:00 I [api_loop] earlier response item ids are not sent back "
                "(RPENT_SEND_REASONING_IDS=0)\n"
            )
        for i in range(1, requests + 1):
            log.write(f"2026-10-27 00:00:00 I [api_loop] === turn {i}/100 ===\n")
            log.write(
                f"2026-10-27 00:00:00 I [api_loop] [usage] in={tokens_in * i} out={tokens_out * i} "
                f"cache_read={cached * i} cache_write=0 requests={i}\n"
            )
            log.flush()
            time.sleep(float(spec.get("sleep_s", 0.02)) / requests)
    if "escalation_log" in bug:
        (out / "escalation.jsonl").write_text('{"request": 1, "role": "worker"}\n')
    png(out / "agentview.png" / "00.png", noise=False)
    timeout = kind == "timeout"
    total = requests + (
        0 if timeout else 1
    )  # the transcript also counts a last reply without a tool call
    stats = {
        "turns_used": requests,
        "tool_calls": requests,
        "total_input_tokens": tokens_in * total,
        "total_output_tokens": tokens_out * total,
        "cache_read_tokens": cached * total,
        "cache_write_tokens": 0,
        "requests": total,
    }
    tag = f"{task}_target_s{seed}"
    (out / f"transcript_{tag}.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "split": "target",
                "seed": seed,
                "model": model,
                "finish": None
                if timeout
                else {"status": "success" if kind == "solve" else "failure"},
                "stats": {"elapsed_s": 1.0} if timeout else stats,
                "messages": [{"role": "user", "content": "Do the task."}],
            }
        )
    )
    elapsed = time.time() - started
    if kind == "infra":
        write_result(
            out,
            task,
            seed,
            valid=False,
            success=False,
            reason="infrastructure_error",
            elapsed=elapsed,
        )
    else:
        write_result(
            out,
            task,
            seed,
            valid=True,
            success=kind == "solve",
            reason="planner_timeout" if timeout else "completed",
            elapsed=elapsed,
        )
    result = json.loads((out / "result.json").read_text())
    result["planner"] = {
        "backend": "api",
        "model": model_logged,
        "reasoning_effort": effort,
        "max_turns": 100,
    }
    (out / "result.json").write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
