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

"""Fake RC-ECC hybrid evaluation episode: the folder RPent leaves, with a genuine escalation log.

Called like ``hybrid_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR`` with the runner's
``ECC_RC_*`` inputs. No model, simulator or network: the requests go through RPent's
real ``EscalationModel`` with scripted worker and senior functions, so
``escalation.jsonl`` (config row, plan turn, check-ins, give-up reviews, errors
turns, ``corrections_chars``) is what the wrapper writes for that rule.

The behaviour per task comes from the JSON file in ``ECC_RC_FAKE_SCENARIO``:
``{task: spec | {"r<rung>": spec | [spec per try]}}`` with

- ``kind``: ``solve`` | ``fail`` (valid failure) | ``timeout`` (valid planner
  timeout) | ``infra`` (invalid result.json) | ``crash`` (no result.json) |
  ``noise`` (noise frames, then waits to be stopped) | ``hang`` (sleeps ``hang_s``);
- ``requests`` (default 20): requests in the episode;
- ``error_results``: request numbers whose tool result carries a top-level ``error``;
- ``giveup_at``: request numbers where the worker proposes ``finish`` without success;
- ``senior_text_at``: request numbers where a senior turn replies with text only;
- ``bug``: wrapper settings that differ from the launch (a broken launcher), e.g.
  ``{"checkin": 5}`` or ``{"errors": 2}``;
- ``file_calls``: ``[[tool, path], ...]`` file-tool calls logged in run.log (not executed);
- ``no_log``: write no escalation.jsonl; ``sleep_s``: wall time per episode.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# RPent comes from the code dir under test, as hybrid_cell.sh's PYTHONPATH makes it (the
# runner scrubs PYTHONPATH from the environment it hands to a cell).
if os.environ.get("ECC_RC_CODE_DIR"):
    sys.path.insert(1, os.environ["ECC_RC_CODE_DIR"])

from fake_cell import png, write_result  # noqa: E402


def spec_for(scenario: dict, task: str, rung: int, n_try: int) -> dict:
    spec = scenario.get(task, {"kind": "fail"})
    if isinstance(spec, dict) and any(
        k.startswith("r") and k[1:].isdigit() for k in spec
    ):
        spec = spec.get(f"r{rung}", spec.get("default", {"kind": "fail"}))
    specs = spec if isinstance(spec, list) else [spec]
    return specs[min(n_try, len(specs)) - 1]


def drive(out: Path, spec: dict, settings: dict) -> int:
    """Run ``requests`` requests through the real escalation wrapper; returns requests made."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel

    from rpent.planner.escalation import EscalationModel

    giveups = set(spec.get("giveup_at", []))
    texts = set(spec.get("senior_text_at", []))
    errors = set(spec.get("error_results", []))
    state = {"n": 0}

    def worker(messages, info):
        if state["n"] in giveups:
            return ModelResponse(
                parts=[
                    ToolCallPart("finish", {"status": "failure", "summary": "stuck"})
                ]
            )
        return ModelResponse(parts=[ToolCallPart("rldx_arm", {"prompt": "Do it."})])

    def senior(messages, info):
        if state["n"] in texts:  # guidance only: the wrapper hands it to the worker
            return ModelResponse(parts=[TextPart("Approach from the left.")])
        return ModelResponse(parts=[ToolCallPart("view_env_state", {})])

    model = EscalationModel(
        FunctionModel(worker, model_name=settings["worker"]),
        FunctionModel(senior, model_name=settings["senior"]),
        checkin=settings["checkin"],
        max_share=settings["max_share"],
        plan=settings["plan"],
        errors=settings["errors"],
        corrections=settings["corrections"],
        log_path=out / "escalation.jsonl",
        senior_max_images=40,
    )
    messages = [ModelRequest.user_text_prompt("Do the task.")]
    for i in range(1, int(spec.get("requests", 20)) + 1):
        state["n"] = i
        asyncio.run(model.request(messages, None, ModelRequestParameters()))
        payload = (
            {"error": "bad step"} if i in errors else {"log": {"result": {"ok": True}}}
        )
        messages.append(
            ModelResponse(
                parts=[
                    ToolCallPart("rldx_arm", {"prompt": "Do it."}, tool_call_id=f"c{i}")
                ]
            )
        )
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        "rldx_arm", [json.dumps(payload)], tool_call_id=f"c{i}"
                    )
                ]
            )
        )
    return int(spec.get("requests", 20))


def main() -> int:
    split, task, seed, out, memory = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        Path(sys.argv[4]),
        Path(sys.argv[5]),
    )
    n_try = int(os.environ.get("ECC_RC_TRY", "1"))
    checkin = int(os.environ["ECC_RC_CHECKIN"])
    errors = int(os.environ["ECC_RC_ERRORS"])
    plan = os.environ["ECC_RC_PLAN"] == "1"
    rung = int(os.environ.get("ECC_RC_FAKE_RUNG") or 0) or _rung_from_path(out)
    scenario = json.loads(Path(os.environ["ECC_RC_FAKE_SCENARIO"]).read_text())
    spec = spec_for(scenario, task, rung, n_try)
    kind = spec.get("kind", "fail")
    corrections_path = os.environ.get("ECC_RC_CORRECTIONS") or ""
    corrections = Path(corrections_path).read_text() if corrections_path else ""
    fake_log = os.environ.get("ECC_RC_FAKE_LOG")

    def event(name: str, **extra) -> None:
        if fake_log:
            with open(fake_log, "a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": name,
                            "task": task,
                            "rung": rung,
                            "try": n_try,
                            "pid": os.getpid(),
                            "t": time.time(),
                            "gpu": os.environ.get("RC_GPU"),
                            "endpoint": os.environ.get("VLA_ENDPOINT"),
                            "seed": seed,
                            "split": split,
                            "memory": str(memory),
                            "plan": plan,
                            "checkin": checkin,
                            "errors": errors,
                            "timeout": os.environ.get("ECC_RC_TIMEOUT_S"),
                            "corrections": corrections_path,
                            **extra,
                        }
                    )
                    + "\n"
                )

    event("start")
    out.mkdir(parents=True, exist_ok=True)
    (out / "env_server.log").write_text("RPC server listening on http://127.0.0.1:0\n")
    started = time.time()
    tag = f"{task}_target_s{seed}"
    if kind == "crash":
        time.sleep(float(spec.get("sleep_s", 0.05)))
        event("end", kind=kind)
        return 1
    settings = {
        "plan": plan,
        "checkin": checkin,
        "errors": errors,
        "max_share": float(os.environ["ECC_RC_MAX_SHARE"]),
        "corrections": corrections,
        "worker": os.environ["ECC_RC_WORKER_MODEL"].split(":", 1)[-1],
        "senior": os.environ["ECC_RC_SENIOR_MODEL"].split(":", 1)[-1],
        **spec.get("bug", {}),
    }
    requests = 0 if spec.get("no_log") else drive(out, spec, settings)
    with (out / "run.log").open("w") as log:
        for i in range(1, requests + 1):
            log.write(f"2026-10-27 00:00:00 I [api_loop] === turn {i}/100 ===\n")
        for tool, path in spec.get(
            "file_calls", []
        ):  # logged like RPent's api loop; nothing is written
            args = json.dumps(
                {
                    "path": path,
                    **({"content": "x"} if tool == "write_text_file" else {}),
                }
            )
            log.write(f"2026-10-27 00:00:00 I [api_loop] [tool>] {tool}({args})\n")
    png(out / "agentview.png" / "00.png", noise=(kind == "noise"))
    if kind == "noise":
        event("noise_written")
        time.sleep(60)
        return 1
    time.sleep(float(spec.get("sleep_s", 0.05)))
    solved = kind in ("solve", "hang")
    (out / f"transcript_{tag}.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "split": "target",
                "seed": seed,
                "model": "fake",
                "elapsed_s": 1.0,
                "finish": {
                    "status": "success" if solved else "failure",
                    "summary": "fake",
                }
                if kind in ("solve", "fail", "hang")
                else None,
                "stats": {"turns_used": requests} if kind != "timeout" else {},
                "messages": [],
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
    elif kind == "timeout":
        write_result(
            out,
            task,
            seed,
            valid=True,
            success=False,
            reason="planner_timeout",
            elapsed=elapsed,
        )
    else:
        write_result(
            out,
            task,
            seed,
            valid=True,
            success=solved,
            reason="completed",
            elapsed=elapsed,
        )
    if kind == "hang":
        time.sleep(float(spec.get("hang_s", 3.0)))
    event("end", kind=kind, rpent=_rpent_origin())
    return 0


def _rpent_origin() -> str | None:
    module = sys.modules.get("rpent.planner.escalation")
    return os.path.realpath(module.__file__) if module else None


def _rung_from_path(out: Path) -> int:
    """``.../tasks/<Task>/r<k>/try<n>`` -> k."""
    name = out.parent.name
    return int(name[1:]) if name.startswith("r") and name[1:].isdigit() else 0


if __name__ == "__main__":
    sys.exit(main())
