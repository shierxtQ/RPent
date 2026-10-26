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

"""Fake final-test episode launcher (a script; pytest collects no test from this module).

Called like ``hybrid_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR`` with the runner's
``ECC_RC_*`` inputs. No model, simulator or network. Unless ``ECC_RC_ESCALATION=0``,
the requests go through RPent's real ``EscalationModel`` with scripted worker and
senior functions, so ``escalation.jsonl`` is what the wrapper writes for the plan
switch, K, N and corrections the launch received; a transcript with the same tool
results is written next to it. With ``ECC_RC_ESCALATION=0`` (Qwen alone) only
``run.log`` usage lines are written, as RPent without the wrapper does.

The output folder ``.../cells/<arm>__<Task>_s<seed>_r<rep>/try<n>`` names the job.
The scenario (JSON file in ``ECC_RC_FAKE_SCENARIO``) maps the job id, then
``<arm>:<Task>``, then ``<arm>``, then ``default`` to a spec or a list of specs (one
per try):

- ``kind``: ``solve`` | ``fail`` | ``timeout`` | ``infra`` | ``crash`` | ``noise`` | ``hang``;
- ``requests`` (default 12), ``error_results`` (request numbers whose tool result has
  a top-level ``error``), ``retry_at`` (request numbers whose call fails argument
  validation), ``giveup_at`` (requests where the worker proposes to give up);
- ``bug``: wrapper settings that differ from the launch, e.g. ``{"checkin": 7}``;
- ``keep_senior``: build the wrapper even for ``ECC_RC_ESCALATION=0`` (a launcher
  without the switch); ``sleep_s``: wall time; ``tokens``: ``[in, out]`` per request.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path


def spec_for(scenario: dict, job: str, arm: str, task: str, n_try: int) -> dict:
    for key in (job, f"{arm}:{task}", arm, "default"):
        if key in scenario:
            spec = scenario[key]
            break
    else:
        spec = {"kind": "fail"}
    specs = spec if isinstance(spec, list) else [spec]
    return dict(specs[min(n_try, len(specs)) - 1])


def drive(out: Path, spec: dict, settings: dict) -> list[dict]:
    """Requests through the real escalation wrapper; returns the transcript messages."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        ToolCallPart,
        ToolReturnPart,
    )
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel

    from rpent.planner.escalation import EscalationModel

    giveups = set(spec.get("giveup_at", []))
    errors = set(spec.get("error_results", []))
    retries = set(spec.get("retry_at", []))
    current = {"n": 0}

    def worker(messages, info):
        if current["n"] in giveups:
            return ModelResponse(
                parts=[
                    ToolCallPart("finish", {"status": "failure", "summary": "stuck"})
                ]
            )
        if current["n"] == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart("view_env_state", {}),
                    ToolCallPart("read_text_file", {"path": "m.md"}),
                ]
            )
        return ModelResponse(parts=[ToolCallPart("rldx_arm", {"prompt": "Do it."})])

    def senior(messages, info):
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
    history = [ModelRequest.user_text_prompt("Do the task.")]
    transcript: list[dict] = [{"role": "user", "content": "Do the task."}]
    for i in range(1, int(spec.get("requests", 12)) + 1):
        current["n"] = i
        response = asyncio.run(model.request(history, None, ModelRequestParameters()))
        calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
        transcript.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"c{i}_{j}",
                        "name": c.tool_name,
                        "input": c.args_as_dict(),
                    }
                    for j, c in enumerate(calls)
                ],
            }
        )
        history.append(response)
        parts = []
        for j, c in enumerate(calls):
            call_id = f"c{i}_{j}"
            if i in retries:
                details = [
                    {
                        "type": "json_invalid",
                        "loc": [],
                        "msg": "Invalid JSON",
                        "input": "{",
                    }
                ]
                parts.append(
                    RetryPromptPart(
                        content=details, tool_name=c.tool_name, tool_call_id=call_id
                    )
                )
                content = json.dumps(details)
            else:
                payload = (
                    {"step": i, "error": "bad step"}
                    if i in errors
                    else {"step": i, "log": {"result": {"ok": True}}}
                )
                parts.append(
                    ToolReturnPart(
                        c.tool_name, [json.dumps(payload)], tool_call_id=call_id
                    )
                )
                content = json.dumps(payload)
            transcript.append(
                {
                    "role": "tool",
                    "name": c.tool_name,
                    "tool_call_id": call_id,
                    "content": content,
                }
            )
        history.append(ModelRequest(parts=parts))
    return transcript


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "fakes"))
    from fake_cell import png, write_result

    sys.path.insert(
        0, os.environ["ECC_RC_CODE_DIR"]
    )  # rpent from the checkout under test
    split, task, seed, out, memory = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        Path(sys.argv[4]),
        Path(sys.argv[5]),
    )
    job = out.parent.name
    arm = job.split("__", 1)[0]
    n_try = int(os.environ.get("ECC_RC_TRY", "1"))
    scenario = json.loads(Path(os.environ["ECC_RC_FAKE_SCENARIO"]).read_text())
    spec = spec_for(scenario, job, arm, task, n_try)
    kind = spec.get("kind", "fail")
    senior_on = os.environ.get("ECC_RC_ESCALATION", "1") != "0" or spec.get(
        "keep_senior"
    )
    corrections_path = os.environ.get("ECC_RC_CORRECTIONS") or ""
    corrections = Path(corrections_path).read_text() if corrections_path else ""
    if corrections_path and os.environ.get("ECC_RC_CORRECTIONS_SHA256"):
        import hashlib

        if (
            hashlib.sha256(Path(corrections_path).read_bytes()).hexdigest()
            != os.environ["ECC_RC_CORRECTIONS_SHA256"]
        ):
            print("fake cell: corrections sha256 mismatch", file=sys.stderr)
            return 2
    if fake_log := os.environ.get("ECC_RC_FAKE_LOG"):
        with open(fake_log, "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "start",
                        "job": job,
                        "arm": arm,
                        "task": task,
                        "seed": seed,
                        "split": split,
                        "try": n_try,
                        "pid": os.getpid(),
                        "t": time.time(),
                        "memory": str(memory),
                        "memory_writable": os.access(memory, os.W_OK),
                        "env": {
                            k: os.environ.get(k)
                            for k in (
                                "ECC_RC_ESCALATION",
                                "ECC_RC_PLAN",
                                "ECC_RC_CHECKIN",
                                "ECC_RC_ERRORS",
                                "ECC_RC_CORRECTIONS",
                                "ECC_RC_CORRECTIONS_SHA256",
                                "ECC_RC_TIMEOUT_S",
                                "ECC_RC_MAX_TURNS",
                                "ECC_RC_MAX_SHARE",
                                "RC_GPU",
                                "VLA_ENDPOINT",
                                "RC_RENDER",
                            )
                        },
                    }
                )
                + "\n"
            )
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if kind == "crash":
        return 1
    tag = f"{task}_target_s{seed}"
    requests = int(spec.get("requests", 12))
    transcript = [{"role": "user", "content": "Do the task."}]
    if senior_on:
        settings = {
            "plan": os.environ["ECC_RC_PLAN"] == "1",
            "checkin": int(os.environ["ECC_RC_CHECKIN"]),
            "errors": int(os.environ["ECC_RC_ERRORS"]),
            "max_share": float(os.environ["ECC_RC_MAX_SHARE"]),
            "corrections": corrections,
            "worker": os.environ["ECC_RC_WORKER_MODEL"].split(":", 1)[-1],
            "senior": os.environ["ECC_RC_SENIOR_MODEL"].split(":", 1)[-1],
            **spec.get("bug", {}),
        }
        transcript = drive(out, spec, settings)
    tokens_in, tokens_out = spec.get("tokens", [1000, 50])
    with (out / "run.log").open("w") as log:
        for i in range(1, requests + 1):
            log.write(f"2026-10-27 00:00:00 I [api_loop] === turn {i}/100 ===\n")
            log.write(
                f"2026-10-27 00:00:00 I [api_loop] [usage] in={tokens_in * i} out={tokens_out * i} cache_read=0 "
                f"cache_write=0 requests={i}\n"
            )
    png(out / "agentview.png" / "00.png", noise=(kind == "noise"))
    if kind == "noise":
        time.sleep(60)  # the runner's render check stops this episode
        return 1
    time.sleep(float(spec.get("sleep_s", 0.05)))
    solved = kind in ("solve", "hang")
    timeout = kind == "timeout"
    (out / f"transcript_{tag}.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "split": "target",
                "seed": seed,
                "finish": None
                if timeout
                else {"status": "success" if solved else "failure", "summary": "fake"},
                "stats": {} if timeout else {"turns_used": requests},
                # a planner timeout keeps only the first user message
                "messages": transcript[:1] if timeout else transcript,
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
    elif timeout:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
