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

"""Fake RC-ECC stage-3 episode: an RPent evaluation folder with RoboCasa-shaped turns.

Called like ``hybrid_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR`` with the runner's
``ECC_RC_*`` inputs. No model, simulator or network. Every request goes through
RPent's real ``EscalationModel`` with scripted worker and senior functions, so
``escalation.jsonl`` is what the wrapper writes for the rule and the corrections
file; the transcript, ``run.log`` (api_loop's format) and ``states.json`` hold the
same turns, with tool results shaped like RoboCasa's (the robot outcome nested in
``log.result``, behind the state).

The episode's behaviour comes from ``ECC_RC_FAKE_SCENARIO``:
``{task: {"<episode>": spec | [spec per try], "default": spec}}`` (the episode is
the output folder's parent, e.g. ``s1_a2``), with

- ``kind``: ``solve`` | ``fail`` | ``timeout`` (planner timeout: the transcript keeps
  only the first user message, run.log keeps the turns) | ``no_turns`` (a timeout
  before any turn) | ``infra`` (invalid result.json) | ``noise`` (noise frames);
- ``requests`` (default 9); ``bug``: wrapper settings that differ from the launch;
- ``peek``: the worker reads the corrections snapshot; ``sleep_s``: wall time at the end.

Each start is logged to ``ECC_RC_FAKE_LOG`` with what the worker and the senior saw.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_cell import png, write_result  # noqa: E402

MARK = "Cloud correction memory for this task"
LANGUAGE = "Close the toaster oven door."


def state_of(step: int) -> dict:
    return {
        "robot0_eef_pos": [
            3.4571986674796 - 0.01 * step,
            -0.5736623401506386,
            1.30081979217628,
        ],
        "robot0_eef_quat": [
            0.6660979630503402,
            0.7322479649835105,
            0.0931107074432701,
            -0.10703652440689523,
        ],
        "robot0_gripper_qpos": [0.02055052244133506, -0.020516032054768756],
        "robot0_base_pos": [
            3.4608176293627717,
            -0.8228094652361997,
            0.7000000383350744,
        ],
        "robot0_base_quat": [0.0, 0.0, 0.7071068286895752, 0.7071066498756409],
    }


def robot_result(
    step: int, *, success: bool, command: dict | None, result: dict | None
) -> dict:
    """``view_env_state`` of a recorded step (robots/robocasa/tools.py), as a tool returns it."""
    return {
        "step": step,
        "task_progress": {
            "door_closed": success,
            "door_angle": 0.0 if success else 0.9,
        },
        "task_language": LANGUAGE,
        "state": state_of(step),
        "robocasa_terminated": success,
        "vla_desync": False,
        "success": success,
        "log": {
            "command": command,
            "result": result,
            "elapsed_s": 1.0 if command else None,
        },
        "images": [
            {
                "role": "calibration_frame",
                "camera": "agentview",
                "artifact": "agentview.png",
            }
        ],
    }


def main() -> int:
    split, task, seed, out, memory = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        Path(sys.argv[4]),
        Path(sys.argv[5]),
    )
    n_try = int(os.environ.get("ECC_RC_TRY", "1"))
    episode = out.parent.name
    scenario = json.loads(Path(os.environ["ECC_RC_FAKE_SCENARIO"]).read_text())
    by_task = scenario.get(task, {})
    spec = by_task.get(episode, by_task.get("default", {"kind": "fail"}))
    specs = spec if isinstance(spec, list) else [spec]
    spec = specs[min(n_try, len(specs)) - 1]
    kind = spec.get("kind", "fail")
    corrections_path = os.environ.get("ECC_RC_CORRECTIONS") or ""
    corrections = Path(corrections_path).read_text() if corrections_path else ""
    seen = {"senior_saw": [], "worker_saw": False, "senior_turns": 0}

    def event(name: str, **extra) -> None:
        if os.environ.get("ECC_RC_FAKE_LOG"):
            with open(os.environ["ECC_RC_FAKE_LOG"], "a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": name,
                            "task": task,
                            "episode": episode,
                            "try": n_try,
                            "pid": os.getpid(),
                            "t": time.time(),
                            "seed": seed,
                            "split": split,
                            "memory": str(memory),
                            "plan": os.environ.get("ECC_RC_PLAN"),
                            "checkin": os.environ.get("ECC_RC_CHECKIN"),
                            "errors": os.environ.get("ECC_RC_ERRORS"),
                            "timeout": os.environ.get("ECC_RC_TIMEOUT_S"),
                            "corrections": corrections_path,
                            "corrections_sha256": hashlib.sha256(
                                corrections.encode()
                            ).hexdigest(),
                            **extra,
                        }
                    )
                    + "\n"
                )

    try:
        os.listdir(out.parent)
        listable = True
    except PermissionError:
        listable = False
    event("start", episode_dir_listable=listable)
    out.mkdir(parents=True, exist_ok=True)
    (out / "env_server.log").write_text("RPC server listening on http://127.0.0.1:0\n")
    started = time.time()
    if kind == "noise":
        png(out / "agentview.png" / "00.png", noise=True)
        event("noise_written")
        time.sleep(60)
        return 1
    png(out / "agentview.png" / "00.png", noise=False)

    # The code under test (the runner passes its checkout), not an installed RPent.
    sys.path.insert(0, os.environ["ECC_RC_CODE_DIR"])
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        ToolCallPart,
        ToolReturnPart,
        UserPromptPart,
    )
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.function import FunctionModel

    from rpent.planner.escalation import EscalationModel

    requests = 0 if kind == "no_turns" else int(spec.get("requests", 9))
    solve = kind == "solve"

    def notes(messages) -> str:
        last = messages[-1]
        return " ".join(
            p.content
            for p in getattr(last, "parts", [])
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )

    def worker(messages, info):
        if MARK in notes(messages):
            seen["worker_saw"] = True
        n = sum(isinstance(m, ModelResponse) for m in messages) + 1
        if n == 1:
            return ModelResponse(
                parts=[
                    TextPart("Reading the initial state."),
                    ToolCallPart("view_env_state", {"step": 0}),
                ]
            )
        if n == 2:
            # ``peek``: the worker reads the senior's corrections snapshot by its path.
            path = (
                str(out.parent / "corrections_used.md")
                if spec.get("peek")
                else f"{memory}/task-specific/{task}.md"
            )
            return ModelResponse(parts=[ToolCallPart("read_text_file", {"path": path})])
        if solve and n == requests:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "finish",
                        {"status": "success", "summary": "Door closed by rldx_arm."},
                    )
                ]
            )
        return ModelResponse(
            parts=[
                TextPart("Trying the full-task skill."),
                ToolCallPart("rldx_arm", {"prompt": LANGUAGE}),
            ]
        )

    def senior(messages, info):
        text = notes(messages)
        seen["senior_turns"] += 1
        if MARK not in text:
            seen["senior_saw"].append("none")
        else:
            seen["senior_saw"].append(
                "match"
                if corrections.strip() and corrections.strip() in text
                else "mismatch"
            )
        return ModelResponse(
            parts=[
                TextPart(
                    "Push the door from below its free edge, then re-check task_progress."
                ),
                ToolCallPart("rldx_arm", {"prompt": LANGUAGE}),
            ]
        )

    settings = {
        "plan": os.environ["ECC_RC_PLAN"] == "1",
        "checkin": int(os.environ["ECC_RC_CHECKIN"]),
        "errors": int(os.environ["ECC_RC_ERRORS"]),
        "max_share": float(os.environ["ECC_RC_MAX_SHARE"]),
        **spec.get("bug", {}),
    }
    model = EscalationModel(
        FunctionModel(
            worker, model_name=os.environ["ECC_RC_WORKER_MODEL"].split(":", 1)[-1]
        ),
        FunctionModel(
            senior, model_name=os.environ["ECC_RC_SENIOR_MODEL"].split(":", 1)[-1]
        ),
        checkin=settings["checkin"],
        max_share=settings["max_share"],
        plan=settings["plan"],
        errors=settings["errors"],
        corrections=corrections,
        log_path=out / "escalation.jsonl",
        senior_max_images=40,
    )
    history = [ModelRequest.user_text_prompt("Do the task.")]
    transcript = [{"role": "user", "content": "Task: fake evaluation episode."}]
    steps = [
        {
            "step_idx": 0,
            "state": state_of(0),
            "terminated": False,
            "truncated": False,
            "artifacts": ["agentview.png"],
            "extras": {
                "task_language": LANGUAGE,
                "success": False,
                "task_progress": {"door_closed": False},
                "vla_desync": True,
            },
        }
    ]
    log = (out / "run.log").open("w")

    def logline(text: str) -> None:
        log.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} I [api_loop] {text}\n")

    finish = None
    for i in range(1, requests + 1):
        response = asyncio.run(model.request(history, None, ModelRequestParameters()))
        history.append(response)
        logline(f"=== turn {i}/100 ===")
        content, tools, returns = [], [], []
        for part in response.parts:
            if isinstance(part, TextPart) and part.content:
                content.append({"type": "text", "text": part.content})
                logline(f"[model] {part.content}")
        for part in response.parts:
            if not isinstance(part, ToolCallPart):
                continue
            call_id = f"call_{i}_{part.tool_name}"
            args = part.args_as_dict()
            content.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": part.tool_name,
                    "input": args,
                }
            )
            logline(f"[tool>] {part.tool_name}({json.dumps(args)[:250]})")
            if part.tool_name == "rldx_arm":
                done = solve and i >= requests - 1
                result = {
                    "ok": True,
                    "prompt": LANGUAGE,
                    "status": "success" if done else "cap",
                    "chunks": 40,
                    "steps_applied": 320,
                    "grasped": False,
                    "gripper_qpos": 0.04,
                    "peak_lift": 0.2,
                }
                command = {"action": "rldx_arm", "prompt": LANGUAGE}
                steps.append(
                    {
                        "step_idx": len(steps),
                        "state": state_of(len(steps)),
                        "terminated": done,
                        "truncated": False,
                        "artifacts": ["agentview.png"],
                        "command": command,
                        "result": result,
                        "elapsed_s": 1.0,
                        "extras": {
                            "task_language": LANGUAGE,
                            "success": done,
                            "task_progress": {
                                "door_closed": done,
                                "door_angle": 0.0 if done else 0.9,
                            },
                            "vla_desync": False,
                        },
                    }
                )
                payload = robot_result(
                    len(steps) - 1, success=done, command=command, result=result
                )
            elif part.tool_name == "view_env_state":
                payload = robot_result(0, success=False, command=None, result=None)
            elif part.tool_name == "read_text_file":
                payload = {
                    "path": args.get("path"),
                    "size": 12,
                    "content": "# fake note\n",
                }
            elif part.tool_name == "finish":
                payload = finish = {"_finish": True, **args}
            else:
                payload = {"ok": True}
            text = json.dumps(payload, indent=2)
            returns.append(ToolReturnPart(part.tool_name, [text], tool_call_id=call_id))
            tools.append(
                {
                    "role": "tool",
                    "name": part.tool_name,
                    "tool_call_id": call_id,
                    "content": text,
                }
            )
        logline(
            f"[usage] in={1000 * i} out={50 * i} cache_read=0 cache_write=0 requests={i}"
        )
        transcript.append({"role": "assistant", "content": content})
        for tool in tools:
            logline(
                f"[tool<] {tool['name']}: {' '.join(tool['content'].split())[:350]}"
            )
            transcript.append(tool)
        history.append(ModelRequest(parts=returns or [UserPromptPart("continue")]))
        if finish:
            logline(f"FINISH called: {finish!r}")
            break
    log.close()
    tag = f"{task}_target_s{seed}"
    (out / "states.json").write_text(
        json.dumps({"run_artifacts": [], "steps": steps}, indent=2)
    )
    timed_out = kind in ("timeout", "no_turns")
    (out / f"transcript_{tag}.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "split": "target",
                "seed": seed,
                "model": "fake",
                "elapsed_s": 1.0,
                "finish": None
                if timed_out
                else (
                    finish
                    or {"_finish": True, "status": "failure", "summary": "Gave up."}
                ),
                "stats": {} if timed_out else {"turns_used": requests},
                "messages": transcript[:1] if timed_out else transcript,
            },
            indent=2,
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
    elif timed_out:
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
            success=solve,
            reason="completed",
            elapsed=elapsed,
        )
    time.sleep(float(spec.get("sleep_s", 0.05)))
    event(
        "end",
        kind=kind,
        senior_saw=seen["senior_saw"],
        worker_saw=seen["worker_saw"],
        senior_turns=seen["senior_turns"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
