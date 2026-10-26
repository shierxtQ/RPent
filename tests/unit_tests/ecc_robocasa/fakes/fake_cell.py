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

"""Fake RC-ECC exploration cell: writes the folder an RPent ``--explore`` run leaves.

Called like ``explore_cell.sh SPLIT TASK SEED OUT_DIR MEMORY_DIR``. No model, no
simulator, no network. The behaviour per task and try comes from the JSON file in
``ECC_RC_FAKE_SCENARIO``: ``{task: spec | [spec for try 1, try 2, ...]}`` with

- ``kind``: ``solve`` | ``fail`` | ``infra`` (invalid result.json) | ``crash`` (no
  result.json) | ``noise`` (noise frames, then waits to be stopped) | ``hang``;
- ``solve_session`` / ``solve_attempt``: where a ``solve`` succeeds;
- ``audit`` (default true), ``family`` (default true), ``family_scope``
  (``task-family``, or ``suite`` as in the cloud task notes), ``global`` (draft ids on
  success), ``global_any`` (draft ids either way),
  ``family_task`` (a task-family draft that names another task);
- ``sleep_s`` per session, ``tokens`` per turn ``[in, out]``, ``turns`` per attempt,
  ``read_paths`` / ``write_paths`` (file-tool calls logged like RPent's api
  loop; writes really happen).

The shapes follow a real Target50 episode folder (``result.json`` schema 1.1,
``states.json`` steps, ``agentview.png/NN.png`` frames, ``run.log`` lines) and the
explore layout of ``rpent/cli/main.py`` (``sessions/session_00N``, ``attempts/``,
the audit and recipe named after the cell tag, inbox drafts under
``<memory>/_internal/inbox/<tag>``).
"""

from __future__ import annotations

import json
import os
import random
import struct
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path


def png(path: Path, *, noise: bool, size: int = 48) -> None:
    rows = []
    rng = random.Random(7)
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            if noise:
                row += bytes(rng.randrange(256) for _ in range(3))
            else:
                row += bytes((80 + x // 4, 90 + y // 4, 100))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def log_line(handle, text: str) -> None:
    handle.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} I [api_loop] {text}\n")
    handle.flush()


def step(index: int, *, success: bool, command=None, result=None) -> dict:
    record = {
        "step_idx": index,
        "state": {
            "robot0_eef_pos": [3.4, -0.5, 1.3],
            "robot0_gripper_qpos": [0.02, -0.02],
        },
        "terminated": success,
        "truncated": False,
        "artifacts": ["agentview.png", "navview.png", "wrist.png"],
        "extras": {
            "task_language": "Do the task.",
            "success": success,
            "task_progress": {},
            "vla_desync": False,
        },
    }
    if command is not None:
        record["command"] = command
        record["result"] = result
        record["elapsed_s"] = 1.0
    return record


def write_result(
    out: Path,
    task: str,
    seed: int,
    *,
    valid: bool,
    success: bool,
    reason: str,
    elapsed: float,
) -> None:
    (out / "result.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "protocol_id": None,
                "evaluation_split": None,
                "task_name": task,
                "environment_split": "target",
                "seed": seed,
                "valid": valid,
                "success": success,
                "success_source": "state.success",
                "termination_reason": reason,
                "elapsed_s": round(elapsed, 1),
                "planner": {
                    "backend": "api",
                    "model": "openai-chat:fake",
                    "reasoning_effort": "none",
                    "max_turns": 100,
                },
                "runtime": {
                    "cell_timeout_seconds": 2400,
                    "rldx_max_chunks": 40,
                    "rldx_settle_patience": 999,
                    "rldx_action_steps_per_chunk": 8,
                },
                "memory": {"audit_status": "missing"},
            },
            indent=2,
        )
        + "\n"
    )


def draft(
    scope: str, tag: str, task: str, name: str, family_scope: str = "task-family"
) -> str:
    evidence = f"evidence:\n  cells: [{tag}]\n  attempts: 3\n  solved_seeds: [0]\n  failed_seeds: []\nconfidence: single-shot\n"
    if scope == "task-family":
        # family_scope "suite" copies the cloud task notes' legacy frontmatter.
        head = f"scope: {family_scope}\nsuite: robocasa\nregime: target\ntask_id: {task}\ntask_language: Do {task}.\n"
    else:
        head = f"scope: global\nkind: strategy\ntitle: Fake lesson {name}\napplies_when: offline tests only\n"
    return f"---\n{head}{evidence}---\nSynthetic note written by the fake cell; no robot was run.\n"


def main() -> int:
    task, seed, out, memory = (
        sys.argv[2],
        int(sys.argv[3]),
        Path(sys.argv[4]),
        Path(sys.argv[5]),
    )
    n_try = int(os.environ.get("ECC_RC_TRY", "1"))
    scenario = json.loads(Path(os.environ["ECC_RC_FAKE_SCENARIO"]).read_text())
    specs = scenario.get(task, {"kind": "fail"})
    specs = specs if isinstance(specs, list) else [specs]
    spec = specs[min(n_try, len(specs)) - 1]
    kind = spec.get("kind", "fail")
    sessions = int(os.environ.get("ECC_RC_SESSIONS", "5"))
    attempts = int(os.environ.get("ECC_RC_ATTEMPTS", "3"))
    sleep_s = float(spec.get("sleep_s", 0.05))
    tokens_in, tokens_out = spec.get("tokens", [1000, 50])
    turns = int(spec.get("turns", 2))
    tag = f"{task}_target_s{seed}"
    fake_log = os.environ.get("ECC_RC_FAKE_LOG")

    def event(name: str, **extra) -> None:
        if fake_log:
            with open(fake_log, "a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": name,
                            "task": task,
                            "try": n_try,
                            "pid": os.getpid(),
                            "t": time.time(),
                            "gpu": os.environ.get("RC_GPU"),
                            "endpoint": os.environ.get("VLA_ENDPOINT"),
                            **extra,
                        }
                    )
                    + "\n"
                )

    event(
        "start",
        env={
            k: os.environ.get(k)
            for k in (
                "ECC_RC_SESSIONS",
                "ECC_RC_ATTEMPTS",
                "ECC_RC_SESSION_TIMEOUT_S",
                "ECC_RC_MAX_TURNS",
                "RC_RENDER",
                "ECC_RC_CRED_FILE",
                "ECC_RC_WORKER_MODEL",
            )
        },
        argv=sys.argv[1:],
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "env_server.log").write_text("RPC server listening on http://127.0.0.1:0\n")
    inbox = memory / "_internal" / "inbox" / tag
    (inbox / "wip").mkdir(parents=True, exist_ok=True)
    started = time.time()
    with (out / "run.log").open("a") as log:
        log.write(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} I [agent] physical agent cmd: fake rpent --robot robocasa "
            f"--task-name {task} --seed {seed} --explore --output-dir {out}\n"
        )
        if kind == "crash":
            time.sleep(sleep_s)
            event("end", kind=kind)
            return 1
        messages = [{"role": "user", "content": "explore"}]
        solve_session = (
            int(spec.get("solve_session", 1)) if kind in ("solve", "hang") else None
        )
        solve_attempt = int(spec.get("solve_attempt", 1))
        last_session = solve_session or sessions
        if kind == "infra":
            last_session = 1
        for s in range(1, last_session + 1):
            if s > 1:
                log.write(
                    f"{datetime.now():%Y-%m-%d %H:%M:%S} I [agent] === handing off to agent {s}/{sessions} ===\n"
                )
            session_dir = out / "sessions" / f"session_{s:03d}"
            steps = [step(0, success=False)]
            png(session_dir / "agentview.png" / "00.png", noise=(kind == "noise"))
            n_attempts = solve_attempt if s == solve_session else attempts
            requests = 0
            for a in range(1, n_attempts + 1):
                if a > 1:
                    steps.append(
                        step(
                            len(steps),
                            success=False,
                            command={"action": "reset"},
                            result={
                                "ok": True,
                                "reset": True,
                                "attempt": a,
                                "notice": "Fresh episode started.",
                            },
                        )
                    )
                for _ in range(turns):
                    requests += 1
                    log_line(log, f"=== turn {requests}/100 ===")
                    for path in spec.get("read_paths", []):
                        log_line(
                            log, f"[tool>] read_text_file({json.dumps({'path': path})})"
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "id": f"call_{requests}",
                                        "name": "read_text_file",
                                        "input": {"path": path},
                                    }
                                ],
                            }
                        )
                    for path in spec.get("write_paths", []):
                        log_line(
                            log,
                            f"[tool>] write_text_file({json.dumps({'path': path, 'content': 'x'})})",
                        )
                        target = Path(path)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(draft("global", tag, task, "intruder"))
                    log_line(
                        log,
                        f"[usage] in={tokens_in * requests} out={tokens_out * requests} cache_read=0 "
                        f"cache_write=0 requests={requests}",
                    )
                solved_here = s == solve_session and a == n_attempts
                steps.append(
                    step(
                        len(steps),
                        success=solved_here,
                        command={"action": "rldx_arm", "prompt": "Do the task."},
                        result={
                            "ok": True,
                            "status": "success" if solved_here else "cap",
                        },
                    )
                )
                png(
                    session_dir / "agentview.png" / f"{len(steps) - 1:02d}.png",
                    noise=(kind == "noise"),
                )
                if not solved_here:
                    (out / "attempts").mkdir(exist_ok=True)
                    (out / "attempts" / f"attempt_{s}_{a}_failed.json").write_text(
                        json.dumps({"attempt": a})
                    )
                    with (inbox / "wip" / "notes.md").open("a") as notes:
                        notes.write(f"- session {s} attempt {a}: the push stalled\n")
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "states.json").write_text(
                json.dumps({"run_artifacts": [], "steps": steps}, indent=2)
            )
            if kind == "noise":
                event("noise_written")
                time.sleep(60)  # the runner's render check stops this cell
                return 1
            time.sleep(sleep_s)
            if kind == "infra":
                break
        solved = kind in ("solve", "hang")
        if solved:
            (out / f"{tag}_recipe.jsonl").write_text(
                '{"action": "rldx_arm", "prompt": "Do the task."}\n'
            )
            log.write(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} I [agent] recipe: {tag}_recipe.jsonl\n"
            )
            if spec.get("audit", True):
                (out / f"{tag}.json").write_text(
                    json.dumps({"task": task, "seed": seed, "status": "success"}) + "\n"
                )
            if spec.get("family", True):
                (inbox / "family.md").write_text(
                    draft(
                        "task-family",
                        tag,
                        task,
                        "f",
                        spec.get("family_scope", "task-family"),
                    )
                )
            if spec.get("family_task"):
                (inbox / "other_family.md").write_text(
                    draft("task-family", tag, spec["family_task"], "x")
                )
            for name in spec.get("global", []):
                (inbox / f"{name}.md").write_text(draft("global", tag, task, name))
        else:
            log.write(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} I [agent] recipe: not written (cell unsolved)\n"
            )
        for name in spec.get(
            "global_any", []
        ):  # global drafts written whether or not the cell solved
            (inbox / f"{name}.md").write_text(draft("global", tag, task, name))
    (out / f"transcript_{tag}.json").write_text(
        json.dumps(
            {
                "task_name": task,
                "split": "target",
                "seed": seed,
                "model": "fake",
                "elapsed_s": 1.0,
                "finish": None,
                "stats": {},
                "messages": messages,
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
    elif kind == "fail" and spec.get("timeout", True):
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
    event("end", kind=kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())
