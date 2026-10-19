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

"""Escalation senior turns through the official Codex CLI (opt-in).

Some Responses gateways keep a conversation on one cache-holding backend only for the official Codex client: on the
Sorux gpt-6-astra gateway, streamed plain API requests were 0-31% cached per episode, while a resumed Codex CLI thread
was 80-99% cached.  ``RPENT_ESCALATION_SENIOR_BACKEND=codex`` makes the escalation senior a :class:`CodexSenior`:

- one Codex thread per episode, append-only, so its prefix stays cacheable;
- each senior request sends only the messages since the senior's previous turn, rendered as text, with the new
  camera images attached (``--image``); the first one also carries the system prompt and the tool list;
- the reply comes back under an output schema as ``{plan, tool_calls[{tool, arguments_json}]}`` and becomes a
  ``TextPart`` plus ``ToolCallPart``s, so the tools run in the shared pydantic-ai loop as before and the worker
  side is unchanged;
- Codex's own tools are disabled and it runs in an empty directory; a reply that cannot be used gets one
  correction turn, a failed Codex process one retry, and then :class:`SeniorUnavailable` makes the worker take
  the request (as after a senior timeout).

``codex exec`` reports token usage cumulatively per thread; each turn's own usage is the difference.  Environment:
``RPENT_CODEX_BIN`` (the ``codex`` binary of ``openai-codex-cli-bin``), ``RPENT_CODEX_TIMEOUT_S`` (per Codex process,
default 180).  ``CODEX_HOME`` is ``<output_dir>/codex_home`` (provider config written at start, mode 600; the key is
read from the environment through ``env_key``); what each turn sent is kept under ``<output_dir>/codex_turns``.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import signal
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

PROVIDER_ID = "rpent_senior"
# Codex's own tools and integrations; the senior only answers in JSON.
DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "browser_use",
    "computer_use",
    "image_generation",
    "multi_agent",
    "in_app_browser",
    "view_image",
    "sleep_tool",
    "tool_suggest",
    "skill_search",
)
# The thread keeps every attached image and resends it with each turn (the API loop's history keeps at most 4 MiB),
# and a slow uplink adds seconds per MB: the PNGs of one long episode reached 17 MB and a senior turn 75 s on a
# Jetson Orin.  Images are re-encoded as JPEG at full resolution (same pixels and image tokens, ~16% of the PNG bytes) and
# attached newest first within 2 MiB per turn and 8 MiB per thread; the newest two are always attached.
JPEG_QUALITY = 90
TURN_IMAGE_BYTES = 2 * 1024 * 1024
THREAD_IMAGE_BYTES = 8 * 1024 * 1024
MIN_IMAGES = 2
RETRY_WAIT_S = 15.0
CHILD_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
    "all_proxy",
    "ALL_PROXY",
)
CONTRACT = (
    "You are the senior planner of a shared robot episode. A faster worker model makes most requests; the harness "
    "gives you single turns (the plan, periodic check-ins, reviews of give-ups and similar events). This "
    "conversation only carries the episode transcript: do not run commands, read or edit files, or use any tool of "
    "your own. Every reply is one JSON object of the output schema: `plan` is your message to the worker (shown to "
    "it as senior guidance) and `tool_calls` are the robot tool calls to make next, each with `tool` and "
    "`arguments_json` (a JSON object, as a string, matching that tool's parameters). The harness runs your calls "
    "and shows you their results, with the worker's later turns, the next time you are asked. Attached images are "
    "numbered in the order they are mentioned in that turn's text. The tools run on the robot host (Linux), not where "
    "this conversation runs: use file paths exactly as the system prompt and the transcript give them (absolute "
    "/home/... paths or paths relative to the repository root), never paths of your own environment."
)

Runner = Callable[
    [list[str], str, dict[str, str], Path, float], Awaitable[tuple[int, str, str]]
]


class SeniorUnavailable(RuntimeError):
    """The senior gave no usable reply; the worker should take this request.

    ``fields`` are added to the failed call's log row (tokens the failed attempts used, backend details).
    """

    def __init__(self, message: str, fields: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.fields = fields or {}


async def run_process(
    args: list[str], stdin: str, env: dict[str, str], cwd: Path, timeout_s: float
) -> tuple[int, str, str]:
    """Run one Codex process; on timeout kill its process group and return code -9."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(cwd),
        start_new_session=True,
    )
    try:
        out, err = await asyncio.wait_for(
            proc.communicate(stdin.encode()), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = await proc.communicate()
        return (
            -9,
            out.decode(errors="replace"),
            err.decode(errors="replace") + f"\ntimeout after {timeout_s:g} s",
        )
    return (
        proc.returncode or 0,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
    )


@dataclass
class _Turn:
    """What one Codex process reported."""

    thread_id: str | None = None
    usage: dict[str, int] | None = None
    reply: str | None = None
    items: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_events(stdout: str) -> _Turn:
    turn = _Turn()
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        turn.thread_id = event.get("thread_id") or turn.thread_id
        kind = event.get("type")
        if kind == "turn.completed" and isinstance(event.get("usage"), dict):
            turn.usage = {
                k: int(v) for k, v in event["usage"].items() if isinstance(v, int)
            }
        elif kind == "item.completed":
            item = event.get("item") or {}
            turn.items.append(str(item.get("type")))
            if item.get("type") == "agent_message":
                turn.reply = item.get("text")
        elif kind in ("error", "turn.failed"):
            detail = event.get("message") or event.get("error")
            turn.errors.append(
                str(detail.get("message") if isinstance(detail, dict) else detail)[:500]
            )
    return turn


def output_schema(tool_names: list[str]) -> dict[str, Any]:
    call = {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool", "arguments_json"],
        "properties": {
            "tool": {"type": "string", "enum": tool_names},
            "arguments_json": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["plan", "tool_calls"],
        "properties": {
            "plan": {"type": "string"},
            "tool_calls": {"type": "array", "items": call},
        },
    }


def parse_reply(
    reply: str | None, tool_names: set[str]
) -> tuple[str, list[tuple[str, str]]]:
    """(plan, [(tool, arguments as a JSON object string)]); ValueError says why the reply cannot be used."""
    try:
        data = json.loads(reply or "")
    except ValueError:
        raise ValueError("the reply was not a JSON object") from None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("plan"), str)
        or not isinstance(data.get("tool_calls"), list)
    ):
        raise ValueError("the reply needs a string `plan` and a list `tool_calls`")
    calls = []
    for index, call in enumerate(data["tool_calls"], 1):
        tool = call.get("tool") if isinstance(call, dict) else None
        if tool not in tool_names:
            raise ValueError(f"tool call {index} names an unknown tool {tool!r}")
        try:
            args = json.loads(call.get("arguments_json") or "{}")
        except ValueError:
            raise ValueError(
                f"arguments_json of tool call {index} ({tool}) is not valid JSON"
            ) from None
        if not isinstance(args, dict):
            raise ValueError(
                f"arguments_json of tool call {index} ({tool}) is not a JSON object"
            )
        calls.append((tool, json.dumps(args, ensure_ascii=False)))
    return data["plan"], calls


def _image_suffix(media_type: str) -> str:
    return {"image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(
        media_type, ".png"
    )


def compact_image(image: BinaryContent) -> BinaryContent:
    """The image as a full-resolution JPEG (see JPEG_QUALITY); unchanged when it cannot be decoded."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image.data)) as decoded:
            out = io.BytesIO()
            decoded.convert("RGB").save(out, "JPEG", quality=JPEG_QUALITY)
    except Exception:
        return image
    return BinaryContent(data=out.getvalue(), media_type="image/jpeg")


class CodexSenior(Model):
    """A pydantic-ai ``Model`` whose requests are turns of one Codex CLI thread (see the module docstring)."""

    def __init__(
        self,
        model_name: str,
        *,
        codex_bin: str,
        output_dir: str | Path,
        base_url: str | None,
        key_env: str,
        reasoning_effort: str = "low",
        timeout_s: float = 180.0,
        runner: Runner = run_process,
        retry_wait_s: float = RETRY_WAIT_S,
    ) -> None:
        super().__init__()
        self._model_name = (
            model_name.split(":", 1)[1]
            if model_name.startswith("openai:")
            else model_name
        )
        self.codex_bin = codex_bin
        self.key_env = key_env
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.runner = runner
        self.retry_wait_s = retry_wait_s
        self.home = Path(output_dir) / "codex_home"
        self.turns_dir = Path(output_dir) / "codex_turns"
        self.workdir = Path(
            tempfile.mkdtemp(prefix="rpent-codex-")
        )  # empty: no AGENTS.md, no repository
        self.thread_id: str | None = None
        self.sent = 0  # messages already shown to the senior
        self.tools_sig: str | None = None
        self.totals = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        self.turn_index = 0
        self.own_calls: set[str] = set()
        self.image_bytes = 0  # attached to this thread so far
        self._write_config(base_url)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "openai"

    # -- setup -------------------------------------------------------------------

    def _write_config(self, base_url: str | None) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        os.chmod(self.home, 0o700)
        lines = [
            f"model = {json.dumps(self._model_name)}",
            f"model_provider = {json.dumps(PROVIDER_ID)}",
            'model_reasoning_summary = "none"',
            'web_search = "disabled"',
            'approval_policy = "never"',
            'sandbox_mode = "read-only"',
            "",
            f"[model_providers.{PROVIDER_ID}]",
            f"name = {json.dumps(PROVIDER_ID)}",
            'wire_api = "responses"',
            f"env_key = {json.dumps(self.key_env)}",
        ]
        if base_url:
            base = base_url.rstrip("/")
            lines.append(
                f"base_url = {json.dumps(base if base.endswith('/v1') else base + '/v1')}"
            )
        path = self.home / "config.toml"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

    def _env(self) -> dict[str, str]:
        env = {name: os.environ[name] for name in CHILD_ENV if name in os.environ}
        env["CODEX_HOME"] = str(self.home)
        env[self.key_env] = os.environ.get(self.key_env, "")
        return env

    # -- rendering -----------------------------------------------------------------

    def _tools_text(self, params: ModelRequestParameters) -> str:
        lines = ["=== robot tools (call them through tool_calls) ==="]
        for tool in params.function_tools:
            schema = json.dumps(
                tool.parameters_json_schema, ensure_ascii=False, separators=(",", ":")
            )
            lines.append(
                f"- {tool.name}: {(tool.description or '').strip()}\n  parameters: {schema}"
            )
        return "\n".join(lines)

    def _render(
        self, messages: list[ModelMessage], params: ModelRequestParameters
    ) -> tuple[str, list[BinaryContent]]:
        """The text and images of everything the senior has not seen yet."""
        items: list[str | BinaryContent] = []
        first = self.thread_id is None
        if first:
            items.append(CONTRACT)
            instructions = next(
                (
                    m.instructions
                    for m in reversed(messages)
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            if instructions:
                items.append("=== system prompt of the episode ===\n" + instructions)
        sig = hashlib.sha256(
            json.dumps(
                [(t.name, t.parameters_json_schema) for t in params.function_tools],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if sig != self.tools_sig:
            items.append(self._tools_text(params))
        items.append(
            "=== episode transcript ===" if first else "=== since your last turn ==="
        )
        for message in messages[self.sent :]:
            if isinstance(message, ModelResponse):
                if (message.model_name or "").startswith(self._model_name):
                    continue  # the senior's own turn is already in this thread
                for part in message.parts:
                    if isinstance(part, TextPart) and part.content.strip():
                        items.append(f"worker: {part.content.strip()}")
                    elif isinstance(part, ToolCallPart):
                        items.append(
                            f"worker call {part.tool_name}: {part.args_as_json_str()}"
                        )
                continue
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    whose = (
                        "your call" if part.tool_call_id in self.own_calls else "call"
                    )
                    items.append(
                        f"result of {whose} {part.tool_name}: {part.model_response_str()}"
                    )
                elif isinstance(part, RetryPromptPart):
                    items.append(
                        f"tool error ({part.tool_name or 'output'}): {part.model_response()}"
                    )
                elif isinstance(part, SystemPromptPart):
                    items.append(f"system: {part.content}")
                elif isinstance(part, UserPromptPart):
                    content = (
                        part.content
                        if isinstance(part.content, list)
                        else [part.content]
                    )
                    for entry in content:
                        if isinstance(entry, str):
                            if entry.strip():
                                items.append(f"user: {entry.strip()}")
                        elif isinstance(
                            entry, BinaryContent
                        ) and entry.media_type.startswith("image/"):
                            items.append(compact_image(entry))
        self.tools_sig = sig
        # Attach the newest images within the turn and thread budgets (always at least MIN_IMAGES); mention the rest.
        budget = min(TURN_IMAGE_BYTES, max(0, THREAD_IMAGE_BYTES - self.image_bytes))
        keep: set[int] = set()
        total = 0
        for rank, index in enumerate(
            i
            for i in reversed(range(len(items)))
            if isinstance(items[i], BinaryContent)
        ):
            size = len(items[index].data)  # type: ignore[union-attr]
            if rank < MIN_IMAGES or total + size <= budget:
                keep.add(index)
                total += size
        self.image_bytes += total
        lines, images = [], []
        for index, item in enumerate(items):
            if isinstance(item, BinaryContent):
                if index in keep:
                    images.append(item)
                    lines.append(f"[camera image {len(images)} attached]")
                else:
                    lines.append("[earlier camera image not attached]")
            else:
                lines.append(item)
        return "\n".join(lines), images

    # -- one request -------------------------------------------------------------------

    def _args(self, schema: Path, images: list[Path], effort: str) -> list[str]:
        args = [
            self.codex_bin,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "-C",
            str(self.workdir),
            "-c",
            f"model_reasoning_effort={json.dumps(effort)}",
            "--output-schema",
            str(schema),
        ]
        for feature in DISABLED_FEATURES:
            args += ["--disable", feature]
        if self.thread_id:
            args += ["resume", self.thread_id]
        args += [f"--image={path}" for path in images]
        return args + ["-"]

    async def _exec(
        self, schema: Path, text: str, images: list[Path], effort: str
    ) -> _Turn:
        code, out, err = await self.runner(
            self._args(schema, images, effort),
            text,
            self._env(),
            self.workdir,
            self.timeout_s,
        )
        turn = parse_events(out)
        if code != 0 and not turn.errors:
            tail = " | ".join(line for line in err.strip().splitlines()[-3:])
            turn.errors.append(f"codex exited {code}: {tail}"[:500])
        if turn.thread_id and not self.thread_id:
            self.thread_id = turn.thread_id
        return turn

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        params = model_request_parameters
        names = [tool.name for tool in params.function_tools]
        effort = str(
            (model_settings or {}).get("openai_reasoning_effort")
            or self.reasoning_effort
        )
        if len(messages) < self.sent:
            self.sent = 0  # the history was rebuilt: show it again in the same thread
        text, images = self._render(messages, params)
        self.turn_index += 1
        folder = self.turns_dir / f"turn-{self.turn_index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "input.txt").write_text(text, encoding="utf-8")
        paths = []
        for number, image in enumerate(images, 1):
            path = folder / f"image-{number}{_image_suffix(image.media_type)}"
            path.write_bytes(image.data)
            paths.append(path)
        schema = folder / "schema.json"
        schema.write_text(json.dumps(output_schema(names)), encoding="utf-8")
        before = dict(self.totals)
        attempts: list[dict[str, Any]] = []
        plan, calls, turn = None, None, _Turn()
        send_text, send_images = text, paths
        retried = corrected = False
        while True:
            turn = await self._exec(schema, send_text, send_images, effort)
            if turn.usage:
                self.totals.update(
                    {k: turn.usage.get(k, self.totals[k]) for k in self.totals}
                )
            attempts.append(
                {
                    "items": turn.items,
                    "errors": turn.errors,
                    "replied": turn.reply is not None,
                }
            )
            if turn.reply is None:
                # The process failed or ended without a reply (error events alone are stream retries): resend once.
                if retried:
                    break
                retried = True
                await asyncio.sleep(self.retry_wait_s)
                continue
            try:
                plan, calls = parse_reply(turn.reply, set(names))
                break
            except ValueError as exc:
                if corrected:
                    break
                corrected = True
                send_text = (
                    f"[harness] Your last reply could not be used: {exc}. Reply again with only the JSON "
                    "object of the output schema; every arguments_json must be a JSON object string."
                )
                send_images = []
        (folder / "attempts.json").write_text(
            json.dumps(attempts, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if turn.reply is not None:
            (folder / "reply.txt").write_text(turn.reply, encoding="utf-8")
        usage = {k: max(0, self.totals[k] - before[k]) for k in self.totals}
        log = {
            "backend": "codex",
            "codex_thread": self.thread_id,
            "codex_turn": self.turn_index,
            "codex_attempts": len(attempts),
            "images_attached": len(paths),
            "codex_items": sorted(
                {i for a in attempts for i in a["items"]} - {"agent_message"}
            ),
        }
        if plan is None or calls is None:
            reasons = [e for a in attempts for e in a["errors"]] or [
                "the reply could not be used"
            ]
            raise SeniorUnavailable(
                f"codex senior gave no usable reply after {len(attempts)} attempts: {reasons[-1]}",
                {
                    "input_tokens": usage["input_tokens"],
                    "cached_input_tokens": usage["cached_input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    **log,
                },
            )
        self.sent = len(messages)
        parts: list[Any] = [TextPart(plan)] if plan.strip() else []
        for tool, args in calls:
            call_id = f"codex_{uuid.uuid4().hex[:24]}"
            self.own_calls.add(call_id)
            parts.append(ToolCallPart(tool, args, tool_call_id=call_id))
        return ModelResponse(
            parts=parts or [TextPart("")],
            usage=RequestUsage(
                input_tokens=usage["input_tokens"],
                cache_read_tokens=usage["cached_input_tokens"],
                output_tokens=usage["output_tokens"],
                details={"reasoning_tokens": usage["reasoning_output_tokens"]},
            ),
            model_name=self._model_name,
            timestamp=datetime.now(timezone.utc),
            provider_name="codex",
            provider_details={"escalation_log": log},
            finish_reason="tool_call" if calls else "stop",
        )


def codex_senior_from_env(
    model_id: str,
    *,
    output_dir: str | Path,
    base_url: str | None,
    key_env: str,
    reasoning_effort: str,
) -> CodexSenior:
    codex_bin = os.environ.get("RPENT_CODEX_BIN")
    if not codex_bin or not Path(codex_bin).is_file():
        raise ValueError(
            "RPENT_ESCALATION_SENIOR_BACKEND=codex requires RPENT_CODEX_BIN, the codex binary"
        )
    return CodexSenior(
        model_id,
        codex_bin=codex_bin,
        output_dir=output_dir,
        base_url=base_url,
        key_env=key_env,
        reasoning_effort=reasoning_effort,
        timeout_s=float(os.environ.get("RPENT_CODEX_TIMEOUT_S") or 180),
    )
