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

"""Worker-led API planning with event-triggered senior turns (opt-in).

Enabled by ``RPENT_ESCALATION=1`` for ``--planner api``. The model built from
``--model``/``--base-url`` is the worker and runs most requests. The senior
shares the same conversation and takes the next request only when:

- ``plan``: memory and the initial state have been read (or at request 6);
- ``stall``: the last four motion results moved the end effector < 1 cm;
- ``errors``: the last two tool results reported an error;
- ``checkin``: the worker has made ``checkin`` requests since the senior;
- ``giveup``: the worker proposed ``finish`` without success. Its reply is
  dropped and the senior decides that turn (at most ``max_vetoes`` times).

``stall``, ``errors`` and ``checkin`` are skipped while the senior already has
``max_share`` of all requests. Senior environment variables:
``RPENT_ESCALATION_MODEL``, ``RPENT_ESCALATION_BASE_URL``,
``RPENT_ESCALATION_KEY_ENV``, ``RPENT_ESCALATION_REASONING_EFFORT`` (default
``low``), ``RPENT_ESCALATION_CHECKIN`` (default 12) and
``RPENT_ESCALATION_MAX_SHARE`` (default 0.2). Every request is logged to
``escalation.jsonl`` in the run output directory.

``RPENT_ESCALATION_SENIOR_MAX_IMAGES`` (optional) keeps only that many of the
newest images in senior requests. The shared history is bounded by image
bytes, which lets long runs with small frames carry about 100 images; the
senior gateway rejects requests above 64-80 images (HTTP 400
``image_compression_failed``). Worker requests are unchanged.

Optional mechanisms (off by default; each can be switched on alone):

- ``RPENT_ESCALATION_PLAN=0`` skips the ``plan`` turn.
- ``RPENT_ESCALATION_PLAN_FILES`` (paths joined by ``os.pathsep``): the
  ``plan`` turn receives these files in full, so the senior's strategy is built
  from the task memory instead of whatever the worker happened to read.
- ``RPENT_ESCALATION_TAKEOVER=K``: at a check-in, error, stall or give-up turn
  the senior may start its reply with ``[TAKE CONTROL]``; it then answers the
  next K requests (reason ``control``) unless it starts a reply with
  ``[HAND BACK]``. ``RPENT_ESCALATION_MAX_TAKEOVERS`` (default 4) bounds the
  takeovers per run. Control turns are not counted against ``max_share``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

MOTION_TOOLS = frozenset(
    {
        "move_to",
        "move_pose",
        "move_to_pair",
        "action_pair",
        "pi0_pick",
        "pi0_doubled",
        "pick",
        "rotate_pitch",
        "rotate_wrist",
    }
)
STALL_WINDOW = 4
STALL_SPAN_M = 0.01
PLAN_FALLBACK_REQUEST = 6
SENIOR_LABEL = "[senior planner guidance]"
TAKE_CONTROL = "[TAKE CONTROL]"
HAND_BACK = "[HAND BACK]"
#: Senior turns that may hand control to the senior (not ``plan`` or ``control``).
TAKEOVER_REASONS = frozenset({"stall", "errors", "checkin", "giveup"})


def _results(messages: list[ModelMessage]) -> Iterator[tuple[str, dict]]:
    """Yield (tool name, parsed JSON result) for tool returns, oldest first."""
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            content = part.content if isinstance(part.content, list) else [part.content]
            parsed: dict = {}
            for item in content:
                if isinstance(item, dict):
                    parsed = item
                    break
                if isinstance(item, str):
                    try:
                        value = json.loads(item)
                    except ValueError:
                        continue
                    if isinstance(value, dict):
                        parsed = value
                        break
            yield part.tool_name, parsed


def _eef(result: dict) -> list[float] | None:
    state = result.get("state")
    pos = state.get("robot0_eef_pos") if isinstance(state, dict) else None
    if isinstance(pos, list) and len(pos) == 3:
        return [float(v) for v in pos]
    return None


def _giveup(response: ModelResponse) -> dict | None:
    for part in response.parts:
        if isinstance(part, ToolCallPart) and part.tool_name == "finish":
            try:
                args = part.args_as_dict()
            except ValueError:
                args = {}
            if str(args.get("status", "")).lower() != "success":
                return args
    return None


def _reply_text(response: ModelResponse) -> str:
    return "\n".join(p.content for p in response.parts if isinstance(p, TextPart))


def load_plan_memory(paths: list[str]) -> str:
    """The listed files in full, each under a ``=== <parent>/<name> ===`` heading."""
    blocks = []
    for raw in paths:
        path = Path(raw)
        name = f"{path.parent.name}/{path.name}"
        try:
            blocks.append(f"=== {name} ===\n{path.read_text(errors='replace').strip()}")
        except OSError:
            blocks.append(f"=== {name} === (not present)")
    return "\n\n".join(blocks)


def _has_tool_call(response: ModelResponse) -> bool:
    return any(isinstance(part, ToolCallPart) for part in response.parts)


def _object_roots(params: ModelRequestParameters) -> ModelRequestParameters:
    """Restore object-typed tool roots for strict OpenAI-compatible endpoints.

    The API planner relaxes numeric leaves for providers that quote numbers,
    and that relaxation also widens the root to ``["object", "string"]``.
    OpenAI Responses rejects any function whose parameters root is not an
    object, so the senior receives the same tools with an object root.
    """

    def fix(tool):
        schema = tool.parameters_json_schema
        if schema.get("type") == "object":
            return tool
        return dataclasses.replace(
            tool, parameters_json_schema={**schema, "type": "object"}
        )

    return dataclasses.replace(
        params,
        function_tools=[fix(t) for t in params.function_tools],
        output_tools=[fix(t) for t in params.output_tools],
    )


def _append_note(messages: list[ModelMessage], note: str) -> list[ModelMessage]:
    """Copy of ``messages`` with ``note`` added to the newest request."""
    out = list(messages)
    for index in range(len(out) - 1, -1, -1):
        if isinstance(out[index], ModelRequest):
            out[index] = dataclasses.replace(
                out[index], parts=[*out[index].parts, UserPromptPart(note)]
            )
            break
    return out


@dataclasses.dataclass
class _ReplayStream(StreamedResponse):
    """Present an already completed response through the streaming interface."""

    _response: ModelResponse = dataclasses.field(kw_only=True)

    def __post_init__(self) -> None:
        self._usage = self._response.usage
        self.provider_response_id = self._response.provider_response_id
        self.provider_details = self._response.provider_details
        self.finish_reason = self._response.finish_reason

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        for index, part in enumerate(self._response.parts):
            yield self._parts_manager.handle_part(vendor_part_id=index, part=part)

    async def close_stream(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return self._response.model_name or ""

    @property
    def provider_name(self) -> str | None:
        return self._response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self._response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self._response.timestamp


def _newest_images(messages: list[ModelMessage], keep: int) -> list[ModelMessage]:
    """Copy of ``messages`` with all but the ``keep`` newest images replaced by a note."""
    located = [
        (mi, pi, ii)
        for mi, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        for pi, part in enumerate(message.parts)
        if isinstance(part, UserPromptPart) and isinstance(part.content, list)
        for ii, item in enumerate(part.content)
        if isinstance(item, BinaryContent) and item.media_type.startswith("image/")
    ]
    drop = set(located[: max(0, len(located) - keep)])
    if not drop:
        return messages
    out = list(messages)
    for mi in sorted({mi for mi, _, _ in drop}):
        parts = list(out[mi].parts)
        for pi, part in enumerate(parts):
            if any((mi, pi, ii) in drop for ii in range(len(part.content or ()))):
                parts[pi] = dataclasses.replace(
                    part,
                    content=[
                        "[earlier camera image omitted for the senior]"
                        if (mi, pi, ii) in drop
                        else item
                        for ii, item in enumerate(part.content)
                    ],
                )
        out[mi] = dataclasses.replace(out[mi], parts=parts)
    return out


class EscalationModel(WrapperModel):
    """Route each request to the worker, or to the senior on an escalation event."""

    def __init__(
        self,
        worker: Model,
        senior: Model,
        *,
        worker_settings: dict[str, Any] | None = None,
        senior_settings: dict[str, Any] | None = None,
        checkin: int = 12,
        max_share: float = 0.2,
        max_vetoes: int = 3,
        log_path: Path | None = None,
        senior_max_images: int | None = None,
        plan_turn: bool = True,
        plan_memory: str | None = None,
        takeover: int = 0,
        max_takeovers: int = 4,
    ) -> None:
        super().__init__(worker)
        if checkin < 1 or not 0 < max_share <= 1 or max_vetoes < 0:
            raise ValueError("invalid escalation budget")
        if takeover < 0 or max_takeovers < 0:
            raise ValueError("invalid takeover budget")
        self.senior = senior
        self.worker_settings = worker_settings or {}
        self.senior_settings = senior_settings or {}
        self.checkin = checkin
        self.max_share = max_share
        self.max_vetoes = max_vetoes
        self.log_path = log_path
        self.senior_max_images = senior_max_images
        self.plan_memory = plan_memory
        self.takeover = takeover
        self.max_takeovers = max_takeovers
        self.takeovers = 0
        # Requests left in the current takeover, and requests the senior answered in control.
        self.control_left = 0
        self.control_requests = 0
        self.requests = 0
        self.senior_requests = 0
        self.since_senior = 0
        self.planned = not plan_turn
        self.vetoes = 0
        # Tool results already reviewed by the senior do not trigger again.
        self.reviewed_results = 0

    # -- routing ---------------------------------------------------------------

    def _trigger(self, messages: list[ModelMessage]) -> str | None:
        if self.control_left > 0:
            return "control"
        results = list(_results(messages))
        if not self.planned:
            names = {name for name, _ in results}
            if {
                "read_text_file",
                "view_env_state",
            } <= names or self.requests + 1 >= PLAN_FALLBACK_REQUEST:
                return "plan"
            return None
        if self.senior_requests >= self.max_share * (self.requests + 1):
            return None
        fresh = results[self.reviewed_results :]
        moves = [p for name, r in fresh if name in MOTION_TOOLS if (p := _eef(r))]
        if len(moves) >= STALL_WINDOW:
            window = moves[-STALL_WINDOW:]
            center = [sum(axis) / len(window) for axis in zip(*window)]
            if max(math.dist(p, center) for p in window) < STALL_SPAN_M:
                return "stall"
        if len(fresh) >= 2 and all("error" in r for _, r in fresh[-2:]):
            return "errors"
        if self.since_senior >= self.checkin:
            return "checkin"
        return None

    def _senior_note(self, reason: str, proposal: dict | None = None) -> str:
        if reason == "control":
            turn = self.takeover - self.control_left + 1
            return (
                "[harness] You are in control of this episode (control turn "
                f"{turn} of {self.takeover}); the worker is paused. Review the latest "
                "results and images and make the next tool call yourself. Start your "
                f"reply with {HAND_BACK} once the worker can safely continue from here."
            )
        why = {
            "plan": "memory and the initial scene have been read; set the overall strategy",
            "stall": "the last robot motions barely moved the end effector",
            "errors": "the last tool calls returned errors",
            "checkin": "periodic review of the worker's progress",
            "giveup": "the worker proposed to end the episode without success",
        }[reason]
        note = (
            "[harness] You are the senior planner. A faster worker model runs most "
            f"turns of this shared episode; you take this turn because {why}. Review "
            "the progress and the latest images, state the corrected plan in a few "
            "sentences for the worker, then make the next tool call yourself. Do not "
            "reply with text only."
        )
        if proposal is not None:
            summary = str(proposal.get("summary", ""))[:1200]
            note += (
                f"\nThe worker proposed finish(status={proposal.get('status')!r}): "
                f"{summary}\nIf the task can still be completed, keep working; "
                "otherwise call finish yourself."
            )
        if reason == "plan" and self.plan_memory:
            note += (
                "\nThe task memory for this run follows in full: references from an "
                "earlier solved episode of this task and cross-task lessons. Treat it "
                "as a procedural prior and re-ground every object, fixture and "
                "coordinate from the live observations. Write the plan for the worker "
                "as short numbered steps (the skill or primitive, the object or "
                "fixture, and how to confirm the step from task_progress), then make "
                "the first tool call yourself.\n\n" + self.plan_memory
            )
        if (
            reason in TAKEOVER_REASONS
            and self.takeover
            and self.takeovers < self.max_takeovers
        ):
            note += (
                "\nIf the episode is off track (no task progress over the recent "
                "attempts, the same skill repeated without effect, an object lost or "
                "misplaced, or the worker misreading the state), start your reply "
                f"with {TAKE_CONTROL}: you then make the next {self.takeover} "
                "requests yourself, and you can return control earlier by starting a "
                f"reply with {HAND_BACK}. If the worker is on track, give brief "
                "guidance and make one call as usual."
            )
        return note

    def _labeled(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Mark senior responses so both models can tell guidance from work."""
        name = self.senior.model_name
        out = []
        for message in messages:
            if isinstance(message, ModelResponse) and (
                message.model_name or ""
            ).startswith(name):
                message = dataclasses.replace(
                    message, parts=[TextPart(SENIOR_LABEL), *message.parts]
                )
            out.append(message)
        return out

    @staticmethod
    def _merge_settings(
        base: ModelSettings | None, extra: dict[str, Any], *, senior: bool
    ) -> ModelSettings:
        settings: dict[str, Any] = dict(base or {})
        if senior:
            # The unified thinking flag is the worker's; the senior uses its own effort.
            settings.pop("thinking", None)
            settings.pop("extra_body", None)
        for key, value in extra.items():
            if key == "extra_body":
                settings["extra_body"] = {**(settings.get("extra_body") or {}), **value}
            else:
                settings[key] = value
        return settings  # type: ignore[return-value]

    async def _call(
        self,
        role: str,
        reason: str | None,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        params: ModelRequestParameters,
    ) -> ModelResponse:
        senior = role == "senior"
        model = self.senior if senior else self.wrapped
        extra = self.senior_settings if senior else self.worker_settings
        if senior:
            params = _object_roots(params)
            if self.senior_max_images is not None:
                messages = _newest_images(messages, self.senior_max_images)
        started = time.monotonic()
        record: dict[str, Any] = {
            "request": self.requests + 1,
            "role": role,
            "reason": reason,
            "model": model.model_name,
        }
        try:
            response = await model.request(
                self._labeled(messages),
                self._merge_settings(model_settings, extra, senior=senior),
                params,
            )
        except Exception as exc:
            self._log(
                record
                | {
                    "latency_s": round(time.monotonic() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}"[:2000],
                }
            )
            raise
        usage = response.usage
        self._log(
            record
            | {
                "latency_s": round(time.monotonic() - started, 3),
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cache_read_tokens,
                "output_tokens": usage.output_tokens,
                "tools": [
                    p.tool_name for p in response.parts if isinstance(p, ToolCallPart)
                ],
            }
        )
        return response

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        reason = self._trigger(messages)
        if reason is None:
            response = await self._call(
                "worker", None, messages, model_settings, model_request_parameters
            )
            proposal = _giveup(response)
            if proposal is None or self.vetoes >= self.max_vetoes:
                self._advance(messages, senior=False)
                return response
            self.vetoes += 1
            self._log(
                {
                    "request": self.requests + 1,
                    "role": "worker",
                    "veto": self.vetoes,
                    "proposal": proposal,
                }
            )
            reason = "giveup"
        else:
            proposal = None

        note = self._senior_note(reason, proposal)
        if reason == "plan" and self.plan_memory:
            self._log({"request": self.requests + 1, "plan_memory_chars": len(self.plan_memory)})
        response = await self._call(
            "senior",
            reason,
            _append_note(messages, note),
            model_settings,
            model_request_parameters,
        )
        self.planned = True
        text = _reply_text(response)
        if reason == "control":
            self.control_left -= 1
            if HAND_BACK in text:
                self.control_left = 0
                self._log({"request": self.requests + 1, "hand_back": self.takeovers})
        elif (
            reason in TAKEOVER_REASONS
            and self.takeover
            and self.takeovers < self.max_takeovers
            and TAKE_CONTROL in text
        ):
            self.takeovers += 1
            self.control_left = self.takeover
            self._log(
                {"request": self.requests + 1, "takeover": self.takeovers, "after": reason}
            )
        if not _has_tool_call(response):
            # A text-only reply would end the run; hand the guidance to the worker.
            guidance = "\n".join(
                p.content for p in response.parts if isinstance(p, TextPart)
            )
            worker = await self._call(
                "worker",
                f"after_{reason}",
                _append_note(messages, f"{SENIOR_LABEL} {guidance}"),
                model_settings,
                model_request_parameters,
            )
            response = dataclasses.replace(
                worker, parts=[TextPart(f"{SENIOR_LABEL} {guidance}"), *worker.parts]
            )
        self._advance(messages, senior=True, control=reason == "control")
        return response

    def _advance(
        self, messages: list[ModelMessage], *, senior: bool, control: bool = False
    ) -> None:
        self.requests += 1
        if senior:
            if control:
                self.control_requests += 1
            else:
                self.senior_requests += 1
            self.since_senior = 0
            self.reviewed_results = sum(1 for _ in _results(messages))
        else:
            self.since_senior += 1

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        # Routing and vetoes need the complete reply, so stream a finished one.
        response = await self.request(
            messages, model_settings, model_request_parameters
        )
        yield _ReplayStream(
            model_request_parameters=model_request_parameters, _response=response
        )

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def escalation_from_env(
    worker: Model, *, output_dir: str | Path, reasoning_effort: str
) -> Model:
    """Wrap ``worker`` when ``RPENT_ESCALATION=1``; otherwise return it unchanged."""
    if os.environ.get("RPENT_ESCALATION") != "1":
        return worker
    from rpent.planner.base import build_api_model
    from rpent.planner.prompt_cache import prompt_cache_from_env
    from rpent.planner.retry import retry_from_env

    model_id = os.environ.get("RPENT_ESCALATION_MODEL")
    key_env = os.environ.get("RPENT_ESCALATION_KEY_ENV")
    if not model_id or not key_env or not os.environ.get(key_env):
        raise ValueError(
            "RPENT_ESCALATION=1 requires RPENT_ESCALATION_MODEL and a non-empty "
            "credential named by RPENT_ESCALATION_KEY_ENV"
        )
    senior = retry_from_env(
        prompt_cache_from_env(
            build_api_model(
                model_id,
                os.environ.get("RPENT_ESCALATION_BASE_URL") or None,
                api_key=os.environ[key_env],
            )
        )
    )
    worker_settings: dict[str, Any] = {}
    if reasoning_effort == "none" and "qwen" in worker.model_name.lower():
        worker_settings["extra_body"] = {"enable_thinking": False}
    senior_settings = {
        "openai_reasoning_effort": os.environ.get(
            "RPENT_ESCALATION_REASONING_EFFORT", "low"
        ),
        # Reasoning item ids do not survive turns written by the other model.
        "openai_send_reasoning_ids": False,
    }
    return EscalationModel(
        worker,
        senior,
        worker_settings=worker_settings,
        senior_settings=senior_settings,
        checkin=int(os.environ.get("RPENT_ESCALATION_CHECKIN", "12")),
        max_share=float(os.environ.get("RPENT_ESCALATION_MAX_SHARE", "0.2")),
        log_path=Path(output_dir) / "escalation.jsonl",
        senior_max_images=(
            int(os.environ["RPENT_ESCALATION_SENIOR_MAX_IMAGES"])
            if os.environ.get("RPENT_ESCALATION_SENIOR_MAX_IMAGES")
            else None
        ),
        plan_turn=os.environ.get("RPENT_ESCALATION_PLAN", "1") != "0",
        plan_memory=(
            load_plan_memory(plan_files.split(os.pathsep))
            if (plan_files := os.environ.get("RPENT_ESCALATION_PLAN_FILES"))
            else None
        ),
        takeover=int(os.environ.get("RPENT_ESCALATION_TAKEOVER", "0")),
        max_takeovers=int(os.environ.get("RPENT_ESCALATION_MAX_TAKEOVERS", "4")),
    )
