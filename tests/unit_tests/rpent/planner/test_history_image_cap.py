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

"""The API planner's optional image-count cap and reasoning-id switch.

``RPENT_MAX_HISTORY_IMAGES`` caps the images every request resends;
``RPENT_SEND_REASONING_IDS=0`` sends no ids of earlier OpenAI Responses items.
Both are off by default. The last test drives the real planner chain
(``build_planner``) against a local fake Responses endpoint and checks the
request bodies on the wire.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pydantic_ai import BinaryContent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from rpent.planner import api_loop
from rpent.planner.api_loop import ApiAgentLoop, _prune_history_images
from rpent.tools.toolkit import ToolResult

PNG = b"\x89PNG\r\n\x1a\n" + b"\0" * 64


def frame(tag: int) -> BinaryContent:
    return BinaryContent(data=PNG + bytes([tag]), media_type="image/png")


def history(n: int) -> list[ModelRequest]:
    return [
        ModelRequest(parts=[UserPromptPart(content=[f"step {i}", frame(i)])])
        for i in range(n)
    ]


def kept_tags(messages: list[Any]) -> list[int]:
    return [
        item.data[-1]
        for message in messages
        for item in message.parts[0].content
        if isinstance(item, BinaryContent)
    ]


def test_without_the_cap_pruning_is_unchanged(monkeypatch):
    monkeypatch.delenv("RPENT_MAX_HISTORY_IMAGES", raising=False)
    monkeypatch.delenv("RPENT_MAX_HISTORY_IMAGE_MIB", raising=False)
    messages = history(50)
    assert _prune_history_images(messages) is messages  # 50 small frames fit 4 MiB
    for empty in ("", "  ", "0", "-3"):
        monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", empty)
        assert _prune_history_images(messages) is messages


def test_the_cap_keeps_the_newest_images_in_order(monkeypatch):
    monkeypatch.delenv("RPENT_MAX_HISTORY_IMAGE_MIB", raising=False)
    monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "3")
    messages = history(10)
    pruned = _prune_history_images(messages)
    assert kept_tags(pruned) == [7, 8, 9]
    dropped = [
        item
        for message in pruned[:7]
        for item in message.parts[0].content
        if isinstance(item, str) and "omitted" in item
    ]
    assert len(dropped) == 7  # every older frame is replaced by the note, text kept
    assert [m.parts[0].content[0] for m in pruned] == [f"step {i}" for i in range(10)]
    assert kept_tags(messages) == list(range(10))  # the history itself is not modified


def test_the_cap_and_the_byte_budget_both_apply(monkeypatch):
    big = BinaryContent(data=b"x" * (3 * 1024 * 1024), media_type="image/png")
    messages = [ModelRequest(parts=[UserPromptPart(content=[big])]) for _ in range(4)]
    monkeypatch.delenv("RPENT_MAX_HISTORY_IMAGE_MIB", raising=False)
    monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "40")
    # the 4 MiB default keeps the 2 newest 3 MiB frames; the cap of 40 does not add any
    assert (
        sum(
            isinstance(i, BinaryContent)
            for m in _prune_history_images(messages)
            for i in m.parts[0].content
        )
        == 2
    )
    monkeypatch.setenv(
        "RPENT_MAX_HISTORY_IMAGES", "1"
    )  # a hard cap, also below the 2-frame minimum
    assert (
        sum(
            isinstance(i, BinaryContent)
            for m in _prune_history_images(messages)
            for i in m.parts[0].content
        )
        == 1
    )
    monkeypatch.setenv(
        "RPENT_MAX_HISTORY_IMAGE_MIB", "0"
    )  # no byte budget: the cap alone
    monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "3")
    assert kept_tags(_prune_history_images(history(6))) == [3, 4, 5]


def test_a_bad_cap_is_refused_when_the_planner_is_built(monkeypatch):
    monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "forty")
    with pytest.raises(ValueError):
        ApiAgentLoop(FunctionModel(lambda m, i: None), dashboard_events=Sink())


def test_the_planner_logs_the_cap(monkeypatch, caplog):
    monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "40")
    with caplog.at_level("INFO"):
        ApiAgentLoop(FunctionModel(lambda m, i: None), dashboard_events=Sink())
    assert (
        "history images: at most 40 per request (RPENT_MAX_HISTORY_IMAGES)"
        in caplog.text
    )
    monkeypatch.delenv("RPENT_MAX_HISTORY_IMAGES")
    caplog.clear()
    with caplog.at_level("INFO"):
        ApiAgentLoop(FunctionModel(lambda m, i: None), dashboard_events=Sink())
    assert "history images" not in caplog.text


def test_reasoning_id_switch_applies_to_openai_responses_models_only(monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from rpent.planner.prompt_cache import PromptCacheModel
    from rpent.planner.retry import RetryingModel
    from rpent.planner.tool_args import ToolArgsRepairModel

    provider = OpenAIProvider(base_url="http://127.0.0.1:9/v1", api_key="fake")
    responses = OpenAIResponsesModel("gpt-6-astra", provider=provider)
    chain = ToolArgsRepairModel(RetryingModel(PromptCacheModel(responses), retries=1))
    qwen = OpenAIChatModel("qwen3.6-27b", provider=provider)
    monkeypatch.delenv("RPENT_SEND_REASONING_IDS", raising=False)
    assert "openai_send_reasoning_ids" not in api_loop._build_model_settings(
        chain, 100, "low"
    )
    monkeypatch.setenv("RPENT_SEND_REASONING_IDS", "1")
    assert "openai_send_reasoning_ids" not in api_loop._build_model_settings(
        chain, 100, "low"
    )
    monkeypatch.setenv("RPENT_SEND_REASONING_IDS", "0")
    settings = api_loop._build_model_settings(chain, 100, "low")
    assert settings == {"max_tokens": 100, "openai_send_reasoning_ids": False}
    # a Qwen chat model keeps its own settings: the thinking switch, no reasoning-id key
    assert api_loop._build_model_settings(qwen, 100, "none") == {
        "max_tokens": 100,
        "extra_body": {"enable_thinking": False},
    }


def test_the_capped_history_is_what_the_model_receives(monkeypatch):
    """A full agent run: every request carries at most the cap of images, newest kept."""
    monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "4")
    seen: list[list[int]] = []

    def model(messages: list[Any], info: Any) -> ModelResponse:
        seen.append(
            [
                item.data[-1]
                for message in messages
                for part in getattr(message, "parts", ())
                if isinstance(part, UserPromptPart) and isinstance(part.content, list)
                for item in part.content
                if isinstance(item, BinaryContent)
            ]
        )
        if len(seen) < 5:
            return ModelResponse(parts=[ToolCallPart("look", {}, f"c{len(seen)}")])
        return ModelResponse(
            parts=[ToolCallPart("finish", {"status": "success", "summary": "ok"}, "f")]
        )

    toolkit = ImageToolkit()
    ApiAgentLoop(FunctionModel(model), dashboard_events=Sink(), timeout_s=30).solve(
        system_prompt="s", user_message="u", toolkit=toolkit, max_turns=10
    )
    # each look adds 3 frames (tags 2-4, 5-7, ...); the 4 newest stay, oldest dropped first
    assert seen == [[], [2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10], [10, 11, 12, 13]]


class Sink:
    enabled = False

    def emit(self, event: Any) -> None:
        pass


class ImageToolkit:
    """``look`` returns three camera frames (agentview, navview, wrist), as a RoboCasa state view."""

    state = None

    def __init__(self) -> None:
        self.n = 0

    def get_tools_spec(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "look",
                "description": "View the state.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "finish",
                "description": "Finish.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                },
            },
        ]

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name == "finish":
            return ToolResult(name, {"_finish": True, **args})
        tags = [self.n * 3 + k + 2 for k in range(3)]
        self.n += 1
        return ToolResult(
            name,
            {
                "step": self.n,
                "_image_cam_bytes": PNG + bytes([tags[0]]),
                "_image_nav_bytes": PNG + bytes([tags[1]]),
                "_image_wrist_bytes": PNG + bytes([tags[2]]),
            },
        )

    def cancel_active_and_wait(self) -> None:
        pass


# -- on the wire ----------------------------------------------------------------------------


class FakeResponses:
    """A streamed OpenAI Responses endpoint: a reasoning item and one function call per reply."""

    def __init__(self, finish_at: int) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self.finish_at = finish_at
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                )
                owner.bodies.append(body)
                owner.headers.append(dict(self.headers))
                n = len(owner.bodies)
                payload = owner.reply(n, body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, n: int, body: dict[str, Any]) -> bytes:
        name, args = (
            ("finish", {"status": "success", "summary": "ok"})
            if n >= self.finish_at
            else ("look", {})
        )
        reasoning = {
            "id": f"rs_{n}",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": f"enc{n}",
        }
        call = {
            "type": "function_call",
            "id": f"fc_{n}",
            "call_id": f"call_{n}",
            "name": name,
            "arguments": json.dumps(args),
            "status": "completed",
        }
        usage = {
            "input_tokens": 1000 * n,
            "input_tokens_details": {"cached_tokens": 100 * n},
            "output_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 5},
            "total_tokens": 1000 * n + 10,
        }
        resp = {
            "id": f"resp_{n}",
            "object": "response",
            "created_at": 0,
            "model": body.get("model"),
            "status": "completed",
            "output": [reasoning, call],
            "usage": usage,
        }
        events = [
            {
                "type": "response.created",
                "response": {
                    **resp,
                    "status": "in_progress",
                    "output": [],
                    "usage": None,
                },
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": reasoning,
            },
            {"type": "response.output_item.done", "output_index": 0, "item": reasoning},
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**call, "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": f"fc_{n}",
                "output_index": 1,
                "delta": call["arguments"],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{n}",
                "output_index": 1,
                "arguments": call["arguments"],
            },
            {"type": "response.output_item.done", "output_index": 1, "item": call},
            {"type": "response.completed", "response": resp},
        ]
        return b"".join(
            f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': i})}\n\n".encode()
            for i, e in enumerate(events)
        )

    def close(self) -> None:
        self.server.shutdown()


def images_in(body: dict[str, Any]) -> int:
    return sum(
        1
        for item in body.get("input") or []
        if isinstance(item, dict) and isinstance(item.get("content"), list)
        for part in item["content"]
        if isinstance(part, dict) and part.get("type") == "input_image"
    )


@pytest.mark.parametrize("switches", [True, False], ids=["cap_and_no_ids", "defaults"])
def test_gpt6_alone_requests_on_the_wire(monkeypatch, tmp_path, switches):
    """The planner a GPT-6-alone episode builds, against a local fake Responses endpoint."""
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.planner.base import build_planner

    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    for name in (
        "RPENT_ESCALATION",
        "RPENT_PROMPT_CACHE",
        "RPENT_MAX_HISTORY_IMAGE_MIB",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RPENT_API_RETRIES", "0")
    monkeypatch.setenv("RPENT_COERCE_TOOL_ARGS", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-senior-key")
    if switches:
        monkeypatch.setenv("RPENT_MAX_HISTORY_IMAGES", "4")
        monkeypatch.setenv("RPENT_SEND_REASONING_IDS", "0")
    else:
        monkeypatch.delenv("RPENT_MAX_HISTORY_IMAGES", raising=False)
        monkeypatch.delenv("RPENT_SEND_REASONING_IDS", raising=False)
    fake = FakeResponses(finish_at=5)
    try:
        monkeypatch.setenv("OPENAI_BASE_URL", fake.url)
        planner = build_planner(
            "api",
            output_dir=str(tmp_path),
            recipe_tag="t",
            robot_name="robocasa",
            model="openai:gpt-6-astra",
            reasoning_effort="low",
            planner_timeout_s=60,
            dashboard_events=NullDashboardEventSink(),
        )
        result = planner.solve(
            system_prompt="Use tools.",
            user_message="Do it.",
            toolkit=ImageToolkit(),
            max_turns=10,
        )
    finally:
        fake.close()
    assert result.error is None, result.error
    assert result.finish_result and result.finish_result["status"] == "success"
    assert len(fake.bodies) == 5 and result.stats["requests"] == 5
    assert not (tmp_path / "escalation.jsonl").exists()  # no escalation wrapper
    for body, headers in zip(fake.bodies, fake.headers):
        assert body["model"] == "gpt-6-astra" and body["stream"] is True
        assert body["reasoning"]["effort"] == "low"
        assert body.get(
            "prompt_cache_key"
        )  # streamed with one cache key per episode, as the senior
        assert "enable_thinking" not in json.dumps(
            body
        )  # Qwen's switch never reaches GPT-6
        lower = {k.lower(): v for k, v in headers.items()}
        assert lower.get("authorization") == "Bearer fake-senior-key"
    assert len({b["prompt_cache_key"] for b in fake.bodies}) == 1
    counts = [images_in(b) for b in fake.bodies]
    later = fake.bodies[1:]
    reasoning_items = [
        i
        for b in later
        for i in b["input"]
        if isinstance(i, dict) and i.get("type") == "reasoning"
    ]
    call_ids = [
        i.get("id")
        for b in later
        for i in b["input"]
        if isinstance(i, dict) and i.get("type") == "function_call"
    ]
    if switches:
        assert counts == [0, 3, 4, 4, 4]
        assert reasoning_items == [] and set(call_ids) == {None}
    else:
        assert counts == [0, 3, 6, 9, 12]
        assert {i["id"] for i in reasoning_items} == {"rs_1", "rs_2", "rs_3", "rs_4"}
        assert set(call_ids) == {"fc_1", "fc_2", "fc_3", "fc_4"}
    assert result.stats["total_input_tokens"] == sum(1000 * n for n in range(1, 6))
    assert result.stats["cache_read_tokens"] == sum(100 * n for n in range(1, 6))
