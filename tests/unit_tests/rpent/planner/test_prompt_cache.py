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

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from rpent.planner.prompt_cache import PromptCacheModel, prompt_cache_from_env
from rpent.planner.retry import RetryingModel, unwrap_retry

MESSAGES = [ModelRequest.user_text_prompt("hi")]


def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(name, raising=False)


def _streaming_model(seen: list[dict[str, Any]]) -> FunctionModel:
    def unary(messages: list[Any], info: AgentInfo) -> ModelResponse:
        raise AssertionError("prompt-cache requests must be streamed")

    async def stream(messages: list[Any], info: AgentInfo):
        del messages
        seen.append(dict(info.model_settings or {}))
        yield "o"
        yield "k"

    return FunctionModel(unary, stream_function=stream)


def _request(model: PromptCacheModel, settings: dict[str, Any] | None = None):
    return asyncio.run(model.request(MESSAGES, settings, ModelRequestParameters()))


def test_requests_are_streamed_with_one_uuid_key_per_run() -> None:
    seen: list[dict[str, Any]] = []
    model = PromptCacheModel(_streaming_model(seen))

    first = _request(model, {"openai_reasoning_effort": "low"})
    second = _request(model)

    assert isinstance(first.parts[0], TextPart)
    assert first.parts[0].content == "ok"
    assert second.parts[0].content == "ok"
    keys = {s["openai_prompt_cache_key"] for s in seen}
    assert len(keys) == 1
    uuid.UUID(keys.pop())
    assert seen[0]["openai_reasoning_effort"] == "low"


def test_caller_supplied_cache_key_wins() -> None:
    seen: list[dict[str, Any]] = []
    model = PromptCacheModel(_streaming_model(seen))

    _request(model, {"openai_prompt_cache_key": "caller-key"})

    assert seen[0]["openai_prompt_cache_key"] == "caller-key"


def test_stream_failure_is_retried_as_one_request() -> None:
    attempts = {"n": 0}

    def unary(messages: list[Any], info: AgentInfo) -> ModelResponse:
        raise AssertionError("must stream")

    async def stream(messages: list[Any], info: AgentInfo):
        del messages, info
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ModelHTTPError(429, "gpt-6-astra", {"message": "quota"})
        yield "ok"

    model = RetryingModel(
        PromptCacheModel(FunctionModel(unary, stream_function=stream)),
        retries=2,
        base_s=0,
    )

    response = asyncio.run(model.request(MESSAGES, None, ModelRequestParameters()))

    assert response.parts[0].content == "ok"
    assert attempts["n"] == 2


def _sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': i})}\n\n".encode()
        for i, e in enumerate(events)
    )


def _function_call(n: int, arguments: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{n}",
        "call_id": f"call_{n}",
        "name": "read_text_file",
        "arguments": arguments,
        "status": "completed",
    }


def test_parallel_tool_args_sent_only_on_done_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gateway stream: call 0 streams deltas, calls 1-2 only send ``done``."""
    _no_proxy(monkeypatch)
    import httpx2
    from openai import AsyncOpenAI
    from pydantic_ai.messages import ToolCallPart
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    paths = ["/a.txt", "/b.txt", "/c.txt"]
    args = [json.dumps({"path": p}) for p in paths]
    events: list[dict[str, Any]] = []
    for n, arguments in enumerate(args):
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": n,
                "item": _function_call(n, ""),
            }
        )
        if n == 0:
            for piece in (arguments[:5], arguments[5:]):
                events.append(
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_0",
                        "output_index": 0,
                        "delta": piece,
                    }
                )
        events.append(
            {
                "type": "response.function_call_arguments.done",
                "item_id": f"fc_{n}",
                "output_index": n,
                "arguments": arguments,
            }
        )
        events.append(
            {
                "type": "response.output_item.done",
                "output_index": n,
                "item": _function_call(n, arguments),
            }
        )
    events.append(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "object": "response",
                "created_at": 0,
                "model": "gpt-6-astra",
                "status": "completed",
                "output": [_function_call(n, a) for n, a in enumerate(args)],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )
    body = _sse(events)

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx2.Response(
            200, content=body, headers={"content-type": "text/event-stream"}
        )

    client = AsyncOpenAI(
        base_url="http://gateway.test/v1",
        api_key="test",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    model = PromptCacheModel(
        OpenAIResponsesModel(
            "gpt-6-astra", provider=OpenAIProvider(openai_client=client)
        )
    )

    response = _request(model)

    calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
    assert [c.args_as_dict() for c in calls] == [{"path": p} for p in paths]


def test_only_openai_responses_models_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _no_proxy(monkeypatch)
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    provider = OpenAIProvider(base_url="http://localhost:1/v1", api_key="test")
    responses = OpenAIResponsesModel("gpt-6-astra", provider=provider)
    chat = OpenAIChatModel("qwen3.6-27b", provider=provider)

    wrapped = prompt_cache_from_env(responses)
    assert isinstance(wrapped, PromptCacheModel)
    assert unwrap_retry(RetryingModel(wrapped)) is responses
    assert prompt_cache_from_env(chat) is chat

    monkeypatch.setenv("RPENT_PROMPT_CACHE", "0")
    assert prompt_cache_from_env(responses) is responses


def test_escalation_senior_is_cache_streamed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _no_proxy(monkeypatch)
    from pydantic_ai.models.openai import OpenAIResponsesModel

    from rpent.planner.escalation import escalation_from_env

    monkeypatch.setenv("RPENT_ESCALATION", "1")
    monkeypatch.setenv("RPENT_ESCALATION_MODEL", "openai:gpt-6-astra")
    monkeypatch.setenv("RPENT_ESCALATION_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("RPENT_ESCALATION_KEY_ENV", "FAKE_SENIOR_KEY")
    monkeypatch.setenv("FAKE_SENIOR_KEY", "test")
    worker = FunctionModel(lambda messages, info: ModelResponse(parts=[]))

    model = escalation_from_env(worker, output_dir=tmp_path, reasoning_effort="none")

    senior = model.senior
    assert isinstance(senior, RetryingModel)
    assert isinstance(senior.wrapped, PromptCacheModel)
    assert isinstance(unwrap_retry(senior), OpenAIResponsesModel)
