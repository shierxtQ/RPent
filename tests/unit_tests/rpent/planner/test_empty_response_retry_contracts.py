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

"""A Responses reply that completes without output is requested again.

The senior chain is the one ``escalation_from_env`` builds (pinned by
``test_prompt_cache.py``): ``RetryingModel(PromptCacheModel(OpenAIResponsesModel))``.
Replies come from an in-process transport; nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import traceback
from typing import Any

import pytest
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from rpent.planner.escalation import EscalationModel
from rpent.planner.prompt_cache import prompt_cache_from_env
from rpent.planner.retry import EMPTY_RESPONSE_PREFIX, RetryingModel, retry_from_env

MESSAGES = [ModelRequest.user_text_prompt("hi")]
REFUSAL = (
    "Invalid prompt: your prompt was flagged as potentially violating our usage "
    "policy. Please try again with a different prompt."
)
_ABSENT = object()


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "HTTP_PROXY",
        "http_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RPENT_API_RETRIES", "3")
    monkeypatch.setenv("RPENT_API_RETRY_BASE_S", "0")
    monkeypatch.delenv("RPENT_PROMPT_CACHE", raising=False)


def _sse(events: list[dict[str, Any]]) -> bytes:
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps({**e, 'sequence_number': i})}\n\n".encode()
        for i, e in enumerate(events)
    )


def _response(output: Any, status: str = "completed", **extra: Any) -> dict[str, Any]:
    body = {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-6-astra",
        "status": status,
        **extra,
    }
    if output is not _ABSENT:
        body["output"] = output
    return body


def _completed(output: Any) -> bytes:
    """A stream that starts normally and completes with ``output``."""
    return _sse(
        [
            {"type": "response.created", "response": _response([], "in_progress")},
            {"type": "response.completed", "response": _response(output)},
        ]
    )


def _tool_call_reply() -> bytes:
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "view_env_state",
        "arguments": "{}",
        "status": "completed",
    }
    return _sse(
        [
            {"type": "response.created", "response": _response([], "in_progress")},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**call, "arguments": ""},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "{}",
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": "{}",
            },
            {"type": "response.output_item.done", "output_index": 0, "item": call},
            {"type": "response.completed", "response": _response([call])},
        ]
    )


class _Gateway:
    """Serve queued replies (status, body, content type) and count requests."""

    def __init__(self, replies: list[tuple[int, bytes, str]]) -> None:
        self.replies = replies
        self.calls = 0

    def handler(self, request: Any) -> Any:
        import httpx2

        self.calls += 1
        status, body, content_type = self.replies[
            min(self.calls, len(self.replies)) - 1
        ]
        return httpx2.Response(
            status, content=body, headers={"content-type": content_type}
        )

    def responses_model(self) -> Any:
        import httpx2
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        client = AsyncOpenAI(
            base_url="http://gateway.test/v1",
            api_key="test",
            max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(self.handler), trust_env=False
            ),
        )
        return OpenAIResponsesModel(
            "gpt-6-astra", provider=OpenAIProvider(openai_client=client)
        )

    def senior(self) -> Any:
        """The senior's wrapper chain as ``escalation_from_env`` builds it."""
        return retry_from_env(prompt_cache_from_env(self.responses_model()))


def _stream(model: Any) -> ModelResponse:
    async def consume() -> ModelResponse:
        async with model.request_stream(
            MESSAGES, None, ModelRequestParameters()
        ) as stream:
            async for _ in stream:
                pass
        return stream.get()

    return asyncio.run(consume())


def _request(model: Any) -> ModelResponse:
    return asyncio.run(model.request(MESSAGES, None, ModelRequestParameters()))


EVENT_STREAM = "text/event-stream"


def test_pydantic_ai_fails_on_a_completion_without_output() -> None:
    """The failure that ended four episodes, reproduced without RPent's wrappers.

    Pydantic AI 2.53 iterates ``response.output`` of the ``response.completed``
    event in ``OpenAIResponsesStreamedResponse._get_event_iterator``. If a
    later release handles a null output itself, this test fails and the guard
    in ``prompt_cache`` can be revisited.
    """
    gateway = _Gateway([(200, _completed(None), EVENT_STREAM)])

    with pytest.raises(TypeError, match="'NoneType' object is not iterable") as raised:
        _stream(gateway.responses_model())

    innermost = traceback.extract_tb(raised.value.__traceback__)[-1]
    assert innermost.filename.replace("\\", "/").endswith(
        "pydantic_ai/models/openai.py"
    )


@pytest.mark.parametrize("output", [None, _ABSENT], ids=["null", "absent"])
def test_a_completion_without_output_is_requested_again(output: Any) -> None:
    gateway = _Gateway(
        [
            (200, _completed(output), EVENT_STREAM),
            (200, _tool_call_reply(), EVENT_STREAM),
        ]
    )
    senior = gateway.senior()

    response = _request(senior)

    assert isinstance(senior, RetryingModel)
    assert gateway.calls == 2
    assert senior.retry_count == 1
    assert [p.tool_name for p in response.parts if isinstance(p, ToolCallPart)] == [
        "view_env_state"
    ]


def test_repeated_empty_completions_end_with_a_named_error() -> None:
    gateway = _Gateway([(200, _completed(None), EVENT_STREAM)])
    senior = gateway.senior()

    with pytest.raises(UnexpectedModelBehavior) as raised:
        _request(senior)

    assert raised.value.message.startswith(EMPTY_RESPONSE_PREFIX)
    assert gateway.calls == 4  # the first request and RPENT_API_RETRIES=3
    assert senior.retry_count == 3


def test_an_empty_output_list_is_a_reply_and_is_not_requested_again() -> None:
    # EscalationModel hands a senior reply without a tool call to the worker.
    gateway = _Gateway([(200, _completed([]), EVENT_STREAM)])
    senior = gateway.senior()

    response = _request(senior)

    assert response.parts == []
    assert gateway.calls == 1
    assert senior.retry_count == 0


@pytest.mark.parametrize(
    "reply, error",
    [
        (
            (
                400,
                json.dumps(
                    {
                        "error": {
                            "message": REFUSAL,
                            "type": "invalid_request_error",
                            "param": None,
                            "code": "invalid_prompt",
                        }
                    }
                ).encode(),
                "application/json",
            ),
            ModelHTTPError,
        ),
        (
            (
                200,
                _sse(
                    [
                        {
                            "type": "error",
                            "code": "invalid_prompt",
                            "message": REFUSAL,
                            "param": None,
                        }
                    ]
                ),
                EVENT_STREAM,
            ),
            ModelAPIError,
        ),
        (
            (
                200,
                _sse(
                    [
                        {
                            "type": "response.created",
                            "response": _response([], "in_progress"),
                        },
                        {
                            "type": "response.failed",
                            "response": _response(
                                [],
                                "failed",
                                error={"code": "invalid_prompt", "message": REFUSAL},
                            ),
                        },
                    ]
                ),
                EVENT_STREAM,
            ),
            ModelAPIError,
        ),
    ],
    ids=["http-400", "stream-error-event", "response-failed"],
)
def test_refused_prompts_are_not_requested_again(
    reply: tuple[int, bytes, str], error: type[Exception]
) -> None:
    gateway = _Gateway([reply, (200, _tool_call_reply(), EVENT_STREAM)])
    senior = gateway.senior()

    with pytest.raises(error, match="flagged as potentially violating"):
        _request(senior)

    assert gateway.calls == 1
    assert senior.retry_count == 0


def test_an_empty_senior_review_no_longer_ends_the_episode(tmp_path) -> None:
    """The worker proposes to give up; GPT-6's first review comes back empty."""

    def worker(messages: list[Any], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(
            parts=[
                ToolCallPart("finish", {"status": "stuck", "summary": "no progress"})
            ]
        )

    gateway = _Gateway(
        [(200, _completed(None), EVENT_STREAM), (200, _tool_call_reply(), EVENT_STREAM)]
    )
    log = tmp_path / "escalation.jsonl"
    model = EscalationModel(
        FunctionModel(worker), gateway.senior(), plan=False, log_path=log
    )

    response = _request(model)

    assert [p.tool_name for p in response.parts if isinstance(p, ToolCallPart)] == [
        "view_env_state"
    ]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    senior_rows = [row for row in rows if row.get("role") == "senior"]
    assert [row["reason"] for row in senior_rows] == ["giveup"]
    assert "error" not in senior_rows[0]
    assert gateway.calls == 2
