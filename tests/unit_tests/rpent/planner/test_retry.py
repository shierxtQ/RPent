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
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from rpent.planner.retry import RetryingModel, retry_from_env, unwrap_retry

MESSAGES = [ModelRequest.user_text_prompt("hi")]


def _failing(statuses: list[int]):
    calls = {"n": 0}

    def model(messages: list[Any], info: AgentInfo) -> ModelResponse:
        del messages, info
        calls["n"] += 1
        if statuses:
            raise ModelHTTPError(statuses.pop(0), "qwen3.6-27b", {"message": "quota"})
        return ModelResponse(parts=[TextPart("ok")])

    return model, calls


def _request(model: RetryingModel) -> ModelResponse:
    return asyncio.run(model.request(MESSAGES, None, ModelRequestParameters()))


def test_rate_limit_is_retried_until_success() -> None:
    function, calls = _failing([429, 503])
    model = RetryingModel(FunctionModel(function), retries=3, base_s=0)

    response = _request(model)

    assert response.parts[0].content == "ok"
    assert calls["n"] == 3
    assert model.retry_count == 2


@pytest.mark.parametrize("statuses", [[400], [429, 429, 429]])
def test_non_retryable_or_exhausted_errors_are_raised(statuses: list[int]) -> None:
    function, calls = _failing(list(statuses))
    model = RetryingModel(FunctionModel(function), retries=2, base_s=0)

    with pytest.raises(ModelHTTPError) as raised:
        _request(model)

    assert raised.value.status_code == statuses[-1]
    assert calls["n"] == min(len(statuses), 3)


def test_stream_retries_only_before_the_stream_is_handed_out() -> None:
    attempts = {"n": 0}

    async def stream(messages: list[Any], info: AgentInfo):
        del messages, info
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ModelHTTPError(429, "qwen3.6-27b", {"message": "quota"})
        yield "ok"

    model = RetryingModel(FunctionModel(stream_function=stream), retries=2, base_s=0)

    async def consume() -> str:
        async with model.request_stream(
            MESSAGES, None, ModelRequestParameters()
        ) as response:
            async for _ in response:
                pass
            return response.get().parts[0].content

    assert asyncio.run(consume()) == "ok"
    assert attempts["n"] == 2


def test_retry_env_switch_and_unwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    base = FunctionModel(lambda messages, info: ModelResponse(parts=[]))

    monkeypatch.setenv("RPENT_API_RETRIES", "0")
    assert retry_from_env(base) is base

    monkeypatch.setenv("RPENT_API_RETRIES", "4")
    wrapped = retry_from_env(base)
    assert isinstance(wrapped, RetryingModel)
    assert wrapped.retries == 4
    assert wrapped.model_name == base.model_name
    assert unwrap_retry(wrapped) is base


def test_retry_backoff_is_excluded_from_active_timeout() -> None:
    from rpent.planner.timing import ActiveClock, run_with_active_timeout

    function, calls = _failing([429, 429])
    model = RetryingModel(FunctionModel(function), retries=2, base_s=0.2)
    clock = ActiveClock(toolkit=object())

    # Backoff sleeps ~0.6s in total; only the excluded clock fits a 0.15s budget.
    response = asyncio.run(
        run_with_active_timeout(
            model.request(MESSAGES, None, ModelRequestParameters()),
            timeout_s=0.15,
            toolkit=object(),
        )
    )

    assert response.parts[0].content == "ok"
    assert calls["n"] == 3
    assert clock.retry_wait_s() >= 0.45
    assert clock.active_s() < 0.15
    assert ActiveClock(toolkit=object()).retry_wait_s() == 0
