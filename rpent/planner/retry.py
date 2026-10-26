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

"""Retry rate-limited and transiently failing model requests.

Parallel RPent tasks share one provider quota; Infini's Qwen endpoint answers
``429 Service quota exceeded`` once a few tasks run together, and a single
429 used to end the episode. The provider SDK's own retries back off for only
a few seconds, shorter than a per-minute quota window, so this wrapper waits
longer (exponential, jittered so parallel tasks do not retry in lockstep).
Backoff time is excluded from the planner's active-time budget.

``RPENT_API_RETRIES`` (default 8, 0 disables), ``RPENT_API_RETRY_BASE_S``
(default 5) and ``RPENT_API_RETRY_MAX_S`` (default 60) tune the policy.

A gateway outage or dropped connection can also arrive without an HTTP status,
inside a streamed response ("upstream_error: The service is temporarily
unavailable", "server_error: Our servers are currently overloaded"), which
Pydantic AI raises as a ``ModelAPIError``. Those whose message contains one of
:data:`TRANSIENT_MESSAGES` are retried too: the streaming wrappers consume the
whole stream inside ``request``, so nothing has reached the caller yet.

Infini's Qwen endpoint also sometimes returns a tool call with empty
arguments (``read_text_file({})``), mostly when the reply opens with text.
Once such a call is in the history the model repeats it, and after a few
identical rounds Infini rejects the request with ``400 Repetitive tool calls
detected``, which ended the episode. With ``RPENT_EMPTY_TOOL_ARGS_RETRIES=N``
(default 0, off), a non-streamed response whose tool call has empty arguments
for a tool with required parameters is requested again, up to N times.

Infini occasionally answers with ``finish_reason: ""``, which fails the Chat
Completions schema and surfaces as ``UnexpectedModelBehavior: Invalid response
from openai chat completions endpoint``; it is retried like a 5xx.
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    UnexpectedModelBehavior,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from rpent.planner.timing import api_retry_wait
from rpent.utils.logging import get_logger

logger = get_logger("api_retry")

#: Rate limiting plus transient gateway/server failures (529: Anthropic overloaded).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})
#: Message fragments (matched case-insensitively anywhere in the message) of
#: transient failures that arrive without an HTTP status: a failed streamed
#: response or a dropped connection.
TRANSIENT_MESSAGES = (
    "server_error",
    "rate_limit_exceeded",
    "upstream_error",
    "stream_read_error",
    "connection error",
    "temporarily unavailable",
    "overloaded",
)


def _empty_required_tool_call(
    response: ModelResponse, parameters: ModelRequestParameters
) -> str | None:
    """Name of a tool called with empty arguments although it requires some."""
    required = {
        tool.name
        for tool in parameters.function_tools
        if (tool.parameters_json_schema or {}).get("required")
    }
    for part in response.parts:
        if (
            isinstance(part, ToolCallPart)
            and part.tool_name in required
            and not part.args_as_dict()
        ):
            return part.tool_name
    return None


#: A malformed provider response (schema validation failed), not a model bug.
INVALID_RESPONSE_PREFIX = "Invalid response from openai chat completions endpoint"


class RetryingModel(WrapperModel):
    """Retry ``wrapped`` requests that fail with a retryable HTTP status."""

    def __init__(
        self,
        wrapped: Model,
        *,
        retries: int = 8,
        base_s: float = 5.0,
        max_s: float = 60.0,
        empty_args_retries: int = 0,
    ) -> None:
        super().__init__(wrapped)
        self.retries = max(0, int(retries))
        self.empty_args_retries = max(0, int(empty_args_retries))
        self.base_s = max(0.0, float(base_s))
        self.max_s = max(self.base_s, float(max_s))
        #: Accumulated over the run, for logs and tests.
        self.retry_count = 0
        self.retry_wait_s = 0.0
        self.empty_args_retry_count = 0

    def _delay_s(self, attempt: int) -> float:
        delay = min(self.max_s, self.base_s * 2**attempt)
        return delay * random.uniform(0.8, 1.2)

    async def _backoff(
        self, exc: ModelAPIError | UnexpectedModelBehavior, attempt: int
    ) -> None:
        delay = self._delay_s(attempt)
        self.retry_count += 1
        self.retry_wait_s += delay
        logger.warning(
            "model %s returned %s; retry %d/%d in %.1fs",
            self.model_name,
            f"HTTP {exc.status_code}"
            if isinstance(exc, ModelHTTPError)
            else f"{exc.message[:120]!r}",
            attempt + 1,
            self.retries,
            delay,
        )
        with api_retry_wait():
            await asyncio.sleep(delay)

    def _retryable(self, exc: ModelAPIError, attempt: int) -> bool:
        if attempt >= self.retries:
            return False
        if isinstance(exc, ModelHTTPError):
            return exc.status_code in RETRYABLE_STATUS
        message = exc.message.lower()
        return any(fragment in message for fragment in TRANSIENT_MESSAGES)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        for empty_attempt in range(self.empty_args_retries + 1):
            response = await self._request_with_backoff(
                messages, model_settings, model_request_parameters
            )
            tool = _empty_required_tool_call(response, model_request_parameters)
            if tool is None or empty_attempt == self.empty_args_retries:
                return response
            self.empty_args_retry_count += 1
            logger.warning(
                "model %s called %s with empty arguments; re-requesting %d/%d",
                self.model_name,
                tool,
                empty_attempt + 1,
                self.empty_args_retries,
            )
        return response

    async def _request_with_backoff(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        attempt = 0
        while True:
            try:
                return await self.wrapped.request(
                    messages, model_settings, model_request_parameters
                )
            except ModelAPIError as exc:
                if not self._retryable(exc, attempt):
                    raise
                await self._backoff(exc, attempt)
                attempt += 1
            except UnexpectedModelBehavior as exc:
                if attempt >= self.retries or not exc.message.startswith(
                    INVALID_RESPONSE_PREFIX
                ):
                    raise
                await self._backoff(exc, attempt)
                attempt += 1

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        # Only failures before the stream is handed out are retried; once the
        # caller consumes events, a retry would duplicate partial output.
        attempt = 0
        while True:
            yielded = False
            try:
                async with self.wrapped.request_stream(
                    messages, model_settings, model_request_parameters, run_context
                ) as stream:
                    yielded = True
                    yield stream
                return
            except ModelAPIError as exc:
                if yielded or not self._retryable(exc, attempt):
                    raise
                await self._backoff(exc, attempt)
                attempt += 1


def retry_from_env(model: Model) -> Model:
    """Wrap ``model`` with the env-configured retry policy (0 retries: unchanged)."""
    retries = int(os.environ.get("RPENT_API_RETRIES", "8"))
    if retries <= 0:
        return model
    return RetryingModel(
        model,
        retries=retries,
        base_s=float(os.environ.get("RPENT_API_RETRY_BASE_S", "5")),
        max_s=float(os.environ.get("RPENT_API_RETRY_MAX_S", "60")),
        empty_args_retries=int(os.environ.get("RPENT_EMPTY_TOOL_ARGS_RETRIES", "0")),
    )


def unwrap_retry(model: Model) -> Model:
    """Return the provider model under retry, prompt-cache and tool-argument repair wrappers."""
    from rpent.planner.prompt_cache import PromptCacheModel
    from rpent.planner.tool_args import ToolArgsRepairModel

    while isinstance(model, (RetryingModel, PromptCacheModel, ToolArgsRepairModel)):
        model = model.wrapped
    return model
