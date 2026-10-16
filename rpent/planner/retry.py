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
"""

from __future__ import annotations

import asyncio
import os
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from rpent.planner.timing import api_retry_wait
from rpent.utils.logging import get_logger

logger = get_logger("api_retry")

#: Rate limiting plus transient gateway/server failures (529: Anthropic overloaded).
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504, 529})


class RetryingModel(WrapperModel):
    """Retry ``wrapped`` requests that fail with a retryable HTTP status."""

    def __init__(
        self,
        wrapped: Model,
        *,
        retries: int = 8,
        base_s: float = 5.0,
        max_s: float = 60.0,
    ) -> None:
        super().__init__(wrapped)
        self.retries = max(0, int(retries))
        self.base_s = max(0.0, float(base_s))
        self.max_s = max(self.base_s, float(max_s))
        #: Accumulated over the run, for logs and tests.
        self.retry_count = 0
        self.retry_wait_s = 0.0

    def _delay_s(self, attempt: int) -> float:
        delay = min(self.max_s, self.base_s * 2**attempt)
        return delay * random.uniform(0.8, 1.2)

    async def _backoff(self, exc: ModelHTTPError, attempt: int) -> None:
        delay = self._delay_s(attempt)
        self.retry_count += 1
        self.retry_wait_s += delay
        logger.warning(
            "model %s returned HTTP %s; retry %d/%d in %.1fs",
            self.model_name,
            exc.status_code,
            attempt + 1,
            self.retries,
            delay,
        )
        with api_retry_wait():
            await asyncio.sleep(delay)

    def _retryable(self, exc: ModelHTTPError, attempt: int) -> bool:
        return exc.status_code in RETRYABLE_STATUS and attempt < self.retries

    async def request(
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
            except ModelHTTPError as exc:
                if not self._retryable(exc, attempt):
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
            except ModelHTTPError as exc:
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
    )


def unwrap_retry(model: Model) -> Model:
    """Return the provider model under a :class:`RetryingModel`."""
    while isinstance(model, RetryingModel):
        model = model.wrapped
    return model
