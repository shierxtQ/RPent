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

"""Make OpenAI Responses requests eligible for gateway prompt caching.

The Codex-compatible gateway (Sorux) only reports cached input for streamed
requests that carry a UUID ``prompt_cache_key``, which is what the Codex
client sends. Pydantic AI's agent loop issues non-streamed requests without a
key, so every request was billed and computed in full. This wrapper streams
each request (collecting the full reply before returning it, so callers still
see one complete :class:`ModelResponse`) and attaches one key per run, so all
requests of an episode share the gateway cache.

``RPENT_PROMPT_CACHE=0`` disables the wrapper.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings


class PromptCacheModel(WrapperModel):
    """Stream every request with a stable ``prompt_cache_key``."""

    def __init__(self, wrapped: Model, *, cache_key: str | None = None) -> None:
        super().__init__(wrapped)
        #: One key per episode: requests of a run share their history prefix.
        self.cache_key = cache_key or str(uuid.uuid4())

    def _with_cache_key(self, model_settings: ModelSettings | None) -> ModelSettings:
        settings: dict[str, Any] = dict(model_settings or {})
        settings.setdefault("openai_prompt_cache_key", self.cache_key)
        return settings  # type: ignore[return-value]

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # Consume the whole stream here: a failure mid-stream raises from this
        # call, so retry wrappers above see it as one failed request.
        async with self.wrapped.request_stream(
            messages, self._with_cache_key(model_settings), model_request_parameters
        ) as stream:
            async for _ in stream:
                pass
        return stream.get()

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(
            messages,
            self._with_cache_key(model_settings),
            model_request_parameters,
            run_context,
        ) as stream:
            yield stream


def prompt_cache_from_env(model: Model) -> Model:
    """Wrap OpenAI Responses models; other providers are returned unchanged."""
    from pydantic_ai.models.openai import OpenAIResponsesModel

    if os.environ.get("RPENT_PROMPT_CACHE", "1") == "0":
        return model
    if not isinstance(model, OpenAIResponsesModel):
        return model
    return PromptCacheModel(model)
