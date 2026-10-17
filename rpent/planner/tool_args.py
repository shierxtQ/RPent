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

"""Repair tool-call argument JSON that a model wrote with leading-zero numbers.

Qwen3.6-27B sometimes writes numbers such as ``"grasp_z_offset": 00.003``.
JSON forbids leading zeros, so the arguments fail to parse, the model is asked
to retry once, and a second identical slip ends the run with
``UnexpectedModelBehavior`` (RoboCasa StirVegetables s2 failed this way on
three attempts in a row). Before Pydantic AI parses a response, arguments that
are not valid JSON have leading zeros removed from numbers outside string
literals; the repair is kept only if the result parses. Valid arguments and
other malformed JSON are left unchanged.

Enabled together with the numeric argument coercion
(``RPENT_COERCE_TOOL_ARGS=1``), which targets the same Qwen number habits.
"""

from __future__ import annotations

import dataclasses
import json
import re

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from rpent.tools.toolkit import _coerce_tool_args_enabled
from rpent.utils.logging import get_logger

logger = get_logger("tool_args")

_NUMBER_CHARS = frozenset("+-0123456789.eE")
_LEADING_ZEROS = re.compile(r"(-?)0+(\d.*)", re.DOTALL)


def repair_json_numbers(text: str) -> str:
    """Strip leading zeros from numbers outside string literals (``00.5`` -> ``0.5``)."""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            in_string = char != '"'
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "-" or char.isdigit():
            j = i
            while j < n and text[j] in _NUMBER_CHARS:
                j += 1
            token = text[i:j]
            match = _LEADING_ZEROS.fullmatch(token)
            out.append(match.group(1) + match.group(2) if match else token)
            i = j
            continue
        out.append(char)
        i += 1
    return "".join(out)


def _repaired(part: ToolCallPart) -> ToolCallPart:
    if not isinstance(part.args, str):
        return part
    try:
        json.loads(part.args)
        return part
    except ValueError:
        pass
    fixed = repair_json_numbers(part.args)
    try:
        json.loads(fixed)
    except ValueError:
        return part
    logger.info("repaired leading-zero numbers in %s arguments", part.tool_name)
    return dataclasses.replace(part, args=fixed)


class ToolArgsRepairModel(WrapperModel):
    """Repair leading-zero numbers in the tool-call arguments ``wrapped`` returns."""

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        parts = [
            _repaired(part) if isinstance(part, ToolCallPart) else part
            for part in response.parts
        ]
        if all(new is old for new, old in zip(parts, response.parts)):
            return response
        return dataclasses.replace(response, parts=parts)


def tool_args_repair_from_env(model: Model) -> Model:
    """Wrap ``model`` when ``RPENT_COERCE_TOOL_ARGS=1``; otherwise return it unchanged."""
    return ToolArgsRepairModel(model) if _coerce_tool_args_enabled() else model
