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

"""Leading-zero repair of tool-call argument JSON."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel

from rpent.planner.tool_args import (
    ToolArgsRepairModel,
    repair_json_numbers,
    tool_args_repair_from_env,
)


def test_leading_zeros_are_removed_outside_strings_only() -> None:
    raw = '{"a": 00.003, "s": "step 007, 00.5", "b": -00.5, "c": [007, 0.5, 0, 10, 1e-05]}'

    assert json.loads(repair_json_numbers(raw)) == {
        "a": 0.003,
        "s": "step 007, 00.5",
        "b": -0.5,
        "c": [7, 0.5, 0, 10, 1e-05],
    }


def test_escaped_quotes_keep_string_contents_untouched() -> None:
    raw = '{"s": "say \\"007\\"", "n": 00.25}'

    assert json.loads(repair_json_numbers(raw)) == {"s": 'say "007"', "n": 0.25}


def _respond(*parts: Any) -> FunctionModel:
    return FunctionModel(lambda messages, info: ModelResponse(parts=list(parts)))


def _request(model: ToolArgsRepairModel) -> ModelResponse:
    return asyncio.run(model.request([], None, ModelRequestParameters()))


def test_wrapper_repairs_invalid_arguments_and_leaves_valid_ones() -> None:
    bad = ToolCallPart(
        "scripted_grasp",
        '{"xyz": [0.5, -0.9, 0.92], "grasp_z_offset": 00.003}',
        "c1",
    )
    good = ToolCallPart("move_to", '{"x": 0.007}', "c2")
    text = TextPart("grasping")

    response = _request(ToolArgsRepairModel(_respond(text, bad, good)))

    assert response.parts[0] == text
    assert response.parts[1].args_as_dict() == {
        "xyz": [0.5, -0.9, 0.92],
        "grasp_z_offset": 0.003,
    }
    assert response.parts[2].args == '{"x": 0.007}'


def test_unrepairable_arguments_are_left_for_the_normal_retry() -> None:
    broken = ToolCallPart("move_to", '{"x": 00.1,', "c1")

    response = _request(ToolArgsRepairModel(_respond(broken)))

    assert response.parts[0].args == '{"x": 00.1,'


def test_repair_follows_the_coerce_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _respond(TextPart("x"))
    monkeypatch.delenv("RPENT_COERCE_TOOL_ARGS", raising=False)
    monkeypatch.delenv("SGD_COERCE_TOOL_ARGS", raising=False)
    assert tool_args_repair_from_env(model) is model

    monkeypatch.setenv("RPENT_COERCE_TOOL_ARGS", "1")
    assert isinstance(tool_args_repair_from_env(model), ToolArgsRepairModel)
