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

import asyncio
import json

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel

from rpent.planner.escalation import SENIOR_LABEL, EscalationModel


class Recorder:
    """FunctionModel callback that records the notes it saw."""

    def __init__(self, reply):
        self.reply = reply
        self.notes: list[str] = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        last = messages[-1]
        self.notes.append(
            " ".join(
                p.content
                for p in last.parts
                if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            )
        )
        return self.reply(messages)


def call(name, **args):
    return ModelResponse(parts=[ToolCallPart(name, args)])


def result(name, payload, index):
    return ModelRequest(
        parts=[ToolReturnPart(name, [json.dumps(payload)], tool_call_id=f"c{index}")]
    )


def build(worker_reply, senior_reply, **kwargs):
    worker = Recorder(worker_reply)
    senior = Recorder(senior_reply)
    model = EscalationModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(senior, model_name="senior"),
        **kwargs,
    )
    return model, worker, senior


def ask(model, messages):
    return asyncio.run(model.request(messages, None, ModelRequestParameters()))


START = [ModelRequest.user_text_prompt("start")]
READY = START + [
    result("view_env_state", {"step": 0}, 1),
    result("read_text_file", {"path": "memory/MEMORY.md"}, 2),
]


def test_plan_goes_to_senior_once_then_worker_continues():
    model, worker, senior = build(lambda m: call("move_to"), lambda m: call("segment"))

    assert ask(model, START).parts[0].tool_name == "move_to"
    assert ask(model, READY).parts[0].tool_name == "segment"
    assert "set the overall strategy" in senior.notes[0]
    assert ask(model, READY).parts[0].tool_name == "move_to"
    assert (model.requests, model.senior_requests) == (3, 1)


def test_worker_giveup_is_vetoed_and_senior_decides():
    model, worker, senior = build(
        lambda m: call("finish", status="failure", summary="cannot grasp"),
        lambda m: call("move_to"),
        max_share=1.0,
    )
    model.planned = True

    response = ask(model, READY)

    assert response.parts[0].tool_name == "move_to"
    assert "cannot grasp" in senior.notes[0]
    assert model.vetoes == 1


def test_text_only_senior_reply_is_handed_to_worker():
    model, worker, senior = build(
        lambda m: call("move_to"),
        lambda m: ModelResponse(parts=[TextPart("approach from the left")]),
    )

    response = ask(model, READY)

    assert response.parts[0].content.startswith(SENIOR_LABEL)
    assert isinstance(response.parts[1], ToolCallPart)
    assert "approach from the left" in worker.notes[-1]


def test_stall_triggers_senior_and_respects_share_cap():
    stalled = READY + [
        result("move_to", {"state": {"robot0_eef_pos": [0.1, 0.2, 1.0]}}, i)
        for i in range(3, 7)
    ]
    model, worker, senior = build(
        lambda m: call("move_to"), lambda m: call("segment"), max_share=1.0
    )
    model.planned = True
    assert ask(model, stalled).parts[0].tool_name == "segment"
    assert "barely moved" in senior.notes[0]
    # The reviewed stall does not trigger again.
    assert ask(model, stalled).parts[0].tool_name == "move_to"

    capped, _, _ = build(
        lambda m: call("move_to"), lambda m: call("segment"), max_share=0.2
    )
    capped.planned = True
    capped.requests, capped.senior_requests = 4, 1
    assert ask(capped, stalled).parts[0].tool_name == "move_to"


def test_streamed_agent_run_uses_routed_reply():
    model, worker, senior = build(
        lambda m: (
            call("noop") if len(m) < 3 else ModelResponse(parts=[TextPart("done")])
        ),
        lambda m: call("noop"),
    )
    agent = Agent(model)

    @agent.tool_plain
    def noop() -> str:
        return "ok"

    async def handler(ctx, events):
        async for _ in events:
            pass

    output = agent.run_sync("start", event_stream_handler=handler)

    assert output.output == "done"
    assert model.requests >= 2


def test_senior_tools_keep_object_root_while_worker_keeps_relaxed_schema():
    from pydantic_ai.tools import ToolDefinition

    seen: dict[str, object] = {}

    def worker(messages, info: AgentInfo) -> ModelResponse:
        seen["worker"] = info.function_tools[0].parameters_json_schema["type"]
        return call("move_to")

    def senior(messages, info: AgentInfo) -> ModelResponse:
        seen["senior"] = info.function_tools[0].parameters_json_schema["type"]
        return call("move_to")

    model = EscalationModel(
        FunctionModel(worker, model_name="worker"),
        FunctionModel(senior, model_name="senior"),
    )
    relaxed = ToolDefinition(
        name="move_to",
        parameters_json_schema={"type": ["object", "string"], "properties": {}},
    )
    params = ModelRequestParameters(function_tools=[relaxed])

    asyncio.run(model.request(START, None, params))
    asyncio.run(model.request(READY, None, params))

    assert seen == {"worker": ["object", "string"], "senior": "object"}
