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


def _image_counts(messages):
    from pydantic_ai.messages import BinaryContent

    images = omitted = 0
    for message in messages:
        for part in getattr(message, "parts", ()):
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                for item in part.content:
                    images += isinstance(item, BinaryContent)
                    omitted += item == "[earlier camera image omitted for the senior]"
    return images, omitted


def test_senior_image_cap_keeps_the_newest_images_and_spares_the_worker():
    from pydantic_ai.messages import BinaryContent

    seen = {}

    def reply(role):
        def respond(messages):
            seen[role] = _image_counts(messages)
            return call("move_to")

        return respond

    frames = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    [f"frame {i}", BinaryContent(bytes([i]) * 8, media_type="image/png")]
                )
            ]
        )
        for i in range(5)
    ]
    model, _, _ = build(reply("worker"), reply("senior"), senior_max_images=2)

    ask(model, START + frames)
    ask(model, READY + frames)

    assert seen["worker"] == (5, 0)
    assert seen["senior"] == (2, 3)


def _routed(model, requests):
    """Tool names of the replies to ``requests`` in order (worker: move_to, senior: segment)."""
    return [ask(model, messages).parts[-1].tool_name for messages in requests]


def _senior_marking(markers):
    """Senior reply that opens with ``markers[key]`` when the note contains ``key``."""

    def reply(messages):
        note = " ".join(
            p.content
            for p in messages[-1].parts
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        text = next((m for key, m in markers.items() if key in note), "guidance")
        return ModelResponse(parts=[TextPart(text), ToolCallPart("segment", {})])

    return reply


def test_plan_turn_can_be_switched_off():
    model, _, senior = build(
        lambda m: call("move_to"), lambda m: call("segment"), plan_turn=False
    )

    assert _routed(model, [READY, READY]) == ["move_to", "move_to"]
    assert senior.notes == []


def test_plan_note_carries_the_task_memory():
    model, _, senior = build(
        lambda m: call("move_to"),
        lambda m: call("segment"),
        plan_memory="=== task-specific/X_s0.json ===\nopen the drawer first",
    )

    _routed(model, [START, READY])

    assert "open the drawer first" in senior.notes[0]
    assert "numbered steps" in senior.notes[0]


def test_checkin_takeover_keeps_the_senior_for_k_requests():
    model, _, senior = build(
        lambda m: call("move_to"),
        _senior_marking({"periodic review": "[TAKE CONTROL] the grasp keeps missing"}),
        checkin=1,
        max_share=1.0,
        takeover=2,
    )

    routed = _routed(model, [START, READY, READY, READY, READY, READY, READY])

    assert routed == [
        "move_to",  # worker reads memory
        "segment",  # plan
        "move_to",
        "segment",  # check-in takes control
        "segment",  # control 1 of 2
        "segment",  # control 2 of 2
        "move_to",  # back to the worker
    ]
    assert "start your reply" in senior.notes[1] and "[TAKE CONTROL]" in senior.notes[1]
    assert "control turn 1 of 2" in senior.notes[2]
    assert (model.takeovers, model.control_requests, model.senior_requests) == (1, 2, 2)


def test_hand_back_ends_the_takeover_early():
    model, _, _ = build(
        lambda m: call("move_to"),
        _senior_marking(
            {"periodic review": "[TAKE CONTROL] off track", "control turn": "[HAND BACK] ok"}
        ),
        checkin=1,
        max_share=1.0,
        takeover=3,
    )

    routed = _routed(model, [START, READY, READY, READY, READY, READY])

    assert routed == ["move_to", "segment", "move_to", "segment", "segment", "move_to"]
    assert model.control_requests == 1


def test_take_control_marker_is_ignored_when_takeover_is_off():
    model, _, senior = build(
        lambda m: call("move_to"),
        _senior_marking({"periodic review": "[TAKE CONTROL] off track"}),
        checkin=1,
        max_share=1.0,
    )

    routed = _routed(model, [START, READY, READY, READY, READY])

    assert routed == ["move_to", "segment", "move_to", "segment", "move_to"]
    assert all("[TAKE CONTROL]" not in note for note in senior.notes)


def test_load_plan_memory_reads_files_in_full_and_marks_missing_ones(tmp_path):
    from rpent.planner.escalation import load_plan_memory

    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "GLOBAL_MEMORY.md").write_text("x" * 90000)

    text = load_plan_memory(
        [str(tmp_path / "global" / "GLOBAL_MEMORY.md"), str(tmp_path / "task-specific" / "T_s0.json")]
    )

    assert "=== global/GLOBAL_MEMORY.md ===\n" + "x" * 90000 in text
    assert "=== task-specific/T_s0.json === (not present)" in text
