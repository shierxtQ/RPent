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

"""Offline tests for the Codex-CLI escalation senior.

A fake runner stands in for the codex binary (scripted JSONL events); no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from rpent.planner import codex_senior as cs
from rpent.planner import escalation as esc

PARAMS = ModelRequestParameters(
    function_tools=[
        ToolDefinition(
            name="move_to",
            description="Move the end effector.",
            parameters_json_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}},
            },
        ),
        ToolDefinition(
            name="view_env_state",
            description="Read object poses.",
            parameters_json_schema={"type": "object", "properties": {}},
        ),
        ToolDefinition(
            name="read_text_file",
            parameters_json_schema={"type": "object", "properties": {}},
        ),
        ToolDefinition(
            name="finish", parameters_json_schema={"type": "object", "properties": {}}
        ),
    ]
)
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64
THREAD = "01a14300-0000-7000-8000-000000000001"


def events(
    reply: str | None, usage: dict[str, int] | None, thread: str = THREAD
) -> str:
    rows: list[dict[str, Any]] = [
        {"type": "thread.started", "thread_id": thread},
        {"type": "turn.started"},
    ]
    if reply is not None:
        rows.append(
            {
                "type": "item.completed",
                "item": {"id": "i0", "type": "agent_message", "text": reply},
            }
        )
    if usage is not None:
        rows.append(
            {
                "type": "turn.completed",
                "usage": {"cache_write_input_tokens": 0, **usage},
            }
        )
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def usage(inp: int, cached: int, out: int = 10, reasoning: int = 0) -> dict[str, int]:
    return {
        "input_tokens": inp,
        "cached_input_tokens": cached,
        "output_tokens": out,
        "reasoning_output_tokens": reasoning,
    }


def reply(plan: str, *calls: tuple[str, dict]) -> str:
    return json.dumps(
        {
            "plan": plan,
            "tool_calls": [
                {"tool": t, "arguments_json": json.dumps(a)} for t, a in calls
            ],
        }
    )


class FakeCodex:
    """Scripted codex processes: each entry is (return code, stdout); records args, stdin and env."""

    def __init__(self, script: list[tuple[int, str]]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        args: list[str],
        stdin: str,
        env: dict[str, str],
        cwd: Path,
        timeout_s: float,
    ):
        self.calls.append({"args": args, "stdin": stdin, "env": env, "cwd": cwd})
        code, out = self.script.pop(0)
        return code, out, "" if code == 0 else "boom"


def senior(
    tmp_path: Path, fake: FakeCodex, monkeypatch: pytest.MonkeyPatch
) -> cs.CodexSenior:
    monkeypatch.setenv("TEST_SENIOR_KEY", "sk-test-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "worker-key-must-not-leak")
    return cs.CodexSenior(
        "openai:gpt-6-astra",
        codex_bin="/bin/codex",
        output_dir=tmp_path,
        base_url="https://senior.example",
        key_env="TEST_SENIOR_KEY",
        runner=fake,
        retry_wait_s=0,
    )


def history() -> list[ModelMessage]:
    return [
        ModelRequest(
            parts=[UserPromptPart("solve the task")], instructions="SYSTEM PROMPT TEXT"
        ),
        ModelResponse(
            parts=[
                TextPart("I will read memory."),
                ToolCallPart("read_text_file", {}, tool_call_id="w1"),
            ],
            model_name="qwen3.6-27b",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "read_text_file", '{"content": "memory"}', tool_call_id="w1"
                )
            ]
        ),
        ModelResponse(
            parts=[ToolCallPart("view_env_state", {}, tool_call_id="w2")],
            model_name="qwen3.6-27b",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("view_env_state", '{"step": 0}', tool_call_id="w2"),
                UserPromptPart(["camera", BinaryContent(PNG, media_type="image/png")]),
                UserPromptPart("[harness] You are the senior planner (note)."),
            ],
            instructions="SYSTEM PROMPT TEXT",
        ),
    ]


def run(model: cs.CodexSenior, messages: list[ModelMessage]) -> ModelResponse:
    return asyncio.run(
        model.request(messages, {"openai_reasoning_effort": "low"}, PARAMS)
    )


def test_first_turn_sends_prompt_tools_transcript_and_images(
    tmp_path, monkeypatch
) -> None:
    fake = FakeCodex(
        [
            (
                0,
                events(
                    reply("Read poses first.", ("move_to", {"x": 0.1})),
                    usage(30000, 4000),
                ),
            )
        ]
    )
    model = senior(tmp_path, fake, monkeypatch)
    response = run(model, history())
    call = fake.calls[0]
    assert call["args"][:2] == ["/bin/codex", "exec"] and "resume" not in call["args"]
    assert call["args"][-1] == "-" and "--output-schema" in call["args"]
    assert all(f in call["args"] for f in cs.DISABLED_FEATURES)
    images = [a for a in call["args"] if a.startswith("--image=")]
    assert len(images) == 1 and Path(images[0].split("=", 1)[1]).read_bytes() == PNG
    text = call["stdin"]
    for expected in (
        cs.CONTRACT,
        "SYSTEM PROMPT TEXT",
        "- move_to: Move the end effector.",
        "worker: I will read memory.",
        "worker call view_env_state: {}",
        'result of call read_text_file: {"content": "memory"}',
        "[camera image 1 attached]",
        "user: [harness] You are the senior planner (note).",
    ):
        assert expected in text
    schema = json.loads(
        Path(call["args"][call["args"].index("--output-schema") + 1]).read_text()
    )
    assert schema["properties"]["tool_calls"]["items"]["properties"]["tool"][
        "enum"
    ] == [t.name for t in PARAMS.function_tools]
    assert (
        isinstance(response.parts[0], TextPart)
        and response.parts[0].content == "Read poses first."
    )
    assert (
        isinstance(response.parts[1], ToolCallPart)
        and response.parts[1].tool_name == "move_to"
    )
    assert response.parts[1].args_as_dict() == {"x": 0.1}
    assert response.model_name == "gpt-6-astra"
    assert (
        response.usage.input_tokens,
        response.usage.cache_read_tokens,
        response.usage.output_tokens,
    ) == (30000, 4000, 10)
    assert response.provider_details["escalation_log"]["codex_thread"] == THREAD
    # The child sees only its own key, never the worker's.
    assert (
        call["env"]["TEST_SENIOR_KEY"] == "sk-test-secret"
        and "OPENAI_API_KEY" not in call["env"]
    )
    assert call["env"]["CODEX_HOME"] == str(tmp_path / "codex_home")


def test_later_turn_resumes_with_the_delta_only(tmp_path, monkeypatch) -> None:
    fake = FakeCodex(
        [
            (
                0,
                events(reply("Move.", ("move_to", {"x": 0.1})), usage(30000, 4000, 10)),
            ),
            (
                0,
                events(
                    reply("Check poses.", ("view_env_state", {})),
                    usage(62000, 35800, 25),
                ),
            ),
        ]
    )
    model = senior(tmp_path, fake, monkeypatch)
    messages = history()
    first = run(model, messages)
    own_id = first.parts[1].tool_call_id
    # What EscalationModel returns for the senior's turn, then the worker continues.
    messages = [
        *messages[:-1],
        ModelRequest(
            parts=[
                p
                for p in messages[-1].parts
                if "(note)" not in str(getattr(p, "content", ""))
            ]
        ),
        first,
        ModelRequest(
            parts=[ToolReturnPart("move_to", '{"ok": true}', tool_call_id=own_id)]
        ),
        ModelResponse(
            parts=[ToolCallPart("move_to", {"x": 0.2}, tool_call_id="w3")],
            model_name="qwen3.6-27b",
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("move_to", '{"ok": false}', tool_call_id="w3"),
                UserPromptPart("[harness] check-in note"),
            ]
        ),
    ]
    second = run(model, messages)
    call = fake.calls[1]
    assert call["args"][call["args"].index("resume") + 1] == THREAD
    text = call["stdin"]
    assert (
        "SYSTEM PROMPT TEXT" not in text
        and "- move_to:" not in text
        and "I will read memory." not in text
    )
    assert text.startswith("=== since your last turn ===")
    assert (
        'result of your call move_to: {"ok": true}' in text
        and 'worker call move_to: {"x":0.2}' in text
    )
    assert "Move." not in text  # the senior's own reply is already in the thread
    assert (
        second.usage.input_tokens,
        second.usage.cache_read_tokens,
        second.usage.output_tokens,
    ) == (32000, 31800, 15)


def test_unusable_reply_gets_one_correction(tmp_path, monkeypatch) -> None:
    fake = FakeCodex(
        [
            (0, events("not json", usage(30000, 0, 5))),
            (
                0,
                events(
                    reply("Fixed.", ("view_env_state", {})), usage(60500, 30000, 12)
                ),
            ),
        ]
    )
    model = senior(tmp_path, fake, monkeypatch)
    response = run(model, history())
    correction = fake.calls[1]
    assert correction["args"][correction["args"].index("resume") + 1] == THREAD
    assert correction["stdin"].startswith(
        "[harness] Your last reply could not be used: the reply was not a JSON object"
    )
    assert not [a for a in correction["args"] if a.startswith("--image=")]
    assert response.parts[1].tool_name == "view_env_state"
    assert (response.usage.input_tokens, response.usage.cache_read_tokens) == (
        60500,
        30000,
    )  # both attempts count
    assert response.provider_details["escalation_log"]["codex_attempts"] == 2


@pytest.mark.parametrize(
    "bad",
    [
        json.dumps(
            {"plan": "x", "tool_calls": [{"tool": "fly", "arguments_json": "{}"}]}
        ),
        json.dumps(
            {"plan": "x", "tool_calls": [{"tool": "move_to", "arguments_json": "[1]"}]}
        ),
        json.dumps({"plan": "x"}),
    ],
)
def test_parse_reply_rejects_unusable_calls(bad: str) -> None:
    with pytest.raises(ValueError):
        cs.parse_reply(bad, {"move_to"})


def test_failed_codex_turn_lets_the_worker_take_the_request(
    tmp_path, monkeypatch
) -> None:
    fake = FakeCodex([(1, ""), (1, "")])
    worker_calls: list[list[ModelMessage]] = []

    def worker_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        worker_calls.append(messages)
        return ModelResponse(parts=[ToolCallPart("move_to", {"x": 0})])

    model = esc.EscalationModel(
        FunctionModel(worker_reply, model_name="qwen3.6-27b"),
        senior(tmp_path, fake, monkeypatch),
        log_path=tmp_path / "escalation.jsonl",
    )
    reads = history()[:4] + [
        ModelRequest(
            parts=[ToolReturnPart("view_env_state", '{"step": 0}', tool_call_id="w2")]
        )
    ]
    response = asyncio.run(model.request(reads, {}, PARAMS))
    assert len(fake.calls) == 2 and len(worker_calls) == 1  # one retry, then the worker
    assert response.parts[0].tool_name == "move_to"
    rows = [
        json.loads(line)
        for line in (tmp_path / "escalation.jsonl").read_text().splitlines()
    ]
    failed = next(r for r in rows if r.get("role") == "senior")
    assert (
        failed["reason"] == "plan"
        and "SeniorUnavailable" in failed["error"]
        and failed["backend"] == "codex"
    )
    assert (
        rows[-1]["role"] == "worker" and rows[-1]["reason"] == "senior_unavailable_plan"
    )
    assert model.senior_requests == 0
    # The missed plan turn is not retried on the next request; the worker continues.
    later = [
        *reads,
        response,
        ModelRequest(
            parts=[ToolReturnPart("move_to", '{"ok": true}', tool_call_id="w9")]
        ),
    ]
    asyncio.run(model.request(later, {}, PARAMS))
    assert len(fake.calls) == 2 and len(worker_calls) == 2


def test_failed_codex_review_keeps_the_vetoed_giveup(tmp_path, monkeypatch) -> None:
    fake = FakeCodex([(1, ""), (1, "")])

    def worker_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart("finish", {"status": "stuck"})])

    model = esc.EscalationModel(
        FunctionModel(worker_reply, model_name="qwen3.6-27b"),
        senior(tmp_path, fake, monkeypatch),
        plan_turn=False,
    )
    response = asyncio.run(model.request(history(), {}, PARAMS))
    assert len(fake.calls) == 2
    assert response.parts[0].tool_name == "finish" and response.parts[
        0
    ].args_as_dict() == {"status": "stuck"}


def test_backend_from_env(tmp_path, monkeypatch) -> None:
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\n")
    worker = FunctionModel(
        lambda m, i: ModelResponse(parts=[TextPart("")]), model_name="qwen3.6-27b"
    )
    monkeypatch.setenv("RPENT_ESCALATION", "1")
    monkeypatch.setenv("RPENT_ESCALATION_MODEL", "openai:gpt-6-astra")
    monkeypatch.setenv("RPENT_ESCALATION_KEY_ENV", "TEST_SENIOR_KEY")
    monkeypatch.setenv("TEST_SENIOR_KEY", "sk-test-secret")
    monkeypatch.setenv("RPENT_ESCALATION_SENIOR_BACKEND", "codex")
    monkeypatch.setenv("RPENT_CODEX_BIN", str(codex))
    model = esc.escalation_from_env(
        worker, output_dir=tmp_path, reasoning_effort="none"
    )
    assert (
        isinstance(model.senior, cs.CodexSenior)
        and model.senior.model_name == "gpt-6-astra"
    )
    monkeypatch.delenv("RPENT_CODEX_BIN")
    with pytest.raises(ValueError, match="RPENT_CODEX_BIN"):
        esc.escalation_from_env(worker, output_dir=tmp_path, reasoning_effort="none")
    monkeypatch.setenv("RPENT_ESCALATION_SENIOR_BACKEND", "claude")
    with pytest.raises(ValueError, match="RPENT_ESCALATION_SENIOR_BACKEND"):
        esc.escalation_from_env(worker, output_dir=tmp_path, reasoning_effort="none")


def test_image_budget_attaches_the_newest(tmp_path, monkeypatch) -> None:
    fake = FakeCodex([(0, events(reply("ok", ("view_env_state", {})), usage(1, 0)))])
    model = senior(tmp_path, fake, monkeypatch)
    big = b"\x89PNG" + b"1" * (3 * 1024 * 1024)
    messages = [
        ModelRequest(
            parts=[UserPromptPart(["a", BinaryContent(big, media_type="image/png")])]
        ),
        ModelRequest(
            parts=[UserPromptPart(["b", BinaryContent(big, media_type="image/png")])]
        ),
        ModelRequest(
            parts=[UserPromptPart(["c", BinaryContent(big, media_type="image/png")])]
        ),
    ]
    run(model, messages)
    text = fake.calls[0]["stdin"]
    assert (
        text.count("[camera image") == 2
        and text.count("[earlier camera image not attached]") == 1
    )
    assert text.index("[earlier camera image not attached]") < text.index(
        "[camera image 1 attached]"
    )


def test_images_become_jpeg_and_the_thread_budget_holds(tmp_path, monkeypatch) -> None:
    from PIL import Image

    def png(seed: int) -> bytes:
        out = __import__("io").BytesIO()
        Image.effect_noise((1024, 1024), 60 + seed).convert("RGB").save(out, "PNG")
        return out.getvalue()

    turns = 12
    fake = FakeCodex(
        [
            (0, events(reply("ok", ("view_env_state", {})), usage(10 * i, 0)))
            for i in range(1, turns + 1)
        ]
    )
    model = senior(tmp_path, fake, monkeypatch)
    messages: list[ModelMessage] = []
    spent_turns = 0
    for turn in range(turns):
        messages = [
            *messages,
            ModelRequest(
                parts=[
                    UserPromptPart(
                        [
                            f"step {turn}",
                            *[
                                BinaryContent(png(turn * 9 + k), media_type="image/png")
                                for k in range(9)
                            ],
                        ]
                    )
                ]
            ),
        ]
        before = model.image_bytes
        run(model, messages)
        attached = [
            Path(a.split("=", 1)[1])
            for a in fake.calls[-1]["args"]
            if a.startswith("--image=")
        ]
        assert attached and all(
            p.suffix == ".jpg" and p.read_bytes()[:2] == b"\xff\xd8" for p in attached
        )
        size = sum(p.stat().st_size for p in attached)
        assert size <= cs.TURN_IMAGE_BYTES or len(attached) == cs.MIN_IMAGES
        if before >= cs.THREAD_IMAGE_BYTES:  # thread budget spent: only the newest two
            spent_turns += 1
            assert len(attached) == cs.MIN_IMAGES
    assert spent_turns > 0


def test_config_is_private_and_holds_no_key(tmp_path, monkeypatch) -> None:
    senior(tmp_path, FakeCodex([]), monkeypatch)
    config = tmp_path / "codex_home" / "config.toml"
    assert stat.S_IMODE(os.stat(config).st_mode) == 0o600
    text = config.read_text()
    assert (
        'env_key = "TEST_SENIOR_KEY"' in text
        and 'base_url = "https://senior.example/v1"' in text
    )
    assert "sk-test-secret" not in text and 'wire_api = "responses"' in text
