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

"""The stage-3 reflection call: retries, refusals and secrets, offline.

In-process with scripted models, and end to end through ``reflect_call.sh`` and
the senior's real model chain against a local fake Responses endpoint (no
request leaves the machine).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from scripts.ecc_robocasa import reflect_client as RC
from tests.unit_tests.ecc_robocasa.conftest import REPO

pytestmark = pytest.mark.timeout(120)
REFUSAL = {
    "error": {
        "message": "Invalid prompt: your prompt was flagged as potentially violating our usage policy.",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_prompt",
    }
}
REQUEST = {
    "model": "openai:gpt-6-astra",
    "reasoning_effort": "low",
    "max_output_tokens": 2048,
    "timeout_s": 60,
    "empty_retries": 2,
}


class Scripted:
    """A model whose replies (text, or an exception) are scripted."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.retry_count = 0

    async def request(self, messages, settings, params):
        self.calls.append(dict(settings))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, float):  # a slow reply
            await asyncio.sleep(reply)
            reply = "late"
        return ModelResponse(
            parts=[TextPart(reply)] if reply else [],
            usage=RequestUsage(
                input_tokens=1000, cache_read_tokens=400, output_tokens=50
            ),
        )


async def no_wait(_: float) -> None:
    return None


def test_a_reply_is_returned_with_the_seniors_settings():
    model = Scripted('{"applied": []}')
    record = RC.call("prompt", REQUEST, model, sleep=no_wait)
    assert record["status"] == "ok" and record["text"] == '{"applied": []}'
    assert record["usage"] == {
        "input_tokens": 1000,
        "cached_input_tokens": 400,
        "output_tokens": 50,
    }
    assert model.calls == [
        {
            "openai_reasoning_effort": "low",
            "openai_send_reasoning_ids": False,
            "max_tokens": 2048,
        }
    ]
    assert record["prompt_sha256"] == hashlib.sha256(b"prompt").hexdigest()


def test_an_empty_reply_is_requested_again_then_reported():
    model = Scripted("", "  ", '{"note": "ok"}')
    record = RC.call("p", REQUEST, model, sleep=no_wait)
    assert (record["status"], record["empty_retries"], len(model.calls)) == ("ok", 2, 3)
    assert (
        len(record["usage_all_attempts"]) == 3
    )  # every billed attempt is kept for the cost
    record = RC.call("p", REQUEST, Scripted("", "", ""), sleep=no_wait)
    assert record["status"] == "empty"


def test_the_timeout_bounds_the_whole_call_including_empty_retries():
    request = {**REQUEST, "timeout_s": 0.5}
    record = RC.call("p", request, Scripted("", 2.0), sleep=no_wait)
    assert record["status"] == "error" and "no reply within" in record["error"]
    assert record["latency_s"] < 1.5


def test_a_content_filter_refusal_is_not_retried_and_secrets_are_scrubbed():
    refusal = ModelHTTPError(status_code=400, model_name="gpt-6-astra", body=REFUSAL)
    model = Scripted(refusal, '{"never": "sent"}')
    record = RC.call("p", REQUEST, model, sleep=no_wait)
    assert record["status"] == "refused" and len(model.calls) == 1
    leak = ModelHTTPError(
        status_code=401,
        model_name="gpt-6-astra",
        body={"error": "bad key sk-secret-123456 for https://relay.example/v1"},
    )
    record = RC.call(
        "p",
        REQUEST,
        Scripted(leak),
        secrets=["sk-secret-123456", "https://relay.example/v1"],
        sleep=no_wait,
    )
    assert record["status"] == "error"
    assert "sk-secret" not in record["error"] and "relay.example" not in record["error"]


# -- end to end: reflect_call.sh -> reflect_client -> the senior's model chain ----------------


class FakeResponses(BaseHTTPRequestHandler):
    """A Responses endpoint streaming scripted replies: "ok", "500", "refuse" or "null_output"."""

    script: list[str] = []
    seen: list[dict] = []
    text = '{"applied": [], "add": [], "revise": [], "remove": [], "note": "fake"}'

    def log_message(self, *args):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        type(self).seen.append(
            {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
        )
        action = type(self).script.pop(0) if type(self).script else "ok"
        if action in ("500", "refuse"):
            data = json.dumps(
                REFUSAL if action == "refuse" else {"error": {"message": "boom"}}
            ).encode()
            self.send_response(400 if action == "refuse" else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        text = type(self).text
        item = {
            "id": "msg_1",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        base = {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": body.get("model"),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": {},
        }
        events = [
            {
                "type": "response.created",
                "response": {**base, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**item, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
            {
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {
                "type": "response.completed",
                "response": {
                    **base,
                    "status": "completed",
                    "output": None if action == "null_output" else [item],
                    "usage": {
                        "input_tokens": 1200,
                        "input_tokens_details": {"cached_tokens": 200},
                        "output_tokens": 80,
                        "output_tokens_details": {"reasoning_tokens": 30},
                        "total_tokens": 1280,
                    },
                },
            },
        ]
        for n, event in enumerate(events):
            event["sequence_number"] = n
            self.wfile.write(
                f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            )
            self.wfile.flush()


@pytest.fixture
def endpoint():
    FakeResponses.script, FakeResponses.seen = [], []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeResponses)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def reflect_call(
    tmp_path: Path, base_url: str, key: str = "fake-senior-key-123456"
) -> tuple[dict, str, int]:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv / "bin" / "python").chmod(0o755)
    cred = tmp_path / "cred.env"
    cred.write_text(
        f"FAKE_SENIOR_URL={base_url}\nFAKE_SENIOR_KEY={key}\nOTHER_KEY=unused\n"
    )
    folder = tmp_path / "reflection"
    folder.mkdir()
    prompt = "Reflect on the episode. Reply with JSON only."
    (folder / "prompt.txt").write_text(prompt)
    (folder / "request.json").write_text(
        json.dumps(
            {
                **REQUEST,
                "prompt_file": "prompt.txt",
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    )
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "ECC_RC_CODE_DIR": str(REPO),
        "ECC_RC_VENV": str(venv),
        "ECC_RC_CRED_FILE": str(cred),
        "ECC_RC_SENIOR_BASE_URL_VAR": "FAKE_SENIOR_URL",
        "ECC_RC_SENIOR_KEY_VAR": "FAKE_SENIOR_KEY",
        "ECC_RC_UNRELATED": "dropped",
        "RPENT_API_RETRY_BASE_S": "0.05",
        "RPENT_API_RETRY_MAX_S": "0.1",
    }
    proc = subprocess.run(
        [
            "bash",
            str(REPO / "scripts" / "ecc_robocasa" / "reflect_call.sh"),
            str(folder / "request.json"),
            str(folder / "reply1.json"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    path = folder / "reply1.json"
    reply = json.loads(path.read_text()) if path.exists() else {}
    # Everything the call left behind: its output and the reply file.
    output = proc.stdout + proc.stderr + (path.read_text() if path.exists() else "")
    return reply, output, proc.returncode


def test_reflect_call_streams_one_responses_request_as_the_senior(tmp_path, endpoint):
    FakeResponses.script = ["500", "null_output", "ok"]
    reply, output, code = reflect_call(tmp_path, endpoint)
    assert code == 0, output
    assert reply["status"] == "ok" and reply["text"] == FakeResponses.text
    assert reply["usage"] == {
        "input_tokens": 1200,
        "cached_input_tokens": 200,
        "output_tokens": 80,
    }
    assert (
        reply["retries"] >= 1
    )  # the empty completion went through the senior's retry wrapper
    assert len(FakeResponses.seen) == 3
    for seen in FakeResponses.seen:
        body = seen["body"]
        assert (
            seen["path"] == "/v1/responses"
            and seen["auth"] == "Bearer fake-senior-key-123456"
        )
        assert body["model"] == "gpt-6-astra" and body["stream"] is True
        assert (
            body["reasoning"]["effort"] == "low" and body["max_output_tokens"] == 2048
        )
        assert body.get("prompt_cache_key") and not body.get("tools")
        assert "Reflect on the episode." in json.dumps(body["input"])
    assert (
        "fake-senior-key" not in output
        and endpoint not in output
        and "127.0.0.1" not in output
    )


def test_reflect_call_reports_a_refusal_after_one_request(tmp_path, endpoint):
    FakeResponses.script = ["refuse"]
    reply, output, code = reflect_call(tmp_path, endpoint)
    assert code == 0, output
    assert reply["status"] == "refused" and "invalid_prompt" in reply["error"]
    assert len(FakeResponses.seen) == 1
    assert "fake-senior-key" not in output


def test_reflect_call_refuses_without_the_credentials(tmp_path, endpoint):
    reply, output, code = reflect_call(tmp_path, "")
    assert code == 2 and reply == {} and "base-URL variable is empty" in output
    assert FakeResponses.seen == []
