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

"""One RC-ECC stage-3 reflection call: a text-only GPT-6 request made the way the senior makes it.

``reflect_call.sh`` runs this module after sourcing the credentials file:

    python -m scripts.ecc_robocasa.reflect_client REQUEST.json REPLY.json

``REQUEST.json`` names the model (``openai:gpt-6-astra``), the reasoning effort,
the output-token cap, the prompt file (next to the request) and its sha256.
The environment gives ``ECC_RC_REFLECT_BASE_URL`` (the senior's base URL) and
``ECC_RC_REFLECT_KEY_ENV`` (the *name* of the variable holding the senior's
key). The model chain is the senior's (``rpent.planner.escalation``
``escalation_from_env``): ``build_api_model`` -> ``prompt_cache_from_env``
(streamed Responses with a prompt-cache key; a completion without output is
raised as a retryable error) -> ``retry_from_env`` (429/5xx and transient
gateway messages, 8 retries by default).

On top of that, a reply without text is requested again (``empty_retries``)
and a content-filter refusal (``invalid_prompt``, "flagged as potentially
violating our usage policy") is never retried: it is reported as
``"status": "refused"``. ``REPLY.json`` holds the status, the reply text,
usage, latency and retry counts. Neither the key nor the base URL is ever
written or printed; error texts are scrubbed of both.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

REFUSAL_MARKERS = ("invalid_prompt", "flagged as potentially violating")
EMPTY_BACKOFF_S = 10.0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def is_refusal(exc: BaseException) -> bool:
    """A content-filter refusal: HTTP 400 ``invalid_prompt`` (or the same text in a stream error)."""
    text = f"{getattr(exc, 'message', '')} {getattr(exc, 'body', '')} {exc}".lower()
    return any(marker in text for marker in REFUSAL_MARKERS)


def scrub(text: str, secrets: list[str]) -> str:
    """``text`` without any of ``secrets`` (and without the base URL's host)."""
    for secret in sorted(
        {s for s in secrets if s and len(s) >= 4}, key=len, reverse=True
    ):
        text = text.replace(secret, "<redacted>")
    return text


def build_model(model_id: str, base_url: str, api_key: str):
    """The senior's chain: provider model, prompt-cache streaming, retries."""
    from rpent.planner.base import build_api_model
    from rpent.planner.prompt_cache import prompt_cache_from_env
    from rpent.planner.retry import retry_from_env

    return retry_from_env(
        prompt_cache_from_env(
            build_api_model(model_id, base_url or None, api_key=api_key)
        )
    )


async def _request(model: Any, prompt: str, settings: dict[str, Any]) -> Any:
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters

    return await model.request(
        [ModelRequest(parts=[UserPromptPart(content=prompt)])],
        settings,
        ModelRequestParameters(),
    )


def _text(response: Any) -> str:
    from pydantic_ai.messages import TextPart

    return "".join(p.content for p in response.parts if isinstance(p, TextPart))


def call(
    prompt: str,
    request: dict[str, Any],
    model: Any,
    *,
    secrets: Sequence[str] = (),
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, Any]:
    """Send ``prompt`` to ``model``; returns the reply record (never raises for model errors).

    ``sleep`` waits between requests for an empty reply (tests pass a no-op).
    """
    settings: dict[str, Any] = {
        "openai_reasoning_effort": str(request.get("reasoning_effort") or "low"),
        # As the senior: reasoning ids are not sent back (escalation_from_env).
        "openai_send_reasoning_ids": False,
    }
    if request.get("max_output_tokens"):
        settings["max_tokens"] = int(request["max_output_tokens"])
    empty_retries = max(0, int(request.get("empty_retries", 3)))
    timeout_s = float(request.get("timeout_s") or 900)
    started = time.monotonic()
    record: dict[str, Any] = {
        "model": request.get("model"),
        "reasoning_effort": settings["openai_reasoning_effort"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "started_at": _now(),
        "empty_retries": 0,
    }

    async def run() -> dict[str, Any]:
        for attempt in range(empty_retries + 1):
            # One deadline for the whole call: every attempt and the waits between them.
            left = timeout_s - (time.monotonic() - started)
            if left <= 0:
                raise asyncio.TimeoutError
            response = await asyncio.wait_for(
                _request(model, prompt, settings), timeout=left
            )
            usage = response.usage
            record["usage"] = {
                "input_tokens": int(usage.input_tokens or 0),
                "cached_input_tokens": int(usage.cache_read_tokens or 0),
                "output_tokens": int(usage.output_tokens or 0),
            }
            record.setdefault("usage_all_attempts", []).append(record["usage"])
            text = _text(response)
            if text.strip():
                return {"status": "ok", "text": text}
            if attempt < empty_retries:
                record["empty_retries"] += 1
                await sleep(EMPTY_BACKOFF_S * (attempt + 1))
        return {
            "status": "empty",
            "text": "",
            "error": "reply without text after retries",
        }

    try:
        outcome = asyncio.run(run())
    except asyncio.TimeoutError:
        outcome = {"status": "error", "error": f"no reply within {timeout_s:.0f} s"}
    except Exception as exc:  # noqa: BLE001 - every model failure becomes a reply record
        kind = "refused" if is_refusal(exc) else "error"
        outcome = {
            "status": kind,
            "error": scrub(f"{type(exc).__name__}: {exc}", list(secrets))[:1500],
        }
    record.update(outcome)
    record["latency_s"] = round(time.monotonic() - started, 3)
    record["retries"] = int(getattr(model, "retry_count", 0) or 0)
    record["ended_at"] = _now()
    return record


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print("usage: reflect_client.py REQUEST.json REPLY.json", file=sys.stderr)
        return 2
    request_path, reply_path = Path(args[0]), Path(args[1])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    prompt = (request_path.parent / request["prompt_file"]).read_text(encoding="utf-8")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != request.get(
        "prompt_sha256"
    ):
        print(
            "reflect_client: the prompt file does not match its sha256", file=sys.stderr
        )
        return 2
    key_env = os.environ.get("ECC_RC_REFLECT_KEY_ENV") or ""
    base_url = os.environ.get("ECC_RC_REFLECT_BASE_URL") or ""
    api_key = os.environ.get(key_env) if key_env else None
    if not api_key or not base_url:
        print(
            "reflect_client: ECC_RC_REFLECT_BASE_URL and the key named by ECC_RC_REFLECT_KEY_ENV "
            "must be set (reflect_call.sh sets them from the credentials file)",
            file=sys.stderr,
        )
        return 2
    secrets = [api_key, base_url, _host(base_url)]
    try:
        model = build_model(str(request["model"]), base_url, api_key)
        record = call(prompt, request, model, secrets=secrets)
    except Exception as exc:  # noqa: BLE001 - reported without the key or the URL
        record = {
            "model": request.get("model"),
            "status": "error",
            "error": scrub(f"{type(exc).__name__}: {exc}", secrets)[:1500],
            "latency_s": 0.0,
            "retries": 0,
            "empty_retries": 0,
        }
    tmp = reply_path.with_name(f".{reply_path.name}.{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, reply_path)
    print(
        f"reflect_client: {record['status']} in {record['latency_s']} s, "
        f"retries {record['retries']}, empty retries {record['empty_retries']}"
    )
    return 0


def _host(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).hostname or ""
    except ValueError:
        return ""


if __name__ == "__main__":
    sys.exit(main())
