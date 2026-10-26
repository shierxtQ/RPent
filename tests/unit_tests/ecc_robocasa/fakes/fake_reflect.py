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

"""Fake stage-3 reflection call: ``reflect_call.sh REQUEST REPLY`` without a model.

Writes the reply record ``reflect_client.py`` writes. The behaviour per episode
comes from ``ECC_RC_FAKE_REFLECT`` (JSON): ``{"<Task>/<episode>": spec |
[spec per call], "default": spec}`` with ``status`` (``ok`` | ``refused`` |
``error`` | ``empty`` | ``crash``: no reply), ``edits`` (the JSON reply; default:
apply every shown entry and add one), ``text`` (a raw reply) and ``sleep_s``.
Every call is logged to ``ECC_RC_FAKE_LOG`` with the request and what the prompt held.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


def main() -> int:
    request_path, reply_path = Path(sys.argv[1]), Path(sys.argv[2])
    request = json.loads(request_path.read_text())
    prompt = (request_path.parent / request["prompt_file"]).read_text()
    episode = request["episode"]
    call = int(re.search(r"reply(\d+)\.json$", reply_path.name).group(1))
    scenario = json.loads(Path(os.environ["ECC_RC_FAKE_REFLECT"]).read_text())
    spec = scenario.get(episode, scenario.get("default", {}))
    specs = spec if isinstance(spec, list) else [spec]
    spec = specs[min(call, len(specs)) - 1]
    entries = json.loads(
        prompt.split("Current correction memory (JSON):\n", 1)[1].split(
            "\n\nEdit the memory", 1
        )[0]
    )
    if os.environ.get("ECC_RC_FAKE_LOG"):
        with open(os.environ["ECC_RC_FAKE_LOG"], "a") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "reflect",
                        "episode": episode,
                        "call": call,
                        "pid": os.getpid(),
                        "t": time.time(),
                        "model": request["model"],
                        "effort": request["reasoning_effort"],
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "sha_ok": hashlib.sha256(prompt.encode()).hexdigest()
                        == request["prompt_sha256"],
                        "outcome_first": "[skill=rldx_arm ok=true status=" in prompt,
                        "run_log_note": "rebuilt from the run log" in prompt,
                        "language": "Task: Close the toaster oven door." in prompt,
                        "entries_shown": [e["id"] for e in entries],
                        "cred_names": [
                            os.environ.get("ECC_RC_SENIOR_BASE_URL_VAR"),
                            os.environ.get("ECC_RC_SENIOR_KEY_VAR"),
                        ],
                        "cred_file_given": bool(os.environ.get("ECC_RC_CRED_FILE")),
                    }
                )
                + "\n"
            )
    time.sleep(float(spec.get("sleep_s", 0.0)))
    status = spec.get("status", "ok")
    if status == "crash":
        return 3
    edits = spec.get("edits") or {
        "applied": [e["id"] for e in entries],
        "add": [
            {
                "trigger": "checkin",
                "situation": f"situation seen in {episode}",
                "diagnosis": "the push started above the free edge",
                "fix": "start 3 cm below the edge",
            }
        ],
        "revise": [],
        "remove": [],
        "note": f"lesson of {episode}",
    }
    record = {
        "model": request["model"],
        "reasoning_effort": request["reasoning_effort"],
        "prompt_sha256": request["prompt_sha256"],
        "status": status,
        "text": spec.get("text", json.dumps(edits)) if status == "ok" else "",
        "usage": {
            "input_tokens": 4000,
            "cached_input_tokens": 1000,
            "output_tokens": 300,
        },
        "usage_all_attempts": [
            {"input_tokens": 4000, "cached_input_tokens": 1000, "output_tokens": 300}
        ],
        "latency_s": 0.01,
        "retries": 0,
        "empty_retries": 0,
    }
    if status == "refused":
        record["error"] = (
            "ModelHTTPError: status_code: 400, body: {'code': 'invalid_prompt'}"
        )
    elif status in ("error", "empty"):
        record["error"] = "ModelHTTPError: status_code: 401"
    reply_path.write_text(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
