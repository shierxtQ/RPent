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

"""The escalation log's first row states the routing switches the run uses."""

from __future__ import annotations

import hashlib
import json

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from rpent.planner.escalation import EscalationModel, escalation_from_env


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _worker():
    return FunctionModel(
        lambda messages, info: ModelResponse(
            parts=[ToolCallPart("view_env_state", {})]
        ),
        model_name="worker",
    )


def test_config_row_is_written_once_at_construction(tmp_path):
    log = tmp_path / "escalation.jsonl"
    EscalationModel(
        _worker(),
        FunctionModel(lambda m, i: ModelResponse(parts=[]), model_name="senior"),
        checkin=6,
        errors=3,
        plan=False,
        corrections="  fix A\n",
        log_path=log,
        senior_max_images=40,
    )
    rows = _rows(log)
    assert rows == [
        {
            "config": {
                "worker": "worker",
                "senior": "senior",
                "plan": False,
                "checkin": 6,
                "errors": 3,
                "max_share": 0.2,
                "max_vetoes": 3,
                "corrections_chars": 5,
                "corrections_sha256": hashlib.sha256(b"fix A").hexdigest(),
                "senior_max_images": 40,
                "plan_memory_chars": 0,
                "takeover": 0,
                "max_takeovers": 4,
            }
        }
    ]


def test_config_row_reflects_the_environment_switches(tmp_path, monkeypatch):
    corrections = tmp_path / "corrections.md"
    corrections.write_text("- entry one\n")
    for name, value in {
        "RPENT_ESCALATION": "1",
        "RPENT_ESCALATION_MODEL": "openai:gpt-6-astra",
        "RPENT_ESCALATION_BASE_URL": "http://127.0.0.1:9/v1",
        "RPENT_ESCALATION_KEY_ENV": "FAKE_SENIOR_KEY",
        "FAKE_SENIOR_KEY": "fake",
        "RPENT_ESCALATION_CHECKIN": "12",
        "RPENT_ESCALATION_ERRORS": "4",
        "RPENT_ESCALATION_PLAN": "0",
        "RPENT_CLOUD_CORRECTIONS": str(corrections),
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("RPENT_ESCALATION_SENIOR_MAX_IMAGES", "RPENT_ESCALATION_PLAN_FILES"):
        monkeypatch.delenv(name, raising=False)
    escalation_from_env(_worker(), output_dir=tmp_path, reasoning_effort="none")
    (row,) = _rows(tmp_path / "escalation.jsonl")
    config = row["config"]
    assert (config["plan"], config["checkin"], config["errors"]) == (False, 12, 4)
    assert config["senior"] == "gpt-6-astra"
    assert config["corrections_chars"] == len("- entry one")
    assert config["corrections_sha256"] == hashlib.sha256(b"- entry one").hexdigest()

    monkeypatch.delenv("RPENT_ESCALATION_PLAN")
    monkeypatch.delenv("RPENT_ESCALATION_ERRORS")
    monkeypatch.setenv("RPENT_CLOUD_CORRECTIONS", str(tmp_path / "missing.md"))
    out = tmp_path / "second"
    escalation_from_env(_worker(), output_dir=out, reasoning_effort="none")
    (row,) = _rows(out / "escalation.jsonl")
    assert (row["config"]["plan"], row["config"]["errors"]) == (True, 2)
    # A missing corrections file still means no corrections; the row makes it visible.
    assert (
        row["config"]["corrections_chars"],
        row["config"]["corrections_sha256"],
    ) == (0, None)
