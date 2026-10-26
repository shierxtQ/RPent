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

"""One hybrid RC-ECC episode: Qwen worker plus GPT-6 senior (stages 2, 3 and the test).

- the call rules (GPT-6 plan turn, check-in every K requests, N consecutive tool
  errors) and the ECC rule's six-rung ladder;
- the environment ``hybrid_cell.sh`` needs (it sources the credentials file itself);
- the escalation evidence check: a finished episode's ``escalation.jsonl`` must
  show the plan on or off, K, N and the cloud corrections it was launched with;
- the per-episode record: outcome, GPT-6 calls by reason, how often the errors
  trigger fired, Qwen and GPT-6 tokens and list-price cost.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.ecc_robocasa import common as C

SPLIT = "target"
#: The senior's routing reasons (``rpent/planner/escalation.py``); ``control`` needs takeover.
SENIOR_REASONS = ("plan", "stall", "errors", "checkin", "giveup")
#: The plan turn comes at this request at the latest (``PLAN_FALLBACK_REQUEST``).
PLAN_FALLBACK_REQUEST = 6
#: The Target50 protocol's planner time per episode (the ported cell launcher).
TARGET50_TIMEOUT_S = {"atomic": 1800, "composite_seen": 3600, "composite_unseen": 3600}


@dataclass(frozen=True)
class Rule:
    """When the worker calls GPT-6: plan turn on/off, check-in every K, N errors in a row."""

    plan: bool
    checkin: int
    errors: int

    def __post_init__(self) -> None:
        if not isinstance(self.plan, bool):
            raise ValueError(f"plan must be true/false, got {self.plan!r}")
        for name in ("checkin", "errors"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def config_name(self, rung: int) -> str:
        """The rung label of ``scripts/ecc/run_ecc_stage2.py`` (``config_name``), e.g. ``c2_gptplan_K12_N4``."""
        return f"c{rung}_{'gptplan' if self.plan else 'qwenplan'}_K{self.checkin}_N{self.errors}"

    def as_json(self) -> dict[str, Any]:
        """The rule as ``scripts/ecc/run_ecc_stage2.py``'s rule files store it."""
        return {
            "gpt_plan": self.plan,
            "checkin_K": self.checkin,
            "errors_N": self.errors,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> Rule:
        return cls(
            plan=value["gpt_plan"], checkin=value["checkin_K"], errors=value["errors_N"]
        )


#: The ECC rule's stage-2 ladder (``CONFIGS`` in ``scripts/ecc/run_ecc_stage2.py``), weakest first.
LEADER_LADDER = (
    Rule(False, 12, 4),
    Rule(True, 12, 4),
    Rule(False, 6, 4),
    Rule(True, 6, 4),
    Rule(True, 6, 3),
    Rule(True, 6, 2),
)


def parse_ladder(value: Sequence[Any] | None) -> tuple[Rule, ...]:
    """Rungs from the config: ``[{plan = "gpt"|"qwen", checkin = K, errors = N}, ...]``."""
    if not value:
        return LEADER_LADDER
    rungs = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"ladder rung must be a table, got {item!r}")
        plan = item.get("plan")
        if plan in ("gpt", "gpt-6", "GPT-6"):
            plan = True
        elif plan in ("qwen", "Qwen"):
            plan = False
        rungs.append(
            Rule(plan=plan, checkin=item.get("checkin"), errors=item.get("errors"))
        )
    if len(set(rungs)) != len(rungs):
        raise ValueError("ladder rungs repeat a configuration")
    return tuple(rungs)


def model_name(model_id: str) -> str:
    """``openai-chat:qwen3.6-27b`` -> ``qwen3.6-27b`` (the name escalation.jsonl logs)."""
    return model_id.split(":", 1)[1] if ":" in model_id else model_id


# ---------------------------------------------------------------------------
# the launcher's environment
# ---------------------------------------------------------------------------


def hybrid_env(
    cfg: Mapping[str, Any],
    *,
    split: str,
    try_n: int,
    endpoint: C.Endpoint,
    rule: Rule,
    timeout_s: int,
    corrections: Path | None = None,
    corrections_sha256: str = "",
    parent: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Everything ``hybrid_cell.sh`` needs; it sources the credentials file itself."""
    from scripts.ecc_robocasa.stage1 import base_env

    cred, worker, senior = cfg["credentials"], cfg["worker"], cfg["senior"]
    if split not in TARGET50_TIMEOUT_S:
        raise ValueError(f"unknown split {split!r}")
    env = base_env(cfg, parent)
    env.update(
        RC_GPU=str(endpoint.gpu),
        VLA_ENDPOINT=endpoint.url,
        ECC_RC_TRY=str(try_n),
        ECC_RC_CRED_FILE=str(cred.get("file") or ""),
        ECC_RC_WORKER_KEY_VAR=str(cred.get("worker_key_var") or ""),
        ECC_RC_SENIOR_BASE_URL_VAR=str(cred.get("senior_base_url_var") or ""),
        ECC_RC_SENIOR_KEY_VAR=str(cred.get("senior_key_var") or ""),
        ECC_RC_WORKER_BASE_URL=str(worker.get("base_url") or ""),
        ECC_RC_WORKER_MODEL=str(worker["model"]),
        ECC_RC_WORKER_EFFORT=str(worker["reasoning_effort"]),
        ECC_RC_SENIOR_MODEL=str(senior["model"]),
        ECC_RC_SENIOR_EFFORT=str(senior["reasoning_effort"]),
        ECC_RC_MAX_SHARE=str(senior["max_share"]),
        ECC_RC_PLAN="1" if rule.plan else "0",
        ECC_RC_CHECKIN=str(rule.checkin),
        ECC_RC_ERRORS=str(rule.errors),
        ECC_RC_CORRECTIONS=str(corrections or ""),
        ECC_RC_CORRECTIONS_SHA256=corrections_sha256 if corrections else "",
        ECC_RC_TIMEOUT_S=str(int(timeout_s)),
        ECC_RC_MAX_TURNS=str(cfg["episode"]["max_turns"]),
        ECC_RC_LLVMPIPE_SCRIPT=str(cfg["paths"].get("llvmpipe_index_script") or ""),
        ECC_RC_MESA_EGL_VENDOR=str(cfg["paths"]["mesa_egl_vendor_json"]),
    )
    return env


def expected_routing(
    cfg: Mapping[str, Any], rule: Rule, *, corrections_text: str = ""
) -> dict[str, Any]:
    """What a finished episode's escalation log must show for ``rule``."""
    import hashlib

    stripped = corrections_text.strip()
    return {
        "plan": rule.plan,
        "checkin": rule.checkin,
        "errors": rule.errors,
        "max_share": float(cfg["senior"]["max_share"]),
        "corrections_chars": len(stripped),
        "corrections_sha256": hashlib.sha256(stripped.encode("utf-8")).hexdigest()
        if stripped
        else None,
        "worker": model_name(str(cfg["worker"]["model"])),
        "senior": model_name(str(cfg["senior"]["model"])),
    }


# ---------------------------------------------------------------------------
# escalation evidence
# ---------------------------------------------------------------------------


def read_rows(path: Path) -> tuple[list[dict[str, Any]], int]:
    """(rows, unparsable lines) of a JSON-lines file; no file -> ([], 0)."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return [], 0
    rows, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad += 1
    return rows, bad


def config_row_problems(
    config: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
    """Differences between the wrapper's config row and the rule the episode was launched with."""
    problems = []
    for key in ("plan", "checkin", "errors", "corrections_chars"):
        if config.get(key) != expected.get(key):
            problems.append(
                f"config row {key}={config.get(key)!r}, expected {expected.get(key)!r}"
            )
    try:
        share = float(config.get("max_share"))
    except (TypeError, ValueError):
        share = -1.0
    if not math.isclose(share, float(expected["max_share"])):
        problems.append(
            f"config row max_share={config.get('max_share')!r}, expected {expected['max_share']}"
        )
    if expected.get("corrections_sha256") != config.get("corrections_sha256"):
        problems.append(
            "config row corrections_sha256 differs from the corrections file"
        )
    for key in ("worker", "senior"):
        if expected.get(key) and config.get(key) != expected[key]:
            problems.append(
                f"config row {key}={config.get(key)!r}, expected {expected[key]!r}"
            )
    if config.get("takeover"):
        problems.append(f"takeover is on (K={config['takeover']}); ECC runs without it")
    if config.get("plan_memory_chars"):
        problems.append("the plan turn is memory-fed; ECC uses the plain plan turn")
    return problems


def check_escalation(
    path: Path,
    expected: Mapping[str, Any],
    *,
    require_config_row: bool = True,
    result_valid: bool = True,
) -> dict[str, Any]:
    """Check one episode's ``escalation.jsonl`` against the rule it was launched with.

    The wrapper's routing state (planned, senior share, requests since the senior)
    is replayed from the logged requests, so every logged turn is checked against
    the expected plan switch, K and share cap: a check-in after fewer than K worker
    requests, a worker turn where a check-in was due, a plan turn with the plan
    off, no plan turn by request 6 with it on, or a control turn are mismatches.
    The config row (written by the wrapper at construction) must match exactly;
    it is the only record of N. Every senior row must carry the expected
    ``corrections_chars`` (absent when there are none). Episodes that ended
    without a valid result may have no log at all.

    Returns ``{"ok", "problems", "evidence", "counts"}``.
    """
    problems: list[str] = []
    counts: dict[str, Any] = {
        "requests": 0,
        "worker_calls": 0,
        "worker_errors": 0,
        "senior_calls": 0,
        "senior_errors": 0,
        "senior_by_reason": dict.fromkeys(SENIOR_REASONS, 0),
        "errors_trigger": 0,
        "vetoes": 0,
        "after_guidance_calls": 0,
    }
    evidence: dict[str, Any] = {"log": path.is_file(), "config_row": None}
    if not path.is_file():
        if result_valid:
            problems.append(
                "no escalation.jsonl although the episode has a valid result: the senior was off"
            )
        return {
            "ok": not problems,
            "problems": problems,
            "evidence": evidence,
            "counts": counts,
        }
    rows, bad = read_rows(path)
    if bad:
        evidence["unparsable_lines"] = bad
    plan, checkin = bool(expected["plan"]), int(expected["checkin"])
    max_share = float(expected["max_share"])
    want_chars = int(expected.get("corrections_chars") or 0)

    configs = [r["config"] for r in rows if isinstance(r.get("config"), dict)]
    if configs:
        evidence["config_row"] = configs[0]
        for config in configs:
            problems.extend(config_row_problems(config, expected))
    elif require_config_row:
        problems.append(
            "escalation.jsonl has no config row (core without the config log?)"
        )

    by_request: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        if "takeover" in row or "hand_back" in row:
            problems.append(f"takeover row at request {row.get('request')}")
            continue
        if "veto" in row:
            counts["vetoes"] += 1
            continue
        role, request = row.get("role"), row.get("request")
        if role not in ("worker", "senior") or not isinstance(request, int):
            continue
        entry = by_request.setdefault(request, {"worker": [], "senior": []})
        entry[role].append(row)
        failed = "error" in row
        if role == "senior":
            counts["senior_errors" if failed else "senior_calls"] += 1
            if not failed and row.get("reason") in SENIOR_REASONS:
                counts["senior_by_reason"][row["reason"]] += 1
            if row.get("corrections_chars", 0) != want_chars:
                problems.append(
                    f"senior request {request}: corrections_chars "
                    f"{row.get('corrections_chars', 0)}, expected {want_chars}"
                )
            if expected.get("senior") and row.get("model") != expected["senior"]:
                problems.append(
                    f"senior request {request} went to model {row.get('model')!r}"
                )
        else:
            counts["worker_errors" if failed else "worker_calls"] += 1
            if str(row.get("reason") or "").startswith("after_"):
                counts["after_guidance_calls"] += 1
            if expected.get("worker") and row.get("model") != expected["worker"]:
                problems.append(
                    f"worker request {request} went to model {row.get('model')!r}"
                )
    counts["errors_trigger"] = counts["senior_by_reason"]["errors"]
    counts["requests"] = max(by_request) if by_request else 0

    planned, seniors, since = not plan, 0, 0
    k_low, k_high, plan_at = 0, None, None
    numbers = sorted(by_request)
    missing = sorted(set(range(1, counts["requests"] + 1)) - set(numbers))
    if missing:
        problems.append(f"requests missing from the log: {missing[:10]}")

    def worker_turn(request: int, capped: bool) -> None:
        nonlocal k_low
        if not planned:
            if plan and request >= PLAN_FALLBACK_REQUEST:
                problems.append(
                    f"no plan turn by request {request} although the plan turn is on"
                )
            return
        if capped:
            return
        if since >= checkin:
            problems.append(
                f"request {request}: {since} worker requests since the senior and below the "
                f"share cap, but no check-in (K={checkin})"
            )
        k_low = max(k_low, since + 1)

    for request in numbers:
        entry = by_request[request]
        capped = seniors >= max_share * request
        plain = [w for w in entry["worker"] if not w.get("reason")]
        if entry["senior"]:
            senior = entry["senior"][0]
            reason = senior.get("reason")
            if reason == "plan":
                if not plan:
                    problems.append(
                        f"plan turn at request {request} although the plan turn is off"
                    )
                elif planned:
                    problems.append(f"second plan turn at request {request}")
                plan_at = request
            elif reason == "giveup":
                worker_turn(request, capped)  # the trigger chose the worker first
            elif reason in ("stall", "errors", "checkin"):
                if not planned:
                    problems.append(
                        f"{reason} turn at request {request} before the plan turn"
                    )
                if capped:
                    problems.append(
                        f"{reason} turn at request {request} above the share cap {max_share}"
                    )
                if reason == "checkin":
                    if since < checkin:
                        problems.append(
                            f"check-in at request {request} after {since} worker requests (K={checkin})"
                        )
                    k_high = since if k_high is None else min(k_high, since)
            else:
                problems.append(
                    f"senior turn with reason {reason!r} at request {request}"
                )
            if "error" in senior:
                break  # the senior call failed: the episode ended here
            seniors += 1
            since = 0
            planned = True
        elif plain:
            worker_turn(request, capped)
            if "error" in plain[0]:
                break
            since += 1
    evidence.update(
        plan_turn_at=plan_at,
        checkin_bounds=[k_low, k_high],
        checkin_gaps_seen=k_high is not None,
    )
    return {
        "ok": not problems,
        "problems": problems,
        "evidence": evidence,
        "counts": counts,
    }


# ---------------------------------------------------------------------------
# the episode record
# ---------------------------------------------------------------------------


def transcript_summary(try_dir: Path) -> dict[str, Any]:
    """Finish status and turns from RPent's transcript (empty after a planner timeout)."""
    out: dict[str, Any] = {"finish_status": None, "turns": None}
    for path in sorted(try_dir.glob("transcript_*.json")):
        record = C.read_json(path, {}) or {}
        finish = record.get("finish") if isinstance(record.get("finish"), dict) else {}
        stats = record.get("stats") if isinstance(record.get("stats"), dict) else {}
        out["finish_status"] = finish.get("status")
        out["finish_summary"] = str(finish.get("summary") or "")[:600] or None
        out["turns"] = stats.get("turns_used")
    if out["turns"] is None:
        try:
            text = (try_dir / "run.log").read_text(errors="replace")
            out["turns"] = len(re.findall(r"=== turn \d+/\d+ ===", text)) or None
        except OSError:
            pass
    return out


def episode_record(
    try_dir: Path, pricing: Mapping[str, float], check: Mapping[str, Any]
) -> dict[str, Any]:
    """Outcome, GPT-6 use and cost of one finished launch."""
    outcome = C.read_outcome(try_dir)
    usage = C.episode_usage(try_dir, pricing) if try_dir.exists() else {}
    counts = check.get("counts") or {}
    return {
        "valid": bool(outcome.get("valid")),
        "success": bool(outcome.get("success")) if outcome.get("valid") else None,
        "termination_reason": outcome.get("termination_reason"),
        "elapsed_s": outcome.get("elapsed_s"),
        **transcript_summary(try_dir),
        "requests": counts.get("requests", 0),
        "gpt_calls": counts.get("senior_calls", 0),
        "gpt_calls_by_reason": counts.get("senior_by_reason", {}),
        "gpt_errors": counts.get("senior_errors", 0),
        "errors_trigger": counts.get("errors_trigger", 0),
        "vetoes": counts.get("vetoes", 0),
        "qwen": {
            "calls": usage.get("worker_calls", 0),
            "input_tokens": usage.get("worker_in", 0),
            "output_tokens": usage.get("worker_out", 0),
            "usd": round(float(usage.get("worker_usd") or 0.0), 4),
        },
        "gpt6": {
            "calls": usage.get("senior_calls", 0),
            "input_tokens": usage.get("senior_in", 0),
            "cached_input_tokens": usage.get("senior_cached", 0),
            "output_tokens": usage.get("senior_out", 0),
            "usd": round(float(usage.get("senior_usd") or 0.0), 4),
        },
        "usd": round(float(usage.get("usd") or 0.0), 4),
        "usage_source": usage.get("source"),
        "evidence": {
            "ok": check.get("ok"),
            "problems": list(check.get("problems") or [])[:20],
            **(check.get("evidence") or {}),
        },
    }


def file_tool_audit(
    try_dir: Path, *, own: Path, memory: Path, code_dir: Path, shared: Sequence[Path]
) -> dict[str, Any]:
    """File-tool paths outside the episode's own folder and the memory corpus it reads.

    Reads of the memory corpus go through RoboCasa's evaluation memory manager,
    which allows only this task's selected files; any other path in ``shared``
    (other tasks' or rungs' folders, other runs, the cloud corpus) is flagged.
    """
    from scripts.ecc_robocasa.stage1 import file_tool_paths

    own_r, memory_r, code_r = (os.path.realpath(p) for p in (own, memory, code_dir))
    shared_r = [os.path.realpath(p) for p in shared]

    def under(real: str, base: str) -> bool:
        return real == base or real.startswith(base + os.sep)

    calls = file_tool_paths(try_dir)
    found: dict[str, set[str]] = {
        "memory_reads": set(),
        "cross_reads": set(),
        "cross_writes": set(),
        "memory_writes": set(),
        "code_dir_writes": set(),
        "outside": set(),
    }
    for tool, path in calls:
        if not path:
            continue
        full = path if os.path.isabs(path) else os.path.join(code_r, path)
        real = os.path.realpath(full)
        write = tool == "write_text_file"
        if under(real, own_r):
            continue
        if under(real, memory_r):
            found["memory_writes" if write else "memory_reads"].add(real)
        elif any(under(real, base) for base in shared_r):
            found["cross_writes" if write else "cross_reads"].add(real)
        elif under(real, code_r):
            if write:
                found["code_dir_writes"].add(real)
        else:
            found["outside"].add(real)
    return {"calls": len(calls), **{k: sorted(v)[:100] for k, v in found.items()}}
