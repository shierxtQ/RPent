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

"""RC-ECC stage 3 reflection: episode digest, prompt, reply parsing and entry bookkeeping.

Ported from ``scripts/ecc/run_ecc_stage3.py`` (the LIBERO-PRO stage-3 runner, 26bdd43; the
source function is named at each port). Kept as it is there: the correction-entry
model and its rendering, the reply schema (``applied`` / ``add`` / ``revise`` /
``remove`` / ``note``) and the bookkeeping of ``apply``:

- an entry counts as used only when the reflection lists it in ``applied``;
- entries applied in a successful episode, and entries written or revised from
  one, become verified;
- an entry applied in ``fails_to_drop`` failed episodes without any success is
  dropped; a revised entry restarts its failure count;
- at most ``max_entries`` entries, at most ``field_chars`` characters per field.

Changed for RoboCasa:

- **Tool outcomes first.** A RoboCasa robot tool returns the new state
  (``view_env_state`` of the recorded step, ``robots/robocasa/toolkit.py``
  ``get_env_state``); the primitive's outcome (``log.result.ok`` / ``status`` /
  ``error``) sits about 550-600 characters into that JSON, behind the robot
  state. The LIBERO digest clips results to 160-300 characters, so GPT-6 would never
  see that a motion failed. :func:`tool_outcome` puts the outcome fields (skill,
  ok, status, error, task_progress, success, ...) first and clips only the rest.
- **Episodes without a transcript.** After a planner timeout RPent's transcript
  keeps only the first user message (``rpent/planner/api_loop.py``). The LIBERO digest
  then finds no planner turn and skips the episode; here an episode whose
  ``run.log`` holds turns is rebuilt from it (texts and calls as logged, tool
  results clipped by the log) with each robot outcome taken in full from
  ``states.json``; only an episode with no planner turn anywhere is skipped.
- **The prompt** describes RoboCasa (mobile manipulator, RLDX skills,
  primitives, task_progress) instead of a LIBERO-PRO arm, takes the task
  language from the episode and the rule as stage 2 froze it, and names the
  episode's termination reason.

Opt-in variants for stage 3 v2 (off by default; with both off every function
returns what it did before):

- **Scene-free entries** (``scene_free``): the prompt asks for rules that hold
  in any scene of the task, and :func:`scene_rejections` checks every added or
  revised entry for scene details (coordinate and pixel lists, pixel positions,
  step numbers, numbers with two or more decimals, object instance ids, the
  episode's own task language). A rejected entry is sent back once
  (:func:`build_fix_prompt`, :func:`merge_rewrites`) and dropped if its rewrite
  still fails.
- **Cross-scene verification** (``cross_scene``): an entry becomes verified
  only when it is applied in a successful episode whose seed differs from the
  seed of the episode that wrote its current text (``text_seed``); an entry
  written or revised from a success stays a candidate, and a verified entry's
  text is frozen (revisions of it are ignored and recorded). Every entry keeps
  its provenance: ``written_seed``, ``text_seed``, ``text_from``,
  ``applied_in`` (episode, seed, success) and ``verified_by``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: run_ecc_stage3.SESSIONS, ATTEMPTS: 5 sessions x 3 attempts = 15 single-shot episodes.
SESSIONS, ATTEMPTS = 5, 3
TRIGGERS = ("plan", "checkin", "errors", "stall", "giveup", "any")
ENTRY_FIELDS = ("situation", "diagnosis", "fix")
#: The JSON keys of an entry that are shown to GPT-6 (run_ecc_stage3.reflect).
SHOWN_KEYS = ("id", "status", "trigger", "situation", "diagnosis", "fix")
#: Senior turns that carry no new information for a digest (a takeover turn is never used here).
SENIOR_ROLE = "senior"


@dataclass(frozen=True)
class Settings:
    """The module constants of run_ecc_stage3 (MAX_ENTRIES ... CONTEXT_AFTER), configurable."""

    max_entries: int = 8
    field_chars: int = 500
    fails_to_drop: int = 2
    context_before: int = 3
    context_after: int = 6
    #: run_ecc_stage3.reflect's ``d["text"][:40000]`` cap on the interventions block.
    interventions_chars: int = 40000
    #: run_ecc_stage3.reflect's ``_short(summary, 600)``.
    summary_chars: int = 600

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> Settings:
        known = {f: section[f] for f in cls.__dataclass_fields__ if f in section}
        settings = cls(**known)
        for name in cls.__dataclass_fields__:
            value = getattr(settings, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"stage3.{name} must be a nonnegative integer")
        if settings.max_entries < 1 or settings.fails_to_drop < 1:
            raise ValueError("stage3.max_entries and stage3.fails_to_drop must be >= 1")
        return settings


def episode_name(n: int, attempts: int = ATTEMPTS) -> str:
    """Label of the ``n``-th episode (0-based): ``s1_a1`` .. ``s5_a3`` (run_ecc_stage3.run_task)."""
    return f"s{n // attempts + 1}_a{n % attempts + 1}"


# ---------------------------------------------------------------------------
# correction memory (run_ecc_stage3.render / save_entries)
# ---------------------------------------------------------------------------


def render(entries: Sequence[Mapping[str, Any]]) -> str:
    """The text appended to every senior note (run_ecc_stage3.render, unchanged)."""
    lines = []
    for e in entries:
        lines.append(
            f"- [{e['id']}] ({e['status']}; trigger: {e['trigger']}) situation: {e['situation']} | "
            f"diagnosis: {e['diagnosis']} | fix: {e['fix']}"
        )
    return "\n".join(lines)


def render_file(entries: Sequence[Mapping[str, Any]]) -> str:
    """The ``<tag>.md`` file content run_ecc_stage3.save_entries writes."""
    return render(entries) + ("\n" if entries else "")


def verified(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The kept memory: verified entries only (run_ecc_stage3.run_task's final step)."""
    return [dict(e) for e in entries if e.get("status") == "verified"]


# ---------------------------------------------------------------------------
# tool results: outcome fields first (RoboCasa fix)
# ---------------------------------------------------------------------------

#: Fields of a primitive's result (``log.result``) shown before anything is clipped, in
#: this order (robots/robocasa/primitives.py, rldx_skill.py result dictionaries).
RESULT_FIELDS = (
    "ok",
    "status",
    "error",
    "reason",
    "stage",
    "stuck",
    "interrupted",
    "final_dist",
    "moved",
    "steps",
    "chunks",
    "steps_applied",
    "grasped",
    "grasp_detected",
    "held_apart",
    "grasp_obj",
    "peak_lift",
    "base_drift",
    "notice",
)
#: Top-level fields of a tool result that read like an outcome (tool errors, finish, reset).
TOP_FIELDS = ("error", "reason", "ok", "status", "notice", "attempt")
#: Top-level fields already summarised (or useless to a text-only reviewer).
SUMMARISED = frozenset(
    {
        "log",
        "state",
        "images",
        "task_progress",
        "task_language",
        "success",
        "robocasa_terminated",
        "step",
        "vla_desync",
        "agent_elapsed_s",
        "content",
        *TOP_FIELDS,
    }
)
OUTCOME_VALUE_CHARS = 300
TASK_PROGRESS_CHARS = 400


def _short(value: Any, n: int) -> str:
    """Whitespace-collapsed text clipped to ``n`` characters (run_ecc_stage3._short)."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = re.sub(r"\s+", " ", text)
    return text if len(text) <= n else text[:n] + "..."


def _as_dict(content: Any) -> dict[str, Any] | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except ValueError:
            return None
        return value if isinstance(value, dict) else None
    return None


def _value(value: Any, n: int = OUTCOME_VALUE_CHARS) -> str:
    """One outcome value as JSON (strings quoted), clipped."""
    return _short(json.dumps(value, ensure_ascii=False, default=str), n)


def _round3(values: Any) -> str | None:
    if (
        isinstance(values, list)
        and values
        and all(isinstance(v, (int, float)) for v in values)
    ):
        return "[" + ",".join(f"{float(v):.3f}" for v in values) + "]"
    return None


def tool_outcome(content: Any) -> tuple[str, str]:
    """``(outcome, rest)`` of one tool result.

    ``outcome`` lists, in a fixed order, the skill that produced the state
    (``log.command.action``), the primitive's result fields (``log.result``:
    ok, status, error, ...), a tool-level error or notice, ``task_progress``,
    ``success``, ``robocasa_terminated``, the step and the end-effector
    position and gripper width. ``rest`` is the remaining JSON, compacted (the
    caller clips it). A result that is not a JSON object has no outcome.
    """
    data = _as_dict(content)
    if data is None:
        return "", _short(content if content is not None else "", 1 << 30)
    parts: list[str] = []
    log = data.get("log") if isinstance(data.get("log"), dict) else None
    if log is not None:
        command = log.get("command") if isinstance(log.get("command"), dict) else None
        if command and command.get("action"):
            parts.append(f"skill={command['action']}")
        result = log.get("result") if isinstance(log.get("result"), dict) else None
        if result:
            for key in RESULT_FIELDS:
                if key in result:
                    parts.append(f"{key}={_value(result[key])}")
    for key in TOP_FIELDS:
        if key in data:
            parts.append(f"{key}={_value(data[key])}")
    if "task_progress" in data:
        parts.append(
            f"task_progress={_value(data['task_progress'], TASK_PROGRESS_CHARS)}"
        )
    for key, label in (("success", "success"), ("robocasa_terminated", "terminated")):
        if key in data:
            parts.append(f"{label}={_value(data[key])}")
    if "step" in data:
        parts.append(f"step={_value(data['step'])}")
    state = data.get("state") if isinstance(data.get("state"), dict) else {}
    eef = _round3(state.get("robot0_eef_pos"))
    if eef:
        parts.append(f"eef={eef}")
    grip = state.get("robot0_gripper_qpos")
    if isinstance(grip, list) and grip and isinstance(grip[0], (int, float)):
        parts.append(f"grip={float(grip[0]):.3f}")
    # Perception and file tools: their own short summaries.
    if isinstance(data.get("summary"), dict):
        parts.append(f"summary={_value(data['summary'])}")
    if "path" in data and ("size" in data or "bytes_written" in data):
        parts.append(f"path={_value(data['path'])}")
    rest = {
        k: v
        for k, v in data.items()
        if k not in SUMMARISED
        and not (k == "summary" and isinstance(v, dict))
        and not (k == "path" and ("size" in data or "bytes_written" in data))
    }
    return " ".join(parts), (json.dumps(rest, ensure_ascii=False) if rest else "")


def show_result(content: Any, n: int) -> str:
    """A tool result for the digest: ``[outcome fields] rest`` with only ``rest`` clipped to ``n``."""
    outcome, rest = tool_outcome(content)
    rest = _short(rest, n) if rest else ""
    if outcome:
        return f"[{outcome}]" + (f" {rest}" if rest else "")
    return rest


# ---------------------------------------------------------------------------
# episode turns: the transcript, or run.log after a planner timeout
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    """One model response: texts and tool calls with their results."""

    texts: list[str]
    #: (tool name, arguments as logged, result content or None)
    calls: list[tuple[str, Any, Any]]


def _turn(turn: Turn, n_text: int, n_arg: int, n_res: int) -> str:
    """One turn as digest lines (run_ecc_stage3._turn, results through :func:`show_result`)."""
    parts = []
    for text in turn.texts:
        if text.strip():
            parts.append(f"  says: {_short(text, n_text)}")
    for name, args, result in turn.calls:
        shown = show_result(result, n_res) if result is not None else "(no result)"
        parts.append(
            f"  call {name}({_short(args if args is not None else {}, n_arg)}) -> {shown}"
        )
    return "\n".join(parts)


def transcript_turns(out: Path) -> tuple[list[Turn], dict[str, Any]]:
    """Assistant turns of the last ``transcript_*.json`` and its finish record."""
    transcripts = sorted(out.glob("transcript_*.json"))
    try:
        record = json.loads(transcripts[-1].read_text()) if transcripts else {}
    except (OSError, ValueError):
        record = {}
    messages = record.get("messages") or []
    results = {
        m.get("tool_call_id"): m.get("content")
        for m in messages
        if isinstance(m, dict) and m.get("role") == "tool"
    }
    turns = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        texts, calls = [], []
        for part in message.get("content") or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                texts.append(str(part.get("text") or ""))
            elif part.get("type") == "tool_use":
                calls.append(
                    (
                        str(part.get("name")),
                        part.get("input"),
                        results.get(part.get("id")),
                    )
                )
        turns.append(Turn(texts, calls))
    finish = record.get("finish") if isinstance(record.get("finish"), dict) else {}
    return turns, finish


_LOG_LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [A-Z] \[([^\]]+)\] ?(.*)$")
_TURN = re.compile(r"^=== turn \d+/\d+ ===$")
_CALL = re.compile(r"^\[tool>\] ([A-Za-z0-9_]+)\((.*)\)$", re.S)
_RESULT = re.compile(r"^\[tool<\] ([A-Za-z0-9_]+): (.*)$", re.S)
_STEP = re.compile(r'^\{\s*"step":\s*(\d+)\s*,')
_FINISH = re.compile(r"^FINISH called: (\{.*\})\s*$", re.S)


def _states(out: Path) -> list[dict[str, Any]]:
    try:
        steps = json.loads((out / "states.json").read_text()).get("steps") or []
    except (OSError, ValueError, AttributeError):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _from_state(step: Mapping[str, Any]) -> dict[str, Any]:
    """The tool-result shape (``view_env_state``) of one ``states.json`` step."""
    extras = step.get("extras") if isinstance(step.get("extras"), dict) else {}
    return {
        "step": step.get("step_idx"),
        "task_progress": extras.get("task_progress", {}),
        "state": step.get("state") or {},
        "robocasa_terminated": step.get("terminated"),
        "success": extras.get("success", False),
        "log": {"command": step.get("command"), "result": step.get("result")},
    }


def run_log_turns(out: Path) -> tuple[list[Turn], dict[str, Any]]:
    """Turns as ``run.log`` records them (``api_loop._log_response`` / ``_log_tool_result``).

    Texts are logged in full (continuation lines included), arguments clipped to
    250 and results to 350 characters. A result that starts with ``{"step": N``
    is a recorded state: it is replaced by step N of ``states.json``, so the
    robot outcome is complete. Turns are numbered in file order (continue notes
    restart the logged counter, the escalation requests do not).
    """
    try:
        lines = (out / "run.log").read_text(errors="replace").splitlines()
    except OSError:
        return [], {}
    steps = {s.get("step_idx"): s for s in _states(out)}
    turns: list[Turn] = []
    finish: dict[str, Any] = {}
    last_text: list[str] | None = None  # the [model] text continuation lines belong to
    for line in lines:
        match = _LOG_LINE.match(line)
        if not match:
            if last_text is not None:
                last_text[-1] += "\n" + line
            continue
        logger, body = match.groups()
        last_text = None
        if logger != "api_loop":
            continue
        if _TURN.match(body):
            turns.append(Turn([], []))
            continue
        if body.startswith("FINISH called: "):
            found = _FINISH.match(body)
            try:
                value = ast.literal_eval(found.group(1)) if found else None
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, dict):
                finish = value
            continue
        if not turns:
            continue
        turn = turns[-1]
        if body.startswith("[model] "):
            turn.texts.append(body[len("[model] ") :])
            last_text = turn.texts
        elif body.startswith("[tool>] "):
            call = _CALL.match(body)
            if call:
                turn.calls.append((call.group(1), call.group(2), None))
        elif body.startswith("[tool<] "):
            result = _RESULT.match(body)
            if not result:
                continue
            name, content = result.groups()
            step = _STEP.match(content)
            if step and int(step.group(1)) in steps:
                content = _from_state(steps[int(step.group(1))])
            for i, (called, args, got) in enumerate(turn.calls):
                if called == name and got is None:
                    turn.calls[i] = (called, args, content)
                    break
    return turns, finish


# ---------------------------------------------------------------------------
# episode digest (run_ecc_stage3.digest)
# ---------------------------------------------------------------------------


def escalation_rows(out: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        lines = (out / "escalation.jsonl").read_text(errors="replace").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def digest(out: Path, settings: Settings = Settings()) -> dict[str, Any]:
    """GPT-6's interventions of one episode, with the turns around them (run_ecc_stage3.digest).

    The turns come from the transcript, or from ``run.log`` when the transcript
    holds no planner turn (``source``). Request k of ``escalation.jsonl`` is the
    k-th model response.
    """
    turns, finish = transcript_turns(out)
    source = "transcript"
    if not turns:
        logged, logged_finish = run_log_turns(out)
        if logged:
            turns, finish, source = logged, finish or logged_finish, "run.log"
        else:
            source = "none"
    rows = escalation_rows(out)
    seniors = [r for r in rows if r.get("role") == SENIOR_ROLE and "error" not in r]
    blocks = []
    for r in seniors:
        try:
            i = int(r["request"]) - 1  # request k is the k-th assistant message
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= i < len(turns):
            continue
        before = "\n".join(
            f" worker turn {j + 1}:\n" + _turn(turns[j], 200, 160, 200)
            for j in range(max(0, i - settings.context_before), i)
        )
        after = "\n".join(
            f" turn {j + 1}:\n" + _turn(turns[j], 160, 120, 160)
            for j in range(i + 1, min(len(turns), i + 1 + settings.context_after))
        )
        blocks.append(
            f"### Intervention at turn {i + 1} (trigger: {r.get('reason')})\n"
            f"Before:\n{before or ' (episode start)'}\n"
            f"Your turn:\n{_turn(turns[i], 1500, 300, 300)}\n"
            f"After:\n{after or ' (episode ended)'}"
        )
    by_reason: dict[str, int] = {}
    for r in seniors:
        by_reason[str(r.get("reason"))] = by_reason.get(str(r.get("reason")), 0) + 1
    return {
        "interventions": len(seniors),
        "by_reason": by_reason,
        "shown": len(blocks),
        "turns": len(turns),
        "source": source,
        "finish": finish or {},
        "text": "\n\n".join(blocks),
    }


def task_language(out: Path) -> str:
    """The episode's task language: ``states.json``, else a tool result, else the audit.

    run_ecc_stage3.task_language read only the model-written audit; the recorded state
    is the environment's own language.
    """
    for step in _states(out)[:1]:
        extras = step.get("extras") if isinstance(step.get("extras"), dict) else {}
        if extras.get("task_language"):
            return str(extras["task_language"])
    turns, _ = transcript_turns(out)
    for turn in turns:
        for _, _, result in turn.calls:
            data = _as_dict(result)
            if data and data.get("task_language"):
                return str(data["task_language"])
    for audit in out.glob("*_s[0-9]*.json"):  # run_ecc_stage3.task_language
        if audit.name.startswith("transcript_"):
            continue
        try:
            lang = json.loads(audit.read_text()).get("task_language")
        except (OSError, ValueError, AttributeError):
            continue
        if lang:
            return str(lang)
    return ""


# ---------------------------------------------------------------------------
# prompt (run_ecc_stage3.PROMPT, adapted to RoboCasa)
# ---------------------------------------------------------------------------

PROMPT = """You are GPT-6, the senior (cloud) planner that a small on-device worker model (Qwen) \
calls for help while it controls a mobile manipulator (a mobile base with one arm and a gripper) \
in a RoboCasa kitchen. The worker acts through tools: the RLDX visuomotor skills rldx_skill (arm \
and base) and rldx_arm (arm, base clamped), which take the task language verbatim; scripted \
primitives (move_to, move_delta, rotate_pitch, set_gripper, release, scripted_grasp, navigate_to, \
move_base); perception (view_env_state, back_project_batch, query_world_map); memory files; and \
finish. A robot tool result reports the primitive's outcome (ok, status, error, ...), the new \
state and task_progress, the current values of the task's success check. You keep a correction \
memory for this one task: short entries of failure situations you diagnosed and the fix that \
works. It is shown only to you at every later intervention.

Task: {language}
Call rule of this task: {rule}
Episode outcome: {outcome} after {turns} turns ({termination}); final summary: {summary}
{source_note}
Your interventions in this episode, with the worker turns around them (each tool result starts \
with its outcome fields in brackets):
{interventions}

Current correction memory (JSON):
{entries}

Edit the memory from this episode. Keep entries concrete: the observable situation (what the \
worker was doing, what the state, task_progress or images showed), the root cause, and the exact \
corrective action (tool, numbers, order). Do not restate generic task strategy that the task \
memory already covers; record what went wrong for the worker and how to correct it. If the \
episode failed, say what your guidance missed and revise or remove entries that did not help. \
Prefer revising an existing entry over adding a near duplicate. Each field at most two sentences. \
At most {max_entries} entries in total.
In "applied", list the ids of existing entries whose situation actually arose in this episode and \
whose fix your guidance applied; leave out entries that were not relevant.

Reply with JSON only:
{{"applied": ["id", ...],
 "add": [{{"trigger": "plan|checkin|errors|stall|giveup|any", "situation": "...", "diagnosis": "...", "fix": "..."}}],
 "revise": [{{"id": "...", "situation": "...", "diagnosis": "...", "fix": "..."}}],
 "remove": ["id", ...],
 "note": "one sentence on what this episode taught"}}"""

RUN_LOG_NOTE = (
    "(The transcript was lost when the episode hit its planner time limit; the turns below "
    "were rebuilt from the run log, whose texts are complete but whose tool arguments and "
    "results are clipped; robot outcomes come from the recorded states.)\n"
)

#: The sentence of PROMPT that asks for concrete entries; the scene-free prompt replaces it.
CONCRETE_RULE = (
    "Keep entries concrete: the observable situation (what the worker was doing, what the "
    "state, task_progress or images showed), the root cause, and the exact corrective action "
    "(tool, numbers, order). "
)
SCENE_FREE_RULE = (
    "The memory is used on other scenes of this task: later episodes run in other kitchen "
    "layouts, with other object positions and possibly another target object or fixture. Write "
    "every entry as a rule that holds in any scene of this task: the situation as the worker "
    "can recognise it (what it was doing, what a tool outcome, task_progress or the images "
    "showed), the root cause, and the fix as a general procedure (which tools, in which order, "
    "and which observations decide the next step). Never write world coordinates, pixel "
    "positions, step or turn numbers, measured values with two or more decimals, object "
    "instance ids (such as dish0) or this scene's poses, and do not quote this scene's task "
    'language: write "the task language verbatim" instead. Entries that do are sent back for '
    "rewriting and dropped if they still do. "
)
#: Inserted before PROMPT's length limits when verified entries are frozen (cross-scene verification).
FROZEN_RULE = (
    'Entries with status "verified" are frozen: do not revise them (a revision is ignored); '
    "remove one only if this episode shows it is wrong. "
)
LIMITS = "Each field at most two sentences."


def prompt_template(*, scene_free: bool = False, frozen_verified: bool = False) -> str:
    """PROMPT, or its stage-3 v2 variant (scene-free entries, frozen verified entries)."""
    template = PROMPT
    if scene_free:
        if CONCRETE_RULE not in template:  # pragma: no cover - guards edits of PROMPT
            raise RuntimeError("PROMPT no longer holds the concrete-entry sentence")
        template = template.replace(CONCRETE_RULE, SCENE_FREE_RULE)
    if frozen_verified:
        if template.count(LIMITS) != 1:  # pragma: no cover - guards edits of PROMPT
            raise RuntimeError("PROMPT no longer holds its length limits once")
        template = template.replace(LIMITS, FROZEN_RULE + LIMITS)
    return template


def build_prompt(
    d: Mapping[str, Any],
    *,
    language: str,
    rule: Mapping[str, Any],
    success: bool,
    termination: str,
    entries: Sequence[Mapping[str, Any]],
    settings: Settings = Settings(),
    scene_free: bool = False,
    frozen_verified: bool = False,
) -> str:
    """The prompt assembly of run_ecc_stage3.reflect, with the RoboCasa fields.

    ``scene_free`` and ``frozen_verified`` select stage 3 v2's instructions
    (:func:`prompt_template`); both off, the prompt is PROMPT's.
    """
    template = (
        PROMPT
        if not (scene_free or frozen_verified)
        else prompt_template(scene_free=scene_free, frozen_verified=frozen_verified)
    )
    return template.format(
        language=language or "(see task memory)",
        rule=json.dumps(dict(rule), ensure_ascii=False),
        outcome="SUCCESS" if success else "FAILURE",
        turns=d["turns"],
        termination=termination or "unknown",
        summary=_short(
            (d.get("finish") or {}).get("summary", ""), settings.summary_chars
        ),
        source_note=RUN_LOG_NOTE if d.get("source") == "run.log" else "",
        interventions=clip_blocks(d["text"], settings.interventions_chars),
        entries=json.dumps(
            [{k: e[k] for k in SHOWN_KEYS} for e in entries],
            ensure_ascii=False,
            indent=1,
        ),
        max_entries=settings.max_entries,
    )


def clip_blocks(text: str, limit: int) -> str:
    """The first ``limit`` characters of the interventions (run_ecc_stage3.reflect's cap),
    cut at an intervention boundary, with a note naming how many were left out."""
    if len(text) <= limit:
        return text
    blocks = text.split("\n\n### Intervention")
    kept, size = [blocks[0]], len(blocks[0])
    for block in blocks[1:]:
        size += len(block) + len("\n\n### Intervention")
        if size > limit:
            break
        kept.append(block)
    if len(kept) == 1 and len(blocks[0]) > limit:
        kept = [blocks[0][:limit] + "..."]
    left_out = len(blocks) - len(kept)
    return "\n\n### Intervention".join(kept) + (
        f"\n\n({left_out} later intervention(s) left out: the prompt keeps the first {limit} characters)"
    )


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# reply and bookkeeping (run_ecc_stage3.parse_edits / apply)
# ---------------------------------------------------------------------------

NO_EDITS: dict[str, Any] = {
    "applied": [],
    "add": [],
    "revise": [],
    "remove": [],
    "note": "",
}


def parse_edits(text: str) -> dict[str, Any]:
    """The edits in GPT-6's reply (run_ecc_stage3.parse_edits).

    As there: the outermost ``{...}`` is parsed, missing keys are empty. Items of
    the wrong type (an ``add`` that is not an object, an id that is a list) are
    dropped and counted under ``malformed`` instead of failing in ``apply``.
    """
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError("no JSON object in reflection reply")
    edits = json.loads(match.group(0))
    if not isinstance(edits, dict):
        raise ValueError("reflection reply JSON is not an object")
    out: dict[str, Any] = {"note": str(edits.get("note") or "")}
    malformed = 0
    for key in ("applied", "remove"):
        value = edits.get(key) or []
        value = [value] if isinstance(value, (str, int)) else value
        if not isinstance(value, list):
            malformed += 1
            value = []
        kept = [
            str(v)
            for v in value
            if isinstance(v, (str, int)) and not isinstance(v, bool)
        ]
        malformed += len(value) - len(kept)
        out[key] = kept
    for key in ("add", "revise"):
        value = edits.get(key) or []
        value = [value] if isinstance(value, dict) else value
        if not isinstance(value, list):
            malformed += 1
            value = []
        kept = [v for v in value if isinstance(v, dict)]
        malformed += len(value) - len(kept)
        out[key] = kept
    if malformed:
        out["malformed"] = malformed
    return out


def apply(
    entries: Sequence[Mapping[str, Any]],
    edits: Mapping[str, Any],
    episode: str,
    success: bool,
    used: set[str],
    settings: Settings = Settings(),
    *,
    seed: int | None = None,
    cross_scene: bool = False,
) -> list[dict[str, Any]]:
    """The edited memory (run_ecc_stage3.apply, unchanged apart from the settings).

    ``cross_scene`` (stage 3 v2, with the episode's ``seed``): an applied entry
    becomes verified only in a successful episode on another seed than its
    ``text_seed``; added and revised entries stay candidates; a verified entry
    is never revised. The counters and the drop rule are run_ecc_stage3's.
    """
    if cross_scene:
        return _apply_cross_scene(
            entries, edits, episode, success, used, settings, seed
        )
    entries = [dict(e) for e in entries]
    # Outcome bookkeeping for entries the senior applied in this episode.
    for e in entries:
        if e["id"] in used:
            e["used_in"] = e.get("used_in", []) + [episode]
            if success:
                e["successes"] = e.get("successes", 0) + 1
                e["status"] = "verified"
            else:
                e["failures"] = e.get("failures", 0) + 1
    remove = set(edits["remove"])
    revised = {str(r.get("id")) for r in edits["revise"]}
    entries = [
        e
        for e in entries
        if e["id"] not in remove
        and (
            e["id"] in revised
            or not (
                e.get("failures", 0) >= settings.fails_to_drop
                and not e.get("successes")
            )
        )
    ]
    by_id = {e["id"]: e for e in entries}
    for r in edits["revise"]:
        e = by_id.get(str(r.get("id")))
        if e is None:
            continue
        for k in ENTRY_FIELDS:
            if r.get(k):
                e[k] = str(r[k])[: settings.field_chars]
        e["revised_in"] = e.get("revised_in", []) + [episode]
        e["failures"] = 0  # the revised fix has not failed yet
        if success:
            e["status"] = "verified"
    stem, next_id = (
        episode.rsplit("/", 1)[-1],
        1,
    )  # ids name the writing episode: s2_a1.3
    for a in edits["add"]:
        if len(entries) >= settings.max_entries:
            break
        if not all(a.get(k) for k in ENTRY_FIELDS):
            continue
        entries.append(
            {
                "id": f"{stem}.{next_id}",
                "trigger": str(a.get("trigger") or "any"),
                **{k: str(a[k])[: settings.field_chars] for k in ENTRY_FIELDS},
                "status": "verified" if success else "candidate",
                "written_in": episode,
                "used_in": [],
                "successes": 0,
                "failures": 0,
            }
        )
        next_id += 1
    return entries


def _cross_scene_verifies(
    e: Mapping[str, Any], success: bool, seed: int | None
) -> bool:
    """Whether applying entry ``e`` in this episode verifies it: a success on another scene."""
    return (
        success
        and seed is not None
        and e.get("status") != "verified"
        and e.get("text_seed") is not None
        and e.get("text_seed") != seed
    )


def frozen_ids(
    entries: Sequence[Mapping[str, Any]],
    used: Collection[str],
    success: bool,
    seed: int | None,
) -> set[str]:
    """Entries whose text a reflection may not revise (cross-scene verification): the verified
    ones, and the ones this episode verifies."""
    return {
        str(e["id"])
        for e in entries
        if e.get("status") == "verified"
        or (str(e["id"]) in used and _cross_scene_verifies(e, success, seed))
    }


def _apply_cross_scene(
    entries: Sequence[Mapping[str, Any]],
    edits: Mapping[str, Any],
    episode: str,
    success: bool,
    used: set[str],
    settings: Settings,
    seed: int | None,
) -> list[dict[str, Any]]:
    """:func:`apply` with cross-scene verification and per-entry provenance."""
    entries = [dict(e) for e in entries]
    for e in entries:
        if e["id"] in used:
            e["used_in"] = e.get("used_in", []) + [episode]
            e["applied_in"] = e.get("applied_in", []) + [
                {"episode": episode, "seed": seed, "success": bool(success)}
            ]
            if _cross_scene_verifies(e, success, seed):
                e["status"] = "verified"
                e["verified_by"] = {"episode": episode, "seed": seed}
            if success:
                e["successes"] = e.get("successes", 0) + 1
            else:
                e["failures"] = e.get("failures", 0) + 1
    remove = set(edits["remove"])
    revised = {
        str(r.get("id"))
        for r in edits["revise"]
        if str(r.get("id"))
        not in {x["id"] for x in entries if x.get("status") == "verified"}
    }
    entries = [
        e
        for e in entries
        if e["id"] not in remove
        and (
            e["id"] in revised
            or not (
                e.get("failures", 0) >= settings.fails_to_drop
                and not e.get("successes")
            )
        )
    ]
    by_id = {e["id"]: e for e in entries}
    for r in edits["revise"]:
        e = by_id.get(str(r.get("id")))
        if e is None or e.get("status") == "verified":
            continue  # a verified entry's text is the one that was verified
        for k in ENTRY_FIELDS:
            if r.get(k):
                e[k] = str(r[k])[: settings.field_chars]
        e["revised_in"] = e.get("revised_in", []) + [episode]
        e["failures"] = 0  # the revised fix has not failed yet
        e["text_seed"] = seed
        e["text_from"] = episode
    stem, next_id = episode.rsplit("/", 1)[-1], 1
    for a in edits["add"]:
        if len(entries) >= settings.max_entries:
            break
        if not all(a.get(k) for k in ENTRY_FIELDS):
            continue
        entries.append(
            {
                "id": f"{stem}.{next_id}",
                "trigger": str(a.get("trigger") or "any"),
                **{k: str(a[k])[: settings.field_chars] for k in ENTRY_FIELDS},
                "status": "candidate",
                "written_in": episode,
                "used_in": [],
                "successes": 0,
                "failures": 0,
                "written_seed": seed,
                "text_seed": seed,
                "text_from": episode,
                "applied_in": [],
                "verified_by": None,
            }
        )
        next_id += 1
    return entries


# ---------------------------------------------------------------------------
# scene-free entries (stage 3 v2): the check, the rewrite request, the merge
# ---------------------------------------------------------------------------

#: (rule, pattern): text that ties an entry to one scene or episode. Each rule's name is what
#: the rewrite request tells GPT-6.
SCENE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "a list of coordinates or pixels",
        re.compile(
            r"[\[(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)*\s*[\])]"
        ),
    ),
    (
        "a pixel position",
        re.compile(r"\b(?:rows?|columns?|cols?|pixels?|px)\s*[=:~≈]?\s*-?\d+", re.I),
    ),
    (
        "a step, turn or seed number of this episode",
        re.compile(
            r"\bstep\s*[-=:#]?\s*\d+|\bsteps\s+\d+|\"step\"\s*:\s*\d+|\bseeds?\s*[-=:#]?\s*\d+|"
            r"\b(?:at|on|by|from|until|after|before|in)\s+(?:turn|request)\s+\d+\b",
            re.I,
        ),
    ),
    (
        "a measured value (a number with two or more decimals)",
        re.compile(r"(?<![\w.])-?\d+\.\d{2,}(?!\d|\.\d)"),
    ),
    (
        "an object instance id",
        re.compile(r"\b(?:[a-z]+_)*[a-z]{2,}_?\d+\b", re.I),
    ),
)
#: Quoted text (straight or typographic quotes): a quoted part of the task language counts too.
QUOTED = re.compile(
    r'"([^"]+)"|\'([^\']+)\'|\u201c([^\u201d]+)\u201d|\u2018([^\u2019]+)\u2019'
)
#: Model and robot names the instance-id pattern would otherwise match.
NOT_INSTANCE_IDS = frozenset({"gpt6", "qwen3", "rldx1", "rc365", "rgb8"})
#: The episode's task language counts as quoted from this many words on.
LANGUAGE_MIN_WORDS = 3
MATCH_CHARS = 80


def _language_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().rstrip(".!?").strip().lower()


def scene_problems(
    text: str, *, language: str = "", allowed: Collection[str] = ()
) -> list[tuple[str, str]]:
    """``(rule, matched text)`` for every scene detail in ``text``; empty when it is scene-free.

    ``allowed`` lists literal numbers that are task-level constants (exempt
    from the decimals rule); ``language`` is the episode's task language,
    which an entry may not contain, nor quote a part of (three words or more).
    """
    found: list[tuple[str, str]] = []
    for name, pattern in SCENE_RULES:
        for match in pattern.finditer(text or ""):
            value = match.group(0)
            if name.startswith("a measured value") and value.lstrip("-") in {
                a.lstrip("-") for a in allowed
            }:
                continue
            if name == "an object instance id" and value.lower() in NOT_INSTANCE_IDS:
                continue
            found.append((name, value[:MATCH_CHARS]))
    key = _language_key(language)
    if len(key.split()) >= LANGUAGE_MIN_WORDS:
        if key in _language_key(text):
            found.append(("this scene's task language", key[:MATCH_CHARS]))
        else:
            for match in QUOTED.finditer(text or ""):
                span = _language_key(next(g for g in match.groups() if g))
                if len(span.split()) >= LANGUAGE_MIN_WORDS and span in key:
                    found.append(("this scene's task language", span[:MATCH_CHARS]))
                    break
    return found


def used_ids(
    entries: Sequence[Mapping[str, Any]],
    edits: Mapping[str, Any],
    shown: Collection[str] | None = None,
) -> set[str]:
    """The ids ``reviewed`` counts as used: listed in ``applied``, existing, and shown."""
    listed = {str(i) for i in edits["applied"]} & {e["id"] for e in entries}
    return listed if shown is None else listed & set(shown)


def scene_rejections(
    edits: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    *,
    language: str = "",
    allowed: Collection[str] = (),
    frozen: Collection[str] = (),
    chars: int | None = None,
) -> list[dict[str, Any]]:
    """The added and revised entries of ``edits`` that are not scene-free, one row per item.

    Only items ``apply`` would act on are checked: adds with all three fields
    (their trigger too, which ``apply`` stores as written), and revisions of
    existing entries that are not frozen. ``chars`` (``Settings.field_chars``)
    also checks each field as ``apply`` stores it, cut to that length. ``item``
    names the item for the rewrite request: ``add <n>`` (1-based position in
    ``add``) or ``revise <id>``.
    """
    rows = []
    for n, a in enumerate(edits["add"], 1):
        if not all(a.get(k) for k in ENTRY_FIELDS):
            continue
        problems = item_problems(a, language, allowed, add=True, chars=chars)
        if problems:
            rows.append(
                {"item": f"add {n}", "kind": "add", "n": n, "problems": problems}
            )
    known = {str(e["id"]) for e in entries}
    for r in edits["revise"]:
        rid = str(r.get("id"))
        if rid not in known or rid in frozen:
            continue
        problems = item_problems(r, language, allowed, chars=chars)
        if problems:
            rows.append(
                {
                    "item": f"revise {rid}",
                    "kind": "revise",
                    "id": rid,
                    "problems": problems,
                }
            )
    return rows


def item_problems(
    item: Mapping[str, Any],
    language: str,
    allowed: Collection[str],
    *,
    add: bool = False,
    chars: int | None = None,
) -> list[dict[str, str]]:
    """The scene problems of one entry or edit item, per field: ``situation``, ``diagnosis``
    and ``fix`` (an add's or a new entry's ``trigger`` too), each as written and, with
    ``chars``, as stored (cut to ``chars``), so a cut can never make a stored field fail."""
    out: list[dict[str, str]] = []
    for k in (("trigger",) if add else ()) + ENTRY_FIELDS:
        if not item.get(k):
            continue
        text = str(item[k])
        variants = [text]
        if chars and k in ENTRY_FIELDS and len(text) > chars:
            variants.append(text[:chars])
        seen: set[tuple[str, str]] = set()
        for variant in variants:
            for rule, match in scene_problems(
                variant, language=language, allowed=allowed
            ):
                if (rule, match) not in seen:
                    seen.add((rule, match))
                    out.append({"field": k, "rule": rule, "match": match})
    return out


FIX_PROMPT = """{prompt}

Your reply was:
{reply}

These entries of your reply are tied to this one scene, so they cannot be kept as they are:
{problems}
Rewrite each of them as a rule that holds in any scene of this task. Keep the lesson; describe \
positions, amounts and targets by what the worker can observe or compute in the current scene \
(for example "the handle's centre in the current image"), and write "the task language verbatim" \
instead of quoting the task. An entry that cannot be written without scene details is dropped.

Reply with JSON only, one item per entry listed above (leave an entry out to drop it):
{{"add": [{{"n": 1, "trigger": "plan|checkin|errors|stall|giveup|any", "situation": "...", "diagnosis": "...", "fix": "..."}}],
 "revise": [{{"id": "...", "situation": "...", "diagnosis": "...", "fix": "..."}}]}}"""


def build_fix_prompt(
    prompt: str, reply: str, rejections: Sequence[Mapping[str, Any]]
) -> str:
    """The rewrite request: the reflection prompt, GPT-6's reply and why each listed item fails."""
    lines = []
    for row in rejections:
        why = "; ".join(
            f'{p["field"]} has {p["rule"]} ("{p["match"]}")' for p in row["problems"]
        )
        lines.append(f"- {row['item']}: {why}")
    return FIX_PROMPT.format(
        prompt=prompt, reply=reply.strip(), problems="\n".join(lines)
    )


def parse_rewrites(text: str) -> dict[str, Any]:
    """The rewritten items of a rewrite reply: ``{"add": {n: item}, "revise": {id: item}}``."""
    out: dict[str, Any] = {"add": {}, "revise": {}}
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return out
    try:
        doc = json.loads(match.group(0))
    except ValueError:
        return out
    if not isinstance(doc, dict):
        return out
    adds = doc.get("add") or []
    for item in adds if isinstance(adds, list) else []:
        if isinstance(item, dict) and str(item.get("n", "")).strip().isdigit():
            out["add"].setdefault(int(str(item["n"]).strip()), item)
    revs = doc.get("revise") or []
    for item in revs if isinstance(revs, list) else []:
        if isinstance(item, dict) and item.get("id") is not None:
            out["revise"].setdefault(str(item["id"]), item)
    return out


def merge_rewrites(
    edits: Mapping[str, Any],
    rejections: Sequence[Mapping[str, Any]],
    rewrites: Mapping[str, Any] | None,
    *,
    language: str = "",
    allowed: Collection[str] = (),
    chars: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``edits`` with every rejected item replaced by its rewrite when that is scene-free, else
    dropped; ``rewrites`` None (a refused rewrite request) drops them all.

    A rewritten item takes the rewrite's fields and the original's where the
    rewrite leaves one out (so an original field that failed fails again), and
    keeps its place; it is checked as :func:`scene_rejections` checks (``chars``
    as there). Returns the merged edits and the outcome: ``kept`` and
    ``dropped`` items, and ``still_rejected`` (the problems of each rewrite
    that failed again).
    """
    rewrites = rewrites or {"add": {}, "revise": {}}
    rejected = {row["item"]: row for row in rejections}
    kept, dropped, still = [], [], []
    adds = []
    for n, a in enumerate(edits["add"], 1):
        row = rejected.get(f"add {n}")
        if row is None:
            adds.append(a)
            continue
        new = rewrites["add"].get(n)
        if new is None:
            dropped.append(row["item"])
            continue
        merged = {
            "trigger": str(new.get("trigger") or a.get("trigger") or "any"),
            **{k: new.get(k) or a.get(k) for k in ENTRY_FIELDS},
        }
        problems = item_problems(merged, language, allowed, add=True, chars=chars)
        if problems or not all(merged.get(k) for k in ENTRY_FIELDS):
            dropped.append(row["item"])
            still.append({"item": row["item"], "problems": problems})
            continue
        adds.append(merged)
        kept.append(row["item"])
    revises = []
    for r in edits["revise"]:
        row = rejected.get(f"revise {r.get('id')}")
        if row is None:
            revises.append(r)
            continue
        new = rewrites["revise"].get(str(r.get("id")))
        if new is None:
            dropped.append(row["item"])
            continue
        merged = {"id": str(r.get("id"))}
        for k in ENTRY_FIELDS:
            if new.get(k) or r.get(k):
                merged[k] = new.get(k) or r.get(k)
        problems = item_problems(merged, language, allowed, chars=chars)
        if problems or len(merged) == 1:
            dropped.append(row["item"])
            still.append({"item": row["item"], "problems": problems})
            continue
        revises.append(merged)
        kept.append(row["item"])
    merged_edits = {**edits, "add": adds, "revise": revises}
    return merged_edits, {"kept": kept, "dropped": dropped, "still_rejected": still}


def decide(d: Mapping[str, Any]) -> str | None:
    """Why an episode is not reflected on (run_ecc_stage3.reflect), or None to reflect."""
    if d["interventions"] == 0:
        return "no senior intervention"
    if d["shown"] == 0:
        return f"no planner turn in the transcript or run.log ({d['turns']} turns)"
    return None


def reviewed(
    entries: Sequence[Mapping[str, Any]],
    reply: str,
    episode: str,
    success: bool,
    settings: Settings = Settings(),
    shown: Collection[str] | None = None,
    *,
    seed: int | None = None,
    cross_scene: bool = False,
    edits: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Edits, used ids and the memory after one reply (run_ecc_stage3.reflect after the call).

    ``shown`` (stage 3's parallel sessions) is the ids of the memory the episode
    ran with when that differs from ``entries``: an id listed in ``applied``
    counts as used only if the episode was shown it, and the others are
    returned as ``applied_not_shown``. Without it, as run_ecc_stage3, where the
    two memories are the same.

    Stage 3 v2: ``edits`` are already parsed (and checked) edits to use instead
    of parsing ``reply``; ``cross_scene`` applies them with cross-scene
    verification on the episode's ``seed`` and also returns
    ``revise_ignored_verified`` (revisions of frozen entries) and
    ``verified_now`` (entries this episode verified).
    """
    if edits is None:
        try:
            edits = parse_edits(reply)
            status = "applied"
        except ValueError as exc:
            edits = dict(NO_EDITS, note=f"unparsed reply: {exc}")
            status = "unparsed"
    else:
        status = "applied"
    listed = {str(i) for i in edits["applied"]} & {e["id"] for e in entries}
    used = used_ids(entries, edits, shown)
    ignored = []
    if cross_scene:
        frozen = frozen_ids(entries, used, success, seed)
        ignored = sorted({str(r.get("id")) for r in edits["revise"]} & frozen)
    after = apply(
        entries,
        edits,
        episode,
        success,
        used,
        settings,
        seed=seed,
        cross_scene=cross_scene,
    )
    out = {
        "status": status,
        "edits": edits,
        "used": sorted(used),
        "entries_after": after,
    }
    if shown is not None:
        out["applied_not_shown"] = sorted(listed - used)
    if cross_scene:
        out["revise_ignored_verified"] = ignored
        out["verified_now"] = sorted(
            str(e["id"])
            for e in after
            if (e.get("verified_by") or {}).get("episode") == episode
        )
    return out
