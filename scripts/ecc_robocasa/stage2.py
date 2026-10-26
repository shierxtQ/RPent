#!/usr/bin/env python3
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

"""RC-ECC stage 2 on RoboCasa Target50: the call-rule ladder (Qwen worker + GPT-6 senior, seed 11).

For every task, one hybrid evaluation episode per rung, from the weakest rule to
the strongest (default: the ECC rule's six rungs (plan, K, N) = (Qwen,12,4),
(GPT-6,12,4), (Qwen,6,4), (GPT-6,6,4), (GPT-6,6,3), (GPT-6,6,2)); the first rung
with a valid success becomes the task's rule, and a task whose rungs all fail
validly gets the strongest rung. Every episode reads the frozen stage-1 corpus
(read-only memory profile, read-only files). Invalid episodes (infrastructure
errors, rendered noise) are launched again under a cap and never count as a rung
failure; a planner timeout is a valid failure. Every finished episode's
escalation log is checked against its rule; a mismatch is an infrastructure
error that stops the run. When every task has its rule, the rules are frozen
with sha256 hashes for stage 3 and the final test.

All tasks run in parallel, rungs one after another within a task. Dry run by
default (prints the plan); ``run --go`` launches. See README.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import stat
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecc_robocasa import common as C  # noqa: E402
from scripts.ecc_robocasa import hybrid_episode as H  # noqa: E402
from scripts.ecc_robocasa import stage1 as S1  # noqa: E402

PRODUCT_SCHEMA = "rc-ecc-s2-product/1"
RULE_SCHEMA = "rc-ecc-s2-rule/1"
S1_PRODUCT_SCHEMA = S1.PRODUCT_SCHEMA
SPLIT = H.SPLIT

STAGE2_DEFAULTS: dict[str, Any] = {
    "seed": 11,
    # The frozen stage-1 product folder (<S1 run>/product); verified before anything starts.
    "s1_product": "",
    "tasks": [],
    # Empty: the ECC rule's six rungs. Else [{plan = "gpt"|"qwen", checkin = K, errors = N}, ...].
    "ladder": [],
    # The rule of a task whose rungs all failed validly (0: the last, strongest rung).
    "fallback_rung": 0,
    "timeout_s_by_split": dict(H.TARGET50_TIMEOUT_S),
    "infra_retries": 2,
    "render_retries": 6,
    # The escalation log's config row is the only record of N; refuse to run without it.
    "require_config_row": True,
    "launcher": ["bash", "{code_dir}/scripts/ecc_robocasa/hybrid_cell.sh"],
}
#: Keys of ``stage2`` that do not define the experiment (paths and task selection).
NON_EXPERIMENT_KEYS = ("s1_product", "tasks", "launcher")


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def stage2_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply the stage-2 defaults to ``cfg`` (in place) and validate them."""
    merged = copy.deepcopy(STAGE2_DEFAULTS)
    for key, value in (cfg.get("stage2") or {}).items():
        if key == "timeout_s_by_split" and isinstance(value, Mapping):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = copy.deepcopy(value)
    code_dir = cfg["paths"].get("code_dir") or ""
    merged["launcher"] = [
        str(item).replace("{code_dir}", code_dir) for item in merged["launcher"]
    ]
    if merged["s1_product"]:
        merged["s1_product"] = str(Path(merged["s1_product"]).expanduser().resolve())
    cfg["stage2"] = merged
    seed = merged["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("stage2.seed must be a nonnegative integer")
    if seed == int(cfg["stage1"]["seed"]):
        raise ValueError(
            "stage2.seed must differ from stage 1's seed: stage 1 left the edge's own solved "
            "recipe for that scene, which would understate the cloud's role (the ECC rule)"
        )
    ladder = H.parse_ladder(merged["ladder"])
    fallback = merged["fallback_rung"]
    if not isinstance(fallback, int) or not 0 <= fallback <= len(ladder):
        raise ValueError(f"stage2.fallback_rung must be 0..{len(ladder)}")
    for split in C.SPLITS:
        value = merged["timeout_s_by_split"].get(split)
        if not isinstance(value, int) or value < 1:
            raise ValueError(
                f"stage2.timeout_s_by_split.{split} must be a positive integer"
            )
    for key in ("infra_retries", "render_retries"):
        if not isinstance(merged[key], int) or merged[key] < 0:
            raise ValueError(f"stage2.{key} must be a nonnegative integer")
    if not merged["launcher"]:
        raise ValueError("stage2.launcher must name the cell launcher")
    return cfg


def experiment_hash(cfg: Mapping[str, Any], s1_corpus_sha256: str) -> str:
    """Everything that defines the stage-2 experiment (not slots, paths or the task list)."""
    s2 = {k: v for k, v in cfg["stage2"].items() if k not in NON_EXPERIMENT_KEYS}
    senior = {
        k: v for k, v in cfg["senior"].items() if k != "checkin"
    }  # K comes from the rung
    return C.canonical_hash(
        {
            "worker": cfg["worker"],
            "senior": senior,
            "episode": cfg["episode"],
            "stage2": s2,
            "s1_corpus_tree_sha256": s1_corpus_sha256,
        }
    )


# ---------------------------------------------------------------------------
# the stage-1 product (read-only input)
# ---------------------------------------------------------------------------


def s1_product(
    path: Path | str, all_tasks: list[tuple[str, str]] | None
) -> dict[str, Any]:
    """Verify the frozen stage-1 product; with ``all_tasks``, also load every task from it."""
    path = Path(path)
    check = S1.verify(path)
    if not check.get("ok"):
        raise SystemExit(f"the stage-1 product at {path} does not verify: {check}")
    doc = C.read_json(path / "PRODUCT.json") or {}
    if doc.get("schema") != S1_PRODUCT_SCHEMA:
        raise SystemExit(f"{path}/PRODUCT.json is not a stage-1 product")
    corpus = path / doc["corpus"]
    info = {
        "path": str(path),
        "corpus": str(corpus),
        "corpus_tree_sha256": doc["corpus_tree_sha256"],
        "product_json_sha256": C.sha256_file(path / "PRODUCT.json"),
        "files": check["files"],
        "run_id": doc.get("run_id"),
        "code_commit": (doc.get("code") or {}).get("commit"),
        "statuses": _count(t.get("status") for t in (doc.get("tasks") or {}).values()),
    }
    if all_tasks is not None:
        validation = S1.validate_corpus(corpus, all_tasks)
        if validation["errors"]:
            raise SystemExit(
                f"stage-1 corpus does not load every task: {validation['errors']}"
            )
        info["tasks_loaded"] = validation["tasks_loaded"]
    return info


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def config_row_supported() -> bool:
    """Whether this code's escalation wrapper writes the config row (checked offline)."""
    import tempfile

    from pydantic_ai.messages import ModelResponse
    from pydantic_ai.models.function import FunctionModel

    from rpent.planner.escalation import EscalationModel

    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "escalation.jsonl"
        EscalationModel(
            FunctionModel(lambda m, i: ModelResponse(parts=[]), model_name="w"),
            FunctionModel(lambda m, i: ModelResponse(parts=[]), model_name="s"),
            log_path=log,
        )
        rows, _ = H.read_rows(log)
    return any(isinstance(r.get("config"), dict) for r in rows)


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


def job_id(task: str, rung: int) -> str:
    return f"{task}__r{rung}"


class Stage2(C.Stage):
    name = "s2"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        layout: C.RunLayout,
        tasks: list[tuple[str, str]],
        product: Mapping[str, Any],
    ):
        self.cfg = cfg
        self.layout = layout
        self.tasks = tasks
        self.split_of = {task: split for split, task in tasks}
        s2 = cfg["stage2"]
        self.seed = int(s2["seed"])
        self.ladder = H.parse_ladder(s2["ladder"])
        self.fallback = int(s2["fallback_rung"]) or len(self.ladder)
        self.product = dict(product)
        self.corpus = Path(product["corpus"])
        self.code_dir = Path(cfg["paths"]["code_dir"])
        n = len(tasks)
        self._jobs = [
            C.Job(
                job_id=job_id(task, k),
                task=task,
                split=split,
                seed=self.seed,
                order=(k - 1) * n
                + i,  # breadth first: every task's rung k before rung k+1
                meta={"rung": k},
            )
            for k in range(1, len(self.ladder) + 1)
            for i, (split, task) in enumerate(tasks)
        ]
        self.by_id = {job.job_id: job for job in self._jobs}

    # -- Stage interface ---------------------------------------------------

    def jobs(self) -> list[C.Job]:
        return list(self._jobs)

    def rule(self, rung: int) -> H.Rule:
        return self.ladder[rung - 1]

    def rung_dir(self, task: str, rung: int) -> Path:
        return self.layout.task_dir(task) / f"r{rung}"

    def try_dir(self, job: C.Job, n: int) -> Path:
        return self.rung_dir(job.task, job.meta["rung"]) / f"try{n}"

    def prepare_try(self, job: C.Job, n: int) -> None:
        if not (self.corpus / "global" / "GLOBAL_MEMORY.md").is_file():
            raise RuntimeError(f"stage-1 corpus missing at {self.corpus}")

    def timeout_s(self, split: str) -> int:
        return int(self.cfg["stage2"]["timeout_s_by_split"][split])

    def launch(
        self, job: C.Job, n: int, endpoint: C.Endpoint
    ) -> tuple[list[str], dict[str, str]]:
        argv = [
            *self.cfg["stage2"]["launcher"],
            job.split,
            job.task,
            str(job.seed),
            str(self.try_dir(job, n)),
            str(self.corpus),
        ]
        env = H.hybrid_env(
            self.cfg,
            split=job.split,
            try_n=n,
            endpoint=endpoint,
            rule=self.rule(job.meta["rung"]),
            timeout_s=self.timeout_s(job.split),
        )
        return argv, env

    def budget_launches(self) -> int:
        return len(self.tasks) * len(self.ladder)

    def on_final(self, job: C.Job, state: dict[str, Any]) -> None:
        """Record the rung's episode, then decide the task if this rung settles it.

        Errors are logged, not raised: running cells must not lose their runner;
        ``settle`` repeats the decision on every poll.
        """
        try:
            if (state.get("final") or {}).get("class") not in ("skipped", "blocked"):
                self.write_record(job, state)
            self.settle_task(job.task)
        except Exception as exc:  # noqa: BLE001 - logged; settle retries it
            C.append_jsonl(
                self.layout.progress,
                {
                    "event": "record_error",
                    "job": job.job_id,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

    # -- states ------------------------------------------------------------

    def state(self, jid: str) -> dict[str, Any]:
        job = self.by_id[jid]
        return C.read_json(self.layout.job_state(jid)) or {
            "job_id": jid,
            "task": job.task,
            "split": job.split,
            "seed": job.seed,
            "tries": [],
            "final": None,
        }

    def eligible(self, job: C.Job, states: Mapping[str, Mapping[str, Any]]) -> bool:
        """Rung 1 always; rung k once rung k-1 failed validly."""
        rung = job.meta["rung"]
        if rung == 1:
            return True
        prev = states[job_id(job.task, rung - 1)].get("final") or {}
        return prev.get("class") == "valid" and not prev.get("success")

    def decision(
        self, task: str, states: Mapping[str, Mapping[str, Any]] | None = None
    ) -> tuple[str, int] | None:
        """``(source, rung)`` once the task's ladder is settled: ``first_success``,
        ``fallback_all_failed`` or ``blocked`` (a rung without a valid result)."""
        for k in range(1, len(self.ladder) + 1):
            jid = job_id(task, k)
            state = (
                states[jid] if states is not None and jid in states else self.state(jid)
            )
            final = state.get("final")
            if not final:
                return None
            cls = final.get("class")
            if cls == "valid" and final.get("success"):
                return "first_success", k
            if cls == "valid":
                continue
            if cls in ("infra_exhausted", "render_exhausted"):
                return "blocked", k
            return None  # skipped/blocked marks without a deciding rung before them
        return "fallback_all_failed", self.fallback

    def settle(self, states: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        for _, task in self.tasks:
            try:
                self.settle_task(task, states)
            except Exception as exc:  # noqa: BLE001 - logged, tried again on the next poll
                C.append_jsonl(
                    self.layout.progress,
                    {
                        "event": "settle_error",
                        "task": task,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

    def settle_task(
        self, task: str, states: Mapping[str, Mapping[str, Any]] | None = None
    ) -> None:
        """Mark the rungs a decision makes unnecessary, and write the rule (idempotent)."""
        decided = self.decision(task, states)
        if decided is None:
            return
        source, rung = decided
        if source in ("first_success", "blocked"):
            for k in range(rung + 1, len(self.ladder) + 1):
                jid = job_id(task, k)
                state = self.state(
                    jid
                )  # fresh from disk: the caller's copy may be stale
                if state.get("final"):
                    continue
                if state["tries"]:
                    C.append_jsonl(
                        self.layout.progress,
                        {
                            "event": "ladder_inconsistent",
                            "job": jid,
                            "decided": [source, rung],
                        },
                    )
                    continue
                state["final"] = {
                    "class": "skipped" if source == "first_success" else "blocked",
                    "success": False,
                    "after_rung": rung,
                    "at": C.now_iso(),
                }
                C.atomic_write_json(self.layout.job_state(jid), state)
        if source == "blocked":
            marker = self.layout.task_dir(task) / "blocked.json"
            if not marker.exists():
                C.atomic_write_json(
                    marker,
                    {
                        "task": task,
                        "rung": rung,
                        "class": self.state(job_id(task, rung))["final"]["class"],
                        "at": C.now_iso(),
                    },
                )
                C.append_jsonl(
                    self.layout.progress,
                    {"event": "blocked", "task": task, "rung": rung},
                )
            return
        path = self.rule_path(task)
        existing = C.read_json(path)
        if existing is not None:
            if existing.get("rung") != rung:
                C.append_jsonl(
                    self.layout.progress,
                    {
                        "event": "rule_conflict",
                        "task": task,
                        "file": existing.get("rung"),
                        "now": rung,
                    },
                )
            return
        doc = self.build_rule(task, source, rung)
        C.atomic_write_json(path, doc)
        C.append_jsonl(
            self.layout.progress,
            {
                "event": "rule",
                "task": task,
                "rung": rung,
                "config": doc["config"],
                "source": source,
            },
        )

    # -- records -------------------------------------------------------------

    def record_path(self, task: str, rung: int) -> Path:
        return self.rung_dir(task, rung) / "episode.json"

    def rule_path(self, task: str) -> Path:
        return self.layout.task_dir(task) / "rule.json"

    def expected(self, rung: int) -> dict[str, Any]:
        return H.expected_routing(self.cfg, self.rule(rung))

    def check_config_row(self, job: C.Job, config: Mapping[str, Any]) -> list[str]:
        return H.config_row_problems(config, self.expected(job.meta["rung"]))

    def check_episode(self, job: C.Job, try_dir: Path) -> dict[str, Any]:
        """Escalation evidence and stage-1 corpus integrity of one finished launch."""
        outcome = C.read_outcome(try_dir)
        check = H.check_escalation(
            try_dir / "escalation.jsonl",
            self.expected(job.meta["rung"]),
            require_config_row=bool(self.cfg["stage2"]["require_config_row"]),
            result_valid=bool(outcome.get("valid")),
        )
        corpus_hash = C.tree_hash(C.tree_files(self.corpus))
        if corpus_hash != self.product["corpus_tree_sha256"]:
            check["problems"].append("the stage-1 corpus changed during the run")
            check["ok"] = False
        return check

    def write_record(self, job: C.Job, state: dict[str, Any]) -> dict[str, Any]:
        rung = job.meta["rung"]
        rule = self.rule(rung)
        final = state["final"]
        tries = state["tries"]
        last = tries[final["try"] - 1]
        try_dir = Path(last["dir"])
        check = self.check_episode(job, try_dir)
        launched = _epoch(last.get("launched_at"))
        record = {
            "task": job.task,
            "split": job.split,
            "seed": job.seed,
            "rung": rung,
            "config": rule.config_name(rung),
            "rule": rule.as_json(),
            "final_class": final["class"],
            "final_try": final["try"],
            "launches": len(tries),
            "infra_retries": sum(1 for t in tries if t.get("class") == "infra"),
            "render_retries": sum(1 for t in tries if t.get("class") == "render"),
            "killed": sum(1 for t in tries if t.get("class") == "killed"),
            "try_dir": str(try_dir),
            "endpoint": last.get("endpoint"),
            "launched_at": last.get("launched_at"),
            "ended_at": last.get("ended_at"),
            "wall_s": round(float(last["ended_ts"]) - launched, 1)
            if launched and last.get("ended_ts")
            else None,
            **H.episode_record(try_dir, self.cfg["pricing"], check),
            "usd_all_launches": round(
                sum(float((t.get("usage") or {}).get("usd") or 0.0) for t in tries), 4
            ),
            "file_tool_audit": H.file_tool_audit(
                try_dir,
                own=try_dir,
                memory=self.corpus,
                code_dir=self.code_dir,
                shared=[
                    self.layout.tasks_dir,
                    Path(self.cfg["paths"].get("runs_dir") or self.layout.root.parent),
                    *(
                        [Path(self.cfg["paths"]["cloud_corpus"])]
                        if self.cfg["paths"].get("cloud_corpus")
                        else []
                    ),
                    *(Path(p) for p in self.cfg["paths"].get("audit_watch") or []),
                ],
            ),
            "recorded_at": C.now_iso(),
        }
        C.atomic_write_json(self.record_path(job.task, rung), record)
        return record

    def build_rule(self, task: str, source: str, rung: int) -> dict[str, Any]:
        """The task's call-rule memory (the fields of ``scripts/ecc/run_ecc_stage2.py``'s RULES_DIR/<tag>.json, plus ours)."""
        rule = self.rule(rung)
        records = [
            rec
            for k in range(1, len(self.ladder) + 1)
            if (rec := C.read_json(self.record_path(task, k))) is not None
        ]
        valid = [r for r in records if r.get("valid")]

        def total(key: str, field: str) -> int:
            return sum(int((r.get(key) or {}).get(field) or 0) for r in valid)

        by_reason: dict[str, int] = {}
        for r in valid:
            for reason, n in (r.get("gpt_calls_by_reason") or {}).items():
                by_reason[reason] = by_reason.get(reason, 0) + int(n)
        return {
            "schema": RULE_SCHEMA,
            "task": task,
            "suite": "robocasa",
            "task_id": task,
            "split": self.split_of[task],
            "environment_split": SPLIT,
            "search_seed": self.seed,
            "rule": rule.as_json(),
            "config": rule.config_name(rung),
            "rung": rung,
            "rule_source": source,
            "solved_in_search": source == "first_success",
            "episodes": len(valid),
            "launches": sum(int(r.get("launches") or 0) for r in records),
            "gpt_calls_in_search": sum(int(r.get("gpt_calls") or 0) for r in valid),
            "gpt_calls_by_reason_in_search": by_reason,
            "errors_trigger_in_search": sum(
                int(r.get("errors_trigger") or 0) for r in valid
            ),
            "errors_trigger_episodes": sum(1 for r in valid if r.get("errors_trigger")),
            "qwen_in_search": {
                "input_tokens": total("qwen", "input_tokens"),
                "output_tokens": total("qwen", "output_tokens"),
            },
            "gpt6_in_search": {
                "input_tokens": total("gpt6", "input_tokens"),
                "cached_input_tokens": total("gpt6", "cached_input_tokens"),
                "output_tokens": total("gpt6", "output_tokens"),
            },
            "usd_in_search": round(
                sum(float(r.get("usd_all_launches") or 0.0) for r in records), 4
            ),
            "history": [
                {
                    "config": r["config"],
                    "rung": r["rung"],
                    "success": r.get("success"),
                    "valid": r.get("valid"),
                    "gpt_calls": r.get("gpt_calls"),
                    "turns": r.get("turns"),
                    "finish_status": r.get("finish_status"),
                    "last_error": None
                    if r.get("termination_reason") == "completed"
                    else r.get("termination_reason"),
                    "termination_reason": r.get("termination_reason"),
                    "requests": r.get("requests"),
                    "gpt_calls_by_reason": r.get("gpt_calls_by_reason"),
                    "errors_trigger": r.get("errors_trigger"),
                    "qwen": r.get("qwen"),
                    "gpt6": r.get("gpt6"),
                    "usd": r.get("usd_all_launches"),
                    "elapsed_s": r.get("elapsed_s"),
                    "wall_s": r.get("wall_s"),
                    "launches": r.get("launches"),
                    "infra_retries": r.get("infra_retries"),
                    "render_retries": r.get("render_retries"),
                    "try_dir": r.get("try_dir"),
                }
                for r in records
            ],
            "ladder": [r.config_name(i) for i, r in enumerate(self.ladder, 1)],
            "s1_product": {
                "corpus_tree_sha256": self.product["corpus_tree_sha256"],
                "product_json_sha256": self.product["product_json_sha256"],
            },
            "written_at": C.now_iso(),
        }


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# the ladder scheduler
# ---------------------------------------------------------------------------


class LadderScheduler(C.Scheduler):
    """``common.Scheduler`` with the ladder's order and the escalation evidence check.

    - one rung of a task at a time; rung k+1 only after rung k failed validly;
    - every launch's escalation log is checked: its config row as soon as it
      appears (a wrong cell is stopped at once), the full replay when it ends;
      a mismatch makes the launch an infrastructure error and writes STOP;
    - a rung's infrastructure budget counts from ``count_from`` (``unblock``).
    """

    stage: Stage2

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._config_checked: set[str] = set()

    @staticmethod
    def class_count(state: Mapping[str, Any], name: str) -> int:
        start = int(state.get("count_from") or 0)
        return sum(1 for t in state["tries"][start:] if t.get("class") == name)

    def launch_ceiling(self) -> int:
        return C.launch_ceiling(
            self.sched,
            self.stage.budget_launches(),
            len(self.stage.tasks),
            self.infra_retries,
            self.render_retries,
        )

    def next_job(self, states: Mapping[str, Mapping[str, Any]]) -> str | None:
        busy = {self.jobs[j].task for j, _ in self.running(states)}
        candidates = [
            self.jobs[j]
            for j, s in states.items()
            if not s["final"]
            and self.jobs[j].task not in busy
            and self.stage.eligible(self.jobs[j], states)
        ]

        def key(job: C.Job) -> tuple[int, int]:
            tries = states[job.job_id]["tries"]
            last_cls = tries[-1].get("class") if tries else None
            return (
                2 if last_cls == "render" else 0 if last_cls == "infra" else 1,
                job.order,
            )

        candidates.sort(key=key)
        return candidates[0].job_id if candidates else None

    def _mismatch(
        self, job_id: str, last: Mapping[str, Any], problems: list[str], when: str
    ) -> None:
        marker = C.infra_marker(last["dir"])
        if not marker.exists():
            C.atomic_write_text(
                marker, f"evidence_mismatch ({when}): {'; '.join(problems[:3])[:600]}\n"
            )
        C.write_stop(
            self.layout,
            "escalation evidence mismatch",
            job=job_id,
            launch=last.get("n"),
            problems=problems[:5],
        )
        self.log(
            {
                "event": "evidence_mismatch",
                "job": job_id,
                "try": last.get("n"),
                "when": when,
                "problems": problems[:5],
            }
        )

    def poll(self) -> None:
        super().poll()
        states = self.all_states()
        for jid, last in self.running(states):
            if last["dir"] in self._config_checked:
                continue
            rows, _ = H.read_rows(Path(last["dir"]) / "escalation.jsonl")
            config = next(
                (r["config"] for r in rows if isinstance(r.get("config"), dict)), None
            )
            if config is None:
                continue
            self._config_checked.add(last["dir"])
            problems = self.stage.check_config_row(self.jobs[jid], config)
            if problems:
                self._mismatch(jid, last, problems, "config row of a running cell")
                if last.get("pid") and C.pid_alive(int(last["pid"])):
                    C.kill_group(
                        int(last["pid"]), grace_s=float(self.sched["kill_grace_s"])
                    )
        self.stage.settle(self.all_states())

    def finish_try(
        self, job_id: str, state: dict[str, Any], exit_code: int | None
    ) -> None:
        last = state["tries"][-1]
        try_dir = Path(last["dir"])
        if not C.killed_marker(try_dir).exists():
            check = self.stage.check_episode(self.jobs[job_id], try_dir)
            last["evidence"] = {
                "ok": check["ok"],
                "problems": check["problems"][:20],
                "counts": check["counts"],
            }
            if not check["ok"]:
                self._mismatch(job_id, last, check["problems"], "finished launch")
        super().finish_try(job_id, state, exit_code)


# ---------------------------------------------------------------------------
# freeze, verify, read
# ---------------------------------------------------------------------------


def _make_read_only(root: Path) -> None:
    S1._make_read_only(root)


def freeze(
    stage: Stage2,
    manifest: Mapping[str, Any],
    *,
    fallback_blocked: bool = False,
    allow_cross_writes: bool = False,
) -> dict[str, Any]:
    """Freeze every task's rule (all tasks decided) into ``product/`` with sha256 hashes."""
    product = stage.layout.product
    if (product / "FROZEN").exists():
        return verify(product)
    stage.settle()
    missing, blocked = [], []
    for _, task in stage.tasks:
        if stage.rule_path(task).exists():
            continue
        (
            blocked
            if (stage.layout.task_dir(task) / "blocked.json").exists()
            else missing
        ).append(task)
    if missing:
        raise SystemExit(
            f"cannot freeze: {len(missing)} task(s) have no rule yet: {', '.join(missing)}"
        )
    if blocked and not fallback_blocked:
        raise SystemExit(
            f"cannot freeze: {len(blocked)} task(s) stopped at a rung without a valid result "
            f"({', '.join(blocked)}); run `unblock` and resume, or freeze --fallback-blocked "
            f"to give them the fallback rung {stage.fallback} (recorded)"
        )
    for task in blocked:
        C.atomic_write_json(
            stage.rule_path(task),
            stage.build_rule(task, "fallback_infra_blocked", stage.fallback),
        )
    records: dict[str, list[dict[str, Any]]] = {}
    writers: dict[str, Any] = {}
    for _, task in stage.tasks:
        records[task] = [
            rec
            for k in range(1, len(stage.ladder) + 1)
            if (rec := C.read_json(stage.record_path(task, k))) is not None
        ]
        for rec in records[task]:
            # Write calls into another task's or rung's folder, another run or the cloud corpus
            # (RPent's file tools allow paths outside the memory root). Write attempts into the
            # read-only stage-1 corpus are refused by the memory manager and its hash is checked
            # below, so they are listed in the records but do not block the freeze.
            audit = rec.get("file_tool_audit") or {}
            if audit.get("cross_writes"):
                writers.setdefault(task, []).extend(audit["cross_writes"])
    if writers and not allow_cross_writes:
        raise SystemExit(
            "refusing to freeze: episodes wrote into other tasks' folders, other runs or the cloud "
            f"corpus (inspect, then freeze --allow-cross-writes): {json.dumps(writers)[:2000]}"
        )
    s1_now = s1_product(stage.product["path"], None)
    if s1_now["corpus_tree_sha256"] != manifest["s1_product"]["corpus_tree_sha256"]:
        raise SystemExit(
            "the stage-1 product changed during the run; refusing to freeze"
        )
    if product.exists():
        S1._make_writable(product)
        shutil.rmtree(product)
    (product / "rules").mkdir(parents=True)
    rules = {}
    for _, task in stage.tasks:
        shutil.copy2(stage.rule_path(task), product / "rules" / f"{task}.json")
        rules[task] = C.read_json(stage.rule_path(task))
    C.atomic_write_json(product / "episodes.json", records)
    files = C.tree_files(product / "rules")
    rules_hash = C.tree_hash(files)
    chosen: dict[str, int] = {}
    by_split: dict[str, dict[str, int]] = {}
    for task, rule in rules.items():
        chosen[rule["config"]] = chosen.get(rule["config"], 0) + 1
        row = by_split.setdefault(rule["split"], {})
        key = (
            rule["config"]
            if rule["rule_source"] == "first_success"
            else f"{rule['config']} ({rule['rule_source']})"
        )
        row[key] = row.get(key, 0) + 1
    doc = {
        "schema": PRODUCT_SCHEMA,
        "stage": "s2",
        "run_id": manifest["run_id"],
        "created_at": C.now_iso(),
        "code": manifest["code"],
        "config_sha256": manifest["config_sha256"],
        "experiment_sha256": manifest["experiment_sha256"],
        "seed": stage.seed,
        "ladder": manifest["ladder"],
        "fallback_rung": stage.fallback,
        "s1_product": manifest["s1_product"],
        "rules": "rules",
        "rules_tree_sha256": rules_hash,
        "episodes_json_sha256": C.sha256_file(product / "episodes.json"),
        "files": files,
        "tasks": {
            task: {
                k: rule.get(k)
                for k in (
                    "split",
                    "rung",
                    "config",
                    "rule",
                    "rule_source",
                    "solved_in_search",
                    "episodes",
                    "launches",
                    "gpt_calls_in_search",
                    "errors_trigger_in_search",
                    "usd_in_search",
                )
            }
            for task, rule in rules.items()
        },
        "summary": {
            "tasks": len(rules),
            "solved_in_search": sum(1 for r in rules.values() if r["solved_in_search"]),
            "episodes": sum(int(r["episodes"]) for r in rules.values()),
            "launches": sum(int(r["launches"]) for r in rules.values()),
            "gpt_calls_in_search": sum(
                int(r["gpt_calls_in_search"]) for r in rules.values()
            ),
            "errors_trigger_in_search": sum(
                int(r["errors_trigger_in_search"]) for r in rules.values()
            ),
            "usd_in_search": round(
                sum(float(r["usd_in_search"]) for r in rules.values()), 2
            ),
            "chosen_config_counts": dict(sorted(chosen.items())),
            "by_split": by_split,
        },
        "blocked_fallback": sorted(blocked),
        "cross_writes_accepted": writers,
    }
    C.atomic_write_json(product / "RULES.json", doc)
    _make_read_only(product / "rules")
    C.atomic_write_text(product / "FROZEN", rules_hash + "\n")
    for name in ("RULES.json", "episodes.json", "FROZEN"):
        (product / name).chmod(0o444)
    C.append_jsonl(
        stage.layout.progress, {"event": "frozen", "rules_tree_sha256": rules_hash}
    )
    return {
        "ok": True,
        "rules_tree_sha256": rules_hash,
        "tasks": len(rules),
        "product": str(product),
    }


def verify(product: Path) -> dict[str, Any]:
    """Recompute the stage-2 product's hashes (what stage 3 and the test do before reading it)."""
    doc = C.read_json(product / "RULES.json")
    if not doc or doc.get("schema") != PRODUCT_SCHEMA:
        return {"ok": False, "problems": ["no RULES.json with the stage-2 schema"]}
    files = C.tree_files(product / doc["rules"])
    problems = []
    if C.tree_hash(files) != doc["rules_tree_sha256"]:
        changed = sorted(set(files.items()) ^ set(doc["files"].items()))
        problems.append(
            f"rules hash mismatch ({len(changed)} differing entries, e.g. {changed[:3]})"
        )
    frozen = (
        (product / "FROZEN").read_text().strip()
        if (product / "FROZEN").exists()
        else None
    )
    if frozen != doc["rules_tree_sha256"]:
        problems.append("FROZEN marker missing or different")
    if C.sha256_file(product / "episodes.json") != doc["episodes_json_sha256"]:
        problems.append("episodes.json changed")
    if sorted(f"{t}.json" for t in doc["tasks"]) != sorted(files):
        problems.append("rules/ does not hold exactly one file per task")
    writable = [
        p
        for p in (product / doc["rules"]).rglob("*")
        if p.stat().st_mode & stat.S_IWUSR
    ]
    if writable:
        problems.append(f"{len(writable)} rules entries are writable")
    return {
        "ok": not problems,
        "problems": problems,
        "rules_tree_sha256": doc["rules_tree_sha256"],
        "tasks": len(doc["tasks"]),
        "s1_corpus_tree_sha256": (doc.get("s1_product") or {}).get(
            "corpus_tree_sha256"
        ),
    }


def load_rules(product: Path | str) -> dict[str, H.Rule]:
    """``{task: Rule}`` from a frozen stage-2 product, after verifying its hashes."""
    product = Path(product)
    check = verify(product)
    if not check["ok"]:
        raise ValueError(
            f"stage-2 product at {product} does not verify: {check['problems']}"
        )
    doc = C.read_json(product / "RULES.json")
    rules = {}
    for task in doc["tasks"]:
        rule = C.read_json(product / doc["rules"] / f"{task}.json")
        rules[task] = H.Rule.from_json(rule["rule"])
    return rules


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def select_tasks(
    cfg: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    all_tasks = C.load_target50(cfg["paths"]["code_dir"])
    wanted = [
        t
        for t in (args.tasks.split(",") if args.tasks else cfg["stage2"]["tasks"])
        if t
    ]
    known = {task for _, task in all_tasks}
    unknown = [t for t in wanted if t not in known]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}")
    tasks = [row for row in all_tasks if not wanted or row[1] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks, all_tasks


def run_layout(cfg: Mapping[str, Any], args: argparse.Namespace) -> C.RunLayout:
    if getattr(args, "run_root", None):
        return C.RunLayout(Path(args.run_root).resolve())
    if not args.run_id:
        raise SystemExit("pass --run-id (or --run-root)")
    return C.RunLayout(
        Path(cfg["paths"]["runs_dir"]).resolve() / C.safe_name(args.run_id)
    )


def ladder_rows(stage_or_cfg: Any) -> list[dict[str, Any]]:
    ladder = (
        stage_or_cfg.ladder
        if isinstance(stage_or_cfg, Stage2)
        else H.parse_ladder(stage_or_cfg["stage2"]["ladder"])
    )
    return [
        {"rung": i, "config": r.config_name(i), **r.as_json()}
        for i, r in enumerate(ladder, 1)
    ]


def budget(cfg: Mapping[str, Any]) -> dict[str, Any]:
    s2 = cfg["stage2"]
    ladder = H.parse_ladder(s2["ladder"])
    return {
        "episodes_per_task": f"1 per rung, at most {len(ladder)}; stop at the first valid success",
        "fallback_rung": int(s2["fallback_rung"]) or len(ladder),
        "timeout_s_by_split": s2["timeout_s_by_split"],
        "max_turns": cfg["episode"]["max_turns"],
        "infra_retries_per_rung": s2["infra_retries"],
        "render_retries_per_rung": s2["render_retries"],
    }


def make_manifest(
    cfg: Mapping[str, Any],
    run_id: str,
    tasks: list[tuple[str, str]],
    product: Mapping[str, Any],
) -> dict[str, Any]:
    code = C.code_identity(cfg["paths"]["code_dir"])
    return {
        "schema": "rc-ecc-run/1",
        "stage": "s2",
        "run_id": run_id,
        "created_at": C.now_iso(),
        "config_path": cfg["_path"],
        "config_sha256": C.config_hash(cfg),
        "experiment_sha256": experiment_hash(cfg, product["corpus_tree_sha256"]),
        "code": code,
        "s1_product": {
            k: product.get(k)
            for k in (
                "path",
                "corpus_tree_sha256",
                "product_json_sha256",
                "run_id",
                "code_commit",
                "statuses",
            )
        },
        "seed": cfg["stage2"]["seed"],
        "split": SPLIT,
        "ladder": ladder_rows(cfg),
        "budget": budget(cfg),
        "tasks": [{"split": s, "task": t} for s, t in tasks],
        "worker": cfg["worker"],
        "senior": {k: v for k, v in cfg["senior"].items() if k != "checkin"},
        "episode": cfg["episode"],
        "endpoints": cfg["endpoints"],
        "scheduler": cfg["scheduler"],
        "history": [
            {
                "at": C.now_iso(),
                "event": "created",
                "code_tree_sha256": code["tree_sha256"],
            }
        ],
    }


def check_resume(
    manifest: Mapping[str, Any], fresh: Mapping[str, Any], allow_code_change: bool
) -> list[str]:
    problems = []
    if manifest["experiment_sha256"] != fresh["experiment_sha256"]:
        problems.append(
            "experiment settings (worker/senior/episode/stage2/stage-1 corpus) differ from the manifest"
        )
    if [t["task"] for t in manifest["tasks"]] != [t["task"] for t in fresh["tasks"]]:
        problems.append("task list differs from the run's manifest")
    if (
        manifest["s1_product"]["corpus_tree_sha256"]
        != fresh["s1_product"]["corpus_tree_sha256"]
    ):
        problems.append("stage-1 corpus differs from the run's manifest")
    if (
        manifest["code"]["tree_sha256"] != fresh["code"]["tree_sha256"]
        and not allow_code_change
    ):
        problems.append(
            "code differs from the run's manifest (pass --allow-code-change to accept and record it)"
        )
    return problems


def print_plan(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    """The dry run: everything --go would do, without creating or starting anything."""
    C.require(cfg, "paths.code_dir", "paths.runs_dir", "stage2.s1_product")
    tasks, all_tasks = select_tasks(cfg, args)
    s2 = cfg["stage2"]
    print(
        f"RC-ECC stage 2 plan (dry run; add --go to launch)\n  config   {cfg['_path']}"
    )
    code = C.code_identity(cfg["paths"]["code_dir"])
    print(
        f"  code     {cfg['paths']['code_dir']}  commit {code['commit']} ({code['commit_source']})  "
        f"tree {code['tree_sha256'][:16]} ({code['files']} files)"
    )
    try:
        product = s1_product(s2["s1_product"], all_tasks)
        print(
            f"  stage-1 product {product['path']}: verified, corpus {product['corpus_tree_sha256'][:16]} "
            f"({product['files']} files), {product.get('tasks_loaded')} tasks load; S1 run "
            f"{product['run_id']} statuses {product['statuses']}"
        )
    except SystemExit as exc:  # the dry run reports it; --go refuses
        product = {
            "path": s2["s1_product"],
            "corpus": str(Path(s2["s1_product"]) / "corpus"),
            "corpus_tree_sha256": "<not frozen>",
            "product_json_sha256": "<not frozen>",
        }
        print(f"  stage-1 product NOT READY (--go refuses until it verifies): {exc}")
    print(
        f"  config sha256 {C.config_hash(cfg)[:16]}  experiment sha256 "
        f"{experiment_hash(cfg, product['corpus_tree_sha256'])[:16]}"
    )
    by_split: dict[str, int] = {}
    for split, _ in tasks:
        by_split[split] = by_split.get(split, 0) + 1
    print(
        f"  tasks    {len(tasks)} ({', '.join(f'{k} {v}' for k, v in by_split.items())}); seed {s2['seed']}, "
        f"split {SPLIT}"
    )
    ladder = ladder_rows(cfg)
    print("  ladder   " + "  ".join(f"{r['rung']}:{r['config']}" for r in ladder))
    b = budget(cfg)
    print(
        f"  budget   {b['episodes_per_task']}; all fail -> rung {b['fallback_rung']}; planner timeout "
        f"{b['timeout_s_by_split']} s, {b['max_turns']} turns per episode"
    )
    print(
        f"  retries  infra {s2['infra_retries']}, render {s2['render_retries']} per rung (never counted as rung failures)"
    )
    print(
        f"  evidence escalation.jsonl config row {'required' if s2['require_config_row'] else 'optional'}; "
        f"this code writes it: {_config_row_note()}"
    )
    layout = (
        run_layout(cfg, args)
        if (args.run_id or getattr(args, "run_root", None))
        else None
    )
    if layout:
        state = (
            "exists: --go resumes it"
            if layout.manifest.exists()
            else "new: --go creates it"
        )
        print(f"  run root {layout.root} ({state})")
    sched = cfg["scheduler"]
    max_slots = min(args.max_slots or sched["max_slots"], sched["max_slots"])
    ceiling = C.launch_ceiling(
        sched,
        len(tasks) * len(ladder),
        len(tasks),
        s2["infra_retries"],
        s2["render_retries"],
    )
    print(
        f"  slots    max {max_slots}; stagger {sched['launch_stagger_s']} s; launch ceiling {ceiling}; "
        f"infra breaker {sched['infra_breaker']}; cost ceiling "
        f"{'$%.0f' % sched['cost_ceiling_usd'] if sched['cost_ceiling_usd'] else 'off'}"
    )
    try:
        usage = C.gpu_usage(cfg)
    except Exception as exc:  # noqa: BLE001 - dry run reports and goes on
        usage = None
        print(f"  GPUs     query failed: {exc}")
    for ep in C.endpoints_from_config(cfg):
        free = C.gpu_free(cfg, ep.gpu, usage)[1] if usage is not None else "not checked"
        print(f"  endpoint {ep.name}: port {ep.port}, {ep.slots} slots; {free}")
    cred = Path(cfg["credentials"].get("file") or "")
    if cred.name:
        info = (
            f"exists, mode {oct(cred.stat().st_mode & 0o777)}"
            if cred.is_file()
            else "MISSING"
        )
        print(f"  credentials file {cred} ({info}; never read by the runner)")
    if cfg["paths"].get("venv"):
        env = {
            **S1.base_env(cfg),
            "PYTHONPATH": cfg["paths"]["code_dir"],
            "RPENT_REPO_ROOT": cfg["paths"]["code_dir"],
        }
        origin = C.import_origin(C.python_for(cfg), cfg["paths"]["code_dir"], env)
        print(
            f"  episode imports: {'OK' if origin.get('ok') else 'NOT FROM THE CODE DIR'} {origin}"
        )
    mine = S1.runner_origin(cfg["paths"]["code_dir"])
    print(
        f"  runner imports:  {'OK' if mine['ok'] else 'NOT FROM THE CODE DIR'} {mine}"
    )
    if tasks:
        split, task = tasks[0]
        stage = Stage2(
            cfg,
            layout or C.RunLayout(Path(cfg["paths"]["runs_dir"]) / "<run-id>"),
            tasks,
            product,
        )
        job = stage.by_id[job_id(task, 1)]
        eps = C.endpoints_from_config(cfg)
        argv, env = stage.launch(
            job, 1, eps[0] if eps else C.Endpoint("gpuN", -1, 0, 0)
        )
        print(f"  first launch: {' '.join(argv)}")
        for k in sorted(
            k
            for k in env
            if k.startswith(("ECC_RC_", "RC_", "VLA_")) or k in cfg["episode"]["env"]
        ):
            print(f"    {k}={env[k]}")
    return 0


def _config_row_note() -> str:
    try:
        return "yes" if config_row_supported() else "NO"
    except Exception as exc:  # noqa: BLE001 - reported in the dry run
        return f"unknown ({type(exc).__name__}: {exc})"


def cmd_run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    if not args.go:
        return print_plan(cfg, args)
    C.require(
        cfg,
        "paths.code_dir",
        "paths.venv",
        "paths.runs_dir",
        "paths.assets",
        "paths.hf_home",
        "paths.rldx_checkpoint",
        "paths.llvmpipe_index_script",
        "credentials.file",
        "credentials.worker_key_var",
        "credentials.senior_base_url_var",
        "credentials.senior_key_var",
        "worker.base_url",
        "endpoints",
        "gpus.allowed",
        "stage2.s1_product",
    )
    if not Path(cfg["credentials"]["file"]).is_file():
        raise SystemExit("credentials file missing")
    try:
        import numpy  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        raise SystemExit(
            "run the runner with the venv's Python: the render-noise check needs numpy and Pillow"
        )
    layout = run_layout(cfg, args)
    tasks, all_tasks = select_tasks(cfg, args)
    product = s1_product(cfg["stage2"]["s1_product"], all_tasks)
    if cfg["stage2"]["require_config_row"] and not config_row_supported():
        raise SystemExit(
            "this code's escalation wrapper writes no config row, so N cannot be checked; use a code "
            "dir with it or set stage2.require_config_row = false"
        )
    fresh = make_manifest(cfg, layout.root.name, tasks, product)
    lock = C.acquire_lock(layout)  # noqa: F841 - held for the process lifetime
    if layout.manifest.exists():
        manifest = C.read_json(layout.manifest)
        if manifest.get("stage") != "s2":
            raise SystemExit(f"{layout.root} is not a stage-2 run")
        problems = check_resume(manifest, fresh, args.allow_code_change)
        if problems:
            raise SystemExit("refusing to resume: " + "; ".join(problems))
        if manifest["code"]["tree_sha256"] != fresh["code"]["tree_sha256"]:
            manifest["history"].append(
                {
                    "at": C.now_iso(),
                    "event": "code_changed",
                    "code": fresh["code"],
                }
            )
        manifest["history"].append(
            {
                "at": C.now_iso(),
                "event": "resumed",
                "pid": os.getpid(),
            }
        )
        C.atomic_write_json(layout.manifest, manifest)
    else:
        env = {
            **S1.base_env(cfg),
            "PYTHONPATH": cfg["paths"]["code_dir"],
            "RPENT_REPO_ROOT": cfg["paths"]["code_dir"],
        }
        origin = C.import_origin(C.python_for(cfg), cfg["paths"]["code_dir"], env)
        if not origin.get("ok"):
            raise SystemExit(
                f"episode processes would not import rpent/robots from the code dir: {origin}"
            )
        fresh["import_origin"] = origin
        fresh["runner_import_origin"] = S1.runner_origin(cfg["paths"]["code_dir"])
        if not fresh["runner_import_origin"]["ok"]:
            raise SystemExit(
                f"the runner does not import from the code dir: {fresh['runner_import_origin']}"
            )
        layout.tasks_dir.mkdir(parents=True, exist_ok=True)
        C.atomic_write_json(
            layout.root / "config.snapshot.json",
            {k: v for k, v in cfg.items() if k != "_path"},
        )
        C.atomic_write_json(layout.manifest, fresh)
        manifest = fresh
    stage = Stage2(cfg, layout, tasks, product)
    servers = C.ServerPool(
        cfg, layout, S1.base_env(cfg), lambda row: C.append_jsonl(layout.progress, row)
    )
    scheduler = LadderScheduler(
        cfg,
        layout,
        stage,
        servers,
        infra_retries=cfg["stage2"]["infra_retries"],
        render_retries=cfg["stage2"]["render_retries"],
        max_slots=args.max_slots,
    )
    C.append_jsonl(
        layout.progress,
        {
            "event": "start",
            "pid": os.getpid(),
            "tasks": len(tasks),
            "max_slots": scheduler.max_slots,
        },
    )
    if args.concurrency:
        C.atomic_write_text(layout.concurrency, f"{args.concurrency}\n")
    for job in stage.jobs():  # finish any record an earlier runner did not complete
        state = scheduler.state(job.job_id)
        if state["final"] and state["final"].get("class") not in ("skipped", "blocked"):
            if not stage.record_path(job.task, job.meta["rung"]).exists():
                stage.on_final(job, state)
    stage.settle()
    states = scheduler.all_states()
    pending = not all(s["final"] for s in states.values())
    up = servers.ensure(force_gpus=args.force_gpu or ()) if pending else []
    print(
        f"endpoints up: {[(ep.name, ep.url, ep.slots) for ep in up]}; "
        f"down: {[(ep.name, ep.note) for ep in servers.endpoints if not ep.up]}",
        flush=True,
    )
    result = "no_endpoint" if pending else "done"
    try:
        if up or scheduler.running(states):
            # Cells cannot list tasks/ while they run, so another task's or rung's folder is not discoverable.
            layout.tasks_dir.chmod(0o311)
            result = scheduler.run()
    finally:
        still = scheduler.running(scheduler.all_states())
        if not still:
            layout.tasks_dir.chmod(0o755)
            if cfg["servers"]["stop_on_exit"]:
                servers.stop_all()
        C.append_jsonl(
            layout.progress, {"event": "end", "result": result, "running": len(still)}
        )
    print(f"runner finished: {result}", flush=True)
    if result == "done" and not args.no_freeze:
        blocked = [
            t for _, t in tasks if (layout.task_dir(t) / "blocked.json").exists()
        ]
        if blocked:
            print(
                f"not frozen: blocked tasks {blocked} (see `unblock` and `freeze --fallback-blocked`)",
                flush=True,
            )
            return 1
        print(json.dumps(freeze(stage, manifest), indent=2), flush=True)
    return 0 if result in ("done", "stopped") else 1


def load_stage(
    cfg: dict[str, Any], args: argparse.Namespace
) -> tuple[C.RunLayout, dict[str, Any], Stage2]:
    layout = run_layout(cfg, args)
    manifest = C.read_json(layout.manifest)
    if manifest is None or manifest.get("stage") != "s2":
        raise SystemExit(f"no stage-2 run at {layout.root}")
    tasks = [(t["split"], t["task"]) for t in manifest["tasks"]]
    product = dict(manifest["s1_product"])
    doc = C.read_json(Path(product["path"]) / "PRODUCT.json") or {}
    product["corpus"] = str(Path(product["path"]) / doc.get("corpus", "corpus"))
    return layout, manifest, Stage2(cfg, layout, tasks, product)


def task_view(
    stage: Stage2, states: Mapping[str, Mapping[str, Any]], task: str
) -> dict[str, Any]:
    """Where one task stands: decided (rung, source), blocked, running rung, or queued rung."""
    rule = C.read_json(stage.rule_path(task))
    rungs = []
    for k in range(1, len(stage.ladder) + 1):
        state = states[job_id(task, k)]
        final = state.get("final") or {}
        tries = state["tries"]
        if final.get("class") == "valid":
            mark = "S" if final.get("success") else "F"
        elif final.get("class") in ("infra_exhausted", "render_exhausted"):
            mark = "X"
        elif final:
            mark = "-"
        elif tries and tries[-1].get("status") in ("launching", "running"):
            mark = "R"
        else:
            mark = "."
        rungs.append(mark)
    view: dict[str, Any] = {"rungs": "".join(rungs), "rule": rule}
    if rule:
        view["status"] = "decided"
    elif (stage.layout.task_dir(task) / "blocked.json").exists() or "X" in rungs:
        view["status"] = "blocked"
    elif "R" in rungs:
        view["status"] = "running"
        view["rung"] = rungs.index("R") + 1
    else:
        view["status"] = "queued"
        view["rung"] = rungs.index(".") + 1 if "." in rungs else None
    return view


def status_text(
    cfg: Mapping[str, Any],
    layout: C.RunLayout,
    manifest: Mapping[str, Any],
    stage: Stage2,
    *,
    tasks_table: bool = False,
) -> str:
    states = {job.job_id: stage.state(job.job_id) for job in stage.jobs()}
    for state in states.values():
        if state["tries"] and state["tries"][-1].get("status") in (
            "launching",
            "running",
        ):
            pid = state["tries"][-1].get("pid")
            state["tries"][-1]["alive"] = bool(pid and C.pid_alive(pid))
    views = {task: task_view(stage, states, task) for _, task in stage.tasks}
    n_rungs = len(stage.ladder)
    decided = sum(1 for v in views.values() if v["status"] == "decided")
    lines = [
        f"RC-ECC S2 {manifest['run_id']}  {C.progress_bar(decided, len(views))} tasks with a rule"
    ]
    head = "  ".join(f"r{k}" for k in range(1, n_rungs + 1))
    lines.append(
        f"  {'split':<17} {'final':>8} {'running':>8} {'queued':>7} {'blocked':>8}   {head}  fb   "
        f"{'infra-retry':>11} {'episodes':>9} {'$':>9}"
    )
    tries_all = [t for s in states.values() for t in s["tries"]]
    for split in [s for s in C.SPLITS if s in {sp for sp, _ in stage.tasks}]:
        names = [t for sp, t in stage.tasks if sp == split]
        counts = dict.fromkeys(("decided", "running", "queued", "blocked"), 0)
        dist = [0] * (n_rungs + 1)
        for task in names:
            view = views[task]
            counts[view["status"]] += 1
            rule = view.get("rule")
            if rule:
                if rule["rule_source"] == "first_success":
                    dist[rule["rung"] - 1] += 1
                else:
                    dist[n_rungs] += 1
        split_tries = [
            t
            for task in names
            for k in range(1, n_rungs + 1)
            for t in states[job_id(task, k)]["tries"]
        ]
        retries = sum(1 for t in split_tries if t.get("class") in ("infra", "render"))
        episodes = sum(
            1
            for task in names
            for k in range(1, n_rungs + 1)
            if (states[job_id(task, k)].get("final") or {}).get("class") == "valid"
        )
        usd = sum(float((t.get("usage") or {}).get("usd") or 0.0) for t in split_tries)
        cells = "  ".join(
            f"{n:>{len(f'r{k + 1}')}}" for k, n in enumerate(dist[:n_rungs])
        )
        lines.append(
            f"  {split:<17} {counts['decided']:>4}/{len(names):<3} {counts['running']:>8} {counts['queued']:>7} "
            f"{counts['blocked']:>8}   {cells}  {dist[n_rungs]:>2}   {retries:>11} {episodes:>9} {usd:>9.2f}"
        )
    by_reason = dict.fromkeys(H.SENIOR_REASONS, 0)
    errors_episodes = 0
    for t in tries_all:
        counts_ = (t.get("evidence") or {}).get("counts") or {}
        if t.get("class") == "valid":
            for reason, n in (counts_.get("senior_by_reason") or {}).items():
                by_reason[reason] = by_reason.get(reason, 0) + int(n)
            errors_episodes += int(bool(counts_.get("errors_trigger")))
    lines.append(
        "  final = the task has its rule; r1..r6 = rule from the first success at that rung; "
        "fb = the fallback rung (no rung succeeded)"
    )
    lines.append(
        "  GPT-6 calls in valid episodes: "
        + ", ".join(f"{r} {n}" for r, n in by_reason.items())
        + f"; errors trigger fired in {errors_episodes} episode(s)"
    )
    running_by_rung: dict[int, int] = {}
    for task, view in views.items():
        if view["status"] == "running":
            running_by_rung[view["rung"]] = running_by_rung.get(view["rung"], 0) + 1
    if running_by_rung:
        lines.append(
            "  running by rung: "
            + ", ".join(f"r{k} {n}" for k, n in sorted(running_by_rung.items()))
        )
    cost = C.total_cost(states, cfg["pricing"])
    worker_usd = sum(
        float((t.get("usage") or {}).get("worker_usd") or 0.0) for t in tries_all
    )
    senior_usd = sum(
        float((t.get("usage") or {}).get("senior_usd") or 0.0) for t in tries_all
    )
    lines.append(
        f"  cost so far (list price, all launches): ${cost:.2f} (finished launches: Qwen ${worker_usd:.2f}, "
        f"GPT-6 ${senior_usd:.2f})"
    )
    holder = C.lock_holder(layout)
    stop = C.read_json(layout.stop) if layout.stop.exists() else None
    lines.append(
        f"  runner: {'pid %s' % holder if holder else 'not running'}; stop file: "
        f"{(stop or {}).get('reason') if stop else 'absent'}"
    )
    load: dict[str, int] = {}
    for s in states.values():
        if s["tries"] and s["tries"][-1].get("status") == "running":
            name = s["tries"][-1].get("endpoint")
            load[name] = load.get(name, 0) + 1
    shown = []
    for ep in C.endpoints_from_config(cfg):
        info = C.read_json(layout.servers_dir / f"{ep.name}.json", {}) or {}
        alive = bool(
            info.get("pid") and C.pid_alive(info["pid"]) and not info.get("stopped_at")
        )
        shown.append(
            f"{ep.name} {'up' if alive else 'down'} {load.get(ep.name, 0)}/{ep.slots}"
        )
    lines.append("  endpoints: " + (", ".join(shown) or "none configured"))
    ceiling = C.launch_ceiling(
        cfg["scheduler"],
        len(stage.tasks) * n_rungs,
        len(stage.tasks),
        cfg["stage2"]["infra_retries"],
        cfg["stage2"]["render_retries"],
    )
    lines.append(f"  launches: {C.counted_launches(states)} of ceiling {ceiling}")
    mismatches = [t for t in tries_all if (t.get("evidence") or {}).get("ok") is False]
    if mismatches:
        lines.append(
            f"  EVIDENCE MISMATCH in {len(mismatches)} launch(es): see evidence_mismatch in progress.jsonl"
        )
    stale = [
        j
        for j, s in states.items()
        if s["tries"] and s["tries"][-1].get("alive") is False
    ]
    if stale:
        lines.append(
            f"  NOTE: {len(stale)} cell(s) marked running have no live process; the next --go scores them"
        )
    blocked = [task for task, v in views.items() if v["status"] == "blocked"]
    if blocked:
        lines.append(f"  blocked (a rung without a valid result): {', '.join(blocked)}")
    flagged = []
    for _, task in stage.tasks:
        for k in range(1, n_rungs + 1):
            audit = (C.read_json(stage.record_path(task, k)) or {}).get(
                "file_tool_audit"
            ) or {}
            kinds = [
                x
                for x in (
                    "cross_writes",
                    "memory_writes",
                    "cross_reads",
                    "code_dir_writes",
                )
                if audit.get(x)
            ]
            if kinds:
                flagged.append(f"{task} r{k} ({', '.join(kinds)})")
    if flagged:
        lines.append(
            f"  file-tool access outside the episode folder: {'; '.join(flagged)}"
        )
    if (layout.product / "FROZEN").exists():
        lines.append(
            f"  product: frozen, rules sha256 {(layout.product / 'FROZEN').read_text().strip()[:16]}"
        )
    if tasks_table:
        lines.append(
            f"  {'split':<17} {'task':<26} {'status':<9} rungs({n_rungs})  rule"
        )
        for split, task in stage.tasks:
            view = views[task]
            rule = view.get("rule")
            what = (
                f"{rule['config']} ({rule['rule_source']})"
                if rule
                else f"rung {view.get('rung')}"
                if view.get("rung")
                else ""
            )
            lines.append(
                f"  {split:<17} {task:<26} {view['status']:<9} {view['rungs']:<10} {what}"
            )
        lines.append(
            "  rungs: S success, F valid failure, R running, X no valid result, - skipped, . not run"
        )
    return "\n".join(lines)


def cmd_status(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    layout, manifest, stage = load_stage(cfg, args)
    print(status_text(cfg, layout, manifest, stage, tasks_table=args.tasks_table))
    return 0


def cmd_stop(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    layout, _, stage = load_stage(cfg, args)
    C.write_stop(layout, "operator", now=bool(args.now))
    print(f"wrote {layout.stop}: no new launches; the runner drains and exits")
    if args.now:
        killed = C.kill_running(
            layout,
            [job.job_id for job in stage.jobs()],
            grace_s=float(cfg["scheduler"]["kill_grace_s"]),
        )
        print(
            f"stopped {killed} running cell(s); they are relaunched (not counted) on the next --go"
        )
    running = []
    for job in stage.jobs():
        tries = stage.state(job.job_id)["tries"]
        if (
            tries
            and tries[-1].get("status") == "running"
            and C.pid_alive(tries[-1].get("pid"))
        ):
            running.append(job.job_id)
    if C.lock_holder(layout) is None and not running:
        servers = C.ServerPool(
            cfg, layout, {}, lambda row: C.append_jsonl(layout.progress, row)
        )
        servers.stop_all()
        print("no runner and no running cell: stopped this run's RLDX servers")
    return 0


def cmd_unblock(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Give a blocked task's rung a fresh infrastructure budget (the runner must not be running)."""
    layout, _, stage = load_stage(cfg, args)
    if C.lock_holder(layout) is not None:
        raise SystemExit("a runner holds the run; stop it first")
    done = []
    for task in args.task:
        if task not in stage.split_of:
            raise SystemExit(f"{task} is not in this run")
        decided = stage.decision(task)
        if not decided or decided[0] != "blocked":
            raise SystemExit(f"{task} is not blocked")
        _, rung = decided
        jid = job_id(task, rung)
        state = stage.state(jid)
        state["count_from"] = len(state["tries"])
        state.setdefault("unblocked", []).append(
            {
                "at": C.now_iso(),
                "previous_final": state["final"],
            }
        )
        state["final"] = None
        C.atomic_write_json(layout.job_state(jid), state)
        for k in range(rung + 1, len(stage.ladder) + 1):
            later = stage.state(job_id(task, k))
            if (later.get("final") or {}).get("class") == "blocked" and not later[
                "tries"
            ]:
                later["final"] = None
                C.atomic_write_json(layout.job_state(job_id(task, k)), later)
        (layout.task_dir(task) / "blocked.json").unlink(missing_ok=True)
        stage.record_path(task, rung).unlink(missing_ok=True)
        C.append_jsonl(
            layout.progress, {"event": "unblocked", "task": task, "rung": rung}
        )
        done.append(f"{task} r{rung}")
    print(f"unblocked: {', '.join(done)}; resume with run --go")
    return 0


def cmd_freeze(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    _, manifest, stage = load_stage(cfg, args)
    result = freeze(
        stage,
        manifest,
        fallback_blocked=args.fallback_blocked,
        allow_cross_writes=args.allow_cross_writes,
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify(Path(args.product).resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_rules(args: argparse.Namespace) -> int:
    """Print every task's rule from a frozen product (after verifying it)."""
    product = Path(args.product).resolve()
    rules = load_rules(product)
    doc = C.read_json(product / "RULES.json")
    for task in rules.keys():
        info = doc["tasks"][task]
        print(
            f"{info['split']:<17} {task:<26} {info['config']:<20} {info['rule_source']}"
        )
    print(json.dumps(doc["summary"], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    def common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config", help="TOML/JSON config outside the repo (or ECC_RC_CONFIG)"
        )
        p.add_argument("--run-id", help="run root name under paths.runs_dir")
        p.add_argument("--run-root", help="explicit run root (instead of --run-id)")

    for name in ("plan", "run"):
        p = sub.add_parser(name, help="dry run (default) / launch with --go")
        common_args(p)
        p.add_argument(
            "--tasks",
            help="comma list of Target50 tasks (default: stage2.tasks or all 50)",
        )
        p.add_argument(
            "--limit", type=int, default=0, help="first N tasks only (smoke)"
        )
        p.add_argument(
            "--max-slots",
            type=int,
            default=0,
            help="lower the concurrent-cell ceiling for this runner",
        )
        p.add_argument(
            "--force-gpu",
            type=int,
            action="append",
            help="start a server on this GPU without the free check",
        )
        p.add_argument(
            "--concurrency",
            type=int,
            default=0,
            help="write <run>/concurrency (live slot limit) before launching",
        )
        p.add_argument(
            "--go", action="store_true", help="really launch (otherwise print the plan)"
        )
        p.add_argument(
            "--no-freeze",
            action="store_true",
            help="do not freeze the rules when every task has one",
        )
        p.add_argument(
            "--allow-code-change",
            action="store_true",
            help="resume although the code tree changed",
        )
    p = sub.add_parser(
        "status", help="progress bar, per-split rung distribution, retries and cost"
    )
    common_args(p)
    p.add_argument(
        "--tasks-table", action="store_true", help="also print one line per task"
    )
    p = sub.add_parser(
        "stop", help="write the stop file (drain); --now also stops running cells"
    )
    common_args(p)
    p.add_argument("--now", action="store_true")
    p = sub.add_parser(
        "unblock", help="fresh infrastructure budget for a blocked task's rung"
    )
    common_args(p)
    p.add_argument("--task", action="append", required=True)
    p = sub.add_parser("freeze", help="freeze the rules (every task decided)")
    common_args(p)
    p.add_argument(
        "--fallback-blocked",
        action="store_true",
        help="give blocked tasks the fallback rung (recorded)",
    )
    p.add_argument(
        "--allow-cross-writes",
        action="store_true",
        help="freeze although an episode wrote outside its folder",
    )
    p = sub.add_parser("verify", help="recompute a frozen stage-2 product's hashes")
    p.add_argument("product")
    p = sub.add_parser("rules", help="print the rules of a frozen stage-2 product")
    p.add_argument("product")
    args = ap.parse_args(argv)
    if args.cmd in (None, "plan"):
        args.go = False
        args.cmd = "run"
        for attr, default in (
            ("tasks", None),
            ("limit", 0),
            ("max_slots", 0),
            ("force_gpu", None),
            ("concurrency", 0),
            ("no_freeze", False),
            ("allow_code_change", False),
            ("run_id", None),
            ("run_root", None),
            ("config", None),
        ):
            if not hasattr(args, attr):
                setattr(args, attr, default)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "rules":
        return cmd_rules(args)
    cfg = stage2_config(C.load_config(C.config_path(args.config)))
    return {
        "run": cmd_run,
        "status": cmd_status,
        "stop": cmd_stop,
        "unblock": cmd_unblock,
        "freeze": cmd_freeze,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
