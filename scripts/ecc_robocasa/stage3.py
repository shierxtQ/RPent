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

"""RC-ECC stage 3 on RoboCasa Target50: the cloud correction memory (GPT-6), seed 12.

Every task runs hybrid episodes (Qwen worker, GPT-6 senior) under its frozen
stage-2 rule, with the frozen stage-1 corpus read-only, on seed 12: up to
5 sessions x 3 attempts = 15 independent single-shot episodes, stopping at the
first valid success (``scripts/ecc/run_ecc_stage3.py``, 26bdd43). Before an
episode the task's correction memory is frozen as ``corrections_used.md`` and
passed as ``RPENT_CLOUD_CORRECTIONS`` (appended to every senior note, never
shown to Qwen). After every valid episode one offline GPT-6 reflection edits
the memory (``reflect.py``); an episode without a senior turn, or without any
planner turn, is not reflected on, and a content-filter refusal is recorded as
a skipped reflection. Invalid episodes (an invalid result, render noise, a
launch the runner stopped) are relaunched under the caps, never reflected on
and never counted. Every valid episode's ``escalation.jsonl`` must show the
rule's plan switch, K, N and the corrections it was launched with; a mismatch
makes the launch an infrastructure error and stops new launches.

``stage3.parallel_attempts = true`` (default false: the order above, as
run_ecc_stage3) runs the attempts of a session at the same time: all of them
with the memory as it was when the session started. Once every episode of the
session is final (invalid ones relaunched under the caps first), its valid
episodes are reflected on one after another in a fixed order (a1, a2, a3),
each reflection editing the memory the previous one left, and the next
session starts with the result. The task stops after the first session with a
valid success, all of whose valid episodes are reflected on. An id a
reflection lists in ``applied`` counts as used only if that episode was shown
the entry as it now stands (an entry an earlier reflection of the same session
wrote or revised was not), so an entry is never verified or failed by an
episode that did not see it. The
scheduler starts a session only when all its attempts can launch (free slots
and endpoint capacity for all of them) and launches its remaining attempts
before anything else, one launch stagger apart, so they start as one block;
relaunches of started sessions come next, then new sessions in the order
(session, task), so every task's first session is started before any second
session.

When every task is final, the product is frozen: ``product/final/<Task>.md``
(verified entries only, rendered as run_ecc_stage3 does; empty when none) for every task,
with a hashed manifest. Dry run by default (prints the plan); ``run --go``
launches. See README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecc_robocasa import common as C  # noqa: E402
from scripts.ecc_robocasa import hybrid_episode as H  # noqa: E402
from scripts.ecc_robocasa import reflect as R  # noqa: E402
from scripts.ecc_robocasa import stage1 as S1  # noqa: E402

STAGE = "s3"
PRODUCT_SCHEMA = "rc-ecc-s3-product/1"
#: Target50 cells (atomic 1-10, composite 1-5) and stage 1's seed 0 are never RSI scenes.
RESERVED_SEEDS = frozenset(range(0, 11))

S3_DEFAULTS: dict[str, Any] = {
    "seed": 12,
    "sessions": R.SESSIONS,
    "attempts_per_session": R.ATTEMPTS,
    # true: a session's attempts run at once with the session's memory, reflected on in order after it
    "parallel_attempts": False,
    "tasks": [],  # empty: all 50 Target50 tasks in manifest order
    "s1_product": "",  # <runs>/s1_full/product (verified with stage1.verify)
    # The frozen stage-2 product folder (<S2 run>/product), verified with stage2.verify; or, for
    # smokes only, s2_rules: a folder of <Task>.json rule files (run_ecc_stage2's schema), recorded as unfrozen.
    "s2_product": "",
    "s2_rules": "",
    "s2_rule_name": "{task}.json",
    "timeout_s_by_split": {},  # default: the Target50 protocol (atomic 1800 s, composite 3600 s)
    "infra_retries": 2,  # relaunches of one episode after an invalid result
    "render_retries": 6,  # relaunches of one episode after rendered noise
    # Runaway guard over episodes and reflections (0: scheduler.cost_ceiling_usd).
    "cost_ceiling_usd": 0,
    "launcher": [],  # episode launcher argv prefix (default: hybrid_cell.sh, as stage 2)
    # run_ecc_stage3's module constants (MAX_ENTRIES, FIELD_CHARS, FAILS_TO_DROP, CONTEXT_*)
    "max_entries": 8,
    "field_chars": 500,
    "fails_to_drop": 2,
    "context_before": 3,
    "context_after": 6,
    "interventions_chars": 40000,
    "summary_chars": 600,
    "reflection": {
        "model": "",  # default: senior.model (the senior's GPT-6)
        "reasoning_effort": "",  # default: senior.reasoning_effort (low)
        "max_output_tokens": 8192,
        "timeout_s": 900,  # one call, the client's retries included
        "empty_retries": 3,  # requests again after a reply without text
        "max_calls": 3,  # calls per episode before the run stops (errors, not refusals)
        "retry_delay_s": 120,
        "max_parallel": 16,
        "cmd": [],  # default: reflect_call.sh
    },
}
#: stage3 keys that define the experiment (paths, caps and launch commands do not).
EXPERIMENT_KEYS = (
    "seed",
    "sessions",
    "attempts_per_session",
    "timeout_s_by_split",
    "max_entries",
    "field_chars",
    "fails_to_drop",
    "context_before",
    "context_after",
    "interventions_chars",
    "summary_chars",
)
REFLECTION_EXPERIMENT_KEYS = (
    "model",
    "reasoning_effort",
    "max_output_tokens",
    "empty_retries",
)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def load_config(path: Path | str) -> dict[str, Any]:
    """The common config plus the ``[stage3]`` section's defaults, validated."""
    cfg = C.load_config(path)
    cfg["stage3"] = C._deep_merge(S3_DEFAULTS, cfg.get("stage3") or {})
    s3 = cfg["stage3"]
    code_dir = cfg["paths"].get("code_dir") or str(REPO_ROOT)
    if not s3["launcher"]:
        s3["launcher"] = ["bash", "{code_dir}/scripts/ecc_robocasa/hybrid_cell.sh"]
    if not s3["reflection"]["cmd"]:
        s3["reflection"]["cmd"] = [
            "bash",
            "{code_dir}/scripts/ecc_robocasa/reflect_call.sh",
        ]
    for key in ("launcher",):
        s3[key] = [str(x).replace("{code_dir}", code_dir) for x in s3[key]]
    s3["reflection"]["cmd"] = [
        str(x).replace("{code_dir}", code_dir) for x in s3["reflection"]["cmd"]
    ]
    for key in ("s1_product", "s2_product", "s2_rules"):
        if isinstance(s3[key], str) and s3[key].startswith("~"):
            s3[key] = str(Path(s3[key]).expanduser())
    validate(cfg)
    return cfg


def validate(cfg: Mapping[str, Any]) -> None:
    s3 = cfg["stage3"]
    seed = s3["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("stage3.seed must be a nonnegative integer")
    s2_seed = int((cfg.get("stage2") or {}).get("seed", 11))
    if seed in RESERVED_SEEDS or seed == s2_seed:
        raise ValueError(
            f"stage3.seed {seed} is a Target50 cell, stage 1's seed or stage 2's seed "
            f"({s2_seed}); stage 3 explores its own seed (12)"
        )
    for key in ("sessions", "attempts_per_session"):
        if not isinstance(s3[key], int) or s3[key] < 1:
            raise ValueError(f"stage3.{key} must be a positive integer")
    if not isinstance(s3["parallel_attempts"], bool):
        raise ValueError("stage3.parallel_attempts must be true or false")
    for key in ("infra_retries", "render_retries"):
        if not isinstance(s3[key], int) or s3[key] < 0:
            raise ValueError(f"stage3.{key} must be a nonnegative integer")
    R.Settings.from_config(s3)
    refl = s3["reflection"]
    for key in ("timeout_s", "max_calls", "max_parallel", "max_output_tokens"):
        if not isinstance(refl[key], int) or refl[key] < 1:
            raise ValueError(f"stage3.reflection.{key} must be a positive integer")
    for split, value in (s3["timeout_s_by_split"] or {}).items():
        if split not in H.TARGET50_TIMEOUT_S or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"stage3.timeout_s_by_split: bad entry {split} = {value!r}"
            )


def budget(cfg: Mapping[str, Any]) -> int:
    s3 = cfg["stage3"]
    return int(s3["sessions"]) * int(s3["attempts_per_session"])


def timeout_for(cfg: Mapping[str, Any], split: str) -> int:
    return int(
        {**H.TARGET50_TIMEOUT_S, **(cfg["stage3"]["timeout_s_by_split"] or {})}[split]
    )


def reflection_model(cfg: Mapping[str, Any]) -> tuple[str, str]:
    refl = cfg["stage3"]["reflection"]
    return (
        str(refl["model"] or cfg["senior"]["model"]),
        str(refl["reasoning_effort"] or cfg["senior"]["reasoning_effort"]),
    )


def experiment_settings(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``[stage3]`` values that define the experiment.

    ``parallel_attempts`` is listed only when on, so a sequential run's
    settings and hash are what they were before the switch existed.
    """
    s3 = cfg["stage3"]
    out = {k: s3[k] for k in EXPERIMENT_KEYS}
    if s3["parallel_attempts"]:
        out["parallel_attempts"] = True
    return out


def experiment_hash(cfg: Mapping[str, Any]) -> str:
    s3 = cfg["stage3"]
    return C.canonical_hash(
        {
            "worker": cfg["worker"],
            # K comes from the task's rule, not senior.checkin.
            "senior": {k: v for k, v in cfg["senior"].items() if k != "checkin"},
            "episode": cfg["episode"],
            "stage3": experiment_settings(cfg),
            "reflection": {k: s3["reflection"][k] for k in REFLECTION_EXPERIMENT_KEYS},
        }
    )


def settings(cfg: Mapping[str, Any]) -> R.Settings:
    return R.Settings.from_config(cfg["stage3"])


def senior_usd(usage: Mapping[str, Any] | None, pricing: Mapping[str, float]) -> float:
    """List price of one GPT-6 reply's usage (the senior's prices)."""
    if not usage:
        return 0.0
    cached = int(usage.get("cached_input_tokens") or 0)
    return (
        (int(usage.get("input_tokens") or 0) - cached)
        * pricing["senior_input_per_mtok"]
        + cached * pricing["senior_cached_input_per_mtok"]
        + int(usage.get("output_tokens") or 0) * pricing["senior_output_per_mtok"]
    ) / 1e6


# ---------------------------------------------------------------------------
# inputs: the frozen stage-1 corpus and the stage-2 rules
# ---------------------------------------------------------------------------


def s1_input(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The verified stage-1 product: its corpus is every episode's ``--memory-dir``."""
    product = cfg["stage3"]["s1_product"]
    if not product:
        raise SystemExit("set stage3.s1_product (the frozen stage-1 product folder)")
    product = Path(product).resolve()
    check = S1.verify(product)
    if not check.get("ok"):
        raise SystemExit(f"the stage-1 product {product} does not verify: {check}")
    doc = C.read_json(product / "PRODUCT.json") or {}
    return {
        "product": str(product),
        "corpus": str(product / doc["corpus"]),
        "corpus_tree_sha256": doc["corpus_tree_sha256"],
        "product_json_sha256": C.sha256_file(product / "PRODUCT.json"),
        "run_id": doc.get("run_id"),
    }


def parse_rule(doc: Any) -> H.Rule:
    """A stage-2 rule file (run_ecc_stage2's schema ``{"rule": {"gpt_plan", "checkin_K", "errors_N"}}``)."""
    if not isinstance(doc, Mapping):
        raise ValueError("not a JSON object")
    rule = doc.get("rule") if isinstance(doc.get("rule"), Mapping) else doc
    aliases = {
        "gpt_plan": ("gpt_plan", "plan"),
        "checkin_K": ("checkin_K", "checkin", "K"),
        "errors_N": ("errors_N", "errors", "N"),
    }
    value = {}
    for key, names in aliases.items():
        found = next((rule[n] for n in names if n in rule), None)
        if found is None:
            raise ValueError(f"no {' / '.join(names)} in the rule")
        value[key] = found
    return H.Rule.from_json(value)


def rules_input(
    cfg: Mapping[str, Any], tasks: list[tuple[str, str]]
) -> tuple[dict[str, H.Rule], dict[str, Any]]:
    """Every task's frozen stage-2 rule, with the sha256 of each rule file.

    From the frozen stage-2 product (``stage3.s2_product``, verified with
    ``stage2.verify``), or from a plain folder of rule files (``stage3.s2_rules``).
    """
    s3 = cfg["stage3"]
    info: dict[str, Any]
    if s3["s2_product"]:
        from scripts.ecc_robocasa import stage2 as S2

        product = Path(s3["s2_product"]).resolve()
        check = S2.verify(product)
        if not check.get("ok"):
            raise SystemExit(f"the stage-2 product {product} does not verify: {check}")
        doc = C.read_json(product / "RULES.json") or {}
        folder = product / doc["rules"]
        info = {
            "kind": "s2_product",
            "product": str(product),
            "rules_json_sha256": C.sha256_file(product / "RULES.json"),
            "rules_tree_sha256": check.get("rules_tree_sha256"),
            "s1_corpus_tree_sha256": check.get("s1_corpus_tree_sha256"),
            "seed": doc.get("seed"),
        }
        name = "{task}.json"
    elif s3["s2_rules"]:
        folder = Path(s3["s2_rules"]).resolve()
        info = {"kind": "rules_folder (not a frozen stage-2 product)"}
        name = s3["s2_rule_name"]
    else:
        raise SystemExit("set stage3.s2_product (the frozen stage-2 product folder)")
    rules, files, problems = {}, {}, []
    for _, task in tasks:
        path = folder / name.format(task=task)
        try:
            rules[task] = parse_rule(json.loads(path.read_text()))
            files[task] = C.sha256_file(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"{task}: {path.name}: {exc}")
    if problems:
        raise SystemExit(
            f"stage-2 rules missing or invalid ({len(problems)}): {problems[:10]}"
        )
    info.update(
        folder=str(folder),
        name=name,
        files=files,
        rules_sha256=C.canonical_hash({t: r.as_json() for t, r in rules.items()}),
    )
    return rules, info


def rung_of(rule: H.Rule) -> int | None:
    return H.LEADER_LADDER.index(rule) + 1 if rule in H.LEADER_LADDER else None


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


#: Task and episode folders while the runner runs: enter and create, but not list (as stage 1's tasks/).
LOCKED_MODE = 0o311


def job_id(task: str, name: str) -> str:
    return f"{task}__{name}"


class Stage3(C.Stage):
    """One job per (task, episode); episode e+1 waits for episode e's reflection
    (parallel sessions: session s+1 waits for every reflection of session s)."""

    name = STAGE

    def __init__(
        self,
        cfg: Mapping[str, Any],
        layout: C.RunLayout,
        tasks: list[tuple[str, str]],
        rules: Mapping[str, H.Rule],
        corpus: Path,
    ):
        self.cfg = cfg
        self.layout = layout
        self.tasks = tasks
        self.rules = dict(rules)
        self.corpus = Path(corpus)
        self.seed = int(cfg["stage3"]["seed"])
        self.budget = budget(cfg)
        self.attempts = int(cfg["stage3"]["attempts_per_session"])
        self.sessions = int(cfg["stage3"]["sessions"])
        self.parallel = bool(cfg["stage3"]["parallel_attempts"])
        self.settings = settings(cfg)
        self.refl = cfg["stage3"]["reflection"]
        self.code_dir = Path(cfg["paths"].get("code_dir") or REPO_ROOT)
        self.split_of = {task: split for split, task in tasks}
        self._jobs = [
            C.Job(
                job_id=job_id(task, R.episode_name(e, self.attempts)),
                task=task,
                split=split,
                seed=self.seed,
                order=i * 100 + e,
                meta={"episode": e, "name": R.episode_name(e, self.attempts)},
            )
            for i, (split, task) in enumerate(tasks)
            for e in range(self.budget)
        ]
        self.by_id = {job.job_id: job for job in self._jobs}
        #: Set by the scheduler that runs this stage (job states, saving).
        self.scheduler: C.Scheduler | None = None
        #: Reflection processes this runner started: episode dir -> Popen.
        self.children: dict[str, subprocess.Popen] = {}

    # -- paths ---------------------------------------------------------------

    def jobs(self) -> list[C.Job]:
        return list(self._jobs)

    def name_of(self, e: int) -> str:
        return R.episode_name(e, self.attempts)

    def episode_dir(self, task: str, e: int) -> Path:
        return self.layout.task_dir(task) / self.name_of(e)

    def try_dir(self, job: C.Job, n: int) -> Path:
        return self.episode_dir(job.task, job.meta["episode"]) / f"try{n}"

    def snapshot(self, task: str, e: int) -> Path:
        return self.episode_dir(task, e) / "corrections_used.md"

    def reflection_path(self, task: str, e: int) -> Path:
        return self.episode_dir(task, e) / "reflection.json"

    def reflection_dir(self, task: str, e: int) -> Path:
        return self.episode_dir(task, e) / "reflection"

    def corrections_dir(self, task: str) -> Path:
        return self.layout.task_dir(task) / "corrections"

    def record_path(self, task: str) -> Path:
        return self.layout.task_dir(task) / "record.json"

    def budget_launches(self) -> int:
        return len(self._jobs)

    # -- sessions (parallel mode) ---------------------------------------------

    def session_of(self, e: int) -> int:
        return e // self.attempts

    def session_episodes(self, s: int) -> range:
        return range(s * self.attempts, (s + 1) * self.attempts)

    def session_started(
        self, task: str, e: int, states: Mapping[str, Mapping[str, Any]]
    ) -> bool:
        """Whether any episode of ``e``'s session has been launched."""
        return any(
            states[job_id(task, self.name_of(p))].get("tries")
            for p in self.session_episodes(self.session_of(e))
        )

    # -- the correction memory chain ----------------------------------------

    def entries_before(self, task: str, e: int) -> list[dict[str, Any]]:
        """The memory reflection ``e`` edits: the previous reflection's ``entries_after``."""
        if e == 0:
            return []
        record = C.read_json(self.reflection_path(task, e - 1))
        if not isinstance(record, dict) or "entries_after" not in record:
            raise RuntimeError(f"{task} {self.name_of(e - 1)} has no reflection record")
        return list(record["entries_after"])

    def memory_used(self, task: str, e: int) -> list[dict[str, Any]]:
        """The memory episode ``e`` runs with: ``entries_before(e)``, or with parallel sessions
        the memory at the start of its session (the previous session's last reflection)."""
        if not self.parallel:
            return self.entries_before(task, e)
        return self.entries_before(task, self.session_of(e) * self.attempts)

    def shown_ids(self, task: str, e: int) -> set[str] | None:
        """Ids of the entries episode ``e`` was shown as they still stand, when that can
        differ from the memory its reflection edits (parallel sessions: entries an earlier
        reflection of the session wrote or revised were not shown); None otherwise."""
        if not self.parallel:
            return None
        now = {str(x["id"]): x for x in self.entries_before(task, e)}
        keys = ("trigger", *R.ENTRY_FIELDS)
        return {
            str(x["id"])
            for x in self.memory_used(task, e)
            if str(x["id"]) in now
            and all(now[str(x["id"])].get(k) == x.get(k) for k in keys)
        }

    def prepare_try(self, job: C.Job, n: int) -> None:
        """Freeze the memory for the episode (try 1) and check it before every launch.

        The snapshot must equal the rendering of the previous reflection's
        entries (parallel sessions: of the previous session's last reflection);
        RPent silently ignores a missing corrections file, so its presence,
        sha256 and size are checked and logged here.
        """
        task, e = job.task, job.meta["episode"]
        path = self.snapshot(task, e)
        memory = self.memory_used(task, e)
        text = R.render_file(memory)
        if not path.exists():
            if n != 1 and (self.episode_dir(task, e) / "try1").exists():
                raise RuntimeError(f"{path} vanished after try 1 launched")
            C.atomic_write_text(path, text)
            path.chmod(0o444)
        on_disk = path.read_text(encoding="utf-8")
        if on_disk != text:
            raise RuntimeError(f"{path} differs from the previous reflection's entries")
        info = corrections_info(path)
        episode = self.episode_dir(task, e) / "episode.json"
        if not episode.exists():
            first = self.session_of(e) * self.attempts
            C.atomic_write_json(
                episode,
                {
                    "task": task,
                    "split": job.split,
                    "episode": self.name_of(e),
                    "index": e,
                    "seed": self.seed,
                    "rule": self.rules[task].as_json(),
                    "rung": rung_of(self.rules[task]),
                    "corrections": {"path": str(path), **info},
                    "entries": len(memory),
                    **(
                        {
                            "session": self.session_of(e) + 1,
                            # the reflection whose entries_after the session runs with
                            "memory_from": self.name_of(first - 1) if first else None,
                        }
                        if self.parallel
                        else {}
                    ),
                    "created_at": C.now_iso(),
                },
            )
        C.append_jsonl(
            self.layout.progress,
            {"event": "corrections", "job": job.job_id, "try": n, **info},
        )
        # Episodes may read any path by name, but cannot list where the senior's memory lies.
        for folder in (self.layout.task_dir(task), self.episode_dir(task, e)):
            folder.chmod(LOCKED_MODE)

    def unlock(self) -> None:
        """Listable task and episode folders again (when the runner exits)."""
        for _, task in self.tasks:
            folder = self.layout.task_dir(task)
            if folder.is_dir():
                folder.chmod(0o755)
            for e in range(self.budget):
                if self.episode_dir(task, e).is_dir():
                    self.episode_dir(task, e).chmod(0o755)

    def launch(
        self, job: C.Job, n: int, endpoint: C.Endpoint
    ) -> tuple[list[str], dict[str, str]]:
        path = self.snapshot(job.task, job.meta["episode"])
        argv = [
            *self.cfg["stage3"]["launcher"],
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
            rule=self.rules[job.task],
            timeout_s=timeout_for(self.cfg, job.split),
            corrections=path,
            corrections_sha256=C.sha256_file(path) if path.exists() else "",
        )
        return argv, env

    # -- evidence ------------------------------------------------------------

    def expected(self, job: C.Job) -> dict[str, Any]:
        """What the episode's escalation log must show: the task's rule and its snapshot."""
        text = self.snapshot(job.task, job.meta["episode"]).read_text(encoding="utf-8")
        return H.expected_routing(self.cfg, self.rules[job.task], corrections_text=text)

    def evidence(self, job: C.Job, n: int) -> dict[str, Any]:
        """Check try ``n``'s escalation log against the rule and the snapshot; kept next to it."""
        expected = self.expected(job)
        result = H.check_escalation(
            self.try_dir(job, n) / "escalation.jsonl", expected, result_valid=True
        )
        C.atomic_write_json(
            self.episode_dir(job.task, job.meta["episode"]) / f"try{n}.evidence.json",
            {"expected": expected, **result, "at": C.now_iso()},
        )
        return result

    # -- readiness, progress -------------------------------------------------

    def ready(self, job: C.Job, states: Mapping[str, Mapping[str, Any]]) -> bool:
        """Episode e launches once e-1 is a valid failure whose reflection is written
        (parallel sessions: once every episode of the previous session is)."""
        e = job.meta["episode"]
        if self.record_path(job.task).exists():
            return False
        if self.parallel:
            s = self.session_of(e)
            previous = self.session_episodes(s - 1) if s else range(0)
        else:
            previous = range(e - 1, e) if e else range(0)
        for p in previous:
            final = states[job_id(job.task, self.name_of(p))].get("final")
            if not final or final.get("class") != "valid" or final.get("success"):
                return False
            if not self.reflection_path(job.task, p).exists():
                return False
        return True

    def tick(self, states: Mapping[str, Mapping[str, Any]]) -> None:
        """Start, poll and apply reflections; finalize tasks that are done."""
        pending = []
        for _, task in self.tasks:
            if self.record_path(task).exists():
                continue
            try:
                action = self._scan(task, states)
                if action and action[0] == "final":
                    self.finalize(task, states, status=action[1], last=action[2])
                elif action:
                    pending.append(action[1:])
            except Exception as exc:  # noqa: BLE001 - logged; the next tick retries
                self._tick_error(task, exc)
        running = sum(
            1
            for task, e, _ in pending
            if (C.read_json(self.reflection_dir(task, e) / "state.json") or {}).get(
                "status"
            )
            == "running"
        )
        for task, e, state in pending:
            try:
                running += self.advance_reflection(task, e, state, running)
            except Exception as exc:  # noqa: BLE001 - logged; the next tick retries
                self._tick_error(task, exc)

    def _tick_error(self, task: str, exc: Exception) -> None:
        C.append_jsonl(
            self.layout.progress,
            {
                "event": "tick_error",
                "task": task,
                "error": f"{type(exc).__name__}: {exc}"[:500],
            },
        )

    def _scan(self, task: str, states: Mapping[str, Mapping[str, Any]]) -> tuple | None:
        """``("final", status, last)``, ``("reflect", task, e, state)``, or None (an episode is
        queued or running)."""
        if self.parallel:
            return self._scan_sessions(task, states)
        for e in range(self.budget):
            state = states[job_id(task, self.name_of(e))]
            final = state.get("final")
            if not final:
                return None
            if final.get("class") == "not_needed":
                continue
            if final.get("class") != "valid":
                return ("final", "infra_exhausted", e)
            if not self.reflection_path(task, e).exists():
                return ("reflect", task, e, state)
            if final.get("success"):
                return ("final", "solved", e)
        return ("final", "unsolved", self.budget - 1)

    def _scan_sessions(
        self, task: str, states: Mapping[str, Mapping[str, Any]]
    ) -> tuple | None:
        """:meth:`_scan` for parallel sessions: nothing happens to a session until every
        episode of it is final; then its reflections run in episode order, one at a time,
        and the task stops after the session if any of its episodes succeeded validly. An
        episode out of relaunches stops the task there (``requeue`` reruns it; the later
        episodes of the session are reflected on after it)."""
        for s in range(self.sessions):
            episodes = list(self.session_episodes(s))
            finals = [
                states[job_id(task, self.name_of(e))].get("final") for e in episodes
            ]
            if all(f and f.get("class") == "not_needed" for f in finals):
                continue
            if not all(finals):
                return None
            for e, final in zip(episodes, finals):
                if final.get("class") == "not_needed":
                    continue
                if final.get("class") != "valid":
                    return ("final", "infra_exhausted", e)
                if not self.reflection_path(task, e).exists():
                    return ("reflect", task, e, states[job_id(task, self.name_of(e))])
            if any(f.get("class") == "valid" and f.get("success") for f in finals):
                return ("final", "solved", episodes[-1])
        return ("final", "unsolved", self.budget - 1)

    # -- reflections ---------------------------------------------------------

    def final_try_dir(self, state: Mapping[str, Any]) -> Path:
        final = state["final"]
        return Path(state["tries"][final["try"] - 1]["dir"])

    def advance_reflection(
        self, task: str, e: int, state: Mapping[str, Any], running: int
    ) -> int:
        """Move episode ``e``'s reflection on; returns how many new calls were started."""
        folder = self.reflection_dir(task, e)
        rstate = C.read_json(folder / "state.json") or {}
        status = rstate.get("status")
        if status == "running":
            self.poll_reflection(task, e, state, rstate)
            return 0
        if status == "failed":
            return 0
        if status == "retry" and time.time() < float(rstate.get("next_at") or 0):
            return 0
        if running >= int(self.refl["max_parallel"]):
            return 0
        return self.start_reflection(task, e, state, rstate)

    def start_reflection(
        self, task: str, e: int, state: Mapping[str, Any], rstate: dict[str, Any]
    ) -> int:
        try_dir = self.final_try_dir(state)
        before = self.entries_before(task, e)
        snapshot = self.snapshot(task, e)
        if snapshot.read_text(encoding="utf-8") != R.render_file(
            self.memory_used(task, e)
        ):
            raise RuntimeError(f"{snapshot} does not match the previous reflection")
        d = R.digest(try_dir, self.settings)
        outcome = C.read_outcome(try_dir)
        success = bool(state["final"].get("success"))
        base = self._record_base(task, e, try_dir, d, outcome, success, before)
        skip = R.decide(d)
        if skip:
            self.write_reflection(
                task,
                e,
                {
                    **base,
                    "status": "skipped",
                    "skipped": skip,
                    "edits": dict(R.NO_EDITS),
                    "used": [],
                    "entries_after": before,
                },
                rstate,
            )
            return 0
        model, effort = reflection_model(self.cfg)
        prompt = R.build_prompt(
            d,
            language=R.task_language(try_dir),
            rule=self.rules[task].as_json(),
            success=success,
            termination=str(outcome.get("termination_reason") or ""),
            entries=before,
            settings=self.settings,
        )
        folder = self.reflection_dir(task, e)
        folder.mkdir(parents=True, exist_ok=True)
        C.atomic_write_text(folder / "prompt.txt", prompt)
        request = {
            "model": model,
            "reasoning_effort": effort,
            "max_output_tokens": int(self.refl["max_output_tokens"]),
            "timeout_s": int(self.refl["timeout_s"]),
            "empty_retries": int(self.refl["empty_retries"]),
            "prompt_file": "prompt.txt",
            "prompt_sha256": R.prompt_sha256(prompt),
            "episode": base["episode"],
        }
        C.atomic_write_json(folder / "request.json", request)
        calls = int(rstate.get("calls") or 0) + 1
        reply, log = folder / f"reply{calls}.json", folder / f"call{calls}.log"
        with log.open("w") as out:
            proc = subprocess.Popen(
                [*self.refl["cmd"], str(folder / "request.json"), str(reply)],
                env=self.reflection_env(),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=str(self.code_dir),
            )
        self.children[str(folder)] = proc
        rstate.update(
            status="running",
            calls=calls,
            pid=proc.pid,
            reply=reply.name,
            log=log.name,
            started_at=C.now_iso(),
            started_ts=time.time(),
            prompt_sha256=request["prompt_sha256"],
        )
        C.atomic_write_json(folder / "state.json", rstate)
        C.append_jsonl(
            self.layout.progress,
            {
                "event": "reflection_start",
                "task": task,
                "episode": self.name_of(e),
                "call": calls,
                "pid": proc.pid,
                "prompt_chars": len(prompt),
            },
        )
        return 1

    def reflection_env(self) -> dict[str, str]:
        """The cell launchers' credential inputs (names and the file path only)."""
        cred = self.cfg["credentials"]
        env = S1.base_env(self.cfg)
        env.update(
            ECC_RC_CRED_FILE=str(cred.get("file") or ""),
            ECC_RC_SENIOR_BASE_URL_VAR=str(cred.get("senior_base_url_var") or ""),
            ECC_RC_SENIOR_KEY_VAR=str(cred.get("senior_key_var") or ""),
        )
        return env

    def poll_reflection(
        self, task: str, e: int, state: Mapping[str, Any], rstate: dict[str, Any]
    ) -> None:
        folder = self.reflection_dir(task, e)
        pid = int(rstate.get("pid") or 0)
        proc = self.children.get(str(folder))
        if proc is not None:
            alive = proc.poll() is None
            if not alive:
                del self.children[str(folder)]
        else:
            alive = (
                bool(pid)
                and C.pid_alive(pid)
                and C.cmdline_has(pid, str(folder / "request.json"))
            )
        if alive:
            limit = float(self.refl["timeout_s"]) + 180
            if time.time() - float(rstate.get("started_ts") or time.time()) > limit:
                C.kill_group(pid, grace_s=10)
                self.children.pop(str(folder), None)
                self.call_failed(
                    task, e, rstate, f"no reply within {limit:.0f} s; stopped"
                )
            return
        reply = C.read_json(folder / str(rstate.get("reply") or ""))
        if not isinstance(reply, dict):
            self.call_failed(
                task, e, rstate, "the reflection call wrote no reply (see its call log)"
            )
            return
        usd = sum(
            senior_usd(u, self.cfg["pricing"])
            for u in reply.get("usage_all_attempts") or []
        )
        rstate["usd"] = round(float(rstate.get("usd") or 0.0) + usd, 6)
        rstate.setdefault("history", []).append(
            {
                k: reply.get(k)
                for k in ("status", "latency_s", "retries", "empty_retries", "error")
            }
            | {"call": rstate.get("calls"), "usd": round(usd, 6)}
        )
        if reply.get("status") not in ("ok", "refused"):
            self.call_failed(
                task, e, rstate, f"{reply.get('status')}: {reply.get('error')}"
            )
            return
        try_dir = self.final_try_dir(state)
        before = self.entries_before(task, e)
        d = R.digest(try_dir, self.settings)
        success = bool(state["final"].get("success"))
        base = self._record_base(
            task, e, try_dir, d, C.read_outcome(try_dir), success, before
        )
        call = {
            "model": reply.get("model"),
            "reasoning_effort": reply.get("reasoning_effort"),
            "prompt_sha256": rstate.get("prompt_sha256"),
            "prompt_chars": len((folder / "prompt.txt").read_text(encoding="utf-8")),
            "usage": reply.get("usage"),
            "calls": rstate.get("calls"),
            "latency_s": reply.get("latency_s"),
            "retries": reply.get("retries"),
            "usd": rstate["usd"],
        }
        if reply["status"] == "refused":
            record = {
                **base,
                **call,
                "status": "skipped",
                "skipped": "content filter refusal",
                "refusal": str(reply.get("error") or "")[:500],
                "edits": dict(R.NO_EDITS),
                "used": [],
                "entries_after": before,
            }
        else:
            text = str(reply.get("text") or "")
            done = R.reviewed(
                before,
                text,
                base["episode"],
                success,
                self.settings,
                shown=self.shown_ids(task, e),
            )
            record = {
                **base,
                **call,
                **done,
                "reply": text,
                "reply_sha256": R.prompt_sha256(text),
            }
        self.write_reflection(task, e, record, rstate)

    def call_failed(self, task: str, e: int, rstate: dict[str, Any], why: str) -> None:
        folder = self.reflection_dir(task, e)
        calls = int(rstate.get("calls") or 0)
        row = {
            "event": "reflection_failed",
            "task": task,
            "episode": self.name_of(e),
            "call": calls,
            "why": why[:500],
        }
        if calls - int(rstate.get("calls_base") or 0) < int(self.refl["max_calls"]):
            rstate.update(
                status="retry",
                next_at=time.time() + float(self.refl["retry_delay_s"]),
                last_error=why[:500],
            )
            row["retry_in_s"] = self.refl["retry_delay_s"]
        else:
            rstate.update(status="failed", last_error=why[:500])
            C.write_stop(
                self.layout,
                "reflection failed",
                task=task,
                episode=self.name_of(e),
                error=why[:300],
            )
        C.atomic_write_json(folder / "state.json", rstate)
        C.append_jsonl(self.layout.progress, row)

    def retry_failed_reflections(self) -> None:
        """Give every failed reflection a fresh call budget (a new runner after a STOP)."""
        for _, task in self.tasks:
            for e in range(self.budget):
                path = self.reflection_dir(task, e) / "state.json"
                rstate = C.read_json(path)
                if isinstance(rstate, dict) and rstate.get("status") == "failed":
                    rstate.update(
                        status="retry",
                        next_at=0,
                        calls_base=int(rstate.get("calls") or 0),
                    )
                    C.atomic_write_json(path, rstate)
                    C.append_jsonl(
                        self.layout.progress,
                        {
                            "event": "reflection_requeued",
                            "task": task,
                            "episode": self.name_of(e),
                        },
                    )

    def _record_base(
        self,
        task: str,
        e: int,
        try_dir: Path,
        d: Mapping[str, Any],
        outcome: Mapping[str, Any],
        success: bool,
        before: list[dict[str, Any]],
    ) -> dict[str, Any]:
        base = {
            "episode": f"{task}/{self.name_of(e)}",
            "index": e,
            "success": success,
            "termination": outcome.get("termination_reason"),
            "try_dir": str(try_dir),
            "interventions": d["interventions"],
            "by_reason": d["by_reason"],
            "shown": d["shown"],
            "turns": d["turns"],
            "digest_source": d["source"],
            "entries_before": before,
        }
        if self.parallel:
            base["session"] = self.session_of(e) + 1
            # the ids the episode ran with (its session's memory): only these can count as used
            base["session_memory"] = [str(x["id"]) for x in self.memory_used(task, e)]
        return base

    def write_reflection(
        self, task: str, e: int, record: dict[str, Any], rstate: dict[str, Any]
    ) -> None:
        """Write the reflection record, then the task's work memory and history (idempotent)."""
        record["at"] = C.now_iso()
        path = self.reflection_path(task, e)
        C.atomic_write_json(path, record)
        path.chmod(0o444)
        rstate.update(status="done", done_at=C.now_iso())
        folder = self.reflection_dir(task, e)
        folder.mkdir(parents=True, exist_ok=True)
        C.atomic_write_json(folder / "state.json", rstate)
        after = record["entries_after"]
        corr = self.corrections_dir(task)
        corr.mkdir(parents=True, exist_ok=True)
        C.atomic_write_json(
            corr / "work.json",
            {
                "task": task,
                "entries": after,
                "after": record["episode"],
                "updated_at": C.now_iso(),
            },
        )
        C.atomic_write_text(corr / "work.md", R.render_file(after))
        C.append_jsonl(
            corr / "history.jsonl",
            {
                "episode": record["episode"],
                "status": record["status"],
                "skipped": record.get("skipped"),
                "success": record["success"],
                "prompt_sha256": record.get("prompt_sha256"),
                "reply": record.get("reply"),
                "edits": record.get("edits"),
                "used": record.get("used"),
                **(
                    {
                        "session_memory": record.get("session_memory"),
                        "applied_not_shown": record.get("applied_not_shown"),
                    }
                    if self.parallel
                    else {}
                ),
                "entries_before": record["entries_before"],
                "entries_after": after,
                "usd": record.get("usd", 0.0),
                "reflection_json_sha256": C.sha256_file(path),
            },
        )
        C.append_jsonl(
            self.layout.progress,
            {
                "event": "reflection_done",
                "task": task,
                "episode": self.name_of(e),
                "status": record["status"],
                "skipped": record.get("skipped"),
                "entries": len(after),
                "verified": len(R.verified(after)),
            },
        )

    # -- the task's end ------------------------------------------------------

    def finalize(
        self,
        task: str,
        states: Mapping[str, Mapping[str, Any]],
        *,
        status: str,
        last: int,
    ) -> None:
        """Mark the episodes not needed, write the kept memory and the task record."""
        episodes, entries = [], []
        for e in range(self.budget):
            state = states[job_id(task, self.name_of(e))]
            final = state.get("final")
            if not final or final.get("class") == "not_needed":
                if e > last and self.scheduler is not None and not state.get("tries"):
                    self.scheduler.save(
                        {
                            "job_id": job_id(task, self.name_of(e)),
                            "task": task,
                            "split": self.split_of[task],
                            "seed": self.seed,
                            "tries": [],
                            "final": {
                                "class": "not_needed",
                                "try": None,
                                "success": False,
                                "reason": status,
                                "at": C.now_iso(),
                            },
                        }
                    )
                continue
            episodes.append(self.episode_summary(task, e, state))
            reflection = C.read_json(self.reflection_path(task, e))
            if isinstance(reflection, dict):
                entries = reflection["entries_after"]
        self._complete_history(task)
        kept = R.verified(entries)
        valid = [h for h in episodes if h["class"] == "valid"]
        solved = any(h["success"] for h in valid)
        final_doc = {  # final/<tag>.json of run_ecc_stage3.run_task
            "task": task,
            "entries": kept,
            "updated_at": C.now_iso(),
            "solved": solved,
            "episodes": len(valid),
            "gpt_calls_in_episodes": sum(h["gpt_calls"] for h in valid),
            "reflections": sum(
                1 for h in valid if h["reflection"] in ("applied", "unparsed")
            ),
            "seed": self.seed,
            "history": episodes,
        }
        folder = self.layout.task_dir(task)
        for name in ("final.json", "final.md"):
            if (folder / name).exists():
                (folder / name).chmod(0o644)
        C.atomic_write_json(folder / "final.json", final_doc)
        C.atomic_write_text(folder / "final.md", R.render_file(kept))
        for name in ("final.json", "final.md"):
            (folder / name).chmod(0o444)
        rule = self.rules[task]
        record = {
            "task": task,
            "split": self.split_of[task],
            "seed": self.seed,
            "rule": rule.as_json(),
            "rung": rung_of(rule),
            "status": status,
            "solved": solved,
            "episodes": len(valid),
            "first_success": next((h["episode"] for h in valid if h["success"]), None),
            "budget": self.budget,
            "launches": sum(h["launches"] for h in episodes),
            "infra_retries": sum(h["infra_retries"] for h in episodes),
            "render_retries": sum(h["render_retries"] for h in episodes),
            "gpt_calls_in_episodes": final_doc["gpt_calls_in_episodes"],
            "errors_trigger": sum(h["errors_trigger"] for h in valid),
            "reflections": {
                k: sum(1 for h in valid if h["reflection"] == k)
                for k in ("applied", "unparsed", "skipped")
            },
            "entries_work": len(entries),
            "entries_kept": len(kept),
            "final_md_sha256": C.sha256_file(folder / "final.md"),
            "usd": {
                "episodes": round(sum(h["usd"] for h in episodes), 4),
                "reflections": round(sum(h["reflection_usd"] for h in episodes), 4),
            },
            "history": episodes,
            "finalized_at": C.now_iso(),
        }
        record["usd"]["total"] = round(
            record["usd"]["episodes"] + record["usd"]["reflections"], 4
        )
        if self.parallel:
            launched = [
                self.session_of(e)
                for e in range(self.budget)
                if states[job_id(task, self.name_of(e))].get("tries")
            ]
            record["parallel_attempts"] = True
            record["sessions_run"] = max(launched) + 1 if launched else 0
        C.atomic_write_json(self.record_path(task), record)
        C.append_jsonl(
            self.layout.progress,
            {
                "event": "task_final",
                "task": task,
                "status": status,
                "episodes": len(valid),
                "entries_kept": len(kept),
            },
        )

    def _complete_history(self, task: str) -> None:
        """Add history rows for reflections a crashed runner wrote but did not log."""
        path = self.corrections_dir(task) / "history.jsonl"
        rows = (
            [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
            if path.exists()
            else []
        )
        logged = {row.get("episode") for row in rows}
        for e in range(self.budget):
            record = C.read_json(self.reflection_path(task, e))
            if not isinstance(record, dict):
                break
            if record["episode"] not in logged:
                C.append_jsonl(
                    path,
                    {
                        "episode": record["episode"],
                        "status": record["status"],
                        "recovered": True,
                        "prompt_sha256": record.get("prompt_sha256"),
                        "reply": record.get("reply"),
                        "edits": record.get("edits"),
                        "used": record.get("used"),
                        "entries_before": record["entries_before"],
                        "entries_after": record["entries_after"],
                        "reflection_json_sha256": C.sha256_file(
                            self.reflection_path(task, e)
                        ),
                    },
                )

    def episode_summary(
        self, task: str, e: int, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        final = state["final"]
        tries = state.get("tries") or []
        reflection = C.read_json(self.reflection_path(task, e)) or {}
        rstate = C.read_json(self.reflection_dir(task, e) / "state.json") or {}
        row: dict[str, Any] = {
            "episode": self.name_of(e),
            "class": final.get("class"),
            "success": bool(final.get("class") == "valid" and final.get("success")),
            "launches": len(tries),
            "infra_retries": sum(1 for t in tries if t.get("class") == "infra"),
            "render_retries": sum(1 for t in tries if t.get("class") == "render"),
            "usd": round(
                sum(float((t.get("usage") or {}).get("usd") or 0.0) for t in tries), 4
            ),
            "reflection": reflection.get("status"),
            "skipped": reflection.get("skipped"),
            "reflection_usd": round(float(rstate.get("usd") or 0.0), 4),
            "entries": len(reflection.get("entries_after") or []),
            "verified": len(R.verified(reflection.get("entries_after") or [])),
            "gpt_calls": 0,
            "errors_trigger": 0,
        }
        if final.get("class") == "valid":
            try_dir = self.final_try_dir(state)
            check = (
                C.read_json(
                    self.episode_dir(task, e) / f"try{final['try']}.evidence.json"
                )
                or {}
            )
            episode = H.episode_record(try_dir, self.cfg["pricing"], check)
            row.update(
                termination=episode["termination_reason"],
                turns=episode["turns"],
                finish_status=episode["finish_status"],
                gpt_calls=episode["gpt_calls"],
                gpt_calls_by_reason=episode["gpt_calls_by_reason"],
                errors_trigger=episode["errors_trigger"],
                digest_source=reflection.get("digest_source"),
                file_access=self.file_access(task, try_dir),
            )
        return row

    def file_access(self, task: str, try_dir: Path) -> dict[str, Any]:
        """File-tool paths outside the episode's own output folder.

        ``senior_memory_reads`` are reads inside the task's folder outside the
        output folder: the corrections snapshot, the reflections and the work
        memory are meant for GPT-6 only.
        """
        access = H.file_tool_audit(
            try_dir,
            own=try_dir,
            memory=self.corpus,
            code_dir=self.code_dir,
            shared=[
                self.layout.tasks_dir,
                Path(self.cfg["paths"].get("runs_dir") or self.layout.root.parent),
                *(
                    Path(v)
                    for v in (
                        self.cfg["stage3"]["s2_product"],
                        self.cfg["stage3"]["s2_rules"],
                    )
                    if v
                ),
            ],
        )
        base = os.path.realpath(self.layout.task_dir(task)) + os.sep
        access["senior_memory_reads"] = [
            p for p in access["cross_reads"] if p.startswith(base)
        ]
        return access

    def reflection_usd(self) -> float:
        usd = 0.0
        for _, task in self.tasks:
            for e in range(self.budget):
                state = C.read_json(self.reflection_dir(task, e) / "state.json")
                if isinstance(state, dict):
                    usd += float(state.get("usd") or 0.0)
        return usd


def corrections_info(path: Path) -> dict[str, Any]:
    """What RPent derives from a corrections file: the stripped text's length and hash."""
    import hashlib

    raw = path.read_bytes()
    text = raw.decode("utf-8").strip()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "chars": len(text),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text
        else None,
    }


class Stage3Scheduler(C.Scheduler):
    """The common scheduler with episode readiness, reflections and the evidence check."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.stage.scheduler = self  # type: ignore[attr-defined]
        self._last_total_check = 0.0
        self._total_usd = 0.0
        self._config_checked: set[str] = set()
        #: The slot limit last reported as too low for a session (parallel sessions).
        self._session_wait_noted: int | None = None
        #: Episodes and reflections together (stage3.cost_ceiling_usd, else the scheduler's);
        #: the base guard's episode-only cost check is replaced by :meth:`guard`.
        self.cost_ceiling = float(
            self.cfg["stage3"].get("cost_ceiling_usd")
            or self.sched.get("cost_ceiling_usd")
            or 0
        )
        self.sched = {**self.sched, "cost_ceiling_usd": 0}

    def launch_ceiling(self) -> int:
        """Every task's episode budget plus each task's (not each episode's) retry allowance."""
        return C.launch_ceiling(
            self.sched,
            self.stage.budget_launches(),
            len(self.stage.tasks),  # type: ignore[attr-defined]
            self.infra_retries,
            self.render_retries,
        )

    def next_job(self, states: Mapping[str, Mapping[str, Any]]) -> str | None:
        busy = {j for j, _ in self.running(states)}
        candidates = [
            self.jobs[j]
            for j, s in states.items()
            if not s["final"]
            and j not in busy
            and self.stage.ready(self.jobs[j], states)  # type: ignore[attr-defined]
        ]

        def key(job: C.Job) -> tuple[int, int]:
            tries = states[job.job_id]["tries"]
            last_cls = tries[-1].get("class") if tries else None
            return (
                2 if last_cls == "render" else 0 if last_cls == "infra" else 1,
                job.order,
            )

        if self.stage.parallel:  # type: ignore[attr-defined]
            return self._next_in_sessions(states, candidates, key, len(busy))
        candidates.sort(key=key)
        return candidates[0].job_id if candidates else None

    def _next_in_sessions(
        self,
        states: Mapping[str, Mapping[str, Any]],
        candidates: list[C.Job],
        key: Any,
        running: int,
    ) -> str | None:
        """Parallel sessions: the rest of a started session first: its attempts not launched
        yet (a session that started with all its slots free launches as one block), then
        relaunches in the base order (infra, killed, render); a new session only when all
        its attempts fit now, in the order (session, task)."""
        stage: Stage3 = self.stage  # type: ignore[assignment]
        started = [
            j
            for j in candidates
            if stage.session_started(j.task, j.meta["episode"], states)
        ]
        if started:

            def rest_first(job: C.Job) -> tuple[int, ...]:
                if not states[job.job_id]["tries"]:
                    return (0, job.order)
                return (1, *key(job))

            return min(started, key=rest_first).job_id
        if not candidates:
            return None

        def order(job: C.Job) -> tuple[int, int]:
            return (stage.session_of(job.meta["episode"]), job.order)

        if not any(ep.up for ep in self.servers.endpoints):
            # Nothing can launch; returning a job lets the loop see that no endpoint is up.
            return min(candidates, key=order).job_id
        need = stage.attempts
        limit = self.slot_limit()
        if limit < need:
            if self._session_wait_noted != limit:
                self._session_wait_noted = limit
                note = f"concurrency {limit} is below the {need} attempts of a session: no session starts"
                self.log({"event": "session_wait", "reason": note})
                print(f"{C.now_iso()} {note}", flush=True)
            return None
        self._session_wait_noted = None
        if min(limit - running, self.free_endpoint_slots(states)) < need:
            return None
        return min(candidates, key=order).job_id

    def free_endpoint_slots(self, states: Mapping[str, Mapping[str, Any]]) -> int:
        """Free slots on the endpoints that are up."""
        load: dict[str, int] = {}
        for _, last in self.running(states):
            load[last.get("endpoint")] = load.get(last.get("endpoint"), 0) + 1
        return sum(
            max(0, ep.slots - load.get(ep.name, 0))
            for ep in self.servers.endpoints
            if ep.up
        )

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
        """Score ended episodes, check new config rows of running ones, move reflections on."""
        super().poll()
        for jid, last in self.running(self.all_states()):
            if last["dir"] in self._config_checked:
                continue
            rows, _ = H.read_rows(Path(last["dir"]) / "escalation.jsonl")
            config = next(
                (r["config"] for r in rows if isinstance(r.get("config"), dict)), None
            )
            if config is None:
                continue
            self._config_checked.add(last["dir"])
            problems = H.config_row_problems(
                config, self.stage.expected(self.jobs[jid])
            )  # type: ignore[attr-defined]
            if problems:
                self._mismatch(jid, last, problems, "config row of a running episode")
                if last.get("pid") and C.pid_alive(int(last["pid"])):
                    C.kill_group(
                        int(last["pid"]), grace_s=float(self.sched["kill_grace_s"])
                    )
        self.stage.tick(self.all_states())  # type: ignore[attr-defined]

    def finish_try(
        self, job_id: str, state: dict[str, Any], exit_code: int | None
    ) -> None:
        last = state["tries"][-1]
        try_dir = Path(last["dir"])
        job = self.jobs[job_id]
        if (
            try_dir.exists()
            and not C.killed_marker(try_dir).exists()
            and not C.infra_marker(try_dir).exists()
        ):
            key = f"{job_id}#{last['n']}"
            try:
                _, self.render_checked[key] = C.check_render(
                    try_dir, self.render_checked.get(key, 0)
                )
            except ImportError:
                pass
            if C.read_outcome(try_dir).get("valid"):
                result = self.stage.evidence(job, last["n"])  # type: ignore[attr-defined]
                if not result["ok"]:
                    self._mismatch(job_id, last, result["problems"], "finished episode")
        super().finish_try(job_id, state, exit_code)

    def guard(self, states: Mapping[str, Mapping[str, Any]]) -> str | None:
        """The common guards, with the cost ceiling over episodes and reflections."""
        reason = super().guard(states)
        if reason:
            return reason
        ceiling = self.cost_ceiling
        if ceiling <= 0:
            return None
        if time.time() - self._last_total_check >= float(self.sched["cost_check_s"]):
            self._total_usd = (
                C.total_cost(states, self.cfg["pricing"]) + self.stage.reflection_usd()
            )  # type: ignore[attr-defined]
            self._last_total_check = time.time()
        if self._total_usd >= ceiling:
            C.write_stop(
                self.layout,
                "cost ceiling",
                usd=round(self._total_usd, 2),
                ceiling=ceiling,
            )
            return "cost ceiling"
        return None


# ---------------------------------------------------------------------------
# product: freeze and verify
# ---------------------------------------------------------------------------


def chain_problems(stage: Stage3, task: str) -> list[str]:
    """The task's memory chain: each reflection edits the previous reflection's entries, and
    each snapshot renders them (parallel sessions: the entries at the start of its session)."""
    problems = []
    rows = []
    history = stage.corrections_dir(task) / "history.jsonl"
    if history.exists():
        rows = [
            json.loads(line)
            for line in history.read_text().splitlines()
            if line.strip()
        ]
    recorded = {row["episode"]: row.get("reflection_json_sha256") for row in rows}
    previous: list[dict[str, Any]] = []
    session_memory: list[dict[str, Any]] = []
    for e in range(stage.budget):
        if e % stage.attempts == 0:
            session_memory = previous
        path = stage.reflection_path(task, e)
        snapshot = stage.snapshot(task, e)
        if not path.exists():
            break
        record = C.read_json(path) or {}
        if record.get("entries_before") != previous:
            problems.append(
                f"{stage.name_of(e)}: entries_before differ from the previous entries_after"
            )
        ran_with = session_memory if stage.parallel else previous
        if snapshot.exists() and snapshot.read_text(encoding="utf-8") != R.render_file(
            ran_with
        ):
            problems.append(
                f"{stage.name_of(e)}: corrections_used.md is not "
                + ("its session's memory" if stage.parallel else "the previous memory")
            )
        if stage.parallel and not set(record.get("used") or []) <= {
            str(x["id"]) for x in ran_with
        }:
            problems.append(
                f"{stage.name_of(e)}: entries counted as used that the episode was not shown"
            )
        if recorded.get(record.get("episode")) not in (None, C.sha256_file(path)):
            problems.append(
                f"{stage.name_of(e)}: reflection.json changed after it was written"
            )
        previous = record.get("entries_after") or []
    return problems


def freeze(
    stage: Stage3,
    manifest: Mapping[str, Any],
    *,
    allow_incomplete: bool = False,
    allow_cross_task_writes: bool = False,
) -> dict[str, Any]:
    """Copy every task's kept memory into ``product/final``, hash it and make it read-only."""
    product = stage.layout.product
    if (product / "FROZEN").exists():
        return verify(product)
    records = {task: C.read_json(stage.record_path(task)) for _, task in stage.tasks}
    missing = [t for t, r in records.items() if not r]
    if missing:
        raise SystemExit(
            f"cannot freeze: {len(missing)} task(s) are not final: {missing[:10]}"
        )
    incomplete = [t for t, r in records.items() if r["status"] == "infra_exhausted"]
    if incomplete and not allow_incomplete:
        raise SystemExit(
            f"cannot freeze: {len(incomplete)} task(s) stopped on infrastructure errors "
            f"(requeue them, or freeze --allow-incomplete): {incomplete}"
        )
    writers = {
        t: [
            h["file_access"]["cross_writes"]
            for h in r["history"]
            if (h.get("file_access") or {}).get("cross_writes")
        ]
        for t, r in records.items()
    }
    writers = {t: w for t, w in writers.items() if w}
    if writers and not allow_cross_task_writes:
        raise SystemExit(
            "refusing to freeze: episodes wrote into another task's folder or another run "
            f"(inspect, then freeze --allow-cross-task-writes): {json.dumps(writers)[:1500]}"
        )
    chains = {t: chain_problems(stage, t) for _, t in stage.tasks}
    chains = {t: p for t, p in chains.items() if p}
    if chains:
        raise SystemExit(
            f"refusing to freeze: broken correction-memory chains: {json.dumps(chains)[:1500]}"
        )
    s1_now = S1.verify(Path(manifest["inputs"]["s1"]["product"]))
    if (
        not s1_now.get("ok")
        or s1_now["corpus_tree_sha256"]
        != manifest["inputs"]["s1"]["corpus_tree_sha256"]
    ):
        raise SystemExit(
            "the stage-1 product changed during the run; refusing to freeze"
        )
    rules = manifest["inputs"]["rules"]
    for task, digest in rules["files"].items():
        path = Path(rules["folder"]) / rules["name"].format(task=task)
        if not path.is_file() or C.sha256_file(path) != digest:
            raise SystemExit(
                f"the stage-2 rule of {task} changed during the run; refusing to freeze"
            )
    if product.exists():
        S1._make_writable(product)
        shutil.rmtree(product)
    (product / "final").mkdir(parents=True)
    for _, task in stage.tasks:
        for suffix in (".md", ".json"):
            shutil.copyfile(
                stage.layout.task_dir(task) / f"final{suffix}",
                product / "final" / f"{task}{suffix}",
            )
    C.atomic_write_json(product / "tasks.json", records)
    summary = summarize(records)
    C.atomic_write_json(product / "summary.json", summary)
    files = C.tree_files(product / "final")
    tree = C.tree_hash(files)
    model, effort = reflection_model(stage.cfg)
    doc = {
        "schema": PRODUCT_SCHEMA,
        "stage": STAGE,
        "run_id": manifest["run_id"],
        "created_at": C.now_iso(),
        "code": manifest["code"],
        "config_sha256": manifest["config_sha256"],
        "experiment_sha256": manifest["experiment_sha256"],
        "seed": stage.seed,
        "budget": manifest["budget"],
        "inputs": manifest["inputs"],
        "reflection": {"model": model, "reasoning_effort": effort},
        "final": "final",
        "final_tree_sha256": tree,
        "files": files,
        "tasks_json_sha256": C.sha256_file(product / "tasks.json"),
        "summary_json_sha256": C.sha256_file(product / "summary.json"),
        "summary": summary,
        "incomplete_accepted": incomplete,
        "cross_task_writes_accepted": writers,
        "tasks": {
            task: {
                k: rec.get(k)
                for k in (
                    "split",
                    "status",
                    "rule",
                    "rung",
                    "episodes",
                    "first_success",
                    "entries_kept",
                    "final_md_sha256",
                )
            }
            for task, rec in records.items()
        },
    }
    C.atomic_write_json(product / "PRODUCT.json", doc)
    S1._make_read_only(product / "final")
    C.atomic_write_text(product / "FROZEN", tree + "\n")
    for name in ("PRODUCT.json", "tasks.json", "summary.json", "FROZEN"):
        (product / name).chmod(0o444)
    C.append_jsonl(
        stage.layout.progress, {"event": "frozen", "final_tree_sha256": tree}
    )
    return {
        "ok": True,
        "final_tree_sha256": tree,
        "files": len(files),
        "product": str(product),
    }


def summarize(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """run_ecc_stage3's ``summary.json`` fields plus the reflections, the errors trigger and cost."""
    recs = list(records.values())
    return {
        "tasks": len(recs),
        "solved": sum(bool(r["solved"]) for r in recs),
        "episodes": sum(r["episodes"] for r in recs),
        "gpt_calls_in_episodes": sum(r["gpt_calls_in_episodes"] for r in recs),
        "reflections": sum(
            r["reflections"]["applied"] + r["reflections"]["unparsed"] for r in recs
        ),
        "reflections_skipped": sum(r["reflections"]["skipped"] for r in recs),
        "verified_entries": sum(r["entries_kept"] for r in recs),
        "tasks_with_entries": sum(bool(r["entries_kept"]) for r in recs),
        "errors_trigger": sum(r["errors_trigger"] for r in recs),
        "infra_exhausted": sorted(
            t for t, r in records.items() if r["status"] == "infra_exhausted"
        ),
        "usd": {
            k: round(sum(r["usd"][k] for r in recs), 2)
            for k in ("episodes", "reflections", "total")
        },
        "by_split": {
            split: {
                "tasks": sum(1 for r in recs if r["split"] == split),
                "solved": sum(1 for r in recs if r["split"] == split and r["solved"]),
            }
            for split in C.SPLITS
        },
    }


def verify(product: Path) -> dict[str, Any]:
    """Recompute a frozen stage-3 product's hashes (what the final test does before reading it)."""
    doc = C.read_json(product / "PRODUCT.json")
    if not doc or doc.get("schema") != PRODUCT_SCHEMA:
        return {"ok": False, "error": "no PRODUCT.json with the stage-3 schema"}
    files = C.tree_files(product / doc["final"])
    problems = []
    if C.tree_hash(files) != doc["final_tree_sha256"]:
        changed = sorted(set(files.items()) ^ set(doc["files"].items()))
        problems.append(
            f"final/ hash mismatch ({len(changed)} differing entries, e.g. {changed[:3]})"
        )
    frozen = (
        (product / "FROZEN").read_text().strip()
        if (product / "FROZEN").exists()
        else None
    )
    if frozen != doc["final_tree_sha256"]:
        problems.append("FROZEN marker missing or different")
    for name, key in (
        ("tasks.json", "tasks_json_sha256"),
        ("summary.json", "summary_json_sha256"),
    ):
        if not (product / name).is_file() or C.sha256_file(product / name) != doc[key]:
            problems.append(f"{name} changed")
    expected = {f"{t}{s}" for t in doc["tasks"] for s in (".md", ".json")}
    if set(files) != expected:
        problems.append(
            f"final/ holds {len(files)} files, expected {len(expected)} (<Task>.md and .json)"
        )
    writable = [
        p
        for p in (product / doc["final"]).rglob("*")
        if p.stat().st_mode & stat.S_IWUSR
    ]
    if writable:
        problems.append(f"{len(writable)} final/ entries are writable")
    return {
        "ok": not problems,
        "problems": problems,
        "final_tree_sha256": doc["final_tree_sha256"],
        "files": len(files),
    }


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def job_states(stage: Stage3, *, mark_alive: bool = False) -> dict[str, dict[str, Any]]:
    states = {}
    for job in stage.jobs():
        state = C.read_json(stage.layout.job_state(job.job_id)) or {
            "job_id": job.job_id,
            "task": job.task,
            "split": job.split,
            "tries": [],
            "final": None,
        }
        if (
            mark_alive
            and state["tries"]
            and state["tries"][-1].get("status") in ("launching", "running")
        ):
            pid = state["tries"][-1].get("pid")
            state["tries"][-1]["alive"] = bool(pid and C.pid_alive(pid))
        states[job.job_id] = state
    return states


def task_overview(
    stage: Stage3, states: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per task: state, current episode, valid episodes, reflections, entries, cost."""
    out = {}
    for split, task in stage.tasks:
        record = C.read_json(stage.record_path(task))
        row: dict[str, Any] = {
            "split": split,
            "status": None,
            "episode": None,
            "valid": 0,
            "solved": False,
            "infra": 0,
            "render": 0,
            "applied": 0,
            "unparsed": 0,
            "skipped": 0,
            "refl_running": 0,
            "refl_retry": 0,
            "refl_failed": 0,
            "entries": 0,
            "verified": 0,
            "errors_trigger": 0,
            "usd_episodes": 0.0,
            "usd_reflections": 0.0,
            "rule": stage.rules[task].as_json(),
            "running_episodes": 0,
        }
        for e in range(stage.budget):
            state = states[job_id(task, stage.name_of(e))]
            tries = state.get("tries") or []
            final = state.get("final") or {}
            row["infra"] += sum(1 for t in tries if t.get("class") == "infra")
            row["render"] += sum(1 for t in tries if t.get("class") == "render")
            row["usd_episodes"] += sum(
                float((t.get("usage") or {}).get("usd") or 0.0) for t in tries
            )
            if tries and tries[-1].get("status") in ("launching", "running"):
                row["episode"], row["status"] = stage.name_of(e), "running"
                row["running_episodes"] += 1
                if stage.parallel:  # the session and how many of its episodes run
                    row["episode"] = (
                        f"s{stage.session_of(e) + 1} ({row['running_episodes']})"
                    )
            if final.get("class") == "valid":
                row["valid"] += 1
                row["solved"] = row["solved"] or bool(final.get("success"))
                evidence = (
                    C.read_json(
                        stage.episode_dir(task, e) / f"try{final['try']}.evidence.json"
                    )
                    or {}
                )
                row["errors_trigger"] += int(
                    (evidence.get("counts") or {}).get("errors_trigger") or 0
                )
            reflection = C.read_json(stage.reflection_path(task, e))
            rstate = C.read_json(stage.reflection_dir(task, e) / "state.json") or {}
            row["usd_reflections"] += float(rstate.get("usd") or 0.0)
            if isinstance(reflection, dict):
                row[
                    {"applied": "applied", "unparsed": "unparsed"}.get(
                        reflection["status"], "skipped"
                    )
                ] += 1
                row["entries"] = len(reflection["entries_after"])
                row["verified"] = len(R.verified(reflection["entries_after"]))
            elif rstate.get("status") == "running":
                row["refl_running"] += 1
                row["episode"], row["status"] = stage.name_of(e), "reflecting"
            elif rstate.get("status") == "retry":
                row["refl_retry"] += 1
                row["episode"], row["status"] = stage.name_of(e), "reflecting"
            elif rstate.get("status") == "failed":
                row["refl_failed"] += 1
                row["episode"], row["status"] = stage.name_of(e), "reflection_failed"
        if record:
            row["status"] = record["status"]
            row["verified"] = record["entries_kept"]
        elif row["status"] is None:
            row["status"] = "queued"
        out[task] = row
    return out


def format_status(
    stage: Stage3, run_id: str, states: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    rows = task_overview(stage, states)
    final = sum(
        1
        for r in rows.values()
        if r["status"] in ("solved", "unsolved", "infra_exhausted")
    )
    lines = [f"RC-ECC S3 {run_id}  {C.progress_bar(final, len(rows))} tasks final"]
    lines.append(
        f"  {'split':<17} {'final':>7} {'running':>8} {'reflecting':>11} {'queued':>7} {'solved':>7} "
        f"{'episodes':>9} {'infra-retry':>12} {'exhausted':>10}"
    )
    for split in C.SPLITS:
        group = [r for r in rows.values() if r["split"] == split]
        if not group:
            continue
        n_final = sum(
            1 for r in group if r["status"] in ("solved", "unsolved", "infra_exhausted")
        )
        lines.append(
            f"  {split:<17} {n_final:>3}/{len(group):<3} "
            f"{sum(r['status'] == 'running' for r in group):>8} "
            f"{sum(r['status'] in ('reflecting', 'reflection_failed') for r in group):>11} "
            f"{sum(r['status'] == 'queued' for r in group):>7} {sum(r['solved'] for r in group):>7} "
            f"{sum(r['valid'] for r in group):>9} {sum(r['infra'] + r['render'] for r in group):>12} "
            f"{sum(r['status'] == 'infra_exhausted' for r in group):>10}"
        )
    tot = {
        k: sum(r[k] for r in rows.values())
        for k in (
            "applied",
            "unparsed",
            "skipped",
            "refl_running",
            "refl_retry",
            "refl_failed",
            "entries",
            "verified",
            "errors_trigger",
            "usd_episodes",
            "usd_reflections",
            "valid",
        )
    }
    lines.append(
        f"  reflections: {tot['applied'] + tot['unparsed'] + tot['skipped']} done "
        f"({tot['applied']} applied, {tot['unparsed']} unparsed, {tot['skipped']} skipped), "
        f"{tot['refl_running']} running, {tot['refl_retry']} waiting to retry, {tot['refl_failed']} failed"
    )
    lines.append(
        f"  correction entries: {tot['entries']} in the current memories, {tot['verified']} verified; "
        f"errors trigger fired {tot['errors_trigger']} times in {tot['valid']} valid episodes"
    )
    if stage.parallel:
        running = [r for r in rows.values() if r["running_episodes"]]
        lines.append(
            f"  parallel sessions ({stage.attempts} attempts at once): {len(running)} running "
            f"({sum(r['running_episodes'] for r in running)} episodes), "
            f"{sum(r['status'] == 'queued' for r in rows.values())} tasks queued (a session starts "
            f"when {stage.attempts} slots are free)"
        )
        try:
            limit = int(stage.layout.concurrency.read_text().strip())
        except (OSError, ValueError):
            limit = None
        if limit is not None and limit < stage.attempts:
            lines.append(
                f"  NOTE: concurrency {limit} is below {stage.attempts}: no new session starts"
            )
    lines.append(
        f"  cost so far (list price): episodes ${tot['usd_episodes']:.2f} + reflections "
        f"${tot['usd_reflections']:.2f} = ${tot['usd_episodes'] + tot['usd_reflections']:.2f}"
    )
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def select_tasks(
    cfg: Mapping[str, Any], args: argparse.Namespace
) -> list[tuple[str, str]]:
    all_tasks = C.load_target50(cfg["paths"]["code_dir"])
    wanted = [
        t
        for t in (
            args.tasks.split(",")
            if getattr(args, "tasks", None)
            else cfg["stage3"]["tasks"]
        )
        if t
    ]
    unknown = sorted(set(wanted) - {t for _, t in all_tasks})
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}")
    tasks = [row for row in all_tasks if not wanted or row[1] in wanted]
    if getattr(args, "limit", 0):
        tasks = tasks[: args.limit]
    return tasks


def run_layout(cfg: Mapping[str, Any], args: argparse.Namespace) -> C.RunLayout:
    if getattr(args, "run_root", None):
        return C.RunLayout(Path(args.run_root).resolve())
    if not getattr(args, "run_id", None):
        raise SystemExit("pass --run-id (or --run-root)")
    return C.RunLayout(
        Path(cfg["paths"]["runs_dir"]).resolve() / C.safe_name(args.run_id)
    )


def make_manifest(
    cfg: Mapping[str, Any],
    run_id: str,
    tasks: list[tuple[str, str]],
    inputs: Mapping[str, Any],
    rules: Mapping[str, H.Rule],
) -> dict[str, Any]:
    code = C.code_identity(cfg["paths"]["code_dir"])
    model, effort = reflection_model(cfg)
    return {
        "schema": "rc-ecc-run/1",
        "stage": STAGE,
        "run_id": run_id,
        "created_at": C.now_iso(),
        "config_path": cfg["_path"],
        "config_sha256": C.config_hash(cfg),
        "experiment_sha256": experiment_hash(cfg),
        "code": code,
        "seed": cfg["stage3"]["seed"],
        "budget": {
            "sessions": cfg["stage3"]["sessions"],
            "attempts_per_session": cfg["stage3"]["attempts_per_session"],
            "episodes_per_task": budget(cfg),
            "stop": "first valid success",
            "timeout_s_by_split": {s: timeout_for(cfg, s) for s in C.SPLITS},
            "max_turns": cfg["episode"]["max_turns"],
            "infra_retries_per_episode": cfg["stage3"]["infra_retries"],
            "render_retries_per_episode": cfg["stage3"]["render_retries"],
            **(
                {
                    "parallel_attempts": True,
                    "stop": "the first session with a valid success, after all its "
                    "episodes are reflected on",
                }
                if cfg["stage3"]["parallel_attempts"]
                else {}
            ),
        },
        "reflection": {
            "model": model,
            "reasoning_effort": effort,
            **{
                k: cfg["stage3"]["reflection"][k]
                for k in REFLECTION_EXPERIMENT_KEYS
                if k not in ("model", "reasoning_effort")
            },
        },
        "settings": experiment_settings(cfg),
        "inputs": dict(inputs),
        "tasks": [
            {
                "split": s,
                "task": t,
                "rule": rules[t].as_json(),
                "rung": rung_of(rules[t]),
            }
            for s, t in tasks
        ],
        "worker": cfg["worker"],
        "senior": cfg["senior"],
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
            "experiment settings (worker/senior/episode/stage3) differ from the run's manifest"
        )
    if manifest["tasks"] != fresh["tasks"]:
        problems.append("task list or rules differ from the run's manifest")
    for key in ("corpus_tree_sha256",):
        if manifest["inputs"]["s1"][key] != fresh["inputs"]["s1"][key]:
            problems.append("the stage-1 corpus differs from the run's manifest")
    if manifest["inputs"]["rules"]["files"] != fresh["inputs"]["rules"]["files"]:
        problems.append("the stage-2 rule files differ from the run's manifest")
    if (
        manifest["code"]["tree_sha256"] != fresh["code"]["tree_sha256"]
        and not allow_code_change
    ):
        problems.append(
            "code differs from the run's manifest (pass --allow-code-change to accept and record it)"
        )
    return problems


def print_plan(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    """The dry run: what --go would do, without creating or starting anything."""
    C.require(cfg, "paths.code_dir", "paths.runs_dir")
    tasks = select_tasks(cfg, args)
    s3 = cfg["stage3"]
    print(
        f"RC-ECC stage 3 plan (dry run; add --go to launch)\n  config   {cfg['_path']}"
    )
    print(
        f"  config sha256 {C.config_hash(cfg)[:16]}  experiment sha256 {experiment_hash(cfg)[:16]}"
    )
    code = C.code_identity(cfg["paths"]["code_dir"])
    print(
        f"  code     {cfg['paths']['code_dir']}  commit {code['commit']} ({code['commit_source']})  "
        f"tree {code['tree_sha256'][:16]} ({code['files']} files)"
    )
    corpus = None
    try:
        s1 = s1_input(cfg)
        corpus = s1["corpus"]
        print(
            f"  stage-1 product {s1['product']}: verified, corpus tree {s1['corpus_tree_sha256'][:16]}"
        )
    except SystemExit as exc:
        print(f"  stage-1 product: NOT READY ({exc})")
    rules: dict[str, H.Rule] = {}
    try:
        rules, info = rules_input(cfg, tasks)
        counts: dict[str, int] = {}
        for rule in rules.values():
            rung = rung_of(rule)
            name = rule.config_name(rung) if rung else f"off-ladder {rule.as_json()}"
            counts[name] = counts.get(name, 0) + 1
        source = (
            "verified stage-2 product" if info["kind"] == "s2_product" else info["kind"]
        )
        print(
            f"  stage-2 rules {info['folder']} ({source}): {len(rules)} tasks; {dict(sorted(counts.items()))}"
        )
    except SystemExit as exc:
        print(f"  stage-2 rules: NOT READY ({exc})")
    by_split: dict[str, int] = {}
    for split, _ in tasks:
        by_split[split] = by_split.get(split, 0) + 1
    print(
        f"  tasks    {len(tasks)} ({', '.join(f'{k} {v}' for k, v in by_split.items())}); seed {s3['seed']}, split {S1.SPLIT}"
    )
    print(
        f"  budget   {s3['sessions']} sessions x {s3['attempts_per_session']} attempts = {budget(cfg)} "
        f"single-shot episodes per task; stop at the first valid success; planner time "
        f"{ {s: timeout_for(cfg, s) for s in C.SPLITS} } s, {cfg['episode']['max_turns']} turns"
    )
    if s3["parallel_attempts"]:
        print(
            f"  sessions parallel: the {s3['attempts_per_session']} attempts of a session run at once with the "
            "memory at the session's start; when all are final, their reflections run in order (a1, a2, ...), "
            "each on the memory the previous one left, and the next session starts with the result; the task "
            "stops after the first session with a valid success (all its episodes reflected on); an applied id "
            "counts as used only if the episode was shown it. A session starts only when all its attempts fit "
            f"(--concurrency counts episodes: use a multiple of {s3['attempts_per_session']})"
        )
    print(
        f"  retries  infra {s3['infra_retries']}, render {s3['render_retries']} per episode (never counted, never reflected on)"
    )
    model, effort = reflection_model(cfg)
    refl = s3["reflection"]
    print(
        f"  reflection {model} ({effort}), after every valid episode; up to {refl['max_calls']} calls per episode, "
        f"{refl['max_parallel']} at once; entries <= {s3['max_entries']}, fields <= {s3['field_chars']} chars, "
        f"dropped after {s3['fails_to_drop']} failures"
    )
    print(f"  reflection command: {' '.join(refl['cmd'])} REQUEST REPLY")
    layout = (
        run_layout(cfg, args)
        if (getattr(args, "run_id", None) or getattr(args, "run_root", None))
        else None
    )
    if layout:
        print(
            f"  run root {layout.root} ({'exists: --go resumes it' if layout.manifest.exists() else 'new: --go creates it'})"
        )
    sched = cfg["scheduler"]
    max_slots = min(args.max_slots or sched["max_slots"], sched["max_slots"])
    ceiling = C.launch_ceiling(
        sched,
        len(tasks) * budget(cfg),
        len(tasks),
        s3["infra_retries"],
        s3["render_retries"],
    )
    cost = float(s3.get("cost_ceiling_usd") or sched.get("cost_ceiling_usd") or 0)
    print(
        f"  slots    max {max_slots}; stagger {sched['launch_stagger_s']} s; launch ceiling {ceiling}; "
        f"infra breaker {sched['infra_breaker']}; cost ceiling "
        f"{'$%.0f' % cost if cost else 'off'} (episodes + reflections)"
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
    mine = S1.runner_origin(cfg["paths"]["code_dir"])
    print(
        f"  runner imports:  {'OK' if mine['ok'] else 'NOT FROM THE CODE DIR'} {mine}"
    )
    if tasks and rules:
        split, task = tasks[0]
        stage = Stage3(
            cfg,
            layout or C.RunLayout(Path(cfg["paths"]["runs_dir"]) / "<run-id>"),
            tasks[:1],
            rules,
            Path(corpus or "<stage-1 corpus>"),
        )
        job = stage.jobs()[0]
        eps = C.endpoints_from_config(cfg)
        argv = [
            *s3["launcher"],
            job.split,
            job.task,
            str(job.seed),
            str(stage.try_dir(job, 1)),
            str(stage.corpus),
        ]
        env = H.hybrid_env(
            cfg,
            split=split,
            try_n=1,
            endpoint=eps[0] if eps else C.Endpoint("gpuN", -1, 0, 0),
            rule=rules[task],
            timeout_s=timeout_for(cfg, split),
            corrections=stage.snapshot(task, 0),
            corrections_sha256="<sha256 of the snapshot>",
        )
        print(f"  first launch: {' '.join(argv)}")
        for k in sorted(
            k
            for k in env
            if k.startswith(("ECC_RC_", "RC_", "VLA_")) or k in cfg["episode"]["env"]
        ):
            print(f"    {k}={env[k]}")
    return 0


def build(
    cfg: dict[str, Any], args: argparse.Namespace, layout: C.RunLayout
) -> tuple[Stage3, dict[str, Any], dict[str, Any]]:
    tasks = select_tasks(cfg, args)
    s1 = s1_input(cfg)
    rules, rules_info = rules_input(cfg, tasks)
    searched_on = rules_info.get("s1_corpus_tree_sha256")
    if searched_on and searched_on != s1["corpus_tree_sha256"]:
        raise SystemExit(
            "the stage-2 rules were searched on another stage-1 corpus "
            f"({searched_on[:16]}) than stage3.s1_product's ({s1['corpus_tree_sha256'][:16]})"
        )
    inputs = {"s1": s1, "rules": rules_info}
    fresh = make_manifest(cfg, layout.root.name, tasks, inputs, rules)
    return Stage3(cfg, layout, tasks, rules, Path(s1["corpus"])), fresh, inputs


def drain_limit_s(stage: Stage3) -> float:
    """How long a runner that stopped launching waits for the last reflections: one call's
    timeout plus 300 s, times the attempts of a session with parallel sessions (a session's
    reflections run one after another)."""
    one = float(stage.refl["timeout_s"]) + 300
    return one * (stage.attempts if stage.parallel else 1)


def drain_reflections(
    stage: Stage3, scheduler: Stage3Scheduler, limit_s: float
) -> None:
    """After the scheduler stopped: reflect on the last finished episodes and finalize their
    tasks (bounded by ``limit_s``). Tasks waiting for an episode launch are left as they are."""
    deadline = time.time() + limit_s
    while True:
        stage.tick(scheduler.all_states())
        states = scheduler.all_states()
        waiting = []
        for _, task in stage.tasks:
            if stage.record_path(task).exists():
                continue
            action = stage._scan(task, states)
            if action is None:
                continue  # the next episode would need a launch
            if action[0] == "reflect":
                rstate = (
                    C.read_json(stage.reflection_dir(task, action[2]) / "state.json")
                    or {}
                )
                if rstate.get("status") == "failed":
                    continue
            waiting.append(task)
        if not waiting or time.time() > deadline:
            return
        time.sleep(min(2.0, float(stage.cfg["scheduler"]["poll_s"])))


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
        "paths.llvmpipe_index_script",
        "credentials.file",
        "credentials.worker_key_var",
        "credentials.senior_base_url_var",
        "credentials.senior_key_var",
        "worker.base_url",
        "endpoints",
        "gpus.allowed",
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
    from scripts.ecc_robocasa.stage2 import config_row_supported

    if not config_row_supported():
        raise SystemExit(
            "this code's escalation wrapper writes no config row: the evidence check (the only "
            "record of N and of the corrections hash) needs it"
        )
    layout = run_layout(cfg, args)
    stage, fresh, _ = build(cfg, args, layout)
    if stage.parallel:
        sched_max = int(cfg["scheduler"]["max_slots"])
        for what, value in (
            ("--concurrency", args.concurrency),
            ("max slots", min(args.max_slots or sched_max, sched_max)),
            ("endpoint slots", sum(ep.slots for ep in C.endpoints_from_config(cfg))),
        ):
            if value < stage.attempts and (value or what != "--concurrency"):
                raise SystemExit(
                    f"{what} {value} is below the {stage.attempts} attempts of a session: with "
                    "parallel sessions no session could start (the concurrency counts episodes)"
                )
    lock = C.acquire_lock(layout)  # noqa: F841 - held for the process lifetime
    if layout.manifest.exists():
        manifest = C.read_json(layout.manifest)
        problems = check_resume(manifest, fresh, args.allow_code_change)
        if problems:
            raise SystemExit("refusing to resume: " + "; ".join(problems))
        if manifest["code"]["tree_sha256"] != fresh["code"]["tree_sha256"]:
            manifest["history"].append(
                {"at": C.now_iso(), "event": "code_changed", "code": fresh["code"]}
            )
        manifest["history"].append(
            {"at": C.now_iso(), "event": "resumed", "pid": os.getpid()}
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
        C.atomic_write_json(
            layout.root / "config.snapshot.json",
            {k: v for k, v in cfg.items() if k != "_path"},
        )
        C.atomic_write_json(layout.manifest, fresh)
        manifest = fresh
    servers = C.ServerPool(
        cfg, layout, S1.base_env(cfg), lambda row: C.append_jsonl(layout.progress, row)
    )
    scheduler = Stage3Scheduler(
        cfg,
        layout,
        stage,
        servers,
        infra_retries=cfg["stage3"]["infra_retries"],
        render_retries=cfg["stage3"]["render_retries"],
        max_slots=args.max_slots,
    )
    C.append_jsonl(
        layout.progress,
        {
            "event": "start",
            "pid": os.getpid(),
            "tasks": len(stage.tasks),
            "max_slots": scheduler.max_slots,
        },
    )
    if args.concurrency:
        C.atomic_write_text(layout.concurrency, f"{args.concurrency}\n")
    layout.tasks_dir.mkdir(parents=True, exist_ok=True)
    stage.retry_failed_reflections()  # an operator restart after a failed reflection retries it
    # Finish what an earlier runner left (reflections, task records).
    stage.tick(scheduler.all_states())
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
    # The last episodes' reflections run on after the scheduler stopped launching.
    drain_reflections(stage, scheduler, drain_limit_s(stage))
    if not scheduler.running(scheduler.all_states()):
        stage.unlock()
    print(f"runner finished: {result}", flush=True)
    done = all(stage.record_path(t).exists() for _, t in stage.tasks)
    if done and not args.no_freeze:
        print(json.dumps(freeze(stage, manifest), indent=2), flush=True)
    return 0 if result in ("done", "stopped") else 1


def load_stage(
    cfg: dict[str, Any], args: argparse.Namespace
) -> tuple[C.RunLayout, dict[str, Any], Stage3]:
    layout = run_layout(cfg, args)
    manifest = C.read_json(layout.manifest)
    if manifest is None:
        raise SystemExit(f"no run at {layout.root}")
    # The run's own mode, whatever the config says now (a resume in the other mode is refused).
    parallel = bool((manifest.get("settings") or {}).get("parallel_attempts"))
    if parallel != cfg["stage3"]["parallel_attempts"]:
        cfg = {**cfg, "stage3": {**cfg["stage3"], "parallel_attempts": parallel}}
    tasks = [(t["split"], t["task"]) for t in manifest["tasks"]]
    rules = {t["task"]: H.Rule.from_json(t["rule"]) for t in manifest["tasks"]}
    return (
        layout,
        manifest,
        Stage3(cfg, layout, tasks, rules, Path(manifest["inputs"]["s1"]["corpus"])),
    )


def cmd_status(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Progress bar, per-split table, reflections, entries, cost, runner and guards."""
    layout, manifest, stage = load_stage(cfg, args)
    states = job_states(stage, mark_alive=True)
    lines = format_status(stage, manifest["run_id"], states)
    holder = C.lock_holder(layout)
    stop = C.read_json(layout.stop) if layout.stop.exists() else None
    lines.append(
        f"  runner: {'pid %s' % holder if holder else 'not running'}; "
        f"stop file: {stop['reason'] if stop else 'absent'}"
    )
    load: dict[str, int] = {}
    for s in states.values():
        if s["tries"] and s["tries"][-1].get("status") == "running":
            load[s["tries"][-1].get("endpoint")] = (
                load.get(s["tries"][-1].get("endpoint"), 0) + 1
            )
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
        len(stage.jobs()),
        len(stage.tasks),
        cfg["stage3"]["infra_retries"],
        cfg["stage3"]["render_retries"],
    )
    lines.append(f"  launches: {C.counted_launches(states)} of ceiling {ceiling}")
    stale = [
        j
        for j, s in states.items()
        if s["tries"] and s["tries"][-1].get("alive") is False
    ]
    if stale:
        lines.append(
            f"  NOTE: {len(stale)} episode(s) marked running have no live process; the next --go scores them"
        )
    flagged = []
    for _, task in stage.tasks:
        record = C.read_json(stage.record_path(task)) or {}
        kinds = sorted(
            {
                kind
                for h in record.get("history") or []
                for kind in (
                    "senior_memory_reads",
                    "cross_writes",
                    "cross_reads",
                    "memory_writes",
                    "code_dir_writes",
                )
                if (h.get("file_access") or {}).get(kind)
            }
        )
        if kinds:
            flagged.append(f"{task} ({', '.join(kinds)})")
    if flagged:
        lines.append(
            f"  file-tool access outside the episode folder: {'; '.join(flagged)}"
        )
    frozen = layout.product / "FROZEN"
    if frozen.exists():
        lines.append(
            f"  product: frozen, final/ sha256 {frozen.read_text().strip()[:16]}"
        )
    print("\n".join(lines))
    if args.tasks_table:
        for task, r in task_overview(stage, states).items():
            rule = r["rule"]
            print(
                f"  {r['split']:<17} {task:<26} {r['status']:<17} ep {r['episode'] or '-':<6} "
                f"valid {r['valid']:>2}  rule {'gpt' if rule['gpt_plan'] else 'qwen'}/K{rule['checkin_K']}/N{rule['errors_N']}  "
                f"refl {r['applied']}a/{r['skipped']}s/{r['unparsed']}u  entries {r['entries']} ({r['verified']} verified)  "
                f"errors-trigger {r['errors_trigger']}  ${r['usd_episodes'] + r['usd_reflections']:.2f}"
            )
    return 0


def cmd_stop(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    layout, _, stage = load_stage(cfg, args)
    C.write_stop(layout, "operator", now=bool(args.now))
    print(
        f"wrote {layout.stop}: no new launches; running episodes and reflections finish"
    )
    if args.now:
        killed = C.kill_running(
            layout,
            [j.job_id for j in stage.jobs()],
            grace_s=float(cfg["scheduler"]["kill_grace_s"]),
        )
        print(
            f"stopped {killed} running episode(s); they are relaunched (not counted) on the next --go"
        )
    if C.lock_holder(layout) is None:
        states = job_states(stage)
        if not any(
            s["tries"]
            and s["tries"][-1].get("status") == "running"
            and C.pid_alive(s["tries"][-1].get("pid"))
            for s in states.values()
        ):
            C.ServerPool(
                cfg, layout, {}, lambda row: C.append_jsonl(layout.progress, row)
            ).stop_all()
            print("no runner and no running episode: stopped this run's RLDX servers")
    return 0


def cmd_freeze(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    _, manifest, stage = load_stage(cfg, args)
    print(
        json.dumps(
            freeze(
                stage,
                manifest,
                allow_incomplete=args.allow_incomplete,
                allow_cross_task_writes=args.allow_cross_task_writes,
            ),
            indent=2,
        )
    )
    return 0


def cmd_requeue(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Give a task that stopped on infrastructure errors a fresh try at its stopped episode."""
    layout, _, stage = load_stage(cfg, args)
    if C.lock_holder(layout) is not None:
        raise SystemExit("stop the runner first")
    record = C.read_json(stage.record_path(args.task))
    if not record or record["status"] != "infra_exhausted":
        raise SystemExit(f"{args.task} did not stop on infrastructure errors")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for e in range(stage.budget):
        jid = job_id(args.task, stage.name_of(e))
        state = C.read_json(layout.job_state(jid)) or {}
        final = state.get("final") or {}
        if final.get("class") in ("infra_exhausted", "render_exhausted"):
            folder = stage.episode_dir(args.task, e)
            keep = folder / "corrections_used.md"
            moved = folder.with_name(f"{folder.name}.requeued-{stamp}")
            folder.rename(moved)
            folder.mkdir()
            shutil.copy2(moved / keep.name, keep)
            layout.job_state(jid).rename(
                layout.job_state(jid).with_name(f"{jid}.json.requeued-{stamp}")
            )
        elif final.get("class") == "not_needed":
            layout.job_state(jid).unlink()
    for name in ("record.json", "final.json", "final.md"):
        path = layout.task_dir(args.task) / name
        if path.exists():
            path.chmod(0o644)
            path.rename(path.with_name(f"{name}.requeued-{stamp}"))
    C.append_jsonl(layout.progress, {"event": "requeued", "task": args.task})
    print(f"requeued {args.task}: its stopped episode runs again on the next --go")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify(Path(args.product).resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def cmd_reflect_only(cfg: dict[str, Any] | None, args: argparse.Namespace) -> int:
    """Build the digest and prompt of one finished episode folder and print them (no call)."""
    out = Path(args.episode_dir)
    entries = (
        (C.read_json(args.entries) or {}).get("entries", []) if args.entries else []
    )
    rule = json.loads(args.rule) if args.rule else {"note": "dry run"}
    s = settings(cfg) if cfg else R.Settings()
    d = R.digest(out, s)
    outcome = C.read_outcome(out)
    prompt = R.build_prompt(
        d,
        language=R.task_language(out),
        rule=rule,
        success=bool(outcome.get("success")),
        termination=str(outcome.get("termination_reason") or ""),
        entries=entries,
        settings=s,
    )
    print(
        json.dumps(
            {
                k: d[k]
                for k in ("interventions", "by_reason", "shown", "turns", "source")
            }
            | {
                "skip": R.decide(d),
                "prompt_chars": len(prompt),
                "prompt_sha256": R.prompt_sha256(prompt),
            },
            indent=1,
        )
    )
    if not args.quiet:
        print(prompt)
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
            help="comma list of Target50 tasks (default: stage3.tasks or all 50)",
        )
        p.add_argument(
            "--limit", type=int, default=0, help="first N tasks only (smoke)"
        )
        p.add_argument(
            "--max-slots",
            type=int,
            default=0,
            help="lower the concurrent-episode ceiling",
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
            help="write <run>/concurrency before launching",
        )
        p.add_argument(
            "--go", action="store_true", help="really launch (otherwise print the plan)"
        )
        p.add_argument(
            "--no-freeze",
            action="store_true",
            help="do not freeze the product when all tasks are final",
        )
        p.add_argument(
            "--allow-code-change",
            action="store_true",
            help="resume although the code tree changed",
        )
    p = sub.add_parser("status", help="progress bar and per-split overview")
    common_args(p)
    p.add_argument(
        "--tasks-table", action="store_true", help="also print one line per task"
    )
    p = sub.add_parser(
        "stop", help="write the stop file (drain); --now also stops running episodes"
    )
    common_args(p)
    p.add_argument("--now", action="store_true")
    p = sub.add_parser("freeze", help="freeze the stage-3 product (all tasks final)")
    common_args(p)
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="freeze although tasks stopped on infrastructure errors",
    )
    p.add_argument("--allow-cross-task-writes", action="store_true")
    p = sub.add_parser(
        "requeue",
        help="rerun the stopped episode of a task that ran out of infrastructure retries",
    )
    common_args(p)
    p.add_argument("--task", required=True)
    p = sub.add_parser("verify", help="recompute a frozen stage-3 product's hashes")
    p.add_argument("product")
    p = sub.add_parser(
        "reflect-only",
        help="print one finished episode's digest and prompt (no model call)",
    )
    p.add_argument("episode_dir")
    p.add_argument("--config")
    p.add_argument(
        "--entries", help="a work.json/final.json whose entries are the current memory"
    )
    p.add_argument("--rule", help="the rule as JSON (default: a dry-run note)")
    p.add_argument("--quiet", action="store_true", help="print the summary only")
    args = ap.parse_args(argv)
    if args.cmd in (None, "plan"):
        args.go, args.cmd = False, "run"
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
    if args.cmd == "reflect-only":
        cfg = (
            load_config(C.config_path(args.config))
            if (args.config or os.environ.get("ECC_RC_CONFIG"))
            else None
        )
        return cmd_reflect_only(cfg, args)
    cfg = load_config(C.config_path(args.config))
    return {
        "run": cmd_run,
        "status": cmd_status,
        "stop": cmd_stop,
        "freeze": cmd_freeze,
        "requeue": cmd_requeue,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
