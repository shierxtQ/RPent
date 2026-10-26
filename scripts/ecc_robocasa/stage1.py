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

"""RC-ECC stage 1 on RoboCasa Target50: edge task memory (Qwen alone, seed 0).

Every task explores once with RPent ``--explore`` (default 5 sessions x 3
attempts, 2400 s per session; RPent ends the cell at the first solved session),
starting from the cloud (GPT-6) memory of that task plus the global layer, in
its own staging corpus. When a task's cell is final, its edge product (the
inbox drafts, Qwen's audit and the exported recipe) is captured. A solved task
whose audit and recipe both exist replaces the cloud task files; any other task
keeps them. After the last task the corpus is assembled from a fresh copy of the
cloud corpus in Target50 order and frozen with sha256 hashes for stages 2-3.

Dry run by default (prints the plan); ``run --go`` launches. See README.md.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import stat
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ecc_robocasa import common as C  # noqa: E402

PRODUCT_SCHEMA = "rc-ecc-s1-product/1"
SPLIT = "target"  # RoboCasa environment split of every Target50 cell
FILE_TOOLS = ("read_text_file", "write_text_file", "list_dir")
#: Environment names never inherited from the operator's shell by an episode.
SCRUB_PREFIXES = ("ECC_RC_", "RPENT_", "ABL_", "RLDX_", "OPENAI_", "MUJOCO_", "VLA_")
SCRUB_NAMES = (
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "PYOPENGL_PLATFORM",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "LP_NUM_THREADS",
    "HF_HOME",
    "HF_HUB_CACHE",
    "ROBOCASA_ASSETS_PATH",
    "MEMORY_CORPUS",
    "RC_GPU",
)


def cell_tag(task: str, seed: int) -> str:
    return f"{task}_{SPLIT}_s{seed}"


# ---------------------------------------------------------------------------
# environment for episodes and servers
# ---------------------------------------------------------------------------


def torch_lib(cfg: Mapping[str, Any]) -> str:
    if cfg["paths"].get("torch_lib"):
        return cfg["paths"]["torch_lib"]
    venv = Path(cfg["paths"]["venv"])
    found = sorted(venv.glob("lib/python3*/site-packages/torch/lib"))
    return (
        str(found[0]) if found else str(venv / "lib/python3.10/site-packages/torch/lib")
    )


def base_env(
    cfg: Mapping[str, Any], parent: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The runner's environment minus stale experiment variables, plus the episode extras."""
    env = {
        k: v
        for k, v in dict(os.environ if parent is None else parent).items()
        if not k.startswith(SCRUB_PREFIXES) and k not in SCRUB_NAMES
    }
    env.update({str(k): str(v) for k, v in cfg["episode"]["env"].items()})
    paths = cfg["paths"]
    if paths.get("venv"):
        env["PATH"] = f"{paths['venv']}/bin:{env.get('PATH', '/usr/bin:/bin')}"
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONDONTWRITEBYTECODE="1",
        RC_RENDER="cpu",
        RC_LP_THREADS=str(cfg["episode"]["lp_threads"]),
        ECC_RC_CODE_DIR=str(paths.get("code_dir") or ""),
        ECC_RC_VENV=str(paths.get("venv") or ""),
        ECC_RC_ALLOWED_GPUS=" ".join(str(g) for g in cfg["gpus"]["allowed"]),
        ECC_RC_ASSETS=str(paths.get("assets") or ""),
        ECC_RC_HF_HOME=str(paths.get("hf_home") or ""),
        ECC_RC_CKPT=str(paths.get("rldx_checkpoint") or ""),
        ECC_RC_CUDA_LIB=str(paths["cuda_lib"]),
        ECC_RC_TORCH_LIB=torch_lib(cfg) if paths.get("venv") else "",
        ECC_RC_TORCH_ARCH=str(cfg["episode"]["torch_cuda_arch_list"]),
    )
    return env


def episode_env(
    cfg: Mapping[str, Any],
    job: C.Job,
    n: int,
    endpoint: C.Endpoint,
    parent: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Everything the exploration launcher needs; it sources the credentials file itself."""
    s1, cred, worker, senior = (
        cfg["stage1"],
        cfg["credentials"],
        cfg["worker"],
        cfg["senior"],
    )
    timeout = s1["session_timeout_s_by_split"].get(job.split, s1["session_timeout_s"])
    env = base_env(cfg, parent)
    env.update(
        RC_GPU=str(endpoint.gpu),
        VLA_ENDPOINT=endpoint.url,
        ECC_RC_TRY=str(n),
        ECC_RC_CRED_FILE=str(cred.get("file") or ""),
        ECC_RC_WORKER_KEY_VAR=str(cred.get("worker_key_var") or ""),
        ECC_RC_SENIOR_BASE_URL_VAR=str(cred.get("senior_base_url_var") or ""),
        ECC_RC_SENIOR_KEY_VAR=str(cred.get("senior_key_var") or ""),
        ECC_RC_WORKER_BASE_URL=str(worker.get("base_url") or ""),
        ECC_RC_WORKER_MODEL=str(worker["model"]),
        ECC_RC_WORKER_EFFORT=str(worker["reasoning_effort"]),
        ECC_RC_SENIOR_MODEL=str(senior["model"]),
        ECC_RC_SENIOR_EFFORT=str(senior["reasoning_effort"]),
        ECC_RC_CHECKIN=str(senior["checkin"]),
        ECC_RC_MAX_SHARE=str(senior["max_share"]),
        ECC_RC_SESSIONS=str(s1["sessions"]),
        ECC_RC_ATTEMPTS=str(s1["attempts_per_session"]),
        ECC_RC_SESSION_TIMEOUT_S=str(int(timeout)),
        ECC_RC_MAX_TURNS=str(cfg["episode"]["max_turns"]),
        ECC_RC_LIST_MEMORY="1" if s1["list_memory"] else "0",
        ECC_RC_LLVMPIPE_SCRIPT=str(cfg["paths"].get("llvmpipe_index_script") or ""),
        ECC_RC_MESA_EGL_VENDOR=str(cfg["paths"]["mesa_egl_vendor_json"]),
    )
    return env


# ---------------------------------------------------------------------------
# staging corpora
# ---------------------------------------------------------------------------


def staging_files(cloud: Path, task: str) -> list[str]:
    """The cloud files one task may read: its own task files and the global layer."""
    from robots.robocasa.memory import task_files

    names = [name for name in task_files(task) if (cloud / name).is_file()]
    names += sorted(
        p.relative_to(cloud).as_posix() for p in (cloud / "global").glob("*.md")
    )
    return names


def build_staging(cloud: Path, task: str, dest: Path, tag: str) -> list[str]:
    """Create ``dest`` holding only ``task``'s cloud files plus global memory.

    Everything is read-only except the cell's inbox ``_internal/inbox/<tag>``,
    so no cell (this task's or another, by absolute path) can change a task's
    cloud memory; the inbox is the only place a cell may write in a corpus.
    """
    from robots.robocasa.memory import TaskMemory, selection_errors

    if dest.exists():
        _make_writable(dest)
        shutil.rmtree(dest)
    names = staging_files(cloud, task)
    for name in names:
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cloud / name, target)
    selection = TaskMemory.load(dest, task, profile="local", split=SPLIT)
    errors = selection_errors(selection.metadata(), task, SPLIT)
    if errors:
        raise ValueError(f"staging corpus for {task}: {errors}")
    inbox = dest / "_internal" / "inbox" / tag
    (inbox / "wip").mkdir(parents=True)
    _make_read_only(dest, keep_writable=inbox)
    return names


# ---------------------------------------------------------------------------
# reading an exploration cell
# ---------------------------------------------------------------------------


def explore_stats(try_dir: Path) -> dict[str, Any]:
    """Sessions and attempts actually used, from each session's ``states.json``.

    An attempt is the initial scene of a session plus every accepted ``reset``.
    RPent counts a session solved when its last recorded state has success, and
    then ends the cell, so the first success is the last attempt of that session.
    """
    per = []
    for session in sorted((try_dir / "sessions").glob("session_*")):
        steps = (C.read_json(session / "states.json", {}) or {}).get("steps") or []
        resets = sum(
            1
            for r in steps
            if (r.get("command") or {}).get("action") == "reset"
            and isinstance(r.get("result"), dict)
            and not r["result"].get("error")
        )
        solved = bool(steps) and bool((steps[-1].get("extras") or {}).get("success"))
        per.append(
            {
                "session": session.name,
                "steps": len(steps),
                "attempts": (1 + resets) if steps else 0,
                "solved": solved,
            }
        )
    solved_at = next((i for i, p in enumerate(per) if p["solved"]), None)
    return {
        "sessions_used": sum(1 for p in per if p["steps"]),
        "attempts_used": sum(p["attempts"] for p in per),
        "per_session": per,
        "solved_session": None if solved_at is None else solved_at + 1,
        "first_success_attempt": None
        if solved_at is None
        else sum(p["attempts"] for p in per[: solved_at + 1]),
    }


_TOOL_CALL = re.compile(
    r"\[tool>\] (read_text_file|write_text_file|list_dir)\((.*)\)\s*$"
)
_TOOL_RESULT = re.compile(
    r"\[tool<\] (read_text_file|write_text_file|list_dir): \{ \"path\": \"((?:[^\"\\]|\\.)*)\""
)
_PATH_ARG = re.compile(r'"path": "((?:[^"\\]|\\.)*)"')


def file_tool_paths(try_dir: Path) -> list[tuple[str, str]]:
    """(tool, path) of every memory/file tool call, from run.log and the transcript."""
    found: list[tuple[str, str]] = []
    try:
        lines = (try_dir / "run.log").read_text(errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        for pattern in (_TOOL_CALL, _TOOL_RESULT):
            match = pattern.search(line)
            if not match:
                continue
            tool, text = match.group(1), match.group(2)
            paths = [text] if pattern is _TOOL_RESULT else _PATH_ARG.findall(text)
            for raw in paths:
                try:
                    found.append((tool, json.loads(f'"{raw}"')))
                except ValueError:
                    found.append((tool, raw))
    for transcript in sorted(try_dir.glob("transcript_*.json")):
        record = C.read_json(transcript, {}) or {}
        for message in record.get("messages") or []:
            content = message.get("content")
            if message.get("role") != "assistant" or not isinstance(content, list):
                continue
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "tool_use"
                    and part.get("name") in FILE_TOOLS
                ):
                    args = part.get("input")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    if isinstance(args, dict) and isinstance(args.get("path"), str):
                        found.append((part["name"], args["path"]))
    return found


def file_tool_audit(
    try_dir: Path, *, own: Path, code_dir: Path, shared: list[Path]
) -> dict[str, Any]:
    """File-tool paths outside the task's own folder.

    RPent's file tools allow any path outside ``<repo_root>/memory``; relative
    paths resolve against the code directory. ``cross_task_reads`` and
    ``cross_task_writes`` are paths in another task's folder, the cloud corpus
    or the runs directory; ``code_dir_writes`` are writes into the code
    checkout; ``outside`` is every other path.
    """
    own_r, code_r = os.path.realpath(own), os.path.realpath(code_dir)
    shared_r = [os.path.realpath(p) for p in shared]

    def under(real: str, base: str) -> bool:
        return real == base or real.startswith(base + os.sep)

    calls = file_tool_paths(try_dir)
    found: dict[str, set[str]] = {
        "cross_task_reads": set(),
        "cross_task_writes": set(),
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
        if any(under(real, base) for base in shared_r):
            found["cross_task_writes" if write else "cross_task_reads"].add(real)
        elif under(real, code_r):
            if write:
                found["code_dir_writes"].add(real)
        else:
            found["outside"].add(real)
    return {"calls": len(calls), **{k: sorted(v)[:100] for k, v in found.items()}}


def normalize_legacy_scope(inbox: Path, task: str) -> list[str]:
    """Rewrite ``scope: suite`` to ``scope: task-family`` in this task's drafts; return their names.

    The cloud task notes the explorer reads (``task-specific/<Task>.md``) carry the
    legacy name of the task-family layer, ``scope: suite``, which RPent's merge
    rejects. A draft that copies it but names exactly this task (suite robocasa,
    this split, this task id) is the edge's task-family note; nothing else changes.
    """
    from rpent.memory.manager import _render, _split_frontmatter, _validate

    identity = {"suite": "robocasa", "regime": SPLIT, "task_id": task}
    changed = []
    for path in sorted(inbox.glob("*.md")) if inbox.is_dir() else []:
        try:
            metadata, body = _split_frontmatter(path)
        except ValueError:
            continue
        if metadata.get("scope") != "suite" or any(
            metadata.get(k) != v for k, v in identity.items()
        ):
            continue
        metadata["scope"] = "task-family"
        try:
            _validate(metadata)
        except ValueError:
            continue  # still invalid: left for the merge to report
        path.write_text(_render(metadata, body))
        changed.append(path.name)
    return changed


def inbox_drafts(inbox: Path, task: str) -> dict[str, Any]:
    """What the explorer left in its inbox (for the record; the publish rule decides)."""
    from rpent.memory.manager import _split_frontmatter, _validate

    out: dict[str, Any] = {
        "family": [],
        "global": [],
        "invalid": [],
        "other_task_family": [],
        "wip_files": 0,
    }
    if not inbox.is_dir():
        return out
    out["wip_files"] = (
        sum(1 for p in (inbox / "wip").rglob("*") if p.is_file())
        if (inbox / "wip").is_dir()
        else 0
    )
    identity = {"suite": "robocasa", "regime": SPLIT, "task_id": task}
    for path in sorted(inbox.glob("*.md")):
        try:
            metadata, _ = _split_frontmatter(path)
            _validate(metadata)
        except ValueError as exc:
            out["invalid"].append({"file": path.name, "error": str(exc)[:200]})
            continue
        if metadata["scope"] == "task-family":
            if all(metadata.get(k) == v for k, v in identity.items()):
                out["family"].append(path.name)
            else:
                out["other_task_family"].append(
                    {
                        "file": path.name,
                        "identity": {k: metadata.get(k) for k in identity},
                    }
                )
        else:
            out["global"].append(path.name)
    return out


# ---------------------------------------------------------------------------
# publishing (the RoboCasa replace rule lives in robots/robocasa/memory.py)
# ---------------------------------------------------------------------------

DEFAULT_PUBLISH = "robots.robocasa.memory:publish_exploration_cell"


def resolve_publish(choice: str) -> tuple[str, Callable[..., dict[str, Any]]]:
    """The publish function, ``module:function`` (``auto`` = the RoboCasa replace).

    It must accept ``(corpus, *, task_name, split, run_dir, solved, inbox)``
    and publish one finished seed-0 exploration cell into ``corpus`` with the
    stage-1 replace rule: a solved cell with its audit and recipe replaces the
    task's cloud files (archived under ``_internal/replaced``); any other cell
    keeps them; global drafts merge by RPent's default rules either way.
    """
    target = DEFAULT_PUBLISH if choice in ("", "auto") else choice
    module, _, attr = target.partition(":")
    return target, getattr(importlib.import_module(module), attr)


def expected_publish(solved: bool, audit: Path, recipe: Path) -> dict[str, str]:
    """The replace rule's decision, recorded before the corpus is assembled."""
    if not solved:
        return {"mode": "keep_cloud", "reason": "unsolved"}
    missing = [
        p.name
        for p in (audit, recipe)
        if not p.is_file() or (p is audit and p.stat().st_size == 0)
    ]
    if missing:
        return {
            "mode": "keep_cloud",
            "reason": f"solved, but missing {', '.join(missing)}",
        }
    return {"mode": "replace", "reason": "solved with audit and recipe"}


def validate_corpus(corpus: Path, tasks: list[tuple[str, str]]) -> dict[str, Any]:
    """Load every task's memory the way evaluation does (local profile); no errors allowed."""
    from robots.robocasa.memory import TaskMemory, selection_errors

    per_task, errors = {}, {}
    for _, task in tasks:
        try:
            selection = TaskMemory.load(corpus, task, profile="local", split=SPLIT)
            problems = selection_errors(selection.metadata(), task, SPLIT)
            per_task[task] = list(selection.selected)
            if problems:
                errors[task] = problems
        except (OSError, ValueError) as exc:
            errors[task] = [str(exc)]
    return {"tasks_loaded": len(per_task), "errors": errors, "selected": per_task}


# ---------------------------------------------------------------------------
# the stage
# ---------------------------------------------------------------------------


class Stage1(C.Stage):
    name = "s1"

    def __init__(
        self,
        cfg: Mapping[str, Any],
        layout: C.RunLayout,
        tasks: list[tuple[str, str]],
        all_tasks: list[tuple[str, str]],
    ):
        self.cfg = cfg
        self.layout = layout
        self.tasks = tasks
        self.all_tasks = all_tasks
        self.seed = int(cfg["stage1"]["seed"])
        self.cloud = Path(cfg["paths"]["cloud_corpus"])
        self.code_dir = Path(cfg["paths"]["code_dir"])
        self._jobs = [
            C.Job(job_id=task, task=task, split=split, seed=self.seed, order=i)
            for i, (split, task) in enumerate(tasks)
        ]

    def jobs(self) -> list[C.Job]:
        return list(self._jobs)

    def staging(self, task: str) -> Path:
        return self.layout.task_dir(task) / "staging"

    def try_dir(self, job: C.Job, n: int) -> Path:
        return self.layout.task_dir(job.task) / f"try{n}"

    def prepare_try(self, job: C.Job, n: int) -> None:
        """Every try starts from the pristine staging corpus (a retry inherits nothing)."""
        staging = self.staging(job.task)
        tag = cell_tag(job.task, self.seed)
        inbox = staging / "_internal" / "inbox" / tag
        if n > 1 and inbox.exists():
            _make_writable(staging)
            shutil.move(
                str(inbox), str(self.layout.task_dir(job.task) / f"try{n - 1}.inbox")
            )
        if n > 1 or not staging.exists():
            build_staging(self.cloud, job.task, staging, tag)

    def launch(
        self, job: C.Job, n: int, endpoint: C.Endpoint
    ) -> tuple[list[str], dict[str, str]]:
        argv = [
            *self.cfg["launch"]["cell"],
            job.split,
            job.task,
            str(job.seed),
            str(self.try_dir(job, n)),
            str(self.staging(job.task)),
        ]
        return argv, episode_env(self.cfg, job, n, endpoint)

    # -- after the last try ------------------------------------------------

    def record_path(self, task: str) -> Path:
        return self.layout.task_dir(task) / "record.json"

    def on_final(self, job: C.Job, state: dict[str, Any]) -> None:
        """Capture the task's record and product, then check that the corpus still assembles.

        Errors are logged, not raised: running cells must not lose their runner.
        A missing record is captured again by the next runner; freeze refuses
        until every task has one.
        """
        if self.record_path(job.task).exists():
            return
        try:
            record = self.capture(job, state)
        except Exception as exc:  # noqa: BLE001 - logged; retried by the next runner
            C.append_jsonl(
                self.layout.progress,
                {
                    "event": "capture_error",
                    "job": job.task,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            return
        C.atomic_write_json(self.record_path(job.task), record)
        C.append_jsonl(
            self.layout.progress,
            {
                "event": "captured",
                "job": job.task,
                "publish": record["publish"]["mode"],
            },
        )
        try:
            summary = assemble(self, None, require_all=False)
            check = {
                k: summary[k]
                for k in (
                    "publish_impl",
                    "replaced",
                    "kept_cloud",
                    "no_valid_result",
                    "not_final",
                )
            }
            check["errors"] = summary["validation"]["errors"]
            C.atomic_write_json(
                self.layout.root / "corpus_check.json", {"at": C.now_iso(), **check}
            )
            if check["errors"]:
                C.append_jsonl(
                    self.layout.progress,
                    {"event": "corpus_check_failed", "errors": check["errors"]},
                )
        except Exception as exc:  # noqa: BLE001 - surfaced in progress.jsonl and at freeze
            C.append_jsonl(
                self.layout.progress,
                {"event": "assemble_error", "error": f"{type(exc).__name__}: {exc}"},
            )

    def capture(self, job: C.Job, state: dict[str, Any]) -> dict[str, Any]:
        """The task's record and its immutable product folder (``tasks/<Task>/product``)."""
        task, tag = job.task, cell_tag(job.task, self.seed)
        final = state["final"]
        last = state["tries"][final["try"] - 1]
        try_dir = Path(last["dir"])
        outcome = C.read_outcome(try_dir)
        stats = explore_stats(try_dir)
        inbox = self.staging(task) / "_internal" / "inbox" / tag
        drafts = inbox_drafts(inbox, task)
        audit, recipe = try_dir / f"{tag}.json", try_dir / f"{tag}_recipe.jsonl"
        audit_ok = C.read_json(audit) is not None if audit.is_file() else False
        solved = final["class"] == "valid" and bool(final.get("success"))
        status = (
            "solved"
            if solved
            else "unsolved"
            if final["class"] == "valid"
            else final["class"]
        )
        if final["class"] == "valid":
            publish = expected_publish(solved, audit, recipe)
        else:  # no valid exploration result: nothing of this cell is published
            publish = {
                "mode": "none",
                "reason": f"{final['class']}: no valid exploration result",
            }
        product = self.layout.task_dir(task) / "product"
        if product.exists():
            shutil.rmtree(product)
        (product / "run").mkdir(parents=True)
        for path in (audit, recipe):
            if path.is_file():
                shutil.copy2(path, product / "run" / path.name)
        normalized: list[str] = []
        if inbox.is_dir():
            shutil.copytree(inbox, product / "inbox")
            normalized = normalize_legacy_scope(product / "inbox", task)
        drafts["scope_suite_normalized"] = normalized
        usage = [t.get("usage") or {} for t in state["tries"]]
        files = C.tree_files(product)
        return {
            "task": task,
            "split": job.split,
            "seed": self.seed,
            "cell_tag": tag,
            "status": status,
            "solved": solved,
            "valid": bool(outcome.get("valid")),
            "termination_reason": outcome.get("termination_reason"),
            "elapsed_s": outcome.get("elapsed_s"),
            "budget": {
                "sessions": self.cfg["stage1"]["sessions"],
                "attempts_per_session": self.cfg["stage1"]["attempts_per_session"],
                "session_timeout_s": self.cfg["stage1"][
                    "session_timeout_s_by_split"
                ].get(job.split, self.cfg["stage1"]["session_timeout_s"]),
                "max_turns_per_session": self.cfg["episode"]["max_turns"],
            },
            **stats,
            "tries": [
                {
                    "try": t["n"],
                    "class": t.get("class"),
                    "endpoint": t.get("endpoint"),
                    "termination_reason": (t.get("outcome") or {}).get(
                        "termination_reason"
                    ),
                    "elapsed_s": (t.get("outcome") or {}).get("elapsed_s"),
                    "usd": round(float((t.get("usage") or {}).get("usd") or 0.0), 4),
                }
                for t in state["tries"]
            ],
            "infra_retries": sum(
                1 for t in state["tries"] if t.get("class") == "infra"
            ),
            "render_retries": sum(
                1 for t in state["tries"] if t.get("class") == "render"
            ),
            "qwen": {
                "input_tokens": (last.get("usage") or {}).get("worker_in", 0),
                "output_tokens": (last.get("usage") or {}).get("worker_out", 0),
                "requests": (last.get("usage") or {}).get("worker_calls", 0),
                "usd": round(float((last.get("usage") or {}).get("usd") or 0.0), 4),
                "usd_all_tries": round(
                    sum(float(u.get("usd") or 0.0) for u in usage), 4
                ),
            },
            "audit_present": audit.is_file(),
            "audit_json_valid": audit_ok,
            "recipe_present": recipe.is_file(),
            "inbox": drafts,
            "publish": publish,
            "file_tool_audit": file_tool_audit(
                try_dir,
                own=self.layout.task_dir(task),
                code_dir=self.code_dir,
                shared=[
                    self.layout.tasks_dir,
                    self.cloud,
                    Path(self.cfg["paths"].get("runs_dir") or self.layout.root.parent),
                    *(Path(p) for p in self.cfg["paths"]["audit_watch"]),
                ],
            ),
            "final_try_dir": str(try_dir),
            "product_files": files,
            "captured_at": C.now_iso(),
        }


# ---------------------------------------------------------------------------
# assembly, freeze, verify
# ---------------------------------------------------------------------------


def assemble(stage: Stage1, dest: Path | None, *, require_all: bool) -> dict[str, Any]:
    """A fresh copy of the cloud corpus, then every task's cell published in Target50 order.

    The order is fixed so that global drafts with the same id merge the same
    way however the cells finished. Tasks without a valid result add nothing.
    With ``dest`` None the corpus is only validated, in a temporary folder
    under the run root that is removed again, so running cells find no
    published memory of other tasks.
    """
    impl_name, impl = resolve_publish(stage.cfg["stage1"]["publish_impl"])
    keep = dest is not None
    if (
        dest is None
    ):  # one folder per process: the runner and an operator may check at once
        dest = stage.layout.root / ".corpus_check" / str(os.getpid()) / "corpus"
        dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    if tmp.exists():
        _make_writable(tmp)
        shutil.rmtree(tmp)
    shutil.copytree(stage.cloud, tmp, symlinks=False)
    published, missing, skipped, mismatched = {}, [], [], []
    for _, task in stage.tasks:
        record = C.read_json(stage.record_path(task))
        if record is None:
            missing.append(task)
            continue
        if record["publish"]["mode"] == "none":
            skipped.append(task)
            continue
        product = stage.layout.task_dir(task) / "product"
        result = impl(
            tmp,
            task_name=task,
            split=SPLIT,
            run_dir=product / "run",
            solved=bool(record["solved"]),
            inbox=product / "inbox" if (product / "inbox").is_dir() else None,
        )
        published[task] = result
        expected = "replaced" if record["publish"]["mode"] == "replace" else "kept"
        if result.get("task_memory") != expected:
            mismatched.append(
                {"task": task, "expected": expected, "got": result.get("task_memory")}
            )
    if require_all and missing:
        _make_writable(tmp)
        shutil.rmtree(tmp)
        raise SystemExit(
            f"cannot freeze: {len(missing)} task(s) have no record (not final, or the capture "
            f"failed: see capture_error in progress.jsonl): {', '.join(missing)}"
        )
    validation = validate_corpus(tmp, stage.all_tasks)
    for item in mismatched:
        validation["errors"].setdefault(item["task"], []).append(
            f"publish decided {item['got']!r}, the record expected {item['expected']!r}"
        )
    for task, result in published.items():
        tag = cell_tag(task, stage.seed)
        selected = validation["selected"].get(task, [])
        native = f"task-specific/{tag}.json" in selected
        cloud = f"task-specific/{task}_s0.json" in selected
        if result.get("task_memory") == "replaced" and (not native or cloud):
            validation["errors"].setdefault(task, []).append(
                "replaced task does not load the edge pair alone"
            )
        if result.get("task_memory") == "kept" and (native or not cloud):
            validation["errors"].setdefault(task, []).append(
                "kept task does not load the cloud pair alone"
            )
    summary = {
        "publish_impl": impl_name,
        "replaced": sorted(
            t for t, r in published.items() if r.get("task_memory") == "replaced"
        ),
        "kept_cloud": sorted(
            t for t, r in published.items() if r.get("task_memory") == "kept"
        ),
        "no_valid_result": skipped,
        "not_final": missing,
        "merge_results": published,
        "validation": validation,
    }
    if not keep:
        _make_writable(dest.parent)
        shutil.rmtree(dest.parent)
        return summary
    C.atomic_write_json(dest.with_name(dest.name + ".assembly.json"), summary)
    if dest.exists():
        _make_writable(dest)
        shutil.rmtree(dest)
    tmp.rename(dest)
    return summary


def _make_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(path.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def _make_read_only(root: Path, keep_writable: Path | None = None) -> None:
    """Remove write permission below ``root``, except inside ``keep_writable``."""
    write = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if keep_writable is not None and (
            path == keep_writable or keep_writable in path.parents
        ):
            continue
        path.chmod(path.stat().st_mode & ~write)
    root.chmod(root.stat().st_mode & ~write)


def freeze(
    stage: Stage1, manifest: Mapping[str, Any], *, allow_cross_task_writes: bool = False
) -> dict[str, Any]:
    """Assemble the final corpus, validate all 50 tasks, hash and make it read-only."""
    product = stage.layout.product
    if (product / "FROZEN").exists():
        return verify(product)
    writers = {}
    for _, task in stage.tasks:
        audit = (C.read_json(stage.record_path(task)) or {}).get(
            "file_tool_audit"
        ) or {}
        if audit.get("cross_task_writes"):
            writers[task] = audit["cross_task_writes"]
    if writers and not allow_cross_task_writes:
        raise SystemExit(
            "refusing to freeze: cells wrote into another task's folder or a shared corpus "
            f"(inspect, then freeze --allow-cross-task-writes): {json.dumps(writers)[:2000]}"
        )
    if product.exists():
        _make_writable(product)
        shutil.rmtree(product)
    product.mkdir(parents=True)
    summary = assemble(stage, product / "corpus", require_all=True)
    (product / "corpus.assembly.json").rename(product / "assembly.json")
    if summary["validation"]["errors"]:
        raise SystemExit(
            f"corpus validation failed: {json.dumps(summary['validation']['errors'])[:2000]}"
        )
    input_files = C.tree_files(stage.cloud)
    if C.tree_hash(input_files) != manifest["cloud_corpus"]["tree_sha256"]:
        raise SystemExit("the cloud corpus changed during the run; refusing to freeze")
    files = C.tree_files(product / "corpus")
    records = {task: C.read_json(stage.record_path(task)) for _, task in stage.tasks}
    C.atomic_write_json(product / "tasks.json", records)
    corpus_hash = C.tree_hash(files)
    doc = {
        "schema": PRODUCT_SCHEMA,
        "stage": "s1",
        "run_id": manifest["run_id"],
        "created_at": C.now_iso(),
        "code": manifest["code"],
        "config_sha256": manifest["config_sha256"],
        "experiment_sha256": manifest["experiment_sha256"],
        "seed": stage.seed,
        "budget": manifest["budget"],
        "cloud_corpus": manifest["cloud_corpus"],
        "publish_impl": summary["publish_impl"],
        "corpus": "corpus",
        "corpus_tree_sha256": corpus_hash,
        "tasks_json_sha256": C.sha256_file(product / "tasks.json"),
        "files": files,
        "tasks": {
            task: {
                k: (rec or {}).get(k)
                for k in (
                    "split",
                    "status",
                    "publish",
                    "first_success_attempt",
                    "sessions_used",
                    "attempts_used",
                )
            }
            for task, rec in records.items()
        },
        "cross_task_writes_accepted": writers,
        "validation": {
            "tasks_loaded": summary["validation"]["tasks_loaded"],
            "errors": {},
        },
    }
    C.atomic_write_json(product / "PRODUCT.json", doc)
    _make_read_only(product / "corpus")
    C.atomic_write_text(product / "FROZEN", corpus_hash + "\n")
    for name in ("PRODUCT.json", "tasks.json", "assembly.json", "FROZEN"):
        (product / name).chmod(0o444)
    C.append_jsonl(
        stage.layout.progress, {"event": "frozen", "corpus_tree_sha256": corpus_hash}
    )
    return {
        "ok": True,
        "corpus_tree_sha256": corpus_hash,
        "files": len(files),
        "product": str(product),
    }


def verify(product: Path) -> dict[str, Any]:
    """Recompute the product's hashes (what stages 2-3 do before reading it)."""
    doc = C.read_json(product / "PRODUCT.json")
    if not doc or doc.get("schema") != PRODUCT_SCHEMA:
        return {"ok": False, "error": "no PRODUCT.json with the stage-1 schema"}
    files = C.tree_files(product / doc["corpus"])
    problems = []
    if C.tree_hash(files) != doc["corpus_tree_sha256"]:
        changed = sorted(set(files.items()) ^ set(doc["files"].items()))
        problems.append(
            f"corpus hash mismatch ({len(changed)} differing entries, e.g. {changed[:3]})"
        )
    frozen = (
        (product / "FROZEN").read_text().strip()
        if (product / "FROZEN").exists()
        else None
    )
    if frozen != doc["corpus_tree_sha256"]:
        problems.append("FROZEN marker missing or different")
    if C.sha256_file(product / "tasks.json") != doc["tasks_json_sha256"]:
        problems.append("tasks.json changed")
    writable = [
        p
        for p in (product / doc["corpus"]).rglob("*")
        if p.stat().st_mode & stat.S_IWUSR
    ]
    if writable:
        problems.append(f"{len(writable)} corpus entries are writable")
    return {
        "ok": not problems,
        "problems": problems,
        "corpus_tree_sha256": doc["corpus_tree_sha256"],
        "files": len(files),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def select_tasks(
    cfg: Mapping[str, Any], args: argparse.Namespace
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    all_tasks = C.load_target50(cfg["paths"]["code_dir"])
    wanted = [
        t
        for t in (args.tasks.split(",") if args.tasks else cfg["stage1"]["tasks"])
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


def budget(cfg: Mapping[str, Any]) -> dict[str, Any]:
    s1 = cfg["stage1"]
    return {
        "sessions": s1["sessions"],
        "attempts_per_session": s1["attempts_per_session"],
        "session_timeout_s": s1["session_timeout_s"],
        "session_timeout_s_by_split": s1["session_timeout_s_by_split"],
        "max_turns_per_session": cfg["episode"]["max_turns"],
        "infra_retries": s1["infra_retries"],
        "render_retries": s1["render_retries"],
        "stop": "RPent ends the cell at the first solved session",
    }


def corpus_summary(cloud: Path) -> dict[str, Any]:
    files = C.tree_files(cloud)
    return {
        "path": str(cloud),
        "tree_sha256": C.tree_hash(files),
        "files": len(files),
        "pairs": sum(
            1
            for f in files
            if f.startswith("task-specific/") and f.endswith("_s0.json")
        ),
        "task_notes": sum(
            1 for f in files if re.fullmatch(r"task-specific/[A-Za-z0-9]+\.md", f)
        ),
        "global": sorted(f for f in files if f.startswith("global/")),
    }


def runner_origin(code_dir: str | Path) -> dict[str, Any]:
    """Where this runner imported ``rpent`` and the RoboCasa memory code from (publishing uses them)."""
    import robots.robocasa.memory as memory
    import rpent

    root = os.path.realpath(str(code_dir)) + os.sep
    found = {
        "rpent": os.path.realpath(rpent.__file__),
        "robots.robocasa.memory": os.path.realpath(memory.__file__),
    }
    return {"ok": all(v.startswith(root) for v in found.values()), **found}


def make_manifest(
    cfg: Mapping[str, Any], run_id: str, tasks: list[tuple[str, str]]
) -> dict[str, Any]:
    code = C.code_identity(cfg["paths"]["code_dir"])
    return {
        "schema": "rc-ecc-run/1",
        "stage": "s1",
        "run_id": run_id,
        "created_at": C.now_iso(),
        "config_path": cfg["_path"],
        "config_sha256": C.config_hash(cfg),
        "experiment_sha256": C.experiment_hash(cfg),
        "code": code,
        "cloud_corpus": corpus_summary(Path(cfg["paths"]["cloud_corpus"])),
        "seed": cfg["stage1"]["seed"],
        "split": SPLIT,
        "budget": budget(cfg),
        "tasks": [
            {"split": s, "task": t, "cell_tag": cell_tag(t, cfg["stage1"]["seed"])}
            for s, t in tasks
        ],
        "worker": cfg["worker"],
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
    manifest: dict[str, Any], fresh: Mapping[str, Any], allow_code_change: bool
) -> list[str]:
    problems = []
    if manifest["experiment_sha256"] != fresh["experiment_sha256"]:
        problems.append(
            "experiment settings (worker/senior/episode/stage1) differ from the run's manifest"
        )
    if [t["task"] for t in manifest["tasks"]] != [t["task"] for t in fresh["tasks"]]:
        problems.append("task list differs from the run's manifest")
    if manifest["cloud_corpus"]["tree_sha256"] != fresh["cloud_corpus"]["tree_sha256"]:
        problems.append("cloud corpus differs from the run's manifest")
    if (
        manifest["code"]["tree_sha256"] != fresh["code"]["tree_sha256"]
        and not allow_code_change
    ):
        problems.append(
            "code differs from the run's manifest (pass --allow-code-change to accept and record it)"
        )
    return problems


def run_layout(cfg: Mapping[str, Any], args: argparse.Namespace) -> C.RunLayout:
    if getattr(args, "run_root", None):
        return C.RunLayout(Path(args.run_root).resolve())
    if not args.run_id:
        raise SystemExit("pass --run-id (or --run-root)")
    return C.RunLayout(
        Path(cfg["paths"]["runs_dir"]).resolve() / C.safe_name(args.run_id)
    )


def print_plan(cfg: Mapping[str, Any], args: argparse.Namespace) -> int:
    """The dry run: everything --go would do, without creating or starting anything."""
    C.require(cfg, "paths.code_dir", "paths.cloud_corpus", "paths.runs_dir")
    tasks, _ = select_tasks(cfg, args)
    s1 = cfg["stage1"]
    print(
        f"RC-ECC stage 1 plan (dry run; add --go to launch)\n  config   {cfg['_path']}"
    )
    print(
        f"  config sha256 {C.config_hash(cfg)[:16]}  experiment sha256 {C.experiment_hash(cfg)[:16]}"
    )
    code = C.code_identity(cfg["paths"]["code_dir"])
    print(
        f"  code     {cfg['paths']['code_dir']}  commit {code['commit']} ({code['commit_source']})  "
        f"tree {code['tree_sha256'][:16]} ({code['files']} files)"
    )
    corpus = corpus_summary(Path(cfg["paths"]["cloud_corpus"]))
    print(
        f"  cloud corpus {corpus['path']}  {corpus['files']} files, {corpus['pairs']} pairs, "
        f"{corpus['task_notes']} task notes, global {corpus['global']}  tree {corpus['tree_sha256'][:16]}"
    )
    by_split = {}
    for split, _ in tasks:
        by_split[split] = by_split.get(split, 0) + 1
    print(
        f"  tasks    {len(tasks)} ({', '.join(f'{k} {v}' for k, v in by_split.items())}); seed {s1['seed']}, split {SPLIT}"
    )
    print(
        f"  budget   {s1['sessions']} sessions x {s1['attempts_per_session']} attempts, {s1['session_timeout_s']} s per session "
        f"{s1['session_timeout_s_by_split'] or ''}, {cfg['episode']['max_turns']} turns per session; stop at the first solved session"
    )
    print(
        f"  retries  infra {s1['infra_retries']}, render {s1['render_retries']} per task (never counted as task failures)"
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
        sched, len(tasks), len(tasks), s1["infra_retries"], s1["render_retries"]
    )
    print(
        f"  slots    max {max_slots}; stagger {sched['launch_stagger_s']} s; launch ceiling {ceiling}; "
        f"infra breaker {sched['infra_breaker']}; cost ceiling "
        f"{'$%.0f' % sched['cost_ceiling_usd'] if sched['cost_ceiling_usd'] else 'off'}"
    )
    try:
        usage = C.gpu_usage(cfg)
    except (OSError, RuntimeError, Exception) as exc:  # noqa: BLE001 - dry run reports and goes on
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
            **base_env(cfg),
            "PYTHONPATH": cfg["paths"]["code_dir"],
            "RPENT_REPO_ROOT": cfg["paths"]["code_dir"],
        }
        origin = C.import_origin(C.python_for(cfg), cfg["paths"]["code_dir"], env)
        print(
            f"  episode imports: {'OK' if origin.get('ok') else 'NOT FROM THE CODE DIR'} {origin}"
        )
    mine = runner_origin(cfg["paths"]["code_dir"])
    print(
        f"  runner imports:  {'OK' if mine['ok'] else 'NOT FROM THE CODE DIR'} {mine}"
    )
    if tasks:
        split, task = tasks[0]
        job = C.Job(job_id=task, task=task, split=split, seed=s1["seed"], order=0)
        layout = layout or C.RunLayout(Path(cfg["paths"]["runs_dir"]) / "<run-id>")
        stage = Stage1(cfg, layout, tasks, tasks)
        eps = C.endpoints_from_config(cfg)
        argv, env = stage.launch(
            job, 1, eps[0] if eps else C.Endpoint("gpuN", -1, 0, 0)
        )
        print(f"  first launch: {' '.join(argv)}")
        shown = sorted(
            k
            for k in env
            if k.startswith(("ECC_RC_", "RC_", "VLA_")) or k in cfg["episode"]["env"]
        )
        for k in shown:
            print(f"    {k}={env[k]}")
    return 0


def cmd_run(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    if not args.go:
        return print_plan(cfg, args)
    C.require(
        cfg,
        "paths.code_dir",
        "paths.venv",
        "paths.cloud_corpus",
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
    fresh = make_manifest(cfg, layout.root.name, tasks)
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
            **base_env(cfg),
            "PYTHONPATH": cfg["paths"]["code_dir"],
            "RPENT_REPO_ROOT": cfg["paths"]["code_dir"],
        }
        origin = C.import_origin(C.python_for(cfg), cfg["paths"]["code_dir"], env)
        if not origin.get("ok"):
            raise SystemExit(
                f"episode processes would not import rpent/robots from the code dir: {origin}"
            )
        fresh["import_origin"] = origin
        fresh["runner_import_origin"] = runner_origin(cfg["paths"]["code_dir"])
        if not fresh["runner_import_origin"]["ok"]:
            raise SystemExit(
                f"the runner does not import from the code dir: {fresh['runner_import_origin']}"
            )
        stage = Stage1(cfg, layout, tasks, all_tasks)
        for _, task in tasks:
            stage.staging(task).parent.mkdir(parents=True, exist_ok=True)
            build_staging(
                stage.cloud, task, stage.staging(task), cell_tag(task, stage.seed)
            )
        C.atomic_write_json(
            layout.root / "config.snapshot.json",
            {k: v for k, v in cfg.items() if k != "_path"},
        )
        C.atomic_write_json(layout.manifest, fresh)
        manifest = fresh
    stage = Stage1(cfg, layout, tasks, all_tasks)
    servers = C.ServerPool(
        cfg, layout, base_env(cfg), lambda row: C.append_jsonl(layout.progress, row)
    )
    scheduler = C.Scheduler(
        cfg,
        layout,
        stage,
        servers,
        infra_retries=cfg["stage1"]["infra_retries"],
        render_retries=cfg["stage1"]["render_retries"],
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
    for job in stage.jobs():  # finish any capture an earlier runner did not complete
        state = scheduler.state(job.job_id)
        if state["final"]:
            stage.on_final(job, state)
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
            # Cells cannot list tasks/ while they run, so another task's folder is not discoverable.
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
        print(json.dumps(freeze(stage, manifest), indent=2), flush=True)
    return 0 if result in ("done", "stopped") else 1


def load_stage(
    cfg: dict[str, Any], args: argparse.Namespace
) -> tuple[C.RunLayout, dict[str, Any], Stage1]:
    layout = run_layout(cfg, args)
    manifest = C.read_json(layout.manifest)
    if manifest is None:
        raise SystemExit(f"no run at {layout.root}")
    all_tasks = C.load_target50(cfg["paths"]["code_dir"])
    tasks = [(t["split"], t["task"]) for t in manifest["tasks"]]
    return layout, manifest, Stage1(cfg, layout, tasks, all_tasks)


def cmd_status(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Blind-safe overview: progress bar, per-split counts, endpoints, guards and cost."""
    layout, manifest, stage = load_stage(cfg, args)
    states = {}
    for job in stage.jobs():
        state = C.read_json(layout.job_state(job.job_id)) or {
            "job_id": job.job_id,
            "split": job.split,
            "tries": [],
            "final": None,
        }
        if state["tries"] and state["tries"][-1].get("status") in (
            "launching",
            "running",
        ):
            pid = state["tries"][-1].get("pid")
            state["tries"][-1]["alive"] = bool(pid and C.pid_alive(pid))
        states[job.job_id] = state
    holder = C.lock_holder(layout)
    stop = C.read_json(layout.stop) if layout.stop.exists() else None
    extra = [
        f"runner: {'pid %s' % holder if holder else 'not running'}; "
        f"stop file: {stop['reason'] if stop else 'absent'}"
    ]
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
    extra.append("endpoints: " + (", ".join(shown) or "none configured"))
    launches = C.counted_launches(states)
    ceiling = C.launch_ceiling(
        cfg["scheduler"],
        len(states),
        len(states),
        cfg["stage1"]["infra_retries"],
        cfg["stage1"]["render_retries"],
    )
    extra.append(f"launches: {launches} of ceiling {ceiling}")
    stale = [
        j
        for j, s in states.items()
        if s["tries"] and s["tries"][-1].get("alive") is False
    ]
    if stale:
        extra.append(
            f"NOTE: {len(stale)} cell(s) marked running have no live process; the next --go scores them"
        )
    flagged = []
    for job in stage.jobs():
        rec = C.read_json(stage.record_path(job.task)) or {}
        audit = rec.get("file_tool_audit") or {}
        kinds = [
            k
            for k in ("cross_task_writes", "cross_task_reads", "code_dir_writes")
            if audit.get(k)
        ]
        if kinds:
            flagged.append(f"{job.task} ({', '.join(kinds)})")
    if flagged:
        extra.append(f"file-tool access outside the task folder: {'; '.join(flagged)}")
    check = C.read_json(layout.root / "corpus_check.json")
    if check:
        extra.append(
            f"corpus check ({check['at']}): {len(check['replaced'])} replaced, "
            f"{len(check['kept_cloud'])} kept cloud, errors {len(check['errors'])}"
        )
    frozen = layout.product / "FROZEN"
    if frozen.exists():
        extra.append(
            f"product: frozen, corpus sha256 {frozen.read_text().strip()[:16]}"
        )
    cost = C.total_cost(states, cfg["pricing"])
    print(
        C.format_status(
            f"RC-ECC S1 {manifest['run_id']}", states, cost_usd=cost, extra_lines=extra
        )
    )
    if args.tasks_table:
        for job in stage.jobs():
            rec = C.read_json(stage.record_path(job.task)) or {}
            s = states[job.job_id]
            live = (
                "running"
                if s["tries"] and s["tries"][-1].get("status") == "running"
                else "queued"
            )
            print(
                f"  {job.split:<17} {job.task:<26} {rec.get('status') or live:<16} tries {len(s['tries'])}  "
                f"sessions {rec.get('sessions_used', '-')}  attempts {rec.get('attempts_used', '-')}  "
                f"first-success {rec.get('first_success_attempt', '-')}  "
                f"publish {(rec.get('publish') or {}).get('mode', '-')}"
            )
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
    running = [
        j.job_id
        for j in stage.jobs()
        if ((C.read_json(layout.job_state(j.job_id)) or {}).get("tries") or [{}])[
            -1
        ].get("status")
        == "running"
        and C.pid_alive(
            ((C.read_json(layout.job_state(j.job_id)) or {}).get("tries") or [{}])[
                -1
            ].get("pid")
        )
    ]
    if C.lock_holder(layout) is None and not running:
        servers = C.ServerPool(
            cfg, layout, {}, lambda row: C.append_jsonl(layout.progress, row)
        )
        servers.stop_all()
        print("no runner and no running cell: stopped this run's RLDX servers")
    return 0


def cmd_assemble(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    """Assemble the corpus from the tasks final so far: validate, or keep a copy with --out.

    A kept copy is readable by running cells (by absolute path); prefer a path
    outside the runs directory while cells run.
    """
    _, _, stage = load_stage(cfg, args)
    out = Path(args.out).resolve() if args.out else None
    summary = assemble(stage, out, require_all=False)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "merge_results"}, indent=2
        )[:6000]
    )
    return 0 if not summary["validation"]["errors"] else 1


def cmd_freeze(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    _, manifest, stage = load_stage(cfg, args)
    result = freeze(
        stage, manifest, allow_cross_task_writes=args.allow_cross_task_writes
    )
    print(json.dumps(result, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify(Path(args.product).resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


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
            help="comma list of Target50 tasks (default: stage1.tasks or all 50)",
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
            help="write <run>/concurrency (live slot limit, at most max-slots) before launching",
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
        "stop", help="write the stop file (drain); --now also stops running cells"
    )
    common_args(p)
    p.add_argument("--now", action="store_true")
    p = sub.add_parser(
        "assemble", help="assemble and validate the corpus from the final tasks so far"
    )
    common_args(p)
    p.add_argument(
        "--out",
        help="keep the assembled corpus in this folder (default: validate only)",
    )
    p = sub.add_parser("freeze", help="freeze the stage-1 product (all tasks final)")
    common_args(p)
    p.add_argument(
        "--allow-cross-task-writes",
        action="store_true",
        help="freeze although a cell wrote into another task's folder (recorded in PRODUCT.json)",
    )
    p = sub.add_parser("verify", help="recompute a frozen product's hashes")
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
    cfg = C.load_config(C.config_path(args.config))
    return {
        "run": cmd_run,
        "status": cmd_status,
        "stop": cmd_stop,
        "assemble": cmd_assemble,
        "freeze": cmd_freeze,
    }[args.cmd](cfg, args)


if __name__ == "__main__":
    sys.exit(main())
