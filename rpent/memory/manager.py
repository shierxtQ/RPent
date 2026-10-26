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

"""Cross-session memory corpus management."""

from __future__ import annotations

import fcntl
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

from rpent.utils.logging import get_logger

logger = get_logger("memory")

SCOPES = {"global", "task-family"}
KINDS = {"primitive", "perception", "strategy", "failure", "infra"}
CONFIDENCE = {"single-shot", "probable", "verified"}
_PREFIXES = ("new_global_", "new_task-family_", "new_", "task-family_")
_KINDS = tuple(f"{kind}_" for kind in sorted(KINDS))


def _split_frontmatter(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(errors="replace")
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("unterminated YAML frontmatter")
    try:
        metadata = yaml.safe_load(text[3:end])
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("frontmatter must be a mapping")
    return metadata, text[end + 4 :]


def _validate(metadata: dict[str, Any]) -> None:
    scope = metadata.get("scope")
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {sorted(SCOPES)}, got {scope!r}")
    if scope == "task-family":
        for field in ("suite", "regime", "task_id", "task_language"):
            if metadata.get(field) in (None, ""):
                raise ValueError(f"task-family memory requires {field!r}")
    else:
        if metadata.get("kind") not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        for field in ("title", "applies_when"):
            if not str(metadata.get(field, "")).strip():
                raise ValueError(f"global memory requires {field!r}")
    if metadata.get("confidence") not in CONFIDENCE:
        raise ValueError(f"confidence must be one of {sorted(CONFIDENCE)}")
    evidence = metadata.get("evidence") or {}
    if not isinstance(evidence, dict):
        raise ValueError("evidence must be a mapping")
    if not isinstance(evidence.get("cells"), list) or not evidence["cells"]:
        raise ValueError("evidence.cells must be a non-empty list")


def _canonical_id(path: Path, metadata: dict[str, Any]) -> str:
    if metadata["scope"] == "task-family":
        return f"task-family_{metadata['suite']}_{metadata['regime']}_t{metadata['task_id']}"
    stem = re.sub(r"_draft$", "", path.stem)
    for prefix in (*_PREFIXES, *_KINDS):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return str(metadata.get("id") or stem).strip()


def _render(metadata: dict[str, Any], body: str) -> str:
    return (
        "---\n"
        + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True)
        + "---\n"
        + body
    )


def _attempt_count(value: Any) -> int:
    """Attempts as an int; drafts sometimes write a list such as ``[3]``."""
    if isinstance(value, list):
        return sum(_attempt_count(item) for item in value)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _evidence_items(evidence: dict[str, Any], key: str) -> list[Any]:
    value = evidence.get(key)
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _merge_evidence(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_evidence = old.get("evidence") or {}
    new_evidence = new.get("evidence") or {}
    cells = sorted(
        {
            *_evidence_items(old_evidence, "cells"),
            *_evidence_items(new_evidence, "cells"),
        },
        key=str,
    )
    evidence = {
        **old_evidence,
        "cells": cells,
        "attempts": _attempt_count(old_evidence.get("attempts"))
        + _attempt_count(new_evidence.get("attempts")),
    }
    for key in ("solved_seeds", "failed_seeds", "contradicted_by"):
        if key in old_evidence or key in new_evidence:
            evidence[key] = sorted(
                {
                    *_evidence_items(old_evidence, key),
                    *_evidence_items(new_evidence, key),
                },
                key=str,
            )
    tasks = {str(cell).rsplit("_s", 1)[0] for cell in cells}
    confidence = (
        "verified"
        if len(cells) >= 3 and len(tasks) >= 2
        else "probable"
        if len(cells) >= 2
        else "single-shot"
    )
    return {**old, "evidence": evidence, "confidence": confidence}


def _has_local_memory(root: Path) -> bool:
    return root.is_dir() and any(path.is_file() for path in root.rglob("*"))


class MemoryManager:
    """Manage merge, validation, and synchronization for one memory corpus.

    root is the corpus root. It is stored resolved and created lazily by
    mutating operations.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        memory_access: str = "read_only",
        inbox_cell_tag: str | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._memory_access = memory_access
        self._inbox_cell_tag = inbox_cell_tag
        #: Thread that holds :meth:`merge_lock` through this manager, if any.
        self._merge_lock_owner: int | None = None

    @property
    def root(self) -> Path:
        """Resolved corpus root."""
        return self._root

    @contextmanager
    def merge_lock(self) -> Iterator[None]:
        """Hold the corpus-wide merge lock, ``_internal/merge.lock``.

        An exclusive ``flock`` serializes merges from every process and thread
        on this host that publishes into the corpus. Nested use by the thread
        that holds it through this manager re-enters it, so a subclass can wrap
        :meth:`merge_memory` in one critical section.
        """
        if self._merge_lock_owner == threading.get_ident():
            yield
            return
        lock_path = self._root / "_internal" / "merge.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            self._merge_lock_owner = threading.get_ident()
            try:
                yield
            finally:
                self._merge_lock_owner = None
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def check_layout(self) -> None:
        """Reject an unmigrated corpus instead of silently omitting its layers.

        A sync may leave old cached directories beside their replacements.
        Only the current directories are exposed by the memory file tools.
        """
        for old, current in (("task_only", "task-specific"), ("suite", "task-family")):
            if (self._root / old).exists() and not (self._root / current).is_dir():
                raise ValueError(
                    f"legacy memory directory {self._root / old}: download the "
                    f"updated corpus with {current}/ into a fresh directory"
                )

    def get_common_tool_bindings(
        self,
    ) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
        """Return memory-aware bindings for shared file tools."""
        from functools import partial

        from rpent.memory import tools as memory_tools
        from rpent.tools import common

        self.check_layout()
        handlers = {
            "read_text_file": partial(
                memory_tools.read_text_file,
                memory_root=self._root,
                memory_access=self._memory_access,
                cell_tag=self._inbox_cell_tag,
            ),
            "write_text_file": partial(
                memory_tools.write_text_file,
                memory_root=self._root,
                memory_access=self._memory_access,
                cell_tag=self._inbox_cell_tag,
            ),
            "list_dir": partial(
                memory_tools.list_dir,
                memory_root=self._root,
                memory_access=self._memory_access,
                cell_tag=self._inbox_cell_tag,
            ),
        }
        bindings: dict[str, tuple[dict[str, Any], Callable[..., Any]]] = {}
        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            handler = handlers.get(name)
            if handler is None:
                continue
            tool_spec = dict(spec)
            tool_spec["description"] += memory_tools.MEMORY_BOUNDARY_NOTE
            bindings[name] = (tool_spec, handler)
        return bindings

    def merge_memory(
        self,
        *,
        cell_tag: str,
        run_state_dir: str | Path,
        solved: bool,
        replace: bool | None = None,
    ) -> dict[str, Any]:
        """Publish one cell's inbox drafts and solved task artifacts.

        Drafts merge into global/task-family; conflicting prose is
        archived under _internal/conflicts. Solved audit/recipe pairs are
        copied to task-specific/. MEMORY.md is refreshed at the end.

        In replace mode (``replace=True``; by default
        ``RPENT_MEMORY_MERGE_REPLACE=1``), a solved cell's own task-family
        write-up and audit/recipe pair replace the ones already published for
        it (the previous files move to _internal/replaced), so an explorer
        seeded with another model's corpus ends up with its own task memory.
        Global notes merge as usual.
        """
        self.check_layout()
        root = self._root
        run_dir = Path(run_state_dir).resolve()
        internal = root / "_internal"
        inbox = internal / "inbox" / cell_tag
        conflicts = internal / "conflicts"
        merged = internal / "merged"
        replaced = internal / "replaced"
        if replace is None:
            replace = os.environ.get("RPENT_MEMORY_MERGE_REPLACE") == "1"
        replace = solved and replace
        tiers = {
            "global": root / "global",
            "task-family": root / "task-family",
            "task": root / "task-specific",
        }
        for directory in (*tiers.values(), conflicts, merged, inbox.parent):
            directory.mkdir(parents=True, exist_ok=True)

        result: dict[str, Any] = {
            "cell": cell_tag,
            "global": 0,
            "task-family": 0,
            "task": 0,
            "evidence": 0,
            "conflicts": 0,
            "skipped": [],
        }
        if replace:
            result["replaced"] = 0
        with self.merge_lock():
            published_draft = False
            for source in sorted(inbox.glob("*.md")):
                try:
                    metadata, body = _split_frontmatter(source)
                    _validate(metadata)
                except ValueError as exc:
                    result["skipped"].append(f"{source.name}: {exc}")
                    continue
                memory_id = _canonical_id(source, metadata)
                metadata["id"] = memory_id
                destination = tiers[metadata["scope"]] / f"{memory_id}.md"
                published_draft = True
                if not destination.exists():
                    destination.write_text(_render(metadata, body))
                    result[metadata["scope"]] += 1
                    continue
                try:
                    old_metadata, old_body = _split_frontmatter(destination)
                except ValueError:
                    # Archive the incoming draft as a conflict.
                    conflict = conflicts / f"{memory_id}__from_{cell_tag}.md"
                    conflict.write_text(_render(metadata, body))
                    result["conflicts"] += 1
                    result["skipped"].append(
                        f"{source.name}: existing non-mergeable note "
                        f"{destination.name} left untouched"
                    )
                    continue
                if replace and metadata["scope"] == "task-family":
                    replaced.mkdir(parents=True, exist_ok=True)
                    shutil.move(
                        str(destination),
                        str(replaced / f"{memory_id}__before_{cell_tag}.md"),
                    )
                    destination.write_text(_render(metadata, body))
                    result["replaced"] += 1
                    continue
                credited = cell_tag in (
                    (old_metadata.get("evidence") or {}).get("cells") or []
                )
                if credited:
                    continue
                merged_metadata = _merge_evidence(old_metadata, metadata)
                destination.write_text(_render(merged_metadata, old_body))
                result["evidence"] += 1
                if body.strip() != old_body.strip():
                    conflict = conflicts / f"{memory_id}__from_{cell_tag}.md"
                    conflict.write_text(_render(metadata, body))
                    result["conflicts"] += 1

            if published_draft:
                archive = merged / cell_tag
                if archive.exists():
                    shutil.rmtree(archive)
                shutil.move(str(inbox), str(archive))

            audit = run_dir / f"{cell_tag}.json"
            recipe = run_dir / f"{cell_tag}_recipe.jsonl"
            if solved and audit.exists() and recipe.exists():
                audit_target = tiers["task"] / audit.name
                recipe_target = tiers["task"] / recipe.name
                if replace and audit_target.exists() and recipe_target.exists():
                    archive = replaced / cell_tag
                    if archive.exists():
                        shutil.rmtree(archive)
                    archive.mkdir(parents=True)
                    shutil.move(str(audit_target), str(archive / audit.name))
                    shutil.move(str(recipe_target), str(archive / recipe.name))
                    result["replaced"] += 1
                if not audit_target.exists() and not recipe_target.exists():
                    shutil.copy2(audit, audit_target)
                    shutil.copy2(recipe, recipe_target)
                    result["task"] = 1
                elif audit_target.exists() != recipe_target.exists():
                    result["skipped"].append(
                        "incomplete existing task audit/recipe pair"
                    )

            self.rebuild_index()
        return result

    def rebuild_index(self) -> Path | None:
        """Regenerate MEMORY.md from global and task-family leaves.

        Leaves without parseable frontmatter are skipped individually; the
        index is rebuilt from the rest. Returns None and writes nothing when
        there are no leaves with parseable frontmatter.
        """
        self.check_layout()
        root = self._root
        groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
            "global": [],
            "task-family": [],
        }
        for scope in groups:
            for path in sorted((root / scope).glob("*.md")):
                text = path.read_text(errors="replace")
                if not text.startswith("---"):
                    continue
                try:
                    metadata, _ = _split_frontmatter(path)
                except ValueError:
                    continue
                groups[scope].append((path.name, metadata))
        index = root / "MEMORY.md"
        if not any(groups.values()):
            return None
        lines = [
            "# Layered memory index",
            "",
            "Generated from memory leaf frontmatter.",
        ]
        for scope, title in (("global", "Global"), ("task-family", "Task-family")):
            lines.extend(("", f"## {title}", ""))
            if not groups[scope]:
                lines.append("_(none)_")
                continue
            for filename, metadata in groups[scope]:
                label = metadata.get("title") or metadata.get("id") or filename
                applies = metadata.get("applies_when") or ""
                suffix = f" — {applies}" if applies else ""
                lines.append(f"- [{label}]({scope}/{filename}){suffix}")
        index = root / "MEMORY.md"
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text("\n".join(lines) + "\n")
        return index

    def validate(self) -> list[str]:
        """Return validation problems for global and task-family leaves.

        Only frontmatter-bearing leaves are schema-validated; plain Markdown
        is allowed silently.
        """
        try:
            self.check_layout()
        except ValueError as exc:
            return [str(exc)]
        root = self._root
        problems: list[str] = []
        ids: dict[str, Path] = {}
        for scope in ("global", "task-family"):
            for path in sorted((root / scope).glob("*.md")):
                text = path.read_text(errors="replace")
                if not text.startswith("---"):
                    continue
                try:
                    metadata, _ = _split_frontmatter(path)
                    _validate(metadata)
                except ValueError as exc:
                    problems.append(f"{path.relative_to(root)}: {exc}")
                    continue
                memory_id = str(metadata.get("id") or "")
                if memory_id != path.stem:
                    problems.append(
                        f"{path.relative_to(root)}: id {memory_id!r} does not match filename"
                    )
                if memory_id in ids:
                    problems.append(
                        f"{path.relative_to(root)}: duplicate id also in "
                        f"{ids[memory_id].relative_to(root)}"
                    )
                ids[memory_id] = path
        return problems

    def sync(
        self,
        *,
        remote_repo: str,
        allow_patterns: tuple[str, ...] | None = None,
    ) -> Path:
        """Sync this memory corpus from its Hugging Face dataset."""
        robot_name = self._root.name
        repo_id = os.environ.get(
            "RPENT_MEMORY_HF_REPO",
            remote_repo,
        )

        if os.environ.get("HF_HUB_OFFLINE") == "1":
            if not _has_local_memory(self._root):
                logger.warning(
                    "HF_HUB_OFFLINE=1 but no local memory was found under %s",
                    self._root,
                )
            return self._root

        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(self._root.parent),
                allow_patterns=list(allow_patterns or (f"{robot_name}/**",)),
            )
        except Exception as exc:
            if _has_local_memory(self._root):
                logger.warning(
                    "could not sync '%s' from '%s': %s; "
                    "continuing with local memory under %s",
                    robot_name,
                    repo_id,
                    exc,
                    self._root,
                )
            else:
                logger.warning(
                    "could not sync '%s' from '%s': %s; "
                    "no local memory was found under %s",
                    robot_name,
                    repo_id,
                    exc,
                    self._root,
                )

        return self._root


__all__ = ["MemoryManager"]
