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

"""RoboCasa task and global memory, shared by prompts and file tools."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.evaluation import write_json_atomic
from rpent.memory import MemoryManager
from rpent.memory.manager import _split_frontmatter, _validate
from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("robocasa_memory")
GLOBAL_FILE = "global/GLOBAL_MEMORY.md"
_TASK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*\Z")
_SPLIT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*\Z")


def task_files(task_name: str) -> tuple[str, str, str]:
    """Return canonical filenames for one task, without directory traversal."""
    if not _TASK_NAME.fullmatch(task_name):
        raise ValueError(f"invalid RoboCasa task name: {task_name!r}")
    return (
        f"task-specific/{task_name}_s0.json",
        f"task-specific/{task_name}_s0_recipe.jsonl",
        f"task-specific/{task_name}.md",
    )


def local_task_files(task_name: str, split: str) -> tuple[str, str, str]:
    """Return the native exploration filenames for the current task and split."""
    published = task_files(task_name)
    if not _SPLIT_NAME.fullmatch(split):
        raise ValueError(f"invalid RoboCasa split: {split!r}")
    return (
        f"task-specific/{task_name}_{split}_s0.json",
        f"task-specific/{task_name}_{split}_s0_recipe.jsonl",
        published[2],
    )


def _layer_file(name: object, directory: str) -> bool:
    if not isinstance(name, str):
        return False
    parts = name.split("/")
    return len(parts) == 2 and parts[0] == directory and parts[1].endswith(".md")


def selection_errors(
    memory: Mapping[str, Any], task_name: str, split: str
) -> list[str]:
    """Check recorded evaluation selection against task/profile boundaries."""
    profile = memory.get("profile", "hf")
    selected = memory.get("selected_files")
    missing = memory.get("missing_layers")
    families = memory.get("task_family", {})
    if (
        not isinstance(profile, str)
        or profile not in {"hf", "local"}
        or not isinstance(selected, list)
        or not all(isinstance(name, str) for name in selected)
        or len(selected) != len(set(selected))
        or not isinstance(missing, list)
        or not isinstance(families, dict)
    ):
        return ["invalid memory selection metadata"]
    published = task_files(task_name)
    native = local_task_files(task_name, split)
    wanted = published
    if profile == "local" and any(name in selected for name in native[:2]):
        wanted = native
    errors = []
    if any(
        (pair[0] in selected) != (pair[1] in selected) for pair in (published, native)
    ):
        errors.append("incomplete task audit/recipe pair")
    if published[0] in selected and native[0] in selected:
        errors.append("both published and native task pairs are selected")
    globals_ = [name for name in selected if _layer_file(name, "global")]
    if profile == "hf":
        if globals_ != [GLOBAL_FILE] or families:
            errors.append("HF evaluation requires its fixed global file")
    elif not globals_:
        errors.append("local evaluation requires global memory")
    identity = {"suite": "robocasa", "regime": split, "task_id": task_name}
    if any(
        not _layer_file(name, "task-family")
        or context != identity
        or name not in selected
        for name, context in families.items()
    ):
        errors.append("task-family identity does not match the current task/split")
    allowed = {*wanted, *globals_, *families}
    if any(name not in allowed for name in selected):
        errors.append("selected memory files violate the task/global boundary")
    if missing != [name for name in wanted if name not in selected]:
        errors.append("missing memory layers do not match the selected task files")
    return errors


@dataclass(frozen=True)
class TaskMemory:
    """The files available to one task for the duration of a run."""

    root: Path
    contents: dict[str, str]
    selected: tuple[str, ...]
    missing: tuple[str, ...]
    profile: str
    task_family: dict[str, dict[str, str]]

    @classmethod
    def load(
        cls,
        root: str | Path,
        task_name: str | None,
        *,
        profile: str = "hf",
        split: str = "target",
    ) -> TaskMemory:
        """Select conventional task/global paths without a dataset manifest.

        With no task yet, validate only the session's root and global
        layer before starting shared services. Task runs must supply their name.
        """
        if profile not in {"hf", "local"}:
            raise ValueError(f"unsupported RoboCasa memory profile: {profile}")
        root = Path(root).expanduser().resolve()
        wanted = task_files(task_name) if task_name is not None else ()
        if not root.is_dir():
            raise ValueError(f"RoboCasa memory directory not found: {root}")
        MemoryManager(root).check_layout()
        pairs = [wanted] if wanted else []
        if wanted and profile == "local":
            pairs.append(local_task_files(task_name, split))
        present_pairs = []
        for pair in pairs:
            present = [
                (root / name).exists() or (root / name).is_symlink()
                for name in pair[:2]
            ]
            if present[0] != present[1]:
                raise ValueError(f"incomplete seed-0 JSON/JSONL pair for {pair[0]}")
            if present[0]:
                present_pairs.append(pair)
        if len(present_pairs) > 1:
            raise ValueError(
                "both published and native task pairs exist; use separate --memory-dir directories"
            )
        if present_pairs:
            wanted = present_pairs[0]

        def read(name: str) -> str:
            path = root / name
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"memory file escapes memory root: {name}")
            try:
                return path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise ValueError(f"cannot read memory file: {name}") from exc

        contents = {
            name: read(name)
            for name in wanted
            if (root / name).exists() or (root / name).is_symlink()
        }
        families = {}
        if profile == "local" and task_name is not None:
            identity = {"suite": "robocasa", "regime": split, "task_id": task_name}
            for path in sorted((root / "task-family").glob("*.md")):
                name = path.relative_to(root).as_posix()
                content = read(name)
                metadata, _ = _split_frontmatter(path)
                if all(metadata.get(key) == value for key, value in identity.items()):
                    _validate(metadata)
                    if metadata["scope"] != "task-family":
                        raise ValueError(f"invalid task-family scope: {name}")
                    contents[name] = content
                    families[name] = identity.copy()
        global_files = (
            [
                path.relative_to(root).as_posix()
                for path in sorted((root / "global").glob("*.md"))
            ]
            if profile == "local"
            else [GLOBAL_FILE]
            if (root / GLOBAL_FILE).exists()
            else []
        )
        if not global_files:
            required = "global/*.md" if profile == "local" else GLOBAL_FILE
            raise ValueError(f"task-global requires {root / required}")
        contents.update({name: read(name) for name in global_files})
        selected = tuple(contents)
        missing = tuple(name for name in wanted if name not in contents)
        for name in missing:
            logger.warning("memory layer not provided: %s", name)
        return cls(root, contents, selected, missing, profile, families)

    def metadata(self) -> dict[str, Any]:
        """Record the selected layers without pinning a data version."""
        return {
            "policy": "task-global",
            "selected_files": list(self.selected),
            "missing_layers": list(self.missing),
            "profile": self.profile,
            "task_family": self.task_family,
        }


def memory_from_variables(variables: Mapping[str, object]) -> TaskMemory:
    """Resolve the same memory selection for prompt and toolkit construction."""
    return TaskMemory.load(
        str(variables["memory_dir"]),
        str(variables["task_name"]),
        profile=str(variables.get("memory_profile") or "hf"),
        split=str(variables.get("split", "target")),
    )


class RoboCasaMemoryManager(MemoryManager):
    """Restrict RPent file tools to the current task and global layer.

    This is a file-tool boundary, not an OS sandbox. Non-memory observations
    remain accessible through the shared tools' normal access rules.
    """

    def __init__(self, memory: TaskMemory, *, output_dir: Path | None = None):
        super().__init__(root=memory.root)
        self.selection = memory
        self._read_files: set[str] = set()
        self._output_dir = output_dir
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            # A reused output directory starts a new audit for this run.
            (output_dir / "memory_reads.jsonl").write_text("", encoding="utf-8")
            write_json_atomic(output_dir / "memory.json", memory.metadata())

    @property
    def unread_files(self) -> tuple[str, ...]:
        """Selected files not yet read completely through RPent's file tool."""
        return tuple(
            name for name in self.selection.selected if name not in self._read_files
        )

    def _relative_path(self, path: str) -> str | None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
        resolved = candidate.resolve()
        if candidate.is_relative_to(self.root) and not resolved.is_relative_to(
            self.root
        ):
            raise PermissionError(f"memory path escapes corpus root: {path}")
        if resolved.is_relative_to(self.root):
            return resolved.relative_to(self.root).as_posix()
        return self._foreign_selected_path(path)

    def _foreign_selected_path(self, path: str) -> str | None:
        """Selected file named by a Windows-style rendering of its path, if any.

        GPT-6 Astra behind some Codex relays rewrites the listed POSIX paths as
        ``C:\\Users\\<user>\\...\\<corpus>\\task-specific\\<Task>.md`` or
        ``/c/Users/<user>/...`` although Codex reports a Linux cwd and bash, so
        every memory read fails. A missing path that ends with the corpus
        folder and a selected file's relative path reads that file.
        """
        if Path(path).exists():
            return None
        normalized = path.replace("\\", "/")
        for name in self.selection.selected:
            if normalized.endswith(f"/{self.root.name}/{name}"):
                return name
        return None

    def get_common_tool_bindings(self):
        """Wrap shared read/list tools; keep existing read-only write policy."""
        bindings = super().get_common_tool_bindings()
        shared_read = bindings["read_text_file"][1]
        shared_list = bindings["list_dir"][1]

        @readonly
        def read_text_file(path: str, max_chars: int = 40000) -> dict:
            relative = self._relative_path(path)
            if relative is not None:
                if relative not in self.selection.selected:
                    raise PermissionError(
                        f"memory is outside current task/global selection: {path}"
                    )
                if (self.root / relative).read_text(
                    encoding="utf-8"
                ) != self.selection.contents[relative]:
                    raise ValueError(f"memory changed during this run: {relative}")
            if relative is not None:
                path = str(self.root / relative)
            result = shared_read(path=path, max_chars=max_chars)
            if relative is not None and isinstance(result.get("content"), str):
                content = result["content"]
                complete = content == self.selection.contents[relative]
                if complete:
                    self._read_files.add(relative)
                if self._output_dir is not None:
                    event = {
                        "path": relative,
                        "complete": complete,
                    }
                    with (self._output_dir / "memory_reads.jsonl").open("a") as handle:
                        handle.write(json.dumps(event) + "\n")
            return result

        @readonly
        def list_dir(path: str = "") -> dict:
            relative = self._relative_path(path) if path else None
            if relative is None:
                return shared_list(path=path)
            prefix = "" if relative == "." else relative.rstrip("/") + "/"
            entries = sorted(
                {
                    name.removeprefix(prefix).split("/")[0]
                    for name in self.selection.selected
                    if name.startswith(prefix)
                }
            )
            if not entries and relative != ".":
                raise PermissionError(
                    f"memory directory is outside current task/global selection: {path}"
                )
            return {
                "path": str((self.root / relative).resolve()),
                "count": len(entries),
                "files": entries,
            }

        for name, handler in (
            ("read_text_file", read_text_file),
            ("list_dir", list_dir),
        ):
            spec = dict(bindings[name][0])
            spec["description"] += (
                " RoboCasa memory is limited to the current task and global memory."
            )
            bindings[name] = (spec, handler)
        return bindings


def _replace_from_env() -> bool:
    return os.environ.get("RPENT_MEMORY_MERGE_REPLACE") == "1"


class RoboCasaExplorationMemoryManager(MemoryManager):
    """Exploration corpus manager whose replace mode follows RoboCasa names.

    A RoboCasa exploration cell ``<Task>_<split>_s0`` writes its audit and
    recipe as ``task-specific/<Task>_<split>_s0.*`` and its task note as a
    task-family leaf. A published corpus names the same task's references
    ``task-specific/<Task>_s0.*`` and keeps its task write-up in
    ``task-specific/<Task>.md``; the local loader refuses a task that has
    both pairs. Without replace mode, merges are the base merge.

    In replace mode (``RPENT_MEMORY_MERGE_REPLACE=1`` or ``replace=True``)
    only the seed-0 cell merges, and a task's memory comes from one side:

    - A solved cell with a non-empty audit and a recipe replaces the task's
      published files: they move to ``_internal/replaced/<Task>_s0/``, and
      the cell's pair and its task-family leaf for this task and split are
      published.
    - Any other cell (unsolved, or solved without its audit or recipe) keeps
      the published files and publishes no task-family leaf.

    Task-family drafts that are not published move to
    ``_internal/held/<cell>/``. Global drafts merge by the base rules either
    way. The merge result adds ``task_memory`` (``"replaced"`` or ``"kept"``),
    ``kept_reason``, ``archived``, ``held`` and ``selected``: the task's
    local-profile selection after the merge.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        task_name: str,
        split: str,
        memory_access: str = "read_only",
        inbox_cell_tag: str | None = None,
    ) -> None:
        super().__init__(
            root, memory_access=memory_access, inbox_cell_tag=inbox_cell_tag
        )
        self._task_name = task_name
        self._split = split
        self._published = task_files(task_name)
        self._native = local_task_files(task_name, split)
        if (
            inbox_cell_tag is not None
            and inbox_cell_tag != self.cell_tag
            and _replace_from_env()
        ):
            raise ValueError(
                "RPENT_MEMORY_MERGE_REPLACE=1 publishes seed-0 references; "
                f"explore {task_name} with --seed 0, not as {inbox_cell_tag}"
            )

    @property
    def cell_tag(self) -> str:
        """The seed-0 exploration cell whose files the loader reads."""
        return Path(self._native[0]).stem

    def merge_memory(
        self,
        *,
        cell_tag: str,
        run_state_dir: str | Path,
        solved: bool,
        replace: bool | None = None,
    ) -> dict[str, Any]:
        """Merge one cell; in replace mode, publish one side's task memory."""
        if replace is None:
            replace = _replace_from_env()
        if not replace:
            return super().merge_memory(
                cell_tag=cell_tag,
                run_state_dir=run_state_dir,
                solved=solved,
                replace=False,
            )
        if cell_tag != self.cell_tag:
            raise ValueError(
                f"replace mode merges only the seed-0 cell {self.cell_tag}, "
                f"not {cell_tag}"
            )
        self.check_layout()
        run_dir = Path(run_state_dir).resolve()
        audit = run_dir / f"{cell_tag}.json"
        recipe = run_dir / f"{cell_tag}_recipe.jsonl"
        if not solved:
            kept = "unsolved"
        elif not audit.is_file() or audit.stat().st_size == 0:
            kept = f"solved, but the explorer wrote no audit {audit.name}"
        elif not recipe.is_file():
            kept = f"solved, but no recipe {recipe.name} was exported"
        else:
            kept = None
        with self.merge_lock():
            held = self._hold_task_family_drafts(cell_tag, kept)
            result = super().merge_memory(
                cell_tag=cell_tag,
                run_state_dir=run_dir,
                solved=kept is None,
                replace=True,
            )
            result["skipped"].extend(f"{name}: held, {why}" for name, why in held)
            archived: list[str] = []
            if kept is None and all(
                (self.root / name).is_file() for name in self._native[:2]
            ):
                archived = self._archive_published()
            # Like the base count: one for the audit/recipe pair, one per note.
            notes = sum(name.endswith(".md") for name in archived)
            result["replaced"] = (
                result.get("replaced", 0) + notes + (len(archived) > notes)
            )
            result.update(
                task_memory="kept" if kept else "replaced",
                kept_reason=kept,
                archived=archived,
                held=[name for name, _ in held],
                selected=list(
                    TaskMemory.load(
                        self.root,
                        self._task_name,
                        profile="local",
                        split=self._split,
                    ).selected
                ),
            )
        return result

    def _hold_task_family_drafts(
        self, cell_tag: str, kept: str | None
    ) -> list[tuple[str, str]]:
        """Move task-family drafts that must not publish out of the inbox."""
        inbox = self.root / "_internal" / "inbox" / cell_tag
        identity = {
            "suite": "robocasa",
            "regime": self._split,
            "task_id": self._task_name,
        }
        held = []
        for source in sorted(inbox.glob("*.md")):
            try:
                metadata, _ = _split_frontmatter(source)
                _validate(metadata)
            except ValueError:
                continue  # the base merge reports invalid drafts
            if metadata["scope"] != "task-family":
                continue
            if kept is not None:
                why = f"task memory kept ({kept})"
            elif any(metadata.get(key) != value for key, value in identity.items()):
                why = f"not a task-family note for {self._task_name}/{self._split}"
            else:
                continue
            destination = self.root / "_internal" / "held" / cell_tag
            destination.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination / source.name))
            held.append((source.name, why))
        return held

    def _archive_published(self) -> list[str]:
        """Move the task's published pair and note to _internal/replaced/."""
        archive = Path("_internal") / "replaced" / Path(self._published[0]).stem
        archived = []
        for name in self._published:
            path = self.root / name
            if path.exists() or path.is_symlink():
                (self.root / archive).mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(self.root / archive / path.name))
                archived.append((archive / path.name).as_posix())
        return archived


def publish_exploration_cell(
    corpus: str | Path,
    *,
    task_name: str,
    split: str,
    run_dir: str | Path,
    solved: bool,
    inbox: str | Path | None = None,
) -> dict[str, Any]:
    """Publish one finished seed-0 exploration cell into ``corpus``, replacing.

    The inbox copy and the merge hold the corpus merge lock, so concurrent
    publishers are serialized.

    Args:
        corpus: Existing memory root to publish into.
        task_name: RoboCasa task of the cell.
        split: RoboCasa split of the cell.
        run_dir: The cell's output directory, which holds the explorer's audit
            and the exported recipe.
        solved: The cell's environment result (``success`` in its
            ``result.json``).
        inbox: When the cell explored against another memory root, such as a
            per-task staging copy run with ``--no-auto-merge-memory``, that
            root's ``_internal/inbox/<Task>_<split>_s0``; it is copied into
            ``corpus`` first.

    Returns:
        The merge result of :class:`RoboCasaExplorationMemoryManager`.

    Raises:
        ValueError: If ``corpus`` does not exist, or the task cannot be loaded
            after the merge.
    """
    manager = RoboCasaExplorationMemoryManager(corpus, task_name=task_name, split=split)
    if not manager.root.is_dir():
        raise ValueError(f"RoboCasa memory directory not found: {manager.root}")
    cell = manager.cell_tag
    with manager.merge_lock():
        if inbox is not None:
            source = Path(inbox).expanduser().resolve()
            target = manager.root / "_internal" / "inbox" / cell
            if source.is_dir() and source != target:
                shutil.copytree(source, target, dirs_exist_ok=True)
        return manager.merge_memory(
            cell_tag=cell, run_state_dir=run_dir, solved=solved, replace=True
        )
