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

"""Replace mode publishes one side's RoboCasa task memory per task.

Corpora are synthetic: published files come from ``make_corpus`` and the
exploration cell's audit, recipe and drafts are written by the test.
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.robocasa.memory import (
    RoboCasaExplorationMemoryManager,
    TaskMemory,
    local_task_files,
    publish_exploration_cell,
    selection_errors,
    task_files,
)
from rpent.memory import MemoryManager

TASK = "OpenDrawer"
OTHER = "StirVegetables"
CELL = f"{TASK}_target_s0"
FAMILY = f"task-family/task-family_robocasa_target_t{TASK}.md"


@pytest.fixture(autouse=True)
def _replace_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RPENT_MEMORY_MERGE_REPLACE", raising=False)


def _cell(
    memory_root: Path,
    run: Path,
    *,
    task: str = TASK,
    audit: str | None = '{"by": "explorer"}\n',
    recipe: bool = True,
    family_task: str | None = TASK,
    global_id: str = "global",
) -> Path:
    """Write one seed-0 cell: run-dir audit and recipe, inbox drafts and notes."""
    tag = f"{task}_target_s0"
    run.mkdir(parents=True, exist_ok=True)
    if audit is not None:
        (run / f"{tag}.json").write_text(audit)
    if recipe:
        (run / f"{tag}_recipe.jsonl").write_text('{"action": "rldx_arm"}\n')
    inbox = memory_root / "_internal" / "inbox" / tag
    (inbox / "wip").mkdir(parents=True, exist_ok=True)
    (inbox / "wip" / "notes.md").write_text("attempt 1: handle missed\n")
    evidence = (
        f"evidence:\n  cells: [{tag}]\n  attempts: 2\n  solved_seeds: [0]\n"
        "confidence: single-shot\n"
    )
    if family_task is not None:
        (inbox / "family.md").write_text(
            "---\nscope: task-family\nsuite: robocasa\nregime: target\n"
            f"task_id: {family_task}\ntask_language: Synthetic task\n"
            + evidence
            + "---\nThe explorer's own technique.\n"
        )
    (inbox / f"{global_id}.md").write_text(
        "---\nscope: global\nkind: strategy\ntitle: Explorer lesson\n"
        "applies_when: Offline test only\n" + evidence + "---\nA cross-task lesson.\n"
    )
    return run


def _snapshot(root: Path, task: str) -> dict[str, bytes]:
    return {
        name: (root / name).read_bytes()
        for name in task_files(task)
        if (root / name).exists()
    }


def _tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "merge.lock"
    }


def _manager(root: Path, task: str = TASK) -> RoboCasaExplorationMemoryManager:
    return RoboCasaExplorationMemoryManager(root, task_name=task, split="target")


def _selection(root: Path, task: str = TASK) -> set[str]:
    return set(TaskMemory.load(root, task, profile="local", split="target").selected)


def test_a_solved_cell_replaces_the_published_task_memory(
    tmp_path, make_corpus, monkeypatch
):
    monkeypatch.setenv("RPENT_MEMORY_MERGE_REPLACE", "1")
    root = make_corpus(tmp_path / "corpus")
    published = _snapshot(root, TASK)
    other = _snapshot(root, OTHER)
    run = _cell(root, tmp_path / "run")

    result = _manager(root).merge_memory(cell_tag=CELL, run_state_dir=run, solved=True)

    native = local_task_files(TASK, "target")
    assert result["task_memory"] == "replaced" and result["kept_reason"] is None
    assert (result["task"], result["task-family"], result["global"]) == (1, 1, 1)
    assert result["replaced"] == 2  # the published pair and the task note
    archive = root / "_internal" / "replaced" / f"{TASK}_s0"
    assert result["archived"] == [
        f"_internal/replaced/{TASK}_s0/{Path(name).name}" for name in published
    ]
    assert {path.name: path.read_bytes() for path in archive.iterdir()} == {
        Path(name).name: data for name, data in published.items()
    }
    assert not any((root / name).exists() for name in task_files(TASK))
    selection = TaskMemory.load(root, TASK, profile="local", split="target")
    assert set(selection.selected) == {
        *native[:2],
        FAMILY,
        "global/GLOBAL_MEMORY.md",
        "global/global.md",
    }
    assert result["selected"] == list(selection.selected)
    assert selection.contents[native[0]] == '{"by": "explorer"}\n'
    assert selection.contents[FAMILY].endswith("The explorer's own technique.\n")
    assert selection_errors(selection.metadata(), TASK, "target") == []
    # Another task keeps its published memory and gains only the global note.
    assert _snapshot(root, OTHER) == other
    assert _selection(root, OTHER) == {
        *task_files(OTHER),
        "global/GLOBAL_MEMORY.md",
        "global/global.md",
    }


def test_an_unsolved_cell_keeps_the_published_task_memory(tmp_path, make_corpus):
    root = make_corpus(tmp_path / "corpus")
    published = _snapshot(root, TASK)
    run = _cell(root, tmp_path / "run")

    result = _manager(root).merge_memory(
        cell_tag=CELL, run_state_dir=run, solved=False, replace=True
    )

    assert result["task_memory"] == "kept" and result["kept_reason"] == "unsolved"
    assert result["replaced"] == 0 and result["archived"] == []
    assert _snapshot(root, TASK) == published
    native_pair = local_task_files(TASK, "target")[:2]
    assert not any((root / name).exists() for name in native_pair)
    # Its task note is held back; its global note merges by the default rules.
    assert result["held"] == ["family.md"]
    assert (root / "_internal" / "held" / CELL / "family.md").is_file()
    assert not (root / FAMILY).exists()
    assert result["global"] == 1
    assert _selection(root) == {
        *task_files(TASK),
        "global/GLOBAL_MEMORY.md",
        "global/global.md",
    }
    assert not (root / "_internal" / "replaced").exists()


@pytest.mark.parametrize(
    "audit, recipe, reason",
    [
        (None, True, f"no audit {CELL}.json"),
        ("", True, f"no audit {CELL}.json"),
        ('{"by": "explorer"}\n', False, f"no recipe {CELL}_recipe.jsonl"),
    ],
    ids=["missing-audit", "empty-audit", "missing-recipe"],
)
def test_a_solved_cell_without_its_audit_or_recipe_keeps_the_published_memory(
    tmp_path, make_corpus, audit, recipe, reason
):
    root = make_corpus(tmp_path / "corpus")
    published = _snapshot(root, TASK)
    run = _cell(root, tmp_path / "run", audit=audit, recipe=recipe)

    result = _manager(root).merge_memory(
        cell_tag=CELL, run_state_dir=run, solved=True, replace=True
    )

    assert result["task_memory"] == "kept"
    assert result["kept_reason"].startswith("solved, but")
    assert reason in result["kept_reason"]
    assert result["task"] == 0 and result["held"] == ["family.md"]
    assert _snapshot(root, TASK) == published
    assert _selection(root) == {
        *task_files(TASK),
        "global/GLOBAL_MEMORY.md",
        "global/global.md",
    }


def test_a_task_note_for_another_task_is_not_published(tmp_path, make_corpus):
    root = make_corpus(tmp_path / "corpus")
    run = _cell(root, tmp_path / "run", family_task=OTHER)

    result = _manager(root).merge_memory(
        cell_tag=CELL, run_state_dir=run, solved=True, replace=True
    )

    assert result["task_memory"] == "replaced"
    assert result["held"] == ["family.md"]
    assert not list((root / "task-family").glob("*.md"))
    assert f"task-family/task-family_robocasa_target_t{OTHER}.md" not in _selection(
        root, OTHER
    )


def test_without_replace_mode_the_merge_is_the_base_merge(tmp_path, make_corpus):
    roots, results = [], []
    for name in ("base", "robocasa"):
        root = make_corpus(tmp_path / name)
        run = _cell(root, tmp_path / f"run-{name}")
        manager = MemoryManager(root) if name == "base" else _manager(root)
        results.append(
            manager.merge_memory(cell_tag=CELL, run_state_dir=run, solved=True)
        )
        roots.append(root)

    assert results[0] == results[1]
    assert _tree(roots[0]) == _tree(roots[1])
    # Both pairs now exist, which is why replace mode is needed.
    with pytest.raises(ValueError, match="both published and native"):
        TaskMemory.load(roots[1], TASK, profile="local")


def test_replace_mode_publishes_only_seed_zero(tmp_path, make_corpus, monkeypatch):
    root = make_corpus(tmp_path / "corpus")
    before = _tree(root)
    monkeypatch.setenv("RPENT_MEMORY_MERGE_REPLACE", "1")

    with pytest.raises(ValueError, match="--seed 0"):
        RoboCasaExplorationMemoryManager(
            root,
            task_name=TASK,
            split="target",
            memory_access="inbox_write",
            inbox_cell_tag=f"{TASK}_target_s3",
        )
    with pytest.raises(ValueError, match="seed-0 cell"):
        _manager(root).merge_memory(
            cell_tag=f"{TASK}_target_s3", run_state_dir=tmp_path / "run", solved=True
        )
    assert _tree(root) == before


def test_republishing_a_cell_keeps_one_pair_and_the_published_archive(
    tmp_path, make_corpus
):
    corpus = make_corpus(tmp_path / "corpus")
    staging = make_corpus(tmp_path / "staging", tasks=(TASK,))
    published = _snapshot(corpus, TASK)
    run = _cell(staging, tmp_path / "run")
    inbox = staging / "_internal" / "inbox" / CELL

    first = publish_exploration_cell(
        corpus, task_name=TASK, split="target", run_dir=run, solved=True, inbox=inbox
    )
    after_first = _tree(corpus)
    second = publish_exploration_cell(
        corpus, task_name=TASK, split="target", run_dir=run, solved=True, inbox=inbox
    )

    assert first["task_memory"] == second["task_memory"] == "replaced"
    assert first["selected"] == second["selected"]
    for name in (*local_task_files(TASK, "target")[:2], FAMILY, "global/global.md"):
        assert _tree(corpus)[name] == after_first[name]
    archive = corpus / "_internal" / "replaced" / f"{TASK}_s0"
    assert {path.name: path.read_bytes() for path in archive.iterdir()} == {
        Path(name).name: data for name, data in published.items()
    }
    # The staging copy only lent its inbox; its files are unchanged.
    assert (inbox / "family.md").is_file()
    assert _snapshot(staging, TASK) == published
    # The merged inbox keeps the handoff notes.
    assert (corpus / "_internal" / "merged" / CELL / "wip" / "notes.md").is_file()


def test_publish_rejects_a_missing_corpus(tmp_path):
    with pytest.raises(ValueError, match="memory directory not found"):
        publish_exploration_cell(
            tmp_path / "missing",
            task_name=TASK,
            split="target",
            run_dir=tmp_path / "run",
            solved=True,
        )
    assert not (tmp_path / "missing").exists()


def test_cli_exploration_auto_merge_replaces_in_replace_mode(
    tmp_path, make_corpus, monkeypatch
):
    """The exploration toolkit's memory is what ``rpent.cli.main`` merges."""
    from robots.robocasa import robot_spec, toolkit
    from robots.robocasa.robot_spec import _parse_config
    from rpent.dashboard.events import NullDashboardEventSink

    monkeypatch.setenv("RPENT_MEMORY_MERGE_REPLACE", "1")
    root = make_corpus(tmp_path / "corpus")
    args = argparse.Namespace(
        task_name=TASK,
        split="target",
        seed=0,
        output_dir=tmp_path / "run",
        memory_dir=root,
        memory_profile=None,
        explore=True,
        explore_sessions=5,
        explore_attempts_per_session=3,
    )
    config = _parse_config(args)
    monkeypatch.setattr(
        toolkit, "RoboCasaToolkit", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    memory = robot_spec.get_toolkit(
        runtime_kwargs={},
        dashboard_events=NullDashboardEventSink(),
        config=config,
        mode="exploration",
        attempts_per_session=3,
    ).memory
    assert isinstance(memory, RoboCasaExplorationMemoryManager)
    write = memory.get_common_tool_bindings()["write_text_file"][1]
    _cell(root, config.output_dir, family_task=None)  # audit, recipe, global note
    write(
        path=config.prompt_vars["memory_inbox"] + "/family.md", content=_family_note()
    )

    result = memory.merge_memory(
        cell_tag=config.recipe_tag, run_state_dir=config.output_dir, solved=True
    )

    assert result["task_memory"] == "replaced"
    assert _selection(root) == {
        *local_task_files(TASK, "target")[:2],
        FAMILY,
        "global/GLOBAL_MEMORY.md",
        "global/global.md",
    }


def _family_note() -> str:
    return (
        "---\nscope: task-family\nsuite: robocasa\nregime: target\n"
        f"task_id: {TASK}\ntask_language: Synthetic task\n"
        f"evidence:\n  cells: [{CELL}]\nconfidence: single-shot\n"
        "---\nWritten through the exploration file tool.\n"
    )


TASKS = (TASK, OTHER, "CloseFridge", "TurnOnStove", "OpenCabinet", "CoffeeSetupMug")


def _publish_one(corpus: str, staging: str, run: str, task: str, solved: bool) -> None:
    tag = f"{task}_target_s0"
    publish_exploration_cell(
        corpus,
        task_name=task,
        split="target",
        run_dir=run,
        solved=solved,
        inbox=Path(staging) / "_internal" / "inbox" / tag,
    )


def test_concurrent_publishers_leave_every_task_loadable(tmp_path, make_corpus):
    corpus = make_corpus(tmp_path / "corpus", tasks=TASKS)
    jobs = []
    for index, task in enumerate(TASKS):
        staging = make_corpus(tmp_path / f"staging-{task}", tasks=(task,))
        run = _cell(
            staging,
            tmp_path / f"run-{task}",
            task=task,
            family_task=task,
            global_id=f"lesson_{task}",
        )
        jobs.append((str(corpus), str(staging), str(run), task, index % 2 == 0))
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_publish_one, args=job) for job in jobs]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
    assert [process.exitcode for process in processes] == [0] * len(TASKS)

    index = (corpus / "MEMORY.md").read_text()
    for _, _, _, task, solved in jobs:
        selected = _selection(corpus, task)
        family = f"task-family/task-family_robocasa_target_t{task}.md"
        if solved:
            assert set(local_task_files(task, "target")[:2]) <= selected
            assert family in selected and not set(task_files(task)) & selected
        else:
            assert set(task_files(task)) <= selected and family not in selected
        assert f"global/lesson_{task}.md" in selected
        assert f"(global/lesson_{task}.md)" in index
