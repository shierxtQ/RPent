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

"""RPENT_EXPLORE_LIST_MEMORY: the explore prompt names the task's published memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from robots.robocasa.prompts import explore
from robots.robocasa.robot_spec import _parse_config, get_robot_spec
from rpent.prompt.utils import format_prompt


def _corpus(root: Path) -> Path:
    (root / "global").mkdir(parents=True)
    (root / "task-specific").mkdir()
    (root / "global" / "GLOBAL_MEMORY.md").write_text("# global\n")
    for task in ("CloseFridge", "OpenDrawer"):
        (root / "task-specific" / f"{task}_s0.json").write_text("{}\n")
        (root / "task-specific" / f"{task}_s0_recipe.jsonl").write_text("{}\n")
        (root / "task-specific" / f"{task}.md").write_text(f"# {task}\n")
    return root


def _variables(tmp_path: Path, memory_dir: Path) -> dict:
    args = argparse.Namespace(
        task_name="CloseFridge",
        split="target",
        seed=0,
        output_dir=tmp_path / "run",
        memory_dir=memory_dir,
        memory_profile="local",
        explore=True,
        explore_sessions=5,
        explore_attempts_per_session=3,
    )
    config = _parse_config(args)
    return {**config.prompt_vars, "output_dir": str(config.output_dir)}


def test_off_by_default_the_explore_prompt_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv(explore.LIST_MEMORY_ENV, raising=False)
    variables = _variables(tmp_path, _corpus(tmp_path / "corpus"))
    rendered = get_robot_spec().prompts.render("system", variables=variables)
    assert rendered == format_prompt(explore.system_prompt(), variables=variables)
    assert "task-specific/CloseFridge_s0.json" not in rendered
    monkeypatch.setenv(explore.LIST_MEMORY_ENV, "0")
    assert get_robot_spec().prompts.render("system", variables=variables) == rendered


def test_on_the_prompt_lists_only_this_tasks_files(tmp_path, monkeypatch):
    monkeypatch.setenv(explore.LIST_MEMORY_ENV, "1")
    root = _corpus(tmp_path / "corpus").resolve()
    rendered = get_robot_spec().prompts.render(
        "system", variables=_variables(tmp_path, root)
    )
    for name in (
        "task-specific/CloseFridge_s0.json",
        "task-specific/CloseFridge_s0_recipe.jsonl",
        "task-specific/CloseFridge.md",
        "global/GLOBAL_MEMORY.md",
    ):
        assert f"- {root / name}\n" in rendered
    assert "OpenDrawer" not in rendered
    assert "MULTI-ATTEMPT EXPLORE" in rendered
