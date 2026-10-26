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

"""Stage-1 product: per-task publish with the RoboCasa replace rule, freeze, verify."""

from __future__ import annotations

import json
import stat

import pytest

from tests.unit_tests.ecc_robocasa.conftest import TASKS

pytestmark = pytest.mark.timeout(300)


def _run_mixed(ecc):
    other_staging = (
        ecc.run_root
        / "tasks"
        / "OpenDrawer"
        / "staging"
        / "task-specific"
        / "OpenDrawer_s0.json"
    )
    ecc.set_scenario(
        {
            "CloseBlenderLid": {"kind": "solve", "global": ["g_lid"]},
            "CloseFridge": {"kind": "fail", "global_any": ["g_fridge"]},
            "CloseToasterOvenDoor": {"kind": "solve", "audit": False},
            "CoffeeSetupMug": {"kind": "infra"},
            "NavigateKitchen": {
                "kind": "solve",
                "family_task": "OpenDrawer",
                "read_paths": [str(other_staging)],
            },
            "OpenDrawer": {"kind": "fail"},
        }
    )
    config = ecc.config(
        tasks=TASKS[:5] + ["OpenDrawer"], max_slots=3, stage1={"infra_retries": 1}
    )
    proc = ecc.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    return config


def test_publish_replaces_only_solved_tasks_and_freezes_a_verified_product(ecc):
    from robots.robocasa.memory import TaskMemory, selection_errors

    config = _run_mixed(ecc)
    product = ecc.run_root / "product"
    corpus = product / "corpus"
    assert (product / "FROZEN").exists()
    doc = json.loads((product / "PRODUCT.json").read_text())
    assert doc["schema"] == "rc-ecc-s1-product/1" and doc["publish_impl"].endswith(
        ":publish_exploration_cell"
    )
    assert doc["corpus_tree_sha256"] == (product / "FROZEN").read_text().strip()
    assert doc["cloud_corpus"]["tree_sha256"] and doc["code"]["tree_sha256"]
    assert doc["validation"]["tasks_loaded"] == 50

    def load(task):
        sel = TaskMemory.load(corpus, task, profile="local", split="target")
        assert selection_errors(sel.metadata(), task, "target") == []
        return set(sel.selected)

    # Solved with audit and recipe: the edge's pair, family note and global drafts; cloud trio archived.
    lid = load("CloseBlenderLid")
    assert {
        "task-specific/CloseBlenderLid_target_s0.json",
        "task-specific/CloseBlenderLid_target_s0_recipe.jsonl",
        "task-family/task-family_robocasa_target_tCloseBlenderLid.md",
        "global/g_lid.md",
        "global/GLOBAL_MEMORY.md",
    } <= lid
    assert not any(
        name.startswith("task-specific/CloseBlenderLid_s0")
        or name == "task-specific/CloseBlenderLid.md"
        for name in lid
    )
    archived = corpus / "_internal" / "replaced" / "CloseBlenderLid_s0"
    assert sorted(p.name for p in archived.iterdir()) == [
        "CloseBlenderLid.md",
        "CloseBlenderLid_s0.json",
        "CloseBlenderLid_s0_recipe.jsonl",
    ]
    # Unsolved: the cloud files stay; its global draft still merges (RPent's default rule).
    fridge = load("CloseFridge")
    assert {
        "task-specific/CloseFridge_s0.json",
        "task-specific/CloseFridge.md",
        "global/g_fridge.md",
    } <= fridge
    assert not any(
        "CloseFridge_target_s0" in name or name.startswith("task-family/")
        for name in fridge
    )
    # Solved without the explorer's audit: kept as cloud.
    toaster = ecc.record("CloseToasterOvenDoor")
    assert toaster["status"] == "solved" and toaster["publish"]["mode"] == "keep_cloud"
    assert "task-specific/CloseToasterOvenDoor_s0.json" in load("CloseToasterOvenDoor")
    # No valid result (infrastructure errors only): nothing published, cloud files stay.
    assert ecc.record("CoffeeSetupMug")["publish"]["mode"] == "none"
    assert "task-specific/CoffeeSetupMug_s0.json" in load("CoffeeSetupMug")
    # A task-family draft naming another task is never published for that task.
    assert not any(name.startswith("task-family/") for name in load("OpenDrawer"))
    assert "task-specific/NavigateKitchen_target_s0.json" in load("NavigateKitchen")
    # The explorer read another task's staging corpus by absolute path: flagged in its record.
    audit = ecc.record("NavigateKitchen")["file_tool_audit"]
    assert any("OpenDrawer" in path for path in audit["cross_task_reads"])
    # A cell cannot change any task's cloud memory in staging: only its inbox is writable.
    staging = ecc.run_root / "tasks" / "OpenDrawer" / "staging"
    for path in (
        staging / "task-specific" / "OpenDrawer_s0.json",
        staging / "global" / "new.md",
    ):
        with pytest.raises(PermissionError):
            path.write_text("tampered\n")
    assert (staging / "_internal" / "inbox" / "OpenDrawer_target_s0" / "wip").is_dir()
    # Tasks outside this run keep their cloud memory too.
    assert "task-specific/OpenStandMixerHead_s0.json" in load("OpenStandMixerHead")

    # Immutable and verifiable.
    assert all(not (p.stat().st_mode & stat.S_IWUSR) for p in corpus.rglob("*"))
    verified = ecc.stage1("verify", str(product))
    assert verified.returncode == 0 and json.loads(verified.stdout)["ok"]
    tasks = json.loads((product / "tasks.json").read_text())
    assert set(tasks) == set(TASKS[:5] + ["OpenDrawer"])

    target = corpus / "global" / "g_lid.md"
    target.chmod(0o644)
    target.write_text(target.read_text() + "tampered\n")
    tampered = ecc.stage1("verify", str(product))
    assert tampered.returncode == 1
    assert any("hash mismatch" in p for p in json.loads(tampered.stdout)["problems"])
    status = ecc.stage1("status", "--config", str(config), "--run-id", "r1")
    assert "product: frozen" in status.stdout


def test_assembly_is_deterministic(ecc):
    from scripts.ecc_robocasa.common import tree_files, tree_hash

    config = _run_mixed(ecc)
    hashes = []
    for n in range(2):
        out = ecc.root / f"assembled{n}"
        proc = ecc.stage1(
            "assemble", "--config", str(config), "--run-id", "r1", "--out", str(out)
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        hashes.append(tree_hash(tree_files(out)))
    assert hashes[0] == hashes[1]
    # Without --out the corpus is only validated: nothing assembled stays in the run root.
    proc = ecc.stage1("assemble", "--config", str(config), "--run-id", "r1")
    assert proc.returncode == 0 and '"errors": {}' in proc.stdout
    assert not any((ecc.run_root / ".corpus_check").iterdir())
    check = json.loads((ecc.run_root / "corpus_check.json").read_text())
    assert (
        check["replaced"] == ["CloseBlenderLid", "NavigateKitchen"]
        and check["errors"] == {}
    )
    frozen = json.loads((ecc.run_root / "product" / "PRODUCT.json").read_text())[
        "corpus_tree_sha256"
    ]
    assert hashes[0] == frozen


def test_a_write_into_another_tasks_folder_blocks_the_freeze(ecc):
    victim_inbox = (
        ecc.run_root
        / "tasks"
        / "CloseFridge"
        / "staging"
        / "_internal"
        / "inbox"
        / "CloseFridge_target_s0"
    )
    ecc.set_scenario(
        {
            "CloseBlenderLid": {
                "kind": "fail",
                "sleep_s": 0.5,
                "write_paths": [str(victim_inbox / "planted.md")],
            },
            "CloseFridge": {"kind": "solve", "sleep_s": 1.5},
        }
    )
    config = ecc.config(tasks=TASKS[:2], max_slots=2)
    proc = ecc.go(config)
    assert "refusing to freeze" in proc.stdout + proc.stderr
    assert not (ecc.run_root / "product" / "FROZEN").exists()
    audit = ecc.record("CloseBlenderLid")["file_tool_audit"]
    assert audit["cross_task_writes"] == [str((victim_inbox / "planted.md").resolve())]
    status = ecc.stage1("status", "--config", str(config), "--run-id", "r1")
    assert "CloseBlenderLid (cross_task_writes)" in status.stdout
    forced = ecc.stage1(
        "freeze", "--config", str(config), "--run-id", "r1", "--allow-cross-task-writes"
    )
    assert forced.returncode == 0, forced.stdout + forced.stderr
    doc = json.loads((ecc.run_root / "product" / "PRODUCT.json").read_text())
    assert list(doc["cross_task_writes_accepted"]) == ["CloseBlenderLid"]


def test_a_task_note_with_the_cloud_notes_legacy_scope_is_kept(ecc):
    """Qwen may copy the cloud note's ``scope: suite``; it is still the edge's task note."""
    ecc.set_scenario(
        {
            "CloseBlenderLid": {"kind": "solve", "family_scope": "suite"},
            "CloseFridge": {"kind": "fail"},
        }
    )
    config = ecc.config(tasks=TASKS[:2])
    proc = ecc.go(config)
    assert "runner finished: done" in proc.stdout, proc.stdout + proc.stderr
    record = ecc.record("CloseBlenderLid")
    assert record["inbox"]["scope_suite_normalized"] == ["family.md"]
    assert record["publish"]["mode"] == "replace"
    corpus = ecc.run_root / "product" / "corpus"
    note = corpus / "task-family" / "task-family_robocasa_target_tCloseBlenderLid.md"
    assert "scope: task-family" in note.read_text()
    product = json.loads((ecc.run_root / "product" / "PRODUCT.json").read_text())
    assert product["validation"]["errors"] == {}
    assembly = json.loads((ecc.run_root / "product" / "assembly.json").read_text())
    selected = assembly["validation"]["selected"]["CloseBlenderLid"]
    assert "task-family/task-family_robocasa_target_tCloseBlenderLid.md" in selected
    assert "task-specific/CloseBlenderLid.md" not in selected
