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

"""The corpus merge lock: re-entrant for its holder, exclusive for others."""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

from rpent.memory import MemoryManager


def test_the_holder_can_merge_inside_the_lock(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path / "memory")

    with manager.merge_lock():
        with manager.merge_lock():
            pass
        result = manager.merge_memory(
            cell_tag="task_s0", run_state_dir=tmp_path / "run", solved=False
        )

    assert result["task"] == 0
    assert (tmp_path / "memory" / "_internal" / "merge.lock").is_file()


def _lock_then_mark(root: str, go, marker: str) -> None:
    go.wait(10)
    with MemoryManager(root).merge_lock():
        Path(marker).write_text("locked")


def test_another_process_waits_for_the_lock(tmp_path: Path) -> None:
    root = tmp_path / "memory"
    marker = tmp_path / "marker"
    context = multiprocessing.get_context("fork")
    go = context.Event()
    # Start the child before locking so it inherits no lock descriptor.
    child = context.Process(target=_lock_then_mark, args=(str(root), go, str(marker)))
    child.start()
    with MemoryManager(root).merge_lock():
        go.set()
        time.sleep(0.5)
        assert not marker.exists()
    child.join(10)

    assert child.exitcode == 0
    assert marker.read_text() == "locked"


def test_another_thread_waits_even_on_the_same_manager(tmp_path: Path) -> None:
    manager = MemoryManager(tmp_path / "memory")
    acquired = threading.Event()

    def lock() -> None:
        with manager.merge_lock():
            acquired.set()

    with manager.merge_lock():
        thread = threading.Thread(target=lock)
        thread.start()
        assert not acquired.wait(0.5)
    thread.join(10)

    assert acquired.is_set()
