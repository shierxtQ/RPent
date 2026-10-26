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

from __future__ import annotations

import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

from rpent.runtime.perception_arbiter import SharedPerceptionArbiter


def test_shared_slot_serializes_clients(tmp_path):
    path = tmp_path / "arbiter.sqlite3"
    starts: list[float] = []
    ends: list[float] = []
    waits: list[float] = []
    lock = threading.Lock()

    def worker(client_id: str) -> None:
        arbiter = SharedPerceptionArbiter(
            path, client_id=client_id, poll_s=0.01, stale_s=30
        )
        with arbiter.hold():
            with lock:
                starts.append(time.monotonic())
            time.sleep(0.05)
            with lock:
                ends.append(time.monotonic())
        waits.append(arbiter.timing().queue_wait_s)

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    starts.sort()
    ends.sort()
    assert starts[1] >= ends[0]
    assert max(waits) >= 0.03


def test_resources_queue_independently_and_report_waits(tmp_path):
    path = tmp_path / "arbiter.sqlite3"
    holder = SharedPerceptionArbiter(path, client_id="a", poll_s=0.01, stale_s=30)
    waiter = SharedPerceptionArbiter(path, client_id="b", poll_s=0.01, stale_s=30)

    with holder.hold("libero_sim"):
        # Another resource is free while the simulator slot is held.
        with waiter.hold("perception"):
            pass
        done = threading.Event()

        def take_sim() -> None:
            with waiter.hold("libero_sim"):
                done.set()

        thread = threading.Thread(target=take_sim)
        thread.start()
        time.sleep(0.1)
        assert not done.is_set()
    thread.join(timeout=5)

    assert done.is_set()
    by_resource = waiter.timing().as_dict()["wait_by_resource"]
    assert set(by_resource) == {"perception", "libero_sim"}
    assert by_resource["libero_sim"] >= 0.05
    assert waiter.current_wait_s() >= by_resource["libero_sim"]


def test_from_env_auto_joins_default_queue(monkeypatch, tmp_path):
    monkeypatch.delenv("RPENT_SHARED_PERCEPTION", raising=False)
    monkeypatch.delenv("RPENT_PERCEPTION_SCHEDULER_PATH", raising=False)
    arbiter = SharedPerceptionArbiter.from_env()
    assert arbiter is not None
    assert arbiter.path == Path("/tmp/rpent-perception-arbiter.sqlite3").resolve()


def test_from_env_off_preserves_direct_mode(monkeypatch):
    monkeypatch.setenv("RPENT_SHARED_PERCEPTION", "off")
    assert SharedPerceptionArbiter.from_env() is None


def test_from_env_auto_accepts_explicit_path(monkeypatch, tmp_path):
    path = tmp_path / "auto.sqlite3"
    monkeypatch.setenv("RPENT_SHARED_PERCEPTION", "auto")
    monkeypatch.setenv("RPENT_PERCEPTION_SCHEDULER_PATH", str(path))
    arbiter = SharedPerceptionArbiter.from_env()
    assert arbiter is not None
    assert arbiter.path == path.resolve()


def _state(path: Path, ticket_id: int) -> str:
    with sqlite3.connect(path) as conn:
        return conn.execute(
            "SELECT state FROM perception_tickets WHERE id=?", (ticket_id,)
        ).fetchone()[0]


def test_a_waiter_outlasting_the_stale_cutoff_is_still_granted(tmp_path):
    path = tmp_path / "arbiter.sqlite3"
    holder = SharedPerceptionArbiter(path, client_id="a", stale_s=30)
    waiter = SharedPerceptionArbiter(path, client_id="b", stale_s=30)
    held = holder._acquire("libero_sim")
    waiting = waiter._insert_ticket("libero_sim")
    assert not waiter._try_grant(waiting, "libero_sim")

    # The wait outlasts stale_s and another client expires the ticket.
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE perception_tickets SET state='expired', heartbeat=heartbeat-100 "
            "WHERE id=?",
            (waiting,),
        )
    holder._release(held)

    assert waiter._try_grant(waiting, "libero_sim")
    assert _state(path, waiting) == "active"


def test_tickets_of_a_killed_client_do_not_block_the_queue(tmp_path):
    path = tmp_path / "arbiter.sqlite3"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    killed = SharedPerceptionArbiter(
        path, client_id=f"{socket.gethostname()}:{dead.pid}:dead", stale_s=900
    )
    other_host = SharedPerceptionArbiter(path, client_id="elsewhere:1:x", stale_s=900)
    waiter = SharedPerceptionArbiter(path, stale_s=900)
    stuck = killed._insert_ticket("libero_sim")
    remote = other_host._insert_ticket("perception")
    waiting = waiter._insert_ticket("libero_sim")

    assert waiter._try_grant(waiting, "libero_sim")
    assert _state(path, stuck) == "expired"
    assert _state(path, remote) == "waiting"  # other hosts are never judged


def test_polling_closes_its_connections(tmp_path):
    path = tmp_path / "arbiter.sqlite3"
    holder = SharedPerceptionArbiter(path, client_id="a", stale_s=30)
    waiter = SharedPerceptionArbiter(path, client_id="b", stale_s=30)
    holder._acquire("libero_sim")
    waiting = waiter._insert_ticket("libero_sim")

    def open_db_fds() -> int:
        fd_dir = "/proc/self/fd"
        return sum(
            1
            for fd in os.listdir(fd_dir)
            if os.path.realpath(os.path.join(fd_dir, fd)).startswith(str(path))
        )

    for _ in range(200):
        waiter._try_grant(waiting, "libero_sim")
    assert open_db_fds() == 0
