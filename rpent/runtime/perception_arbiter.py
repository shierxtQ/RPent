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

"""Optional cross-process scheduler for shared Pi0.5/SAM3 services.

The scheduler is automatic for LIBERO by default.  An explicit
``RPENT_SHARED_PERCEPTION=off`` keeps the historical direct-client path.
Enabled clients use one stable SQLite queue on the local host so separate
LIBERO processes share a fair, crash-tolerant perception slot without adding a
runtime dependency.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", "disabled"}
_AUTO = {"auto", "default", ""}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS perception_tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id TEXT NOT NULL,
    resource TEXT NOT NULL,
    state TEXT NOT NULL,
    requested_at REAL NOT NULL,
    granted_at REAL,
    heartbeat REAL,
    released_at REAL
);
CREATE INDEX IF NOT EXISTS perception_waiting
    ON perception_tickets(resource, state, id);
"""


@dataclass(frozen=True)
class PerceptionTiming:
    """Local timing and queue counters for one client process."""

    enabled: bool
    client_id: str
    scheduler_path: str | None
    queue_wait_s: float
    acquisitions: int
    #: ``queue_wait_s`` split by resource (e.g. ``perception``, ``libero_sim``).
    wait_by_resource: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "client_id": self.client_id,
            "scheduler_path": self.scheduler_path,
            "queue_wait_s": round(self.queue_wait_s, 6),
            "acquisitions": self.acquisitions,
            "wait_by_resource": {
                name: round(value, 6) for name, value in self.wait_by_resource.items()
            },
        }


class _Lease(AbstractContextManager):
    def __init__(self, arbiter: "SharedPerceptionArbiter", resource: str):
        self._arbiter = arbiter
        self._resource = resource
        self._ticket_id: int | None = None

    def __enter__(self) -> "_Lease":
        self._ticket_id = self._arbiter._acquire(self._resource)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._ticket_id is not None:
            self._arbiter._release(self._ticket_id)
            self._ticket_id = None


class SharedPerceptionArbiter:
    """Fair, process-safe scheduler for the shared perception GPU slot."""

    def __init__(
        self,
        path: str | Path,
        *,
        client_id: str | None = None,
        poll_s: float = 0.05,
        stale_s: float = 900.0,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.client_id = (
            client_id
            or os.getenv("RPENT_PERCEPTION_TASK_ID")
            or (f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}")
        )
        self.poll_s = max(0.01, float(poll_s))
        self.stale_s = max(30.0, float(stale_s))
        self._lock = threading.Lock()
        self._queue_wait_s = 0.0
        self._wait_by_resource: dict[str, float] = {}
        self._acquisitions = 0
        self._active_wait_started: float | None = None
        self._init_db()

    @classmethod
    def from_env(cls) -> "SharedPerceptionArbiter | None":
        """Construct the local arbiter unless explicitly disabled.

        ``auto`` (including an unset variable) uses the stable host-local
        scheduler path, so a task can join after another RPent process has
        already started.  ``0``/``off`` preserves the historical direct
        client path; explicit true values remain accepted for compatibility.
        """
        mode = os.getenv("RPENT_SHARED_PERCEPTION", "auto").strip().lower()
        if mode in _FALSE:
            return None
        if mode not in _TRUE | _AUTO:
            raise ValueError(
                f"RPENT_SHARED_PERCEPTION must be auto, 1/on, or 0/off; got {mode!r}"
            )
        path = os.getenv(
            "RPENT_PERCEPTION_SCHEDULER_PATH",
            "/tmp/rpent-perception-arbiter.sqlite3",
        )
        return cls(
            path,
            poll_s=float(os.getenv("RPENT_PERCEPTION_POLL_S", "0.05")),
            stale_s=float(os.getenv("RPENT_PERCEPTION_LEASE_S", "900")),
        )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000")
        # WAL is persistent per database. Concurrent process startup can race
        # while changing the journal mode; the busy timeout protects normal
        # transactions, and a startup race may safely use the existing mode.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower():
                conn.close()
                raise
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def hold(self, resource: str = "perception") -> AbstractContextManager:
        """Return a context that waits fairly for the shared resource."""
        if not resource:
            raise ValueError("resource must be non-empty")
        return _Lease(self, resource)

    def _insert_ticket(self, resource: str) -> int:
        now = time.time()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO perception_tickets "
                "(client_id, resource, state, requested_at, heartbeat) "
                "VALUES (?, ?, 'waiting', ?, ?)",
                (self.client_id, resource, now, now),
            )
            return int(cur.lastrowid)

    def _try_grant(self, ticket_id: int, resource: str) -> bool:
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cutoff = now - self.stale_s
            conn.execute(
                "UPDATE perception_tickets SET state='expired', released_at=? "
                "WHERE resource=? AND state IN ('active', 'waiting') "
                "AND COALESCE(heartbeat, requested_at) < ?",
                (now, resource, cutoff),
            )
            active = conn.execute(
                "SELECT id FROM perception_tickets "
                "WHERE resource=? AND state='active' LIMIT 1",
                (resource,),
            ).fetchone()
            head = conn.execute(
                "SELECT id FROM perception_tickets "
                "WHERE resource=? AND state='waiting' ORDER BY id LIMIT 1",
                (resource,),
            ).fetchone()
            if active is None and head is not None and int(head[0]) == ticket_id:
                conn.execute(
                    "UPDATE perception_tickets SET state='active', granted_at=?, heartbeat=? WHERE id=?",
                    (now, now, ticket_id),
                )
                conn.commit()
                return True
            conn.commit()
        return False

    def _acquire(self, resource: str) -> int:
        ticket_id = self._insert_ticket(resource)
        started = time.perf_counter()
        with self._lock:
            self._active_wait_started = started
        try:
            while not self._try_grant(ticket_id, resource):
                time.sleep(self.poll_s)
        except BaseException:
            self._cancel(ticket_id)
            raise
        finally:
            waited = time.perf_counter() - started
            with self._lock:
                self._queue_wait_s += waited
                self._wait_by_resource[resource] = (
                    self._wait_by_resource.get(resource, 0.0) + waited
                )
                self._acquisitions += 1
                self._active_wait_started = None
        return ticket_id

    def _cancel(self, ticket_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE perception_tickets SET state='cancelled', released_at=? WHERE id=?",
                (time.time(), ticket_id),
            )

    def _release(self, ticket_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE perception_tickets SET state='released', released_at=?, heartbeat=? WHERE id=?",
                (time.time(), time.time(), ticket_id),
            )

    def current_wait_s(self) -> float:
        """Return completed plus currently active queue wait."""
        with self._lock:
            active = (
                time.perf_counter() - self._active_wait_started
                if self._active_wait_started is not None
                else 0.0
            )
            return self._queue_wait_s + active

    def timing(self) -> PerceptionTiming:
        with self._lock:
            waited = self._queue_wait_s
            by_resource = dict(self._wait_by_resource)
            acquisitions = self._acquisitions
        return PerceptionTiming(
            enabled=True,
            client_id=self.client_id,
            scheduler_path=str(self.path),
            queue_wait_s=waited,
            acquisitions=acquisitions,
            wait_by_resource=by_resource,
        )
