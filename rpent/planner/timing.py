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

"""Helpers for active-time planner budgets.

Active time is wall time minus waits the task does not control: queueing for
the shared perception GPU (reported by the toolkit) and backoff after a
rate-limited or transiently failing model request (recorded by
:mod:`rpent.planner.retry`). Both are measured from when a clock starts, so a
process that runs several episodes does not carry earlier waits forward.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Awaitable, TypeVar

T = TypeVar("T")

_retry_lock = threading.Lock()
_retry_wait_s = 0.0
_retry_waits_in_progress: dict[object, float] = {}


@contextmanager
def api_retry_wait() -> Iterator[None]:
    """Mark a model-request retry backoff as excluded time while it runs."""
    global _retry_wait_s
    token = object()
    with _retry_lock:
        _retry_waits_in_progress[token] = time.perf_counter()
    try:
        yield
    finally:
        with _retry_lock:
            started = _retry_waits_in_progress.pop(token)
            _retry_wait_s += time.perf_counter() - started


def api_retry_wait_s() -> float:
    """Return completed plus in-progress retry backoff in this process."""
    with _retry_lock:
        now = time.perf_counter()
        return _retry_wait_s + sum(
            now - started for started in _retry_waits_in_progress.values()
        )


def excluded_wait_s(toolkit: Any) -> float:
    """Read local resource wait without requiring a specialized toolkit."""
    try:
        value = float(toolkit.excluded_wait_s())
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return max(0.0, value)


class ActiveClock:
    """Wall clock that excludes perception queueing and API retry backoff."""

    def __init__(self, toolkit: Any) -> None:
        self.toolkit = toolkit
        self.started = time.perf_counter()
        self._queue0 = excluded_wait_s(toolkit)
        self._retry0 = api_retry_wait_s()

    def queue_wait_s(self) -> float:
        return max(0.0, excluded_wait_s(self.toolkit) - self._queue0)

    def retry_wait_s(self) -> float:
        return max(0.0, api_retry_wait_s() - self._retry0)

    def excluded_s(self) -> float:
        return self.queue_wait_s() + self.retry_wait_s()

    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    def active_s(self) -> float:
        return max(0.0, self.elapsed_s() - self.excluded_s())

    def stats(self) -> dict[str, Any]:
        """Timing fields for planner stats."""
        return {
            "elapsed_s": round(self.elapsed_s(), 3),
            "active_elapsed_s": round(self.active_s(), 3),
            "excluded_wait_s": round(self.excluded_s(), 3),
            "perception_wait_s": round(self.queue_wait_s(), 3),
            "api_retry_wait_s": round(self.retry_wait_s(), 3),
            "resource_timing": (
                self.toolkit.resource_timing()
                if callable(getattr(self.toolkit, "resource_timing", None))
                else {}
            ),
        }


async def run_with_active_timeout(
    awaitable: Awaitable[T],
    *,
    timeout_s: float,
    toolkit: Any,
    poll_s: float = 0.2,
) -> T:
    """Run an awaitable with a timeout that excludes queue and retry waits.

    With the default toolkit and no retries this is equivalent to a normal
    wall-clock timeout.
    """
    task = asyncio.create_task(awaitable)
    clock = ActiveClock(toolkit)
    try:
        while not task.done():
            remaining = timeout_s - clock.active_s()
            if remaining <= 0:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.TimeoutError
            try:
                return await asyncio.wait_for(
                    asyncio.shield(task), timeout=min(max(0.01, poll_s), remaining)
                )
            except asyncio.TimeoutError:
                continue
        return await task
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        raise


def wait_thread_with_active_timeout(
    thread, *, timeout_s: float, toolkit: Any, poll_s: float = 0.2
) -> bool:
    """Join a blocking SDK worker while excluding shared-resource queue waits."""
    clock = ActiveClock(toolkit)
    while thread.is_alive():
        remaining = timeout_s - clock.active_s()
        if remaining <= 0:
            return False
        thread.join(timeout=min(max(0.01, poll_s), remaining))
    return True
