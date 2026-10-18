"""Time source for polling, backoff and timing measurements; faked in tests."""

import asyncio
import math
import threading
import time
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Seconds plus an awaitable sleep, so runtimes never read wall time directly."""

    def now(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class MonotonicClock(Clock):
    def now(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class StoreClock(Clock):
    """A store's clock as one node estimates it from the ``Date`` header of its responses.

    ``now`` is ``local.now()`` plus an offset. An observation is a request sent at local time
    ``sent`` whose response arrived at ``received`` with ``date``, the server's time truncated
    to the second: the offset lies between ``date - received`` and ``date + 1 - sent``.
    Observations are intersected, so the window narrows toward the round trip as responses are
    stamped at different fractions of a second. The offset stays at ``offset`` while every
    observation allows it, so a node whose wall clock already agrees with the store keeps it;
    once one rules it out, the offset follows the window's middle. An observation outside the
    window means the local clock drifted or stepped, and the window starts over from it.
    ``observe`` may be called from any thread.
    """

    def __init__(self, local: Clock, offset: float) -> None:
        self._local = local
        self._prior: float | None = offset
        self._offset = offset
        self._lowest = -math.inf
        self._highest = math.inf
        self._lock = threading.Lock()

    def observe(self, sent: float, received: float, date: float) -> None:
        lowest, highest = date - received, date + 1 - sent
        with self._lock:
            if max(lowest, self._lowest) <= min(highest, self._highest):
                lowest, highest = max(lowest, self._lowest), min(highest, self._highest)
            self._lowest, self._highest = lowest, highest
            if self._prior is not None and not lowest <= self._prior <= highest:
                self._prior = None
            self._offset = self._prior if self._prior is not None else (lowest + highest) / 2

    def now(self) -> float:
        return self._local.now() + self._offset

    async def sleep(self, seconds: float) -> None:
        await self._local.sleep(seconds)
