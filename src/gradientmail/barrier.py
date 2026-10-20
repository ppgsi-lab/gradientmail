"""Hold nodes at a named point until a number of them have arrived, then release them together."""

import asyncio
import math
from collections.abc import Sequence
from functools import partial

from gradientmail.clock import Clock
from gradientmail.mailbox import Mailbox, retrying

__all__ = ["Barrier"]

_GOLDEN = (math.sqrt(5) - 1) / 2


class Barrier:
    """A rendezvous through the mailbox: ``wait(name, parties=n)`` returns once ``n`` nodes
    arrived, on every node at the same instant of the mailbox's clock.

    Nodes share a run-scoped mailbox, with a distinct node id per node. An arrival is the
    create-once object ``barriers/{name}/{node_id}`` holding its time on ``mailbox.clock()``.
    ``wait`` creates this node's and lists the prefix once per ``poll_interval``, at instants
    of that clock every node computes alike, until it holds ``parties`` keys or more. The
    barrier opens at the ``parties``-th earliest arrival time, and every node that finds it open
    reads the arrivals and computes the same release: ``margin`` seconds after the first poll
    instant more than ``margin`` past the opening. ``margin`` bounds a write reaching every
    listing, and a listing plus its reads returning, both widened by how far the nodes' clocks
    disagree; nodes must share ``poll_interval`` and ``margin``. A node whose requests outlast
    ``margin`` finds the release passed and returns at once.

    Arrivals are never removed: a name serves once per run, a loop names each barrier
    (``f"epoch-{n}"``) and a node arriving after the release passes at once. Transient mailbox
    failures are retried.
    """

    def __init__(
        self,
        *,
        mailbox: Mailbox,
        node_id: str,
        poll_interval: float = 5.0,
        margin: float = 1.0,
    ) -> None:
        self._mailbox = mailbox
        self._node_id = node_id
        self._poll_interval = poll_interval
        self._margin = margin

    def wait(self, name: str, *, parties: int, timeout: float | None = None) -> frozenset[str]:
        """Record this node's arrival at ``name`` and block until the barrier releases.

        ``parties`` is a count the caller fixes; it need not match the algorithm's quorum.
        Returns the node ids this node found arrived when it saw the barrier open. Raises
        ``TimeoutError`` after ``timeout`` seconds without opening; the arrival stays recorded,
        so calling again resumes the wait. Once the barrier is open, a call returns at the
        release, or at once when the release has passed.
        """
        if parties < 1:
            raise ValueError("parties must be positive")
        return asyncio.run(self._wait(name, parties, timeout))

    async def _wait(self, name: str, parties: int, timeout: float | None) -> frozenset[str]:
        prefix = f"barriers/{name}/"
        clock = self._mailbox.clock()
        deadline = None if timeout is None else clock.now() + timeout
        arrival = repr(clock.now()).encode()
        await retrying(
            partial(self._mailbox.put_if_absent, f"{prefix}{self._node_id}", arrival),
            clock=clock,
        )
        while True:
            arrived = await retrying(partial(self._mailbox.list, prefix), clock=clock)
            if len(arrived) >= parties:
                release = await self._release(arrived, parties, clock)
                await clock.sleep(max(0.0, release - clock.now()))
                return frozenset(key.removeprefix(prefix) for key in arrived)
            now = clock.now()
            delay = self._poll_after(now) - now
            if deadline is not None:
                remaining = deadline - now
                if remaining <= 0:
                    raise TimeoutError(
                        f"{len(arrived)} of {parties} nodes arrived at {name!r} within {timeout} s"
                    )
                delay = min(delay, remaining)
            await clock.sleep(delay)

    async def _release(self, arrived: Sequence[str], parties: int, clock: Clock) -> float:
        async with asyncio.TaskGroup() as group:
            reads = [
                group.create_task(retrying(partial(self._mailbox.get, key), clock=clock))
                for key in arrived
            ]
        times = sorted(float(body) for read in reads if (body := read.result()) is not None)
        return self._poll_after(times[parties - 1] + self._margin) + self._margin

    def _poll_after(self, instant: float) -> float:
        # Poll k falls in [k·P, (k+1)·P) at a fraction of a second that advances by the golden
        # ratio, so responses are stamped all across the second and the clock's estimate
        # narrows from both ends; polls on whole seconds would leave one end a second wide.
        # The third candidate covers a floor that rounding put one interval low.
        interval = self._poll_interval
        k = math.floor(instant / interval)
        polls = (j * interval + (j * _GOLDEN % 1) * min(interval, 1.0) for j in (k, k + 1, k + 2))
        return next(poll for poll in polls if poll > instant)
