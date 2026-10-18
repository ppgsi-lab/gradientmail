"""Hold nodes at a named point until a number of them have arrived."""

import asyncio
from functools import partial

from gradientmail.clock import Clock, MonotonicClock
from gradientmail.mailbox import Mailbox, retrying

__all__ = ["Barrier"]


class Barrier:
    """A rendezvous through the mailbox: ``wait(name, parties=n)`` returns once ``n`` nodes arrived.

    Nodes share a run-scoped mailbox, with a distinct learner ID per node. An arrival is the
    create-once object ``barriers/{name}/{learner_id}``; ``wait`` creates this node's and lists
    the prefix every ``poll_interval`` seconds until it holds ``parties`` keys or more. Arrivals
    are never removed: a name serves once per run, a loop names each barrier (``f"epoch-{n}"``)
    and a node arriving after the barrier opened passes at once. Transient mailbox failures are
    retried. ``clock`` is the time source for polling and the timeout.
    """

    def __init__(
        self,
        *,
        mailbox: Mailbox,
        learner_id: str,
        poll_interval: float = 1.0,
        clock: Clock | None = None,
    ) -> None:
        self._mailbox = mailbox
        self._learner_id = learner_id
        self._poll_interval = poll_interval
        self._clock = clock if clock is not None else MonotonicClock()

    def wait(self, name: str, *, parties: int, timeout: float | None = None) -> frozenset[str]:
        """Record this node's arrival at ``name`` and block until ``parties`` nodes have arrived.

        ``parties`` is a count the caller fixes; it need not match the algorithm's quorum.
        Returns the learner IDs that had arrived when the barrier opened. Raises ``TimeoutError``
        after ``timeout`` seconds without opening; the arrival stays recorded, so calling again
        resumes the wait. Calling again once the barrier is open returns at once.
        """
        if parties < 1:
            raise ValueError("parties must be positive")
        return asyncio.run(self._wait(name, parties, timeout))

    async def _wait(self, name: str, parties: int, timeout: float | None) -> frozenset[str]:
        prefix = f"barriers/{name}/"
        deadline = None if timeout is None else self._clock.now() + timeout
        await retrying(
            partial(self._mailbox.put_if_absent, f"{prefix}{self._learner_id}", b""),
            clock=self._clock,
        )
        while True:
            arrived = await retrying(partial(self._mailbox.list, prefix), clock=self._clock)
            if len(arrived) >= parties:
                return frozenset(key.removeprefix(prefix) for key in arrived)
            delay = self._poll_interval
            if deadline is not None:
                remaining = deadline - self._clock.now()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{len(arrived)} of {parties} nodes arrived at {name!r} within {timeout} s"
                    )
                delay = min(delay, remaining)
            await self._clock.sleep(delay)
