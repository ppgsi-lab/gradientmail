"""Work shared among a run's nodes: units claimed through the mailbox and loaded ahead of training.

``SharedQueue`` is what the caller hands ``Session.run`` in place of a dataloader: units of work
by id, the same ids on every node, and how to load one into batches. Only the ids cross the
mailbox; the values stay local. The session consumes it through a ``Feed``, which claims one
unit at a time on a thread of its own, as ``units/{id}.json`` (``protocol`` module): it lists
the claims, takes the first unit in this node's mapping order that nobody holds and loads it,
listing again once the unit is loaded, so a node that trains faster claims more units. Up to
``prefetch`` loaded batches wait for training, each tagged with the unit it came from. A claim
stays when its unit is finished or when its node fails, so a unit a node dies in is lost to the
run; ``Feed.release`` gives back the units a node claimed and never started. A node that stops
training (``Session``'s ``"either"`` role, once elected the run's syncer) pauses its feed, which
ends at the next unit's first batch, so the unit in progress is trained to its end and the
units loaded ahead go back.
"""

import asyncio
import re
from collections import deque
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from threading import Condition, Thread
from typing import Self

from gradientmail.protocol import Exchange

__all__ = ["Feed", "SharedQueue"]


@dataclass(frozen=True)
class SharedQueue[W, Batch]:
    """Units of work shared by a run's nodes, for ``Session.run`` to claim and train.

    ``units`` maps an id, ``[a-z0-9-]{1,64}`` and the same on every node, to the value ``load``
    turns into batches. A node visits the units in its own mapping's order, so nodes that list
    them in different orders start on different units. At most ``prefetch`` loaded batches wait
    for training.
    """

    units: Mapping[str, W]
    load: Callable[[W], Iterator[Batch]]
    prefetch: int = 8

    def __post_init__(self) -> None:
        if self.prefetch < 1:
            raise ValueError("prefetch must be positive")
        if invalid := sorted(unit for unit in self.units if not _UNIT_ID.fullmatch(unit)):
            raise ValueError(f"unit ids must match [a-z0-9-]{{1,64}}, got {invalid}")


class Feed[W, Batch]:
    """One node's share of a ``SharedQueue``: a context manager that iterates ``(unit, batch)``.

    Entering starts the thread that claims units through ``exchange`` as ``node_id`` and loads
    them. Iteration ends once no unclaimed unit is left and the buffer is drained, which means
    no work remains to claim, not that the other nodes have finished theirs; a loader or mailbox
    failure is raised after the batches buffered before it. ``pause`` is asked before the first
    batch of each unit is handed out: once it holds, iteration ends there instead, with
    ``paused`` set, and that unit stays unstarted. Exiting stops the thread and joins it,
    waiting for the mailbox or loader call in progress to return.
    """

    def __init__(
        self,
        queue: SharedQueue[W, Batch],
        *,
        exchange: Exchange,
        node_id: str,
        pause: Callable[[], bool] = lambda: False,
    ) -> None:
        self._queue = queue
        self._exchange = exchange
        self._node_id = node_id
        self._pause = pause
        self._condition = Condition()
        self._pending: deque[tuple[str, Batch]] = deque()
        self._claimed: list[str] = []
        self._started: set[str] = set()
        self._stopped = False
        self._finished = False
        self._paused = False
        self._failure: BaseException | None = None
        self._thread = Thread(target=self._produce, name=f"gradientmail-queue-{node_id}")

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop()

    def __iter__(self) -> Iterator[tuple[str, Batch]]:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopped or self._pending or self._finished)
                if self._stopped:
                    return
                if not self._pending:
                    if self._failure is not None:
                        raise self._failure
                    return
                unit, batch = self._pending[0]
                # Units load one after another: a batch of a unit not started yet begins the next.
                if unit not in self._started and self._pause():
                    self._paused = True
                    return
                self._pending.popleft()
                self._started.add(unit)
                self._condition.notify_all()
            yield unit, batch

    @property
    def paused(self) -> bool:
        """Whether iteration ended because ``pause`` held as a unit was about to start."""
        return self._paused

    def release(self) -> None:
        """Stop loading, drop the buffered batches and delete the claims on the units whose
        first batch was never handed out, so that any node may train them."""
        self._stop()
        unstarted = [unit for unit in self._claimed if unit not in self._started]
        asyncio.run(self._exchange.release_units(unstarted))

    def _stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join()
        self._pending.clear()

    def _produce(self) -> None:
        unit = ""

        def loaded() -> Generator[Batch, None, None]:
            nonlocal unit
            with asyncio.Runner() as runner:
                while True:
                    taken = runner.run(self._exchange.unit_claims())
                    for candidate, work in self._queue.units.items():
                        if candidate not in taken and runner.run(
                            self._exchange.claim_unit(candidate, self._node_id)
                        ):
                            unit = candidate
                            with self._condition:
                                self._claimed.append(candidate)
                            # Delegated, not wrapped, so that closing the source closes the
                            # loader too.
                            yield from self._queue.load(work)
                            break
                    else:
                        return

        try:
            with closing(loaded()) as source:
                while True:
                    with self._condition:
                        self._condition.wait_for(
                            lambda: self._stopped or len(self._pending) < self._queue.prefetch
                        )
                        if self._stopped:
                            return
                    batch = next(source)
                    with self._condition:
                        self._pending.append((unit, batch))
                        self._condition.notify_all()
        except StopIteration:
            pass
        except BaseException as error:
            self._failure = error
        finally:
            with self._condition:
                self._finished = True
                self._condition.notify_all()


_UNIT_ID = re.compile(r"[a-z0-9-]{1,64}")
