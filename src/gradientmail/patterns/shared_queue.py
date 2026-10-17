"""Distribute work through mailbox claims and prefetch its loaded items."""

import asyncio
from collections import deque
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import closing, contextmanager
from threading import Condition, Thread

from gradientmail.mailbox import Mailbox

__all__ = ["shared_queue"]


@contextmanager
def shared_queue[Work, Batch](
    units: Mapping[str, Work],
    *,
    mailbox: Mailbox,
    learner_id: str,
    load: Callable[[Work], Iterator[Batch]],
    prefetch: int = 8,
) -> Iterator[Iterator[Batch]]:
    """Reserve units in mapping order and load their items on a background thread.

    Nodes share unit identifiers and a run-scoped mailbox, with a distinct learner ID
    per node. Values stay local; reservations use ``units/{identifier}`` keys. Exclusive
    ownership requires atomic ``put_if_absent``. Claims remain after completion or failure.

    At most ``prefetch`` loaded items wait for consumption. Loader and mailbox failures
    propagate after buffered items. Exhaustion means no unclaimed work remains, not that
    other nodes have finished. Exiting stops prefetch and joins the thread, waiting for
    any current mailbox or loader operation to return.
    """
    if prefetch < 1:
        raise ValueError("prefetch must be positive")

    owner = learner_id.encode()
    condition = Condition()
    pending = deque[Batch]()
    stopped = False
    finished = False
    failure: BaseException | None = None

    async def claim(key: str) -> bool:
        return await mailbox.put_if_absent(key, owner) and await mailbox.get(key) == owner

    def loaded() -> Generator[Batch, None, None]:
        with asyncio.Runner() as runner:
            while True:
                taken = set(runner.run(mailbox.list("units/")))
                for identifier, work in units.items():
                    key = f"units/{identifier}"
                    if key not in taken and runner.run(claim(key)):
                        yield from load(work)
                        break
                else:
                    return

    def produce() -> None:
        nonlocal finished, failure
        try:
            with closing(loaded()) as source:
                while True:
                    with condition:
                        condition.wait_for(lambda: stopped or len(pending) < prefetch)
                        if stopped:
                            return
                    batch = next(source)
                    with condition:
                        pending.append(batch)
                        condition.notify_all()
        except StopIteration:
            pass
        except BaseException as error:
            failure = error
        finally:
            with condition:
                finished = True
                condition.notify_all()

    def consume() -> Iterator[Batch]:
        while True:
            with condition:
                condition.wait_for(lambda: stopped or pending or finished)
                if stopped:
                    return
                if not pending:
                    if failure is not None:
                        raise failure
                    return
                batch = pending.popleft()
                condition.notify_all()
            yield batch

    thread = Thread(target=produce, name=f"shared-queue-{learner_id}")
    thread.start()
    try:
        yield consume()
    finally:
        with condition:
            stopped = True
            condition.notify_all()
        thread.join()
        pending.clear()
