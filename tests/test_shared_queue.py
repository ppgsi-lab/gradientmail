import asyncio
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread, current_thread

import pytest
from fakes import FaultyMailbox

from gradientmail.mailbox import Mailbox, MailboxUnavailable
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.patterns.shared_queue import shared_queue


def describe_shared_queue():
    def it_loads_in_mapping_order_and_skips_existing_claims(mailbox: Mailbox):
        asyncio.run(mailbox.put_if_absent("units/taken", b"other"))
        units = {"z": (None, 3), "taken": (100,), "empty": (), "a": (1, 2)}

        with shared_queue(units, mailbox=mailbox, learner_id="node", load=iter) as batches:
            assert tuple(batches) == (None, 3, 1, 2)

        assert asyncio.run(mailbox.get("units/z")) == b"node"
        assert asyncio.run(mailbox.get("units/taken")) == b"other"
        with shared_queue(units, mailbox=mailbox, learner_id="later", load=iter) as batches:
            assert tuple(batches) == ()

    def it_finishes_when_there_are_no_units(mailbox: Mailbox):
        units: dict[str, tuple[int, ...]] = {}
        with shared_queue(units, mailbox=mailbox, learner_id="node", load=iter) as batches:
            assert tuple(batches) == ()

    def it_leaves_more_work_to_the_available_node(mailbox: Mailbox):
        started = Event()
        release = Event()
        units = {str(i): (2 * i, 2 * i + 1) for i in range(4)}

        def slow_load(work: tuple[int, ...]) -> Iterator[int]:
            started.set()
            assert release.wait(5)
            yield from work

        with shared_queue(units, mailbox=mailbox, learner_id="slow", load=slow_load) as slow:
            try:
                assert started.wait(5)
                with shared_queue(units, mailbox=mailbox, learner_id="fast", load=iter) as fast:
                    assert tuple(fast) == (2, 3, 4, 5, 6, 7)
            finally:
                release.set()
            assert tuple(slow) == (0, 1)

    def it_reserves_each_unit_once_under_concurrent_consumption(mailbox: Mailbox):
        ready = Barrier(4)
        units = {str(i): (i,) for i in range(16)}

        def consume(rank: int) -> tuple[int, ...]:
            first = True

            def load(work: tuple[int, ...]) -> Iterator[int]:
                nonlocal first
                if first:
                    first = False
                    ready.wait(timeout=5)
                yield from work

            with shared_queue(
                units, mailbox=mailbox, learner_id=f"node-{rank}", load=load
            ) as batches:
                return tuple(batches)

        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(consume, range(4)))

        assert all(results)
        assert sorted(batch for result in results for batch in result) == list(range(16))

    def it_bounds_prefetch_and_closes_the_loader_on_early_exit():
        loaded = tuple(Event() for _ in range(4))
        closed = Event()
        workers: list[Thread] = []

        def load(work: range) -> Iterator[int]:
            workers.append(current_thread())
            try:
                for batch in work:
                    loaded[batch].set()
                    yield batch
            finally:
                closed.set()

        with shared_queue(
            {"work": range(4)},
            mailbox=MemoryMailbox(),
            learner_id="node",
            load=load,
            prefetch=2,
        ) as batches:
            assert loaded[1].wait(5)
            assert not loaded[2].wait(0.05)
            assert next(batches) == 0
            assert loaded[2].wait(5)
            assert not loaded[3].wait(0.05)

        assert closed.is_set()
        assert workers[0] is not current_thread()
        assert not workers[0].is_alive()

    def it_closes_the_worker_when_the_consumer_raises():
        closed = Event()

        def load(work: range) -> Iterator[int]:
            try:
                yield from work
            finally:
                closed.set()

        with (
            pytest.raises(ValueError, match="training failed"),
            shared_queue(
                {"work": range(10)},
                mailbox=MemoryMailbox(),
                learner_id="node",
                load=load,
                prefetch=1,
            ) as batches,
        ):
            assert next(batches) == 0
            raise ValueError("training failed")

        assert closed.is_set()

    def it_delivers_buffered_items_before_propagating_loader_failure():
        mailbox = MemoryMailbox()

        def load(work: int) -> Iterator[int]:
            yield work
            raise ValueError("loading failed")

        with shared_queue({"work": 42}, mailbox=mailbox, learner_id="node", load=load) as batches:
            assert next(batches) == 42
            with pytest.raises(ValueError, match="loading failed"):
                next(batches)

        assert asyncio.run(mailbox.get("units/work")) == b"node"

    def it_propagates_mailbox_failure():
        mailbox = FaultyMailbox(MemoryMailbox(), fail_after=2)

        with (
            shared_queue({"work": (42,)}, mailbox=mailbox, learner_id="node", load=iter) as batches,
            pytest.raises(MailboxUnavailable),
        ):
            next(batches)

    def it_skips_a_claim_that_does_not_confirm_ownership():
        class UnconfirmedMailbox(MemoryMailbox):
            async def get(self, key: str) -> bytes | None:
                if key == "units/contested":
                    return b"other"
                return await super().get(key)

        with shared_queue(
            {"contested": (1,), "available": (2,)},
            mailbox=UnconfirmedMailbox(),
            learner_id="node",
            load=iter,
        ) as batches:
            assert tuple(batches) == (2,)

    @pytest.mark.parametrize("prefetch", [0, -1])
    def it_rejects_nonpositive_prefetch(prefetch: int):
        with (
            pytest.raises(ValueError, match="prefetch must be positive"),
            shared_queue(
                {"work": (1,)},
                mailbox=MemoryMailbox(),
                learner_id="node",
                load=iter,
                prefetch=prefetch,
            ),
        ):
            pass
