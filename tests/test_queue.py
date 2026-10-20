"""``SharedQueue``: units claimed through the mailbox and loaded ahead of training.

``Feed`` is how a session consumes the queue, so the claiming, prefetching and releasing are
tested on it directly; the last block runs the queue through ``Session.run`` on two nodes.
"""

import asyncio
import json
import time
from array import array
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Thread, current_thread

import pytest
from fakes import FakeClock, FaultyMailbox

import gradientmail as gm
from gradientmail.mailbox import Mailbox
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import Exchange
from gradientmail.queue import Feed, SharedQueue


def feed[W, Batch](
    units: Mapping[str, W],
    load: Callable[[W], Iterator[Batch]],
    *,
    mailbox: Mailbox,
    node_id: str = "node",
    prefetch: int = 8,
    pause: Callable[[], bool] = lambda: False,
) -> Feed[W, Batch]:
    exchange = Exchange(mailbox, clock=mailbox.clock())
    return Feed(SharedQueue(units, load, prefetch), exchange=exchange, node_id=node_id, pause=pause)


def owner(mailbox: Mailbox, unit: str) -> bytes | None:
    return asyncio.run(mailbox.get(f"units/{unit}.json"))


def describe_shared_queue():
    @pytest.mark.parametrize("prefetch", [0, -1])
    def it_rejects_nonpositive_prefetch(prefetch: int):
        with pytest.raises(ValueError, match="prefetch must be positive"):
            SharedQueue({"work": (1,)}, iter, prefetch)

    def it_rejects_unit_ids_outside_the_key_alphabet():
        with pytest.raises(ValueError, match=r"got \['Work', 'a/b'\]"):
            SharedQueue({"ok": (1,), "a/b": (2,), "Work": (3,)}, iter)


def describe_feed():
    def it_claims_in_mapping_order_and_skips_units_claimed_elsewhere(mailbox: Mailbox):
        asyncio.run(mailbox.put_if_absent("units/taken.json", b'"other"'))
        units = {"z": (None, 3), "taken": (100,), "empty": (), "a": (1, 2)}

        with feed(units, iter, mailbox=mailbox) as batches:
            assert tuple(batches) == (("z", None), ("z", 3), ("a", 1), ("a", 2))

        assert owner(mailbox, "z") == b'"node"'
        assert owner(mailbox, "taken") == b'"other"'
        with feed(units, iter, mailbox=mailbox, node_id="later") as batches:
            assert tuple(batches) == ()

    def it_finishes_when_there_are_no_units(mailbox: Mailbox):
        units: dict[str, tuple[int, ...]] = {}
        with feed(units, iter, mailbox=mailbox) as batches:
            assert tuple(batches) == ()

    def it_leaves_more_work_to_the_available_node(mailbox: Mailbox):
        started = Event()
        release = Event()
        units = {str(i): (2 * i, 2 * i + 1) for i in range(4)}

        def slow_load(work: tuple[int, ...]) -> Iterator[int]:
            started.set()
            assert release.wait(5)
            yield from work

        with feed(units, slow_load, mailbox=mailbox, node_id="slow") as slow:
            try:
                assert started.wait(5)
                with feed(units, iter, mailbox=mailbox, node_id="fast") as fast:
                    assert [batch for _, batch in fast] == [2, 3, 4, 5, 6, 7]
            finally:
                release.set()
            assert [batch for _, batch in slow] == [0, 1]

    def it_claims_each_unit_once_under_concurrent_consumption(mailbox: Mailbox):
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

            with feed(units, load, mailbox=mailbox, node_id=f"node-{rank}") as batches:
                return tuple(batch for _, batch in batches)

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

        with feed({"work": range(4)}, load, mailbox=MemoryMailbox(), prefetch=2) as batches:
            assert loaded[1].wait(5)
            assert not loaded[2].wait(0.05)
            assert next(iter(batches)) == ("work", 0)
            assert loaded[2].wait(5)
            assert not loaded[3].wait(0.05)

        assert closed.is_set()
        assert workers[0] is not current_thread()
        assert not workers[0].is_alive()

    def it_closes_the_loader_when_the_consumer_raises():
        closed = Event()

        def load(work: range) -> Iterator[int]:
            try:
                yield from work
            finally:
                closed.set()

        with (
            pytest.raises(ValueError, match="training failed"),
            feed({"work": range(10)}, load, mailbox=MemoryMailbox(), prefetch=1) as batches,
        ):
            assert next(iter(batches)) == ("work", 0)
            raise ValueError("training failed")

        assert closed.is_set()

    def it_delivers_buffered_batches_before_raising_a_loader_failure():
        mailbox = MemoryMailbox()

        def load(work: int) -> Iterator[int]:
            yield work
            raise ValueError("loading failed")

        with feed({"work": 42}, load, mailbox=mailbox) as batches:
            pulled = iter(batches)
            assert next(pulled) == ("work", 42)
            with pytest.raises(ValueError, match="loading failed"):
                next(pulled)

        assert owner(mailbox, "work") == b'"node"'

    def it_confirms_a_claim_whose_response_was_lost(clock: FakeClock):
        # The second operation, the claim's write, lands and its response is lost; the retry
        # finds the key taken and the read-back shows that the claim is this node's.
        mailbox = FaultyMailbox(MemoryMailbox(clock=clock), fail_after=2)

        with feed({"work": (42,)}, iter, mailbox=mailbox) as batches:
            assert tuple(batches) == (("work", 42),)

    def it_skips_a_claim_that_does_not_confirm_ownership():
        class UnconfirmedMailbox(MemoryMailbox):
            async def get(self, key: str) -> bytes | None:
                if key == "units/contested.json":
                    return b'"other"'
                return await super().get(key)

        units = {"contested": (1,), "available": (2,)}
        with feed(units, iter, mailbox=UnconfirmedMailbox()) as batches:
            assert tuple(batches) == (("available", 2),)

    def it_gives_back_the_units_it_claimed_but_never_started(mailbox: Mailbox):
        loading_second = Event()

        def load(work: tuple[int, ...]) -> Iterator[int]:
            if work == (3, 4):
                loading_second.set()
            yield from work

        units = {"first": (1, 2), "second": (3, 4)}
        with feed(units, load, mailbox=mailbox) as batches:
            assert next(iter(batches)) == ("first", 1)
            assert loading_second.wait(5)
            batches.release()
            assert tuple(batches) == ()

        assert owner(mailbox, "first") == b'"node"'
        assert owner(mailbox, "second") is None
        with feed(units, load, mailbox=mailbox, node_id="other") as batches:
            assert tuple(batches) == (("second", 3), ("second", 4))

    def when_paused():
        def it_ends_after_the_unit_in_progress_and_leaves_the_next_one_unstarted(mailbox: Mailbox):
            pausing = Event()
            units = {"first": (1, 2, 3), "second": (4, 5), "third": (6,)}

            with feed(units, iter, mailbox=mailbox, pause=pausing.is_set) as batches:
                pulled = iter(batches)
                assert next(pulled) == ("first", 1)
                pausing.set()
                assert tuple(pulled) == (("first", 2), ("first", 3))
                assert batches.paused
                # The second unit's first batch was waiting: its unit was claimed, not started.
                assert owner(mailbox, "second") == b'"node"'
                batches.release()

            assert owner(mailbox, "first") == b'"node"'
            assert owner(mailbox, "second") is None
            with feed(units, iter, mailbox=mailbox, node_id="other") as batches:
                assert tuple(batches) == (("second", 4), ("second", 5), ("third", 6))
                assert not batches.paused

        def it_hands_out_nothing_when_paused_before_the_first_unit(mailbox: Mailbox):
            with feed({"work": (1, 2)}, iter, mailbox=mailbox, pause=lambda: True) as batches:
                assert tuple(batches) == ()
                assert batches.paused
                batches.release()

            assert owner(mailbox, "work") is None


def describe_session_on_a_shared_queue():
    def it_trains_every_unit_once_across_the_nodes_and_marks_each_node_done():
        mailbox = MemoryMailbox()
        algorithm = gm.DiLoCo(sync_every=4, fragments=2)
        units = {f"unit-{i}": range(4 * i, 4 * i + 4) for i in range(6)}
        trained: dict[str, list[int]] = {"one": [], "two": []}

        def train(name: str) -> None:
            model = gm.MemoryAdapter(
                {
                    "a": gm.TensorBytes("F32", (2,), array("f", (1.0, 2.0)).tobytes()),
                    "b": gm.TensorBytes("F32", (2,), array("f", (3.0, 4.0)).tobytes()),
                }
            )
            session = gm.Session(
                model, algorithm=algorithm, mailbox=mailbox, node_id=name, poll_interval=0.01
            )

            def step(batch: int) -> int:
                trained[name].append(batch)
                time.sleep(0.001)
                return 1

            session.run(step, gm.SharedQueue(units, load=iter))

        with ThreadPoolExecutor(max_workers=2) as pool:
            for running in [pool.submit(train, name) for name in trained]:
                running.result()

        assert sorted(trained["one"] + trained["two"]) == list(range(24))
        owners = {unit: json.loads(owner(mailbox, unit) or b"null") for unit in units}
        assert all(set(units[unit]) <= set(trained[node]) for unit, node in owners.items())
        assert asyncio.run(mailbox.list("done/")) == ["done/one.json", "done/two.json"]
