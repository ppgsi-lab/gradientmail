import asyncio
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fakes import ClockedMailbox, FakeClock, FaultyMailbox

from gradientmail.clock import MonotonicClock, StoreClock
from gradientmail.mailbox import Mailbox
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.patterns.barrier import Barrier


def arrive(mailbox: Mailbox, name: str, *learners: str, ago: float = 0.0) -> None:
    at = repr(mailbox.clock().now() - ago).encode()

    async def put() -> None:
        for learner in learners:
            await mailbox.put_if_absent(f"barriers/{name}/{learner}", at)

    asyncio.run(put())


def describe_barrier():
    def it_holds_until_parties_have_arrived(mailbox: Mailbox):
        first = Barrier(mailbox=mailbox, learner_id="first", poll_interval=0.01, margin=0.01)
        last = Barrier(mailbox=mailbox, learner_id="last", poll_interval=0.01, margin=0.01)

        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(first.wait, "start", parties=2)
            with pytest.raises(TimeoutError):
                waiting.result(timeout=0.1)
            assert last.wait("start", parties=2) == frozenset({"first", "last"})
            assert waiting.result(timeout=5) == frozenset({"first", "last"})

    def it_passes_at_once_after_the_release(mailbox: Mailbox):
        arrive(mailbox, "start", "a", "b", ago=10.0)
        late = Barrier(mailbox=mailbox, learner_id="late")

        assert late.wait("start", parties=2, timeout=0) == frozenset({"a", "b", "late"})

    def it_returns_at_once_when_called_again(mailbox: Mailbox):
        barrier = Barrier(mailbox=mailbox, learner_id="node", poll_interval=0.01, margin=0.01)

        assert barrier.wait("start", parties=1) == frozenset({"node"})
        assert barrier.wait("start", parties=1, timeout=0) == frozenset({"node"})

    def it_keeps_barriers_apart_by_name(mailbox: Mailbox):
        arrive(mailbox, "start", "a", ago=10.0)
        barrier = Barrier(mailbox=mailbox, learner_id="b", poll_interval=0.01, margin=0.01)

        with pytest.raises(TimeoutError):
            barrier.wait("done", parties=2, timeout=0)
        assert barrier.wait("start", parties=2, timeout=0) == frozenset({"a", "b"})

    def it_releases_two_margins_and_at_most_two_polls_after_the_opening(clock: FakeClock):
        mailbox = MemoryMailbox(clock=clock)
        clock.advance(100.0)
        arrive(mailbox, "start", "a", ago=3.0)
        barrier = Barrier(mailbox=mailbox, learner_id="b", poll_interval=1.0, margin=0.25)

        assert barrier.wait("start", parties=2) == frozenset({"a", "b"})
        assert 0.5 < clock.now() - 100.0 < 0.5 + 2 * 1.0

    def it_releases_nodes_whose_clocks_disagree_at_one_instant():
        # The store reads monotonic time plus EPOCH. The nodes' wall clocks are seconds apart;
        # each corrects its own from five minutes of responses with 2 to 10 ms one-way trips.
        epoch = 1_792_000_000.0
        rng = random.Random(0)
        store = MemoryMailbox()
        nodes = {}
        for learner, wall_error in (("a", -2.5), ("b", 0.0), ("c", 3.5)):
            clock = StoreClock(MonotonicClock(), offset=epoch + wall_error)
            start = time.monotonic() - 300.0
            for second in range(300):
                stamped = start + second + rng.random()
                forward, back = rng.uniform(0.002, 0.01), rng.uniform(0.002, 0.01)
                clock.observe(stamped - forward, stamped + back, math.floor(stamped + epoch))
            barrier = Barrier(
                mailbox=ClockedMailbox(store, clock),
                learner_id=learner,
                poll_interval=0.2,
                margin=0.05,
            )
            nodes[learner] = barrier
        released: dict[str, float] = {}

        def run(learner: str, delay: float) -> None:
            time.sleep(delay)
            nodes[learner].wait("start", parties=3, timeout=5)
            released[learner] = time.monotonic()

        threads = [
            threading.Thread(target=run, args=(learner, delay))
            for learner, delay in (("a", 0.0), ("b", 0.07), ("c", 0.13))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert sorted(released) == ["a", "b", "c"]
        assert max(released.values()) - min(released.values()) < 0.04

    def it_times_out_and_keeps_the_arrival(clock: FakeClock):
        mailbox = MemoryMailbox(clock=clock)
        barrier = Barrier(mailbox=mailbox, learner_id="node", poll_interval=0.1)

        with pytest.raises(TimeoutError, match=r"1 of 2 nodes arrived at 'start' within 1\.0 s"):
            barrier.wait("start", parties=2, timeout=1.0)

        assert clock.now() == pytest.approx(1.0)
        assert asyncio.run(mailbox.get("barriers/start/node")) == b"0.0"
        arrive(mailbox, "start", "other")
        assert barrier.wait("start", parties=2, timeout=0) == frozenset({"node", "other"})

    def it_retries_a_transient_mailbox_failure(clock: FakeClock):
        mailbox = FaultyMailbox(MemoryMailbox(clock=clock), fail_after=1)
        barrier = Barrier(mailbox=mailbox, learner_id="node")

        assert barrier.wait("start", parties=1) == frozenset({"node"})

    @pytest.mark.parametrize("parties", [0, -1])
    def it_rejects_nonpositive_parties(parties: int):
        barrier = Barrier(mailbox=MemoryMailbox(), learner_id="node")

        with pytest.raises(ValueError, match="parties must be positive"):
            barrier.wait("start", parties=parties)
