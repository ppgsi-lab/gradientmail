import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from fakes import FakeClock, FaultyMailbox

from gradientmail.mailbox import Mailbox
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.patterns.barrier import Barrier


def arrive(mailbox: Mailbox, name: str, *learners: str) -> None:
    async def put() -> None:
        for learner in learners:
            await mailbox.put_if_absent(f"barriers/{name}/{learner}", b"")

    asyncio.run(put())


def describe_barrier():
    def it_holds_until_parties_have_arrived(mailbox: Mailbox):
        first = Barrier(mailbox=mailbox, learner_id="first", poll_interval=0.01)
        last = Barrier(mailbox=mailbox, learner_id="last", poll_interval=0.01)

        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(first.wait, "start", parties=2)
            with pytest.raises(TimeoutError):
                waiting.result(timeout=0.1)
            assert last.wait("start", parties=2) == frozenset({"first", "last"})
            assert waiting.result(timeout=5) == frozenset({"first", "last"})

    def it_passes_at_once_after_opening(mailbox: Mailbox):
        arrive(mailbox, "start", "a", "b")
        late = Barrier(mailbox=mailbox, learner_id="late")

        assert late.wait("start", parties=2, timeout=0) == frozenset({"a", "b", "late"})

    def it_returns_at_once_when_called_again(mailbox: Mailbox):
        barrier = Barrier(mailbox=mailbox, learner_id="node")

        assert barrier.wait("start", parties=1) == frozenset({"node"})
        assert barrier.wait("start", parties=1, timeout=0) == frozenset({"node"})

    def it_keeps_barriers_apart_by_name(mailbox: Mailbox):
        arrive(mailbox, "start", "a")
        barrier = Barrier(mailbox=mailbox, learner_id="b")

        with pytest.raises(TimeoutError):
            barrier.wait("done", parties=2, timeout=0)
        assert barrier.wait("start", parties=2, timeout=0) == frozenset({"a", "b"})

    def it_times_out_and_keeps_the_arrival(clock: FakeClock):
        mailbox = MemoryMailbox()
        barrier = Barrier(mailbox=mailbox, learner_id="node", poll_interval=0.1, clock=clock)

        with pytest.raises(TimeoutError, match=r"1 of 2 nodes arrived at 'start' within 1\.0 s"):
            barrier.wait("start", parties=2, timeout=1.0)

        assert clock.now() == pytest.approx(1.0)
        assert asyncio.run(mailbox.get("barriers/start/node")) == b""
        arrive(mailbox, "start", "other")
        assert barrier.wait("start", parties=2, timeout=0) == frozenset({"node", "other"})

    def it_retries_a_transient_mailbox_failure(clock: FakeClock):
        mailbox = FaultyMailbox(MemoryMailbox(), fail_after=1)
        barrier = Barrier(mailbox=mailbox, learner_id="node", clock=clock)

        assert barrier.wait("start", parties=1) == frozenset({"node"})
        assert clock.now() == pytest.approx(0.5)

    @pytest.mark.parametrize("parties", [0, -1])
    def it_rejects_nonpositive_parties(parties: int):
        barrier = Barrier(mailbox=MemoryMailbox(), learner_id="node")

        with pytest.raises(ValueError, match="parties must be positive"):
            barrier.wait("start", parties=parties)
