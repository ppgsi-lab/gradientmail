"""LearnerState transitions in isolation; the Learner runtime over MemoryMailbox and MemoryAdapter.

The runtime boundary is what the syncer and the user's model observe: the bytes the adapter
holds, the snapshot records the learner publishes and ``rounds_seen``. Round records are
written through the real Exchange, as the syncer writes them. Tests that need two polls to
see a precise sequence of records drive ``join`` and ``poll`` on their own loop; the rest run
the communication thread.
"""

import asyncio
import struct
import time
from array import array
from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest
from fakes import FakeClock, SlowMailbox, TrippedMailbox

from gradientmail.adapters import MemoryAdapter
from gradientmail.algorithms import Algorithm, Estimates, Participant, RoundSchedule
from gradientmail.algorithms.diloco import DiLoCo
from gradientmail.events import (
    Callback,
    Event,
    FragmentAligned,
    PollFailed,
    PublishCancelled,
    PublishFailed,
    RoundRecorded,
    SnapshotPublished,
    log_events,
)
from gradientmail.learner import (
    Learner,
    LearnerState,
    Publishers,
    advanced,
    applied,
    due_round,
    overdue,
)
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import (
    Exchange,
    RoundRecordDraft,
    RunConfigMismatch,
    Snapshot,
    SnapshotDraft,
)
from gradientmail.serialization import SchemaMismatch, TensorBytes


def f32(*values: float) -> TensorBytes:
    return TensorBytes("F32", (len(values),), array("f", values).tobytes())


# Two equal-sized tensors give the plan (("a",), ("b",)); a round every two steps, "a" first.
ALGORITHM = DiLoCo(sync_every=4, fragments=2, in_flight=2)
SCHEDULE = RoundSchedule(sync_every=4, fragments=2, in_flight=2)
TENSORS = {"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}
A1 = {"a": f32(10.0, 20.0)}
B1 = {"b": f32(30.0, 40.0)}
A2 = {"a": f32(100.0, 200.0)}

FRESH = LearnerState(
    local_step=0,
    steps=(0, 0),
    tokens=(0, 0),
    applied_version=(0, 0),
    rounds_seen=-1,
    last_published=-1,
    published=(),
    step_seconds=0.0,
)
NOBODY = Publishers(-1, frozenset())


def due(state: LearnerState, *, quorum: int = 1, publishers: Publishers = NOBODY) -> int | None:
    return due_round(state, SCHEDULE, quorum=quorum, publishers=publishers)


def other_snapshot(round: int, learner: str) -> SnapshotDraft:
    """Another learner's snapshot of ``round``, as it lands in the store."""
    return SnapshotDraft(
        learner=learner,
        round=round,
        local_step=4,
        steps=4,
        tokens=12,
        base_version=0,
        rounds_seen=-1,
        step_seconds=0.1,
        schema_digest=MemoryAdapter(TENSORS).schema().digest(),
        final=False,
    )


def learner(
    mailbox: MemoryMailbox,
    clock: FakeClock,
    tensors: Mapping[str, TensorBytes] = TENSORS,
    *,
    algorithm: Algorithm = ALGORITHM,
    learner_id: str = "learner-a",
    on_event: Callback = log_events,
) -> tuple[Learner, MemoryAdapter]:
    adapter = MemoryAdapter(tensors)
    return (
        Learner(
            adapter=adapter,
            algorithm=algorithm,
            mailbox=mailbox,
            learner_id=learner_id,
            clock=clock,
            poll_interval=0.01,
            lease_every=0.05,
            lease_timeout=0.1,
            on_event=on_event,
        ),
        adapter,
    )


async def spin(until: Callable[[], bool]) -> None:
    """Yield to the loop until ``until`` holds; the clock moves only when the learner sleeps."""
    for _ in range(10_000):
        if until():
            return
        await asyncio.sleep(0)
    raise AssertionError("the learner never reached the expected point")


def joined(
    mailbox: MemoryMailbox,
    clock: FakeClock,
    tensors: Mapping[str, TensorBytes] = TENSORS,
    *,
    algorithm: Algorithm = ALGORITHM,
    on_event: Callback = log_events,
) -> tuple[Learner, MemoryAdapter]:
    created, adapter = learner(mailbox, clock, tensors, algorithm=algorithm, on_event=on_event)
    asyncio.run(created.join())
    return created, adapter


def started(
    mailbox: MemoryMailbox,
    clock: FakeClock,
    tensors: Mapping[str, TensorBytes] = TENSORS,
    *,
    algorithm: Algorithm = ALGORITHM,
    on_event: Callback = log_events,
) -> tuple[Learner, MemoryAdapter]:
    created, adapter = learner(mailbox, clock, tensors, algorithm=algorithm, on_event=on_event)
    created.start()
    return created, adapter


def commit(
    exchange: Exchange,
    round: int,
    theta: Mapping[str, TensorBytes],
    participants: tuple[Participant, ...] = (),
    update: Mapping[str, TensorBytes] | None = None,
) -> None:
    """The syncer's side of round ``round``: a committed record pointing at ``theta``."""
    draft = RoundRecordDraft(
        round=round,
        step=SCHEDULE.step_of(round),
        fragment=SCHEDULE.fragment_of(round),
        version=SCHEDULE.version_after(round),
        participants=participants,
        rejected=(),
        cosine=None,
        quorum_seconds=0.0,
        grace_seconds=0.0,
        estimates=Estimates(),
    )
    asyncio.run(exchange.commit(draft, theta, {}, update))


def snapshot(exchange: Exchange, round: int) -> Snapshot:
    (only,) = asyncio.run(exchange.snapshots(round))
    return only


def step_until(trained: Learner, samples: int, condition: Callable[[], bool]) -> int:
    """Train until ``condition`` holds; returns the number of steps that took."""
    deadline = time.monotonic() + 10.0
    taken = 0
    while not condition():
        assert time.monotonic() < deadline, "the learner never reached the expected state"
        trained.step(samples)
        taken += 1
        time.sleep(0.001)
    return taken


def describe_due_round():
    def when_fewer_steps_than_the_gap():
        def it_is_none():
            assert due(replace(FRESH, local_step=1, steps=(1, 1))) is None
            assert due(replace(FRESH, local_step=2, steps=(2, 2))) == 0

    def when_in_flight_rounds_are_outstanding():
        def it_stops_at_rounds_seen_plus_tau():
            ahead = replace(
                FRESH,
                local_step=20,
                steps=(20, 20),
                rounds_seen=0,
                last_published=1,
                published=((1, 4),),
            )

            assert due(ahead) == 2
            assert due(replace(ahead, last_published=2, published=((1, 4), (2, 6)))) is None

    def when_the_fragment_was_just_overwritten():
        def it_waits_for_a_step_on_it():
            overwritten = replace(
                FRESH,
                local_step=20,
                steps=(0, 20),
                applied_version=(1, 1),
                rounds_seen=1,
                last_published=1,
                published=((1, 4),),
            )

            assert due(overwritten) is None
            assert due(advanced(overwritten, 8, 0.1)) == 2

    def when_the_round_is_one_snapshot_short_of_quorum():
        def it_is_due_with_the_steps_trained_so_far():
            early = replace(FRESH, local_step=1, steps=(1, 1))
            two_others = frozenset({"learner-b", "learner-c"})

            assert due(early, quorum=3, publishers=Publishers(0, two_others)) == 0
            assert due(early, quorum=3, publishers=Publishers(0, frozenset({"learner-b"}))) is None
            assert due(early, quorum=3, publishers=Publishers(1, two_others)) is None


def describe_overdue():
    def it_holds_once_an_unaligned_round_is_that_many_steps_old():
        state = replace(
            FRESH, local_step=6, rounds_seen=0, last_published=2, published=((1, 4), (2, 6))
        )

        assert not overdue(state, 3)
        assert overdue(state, 2)
        assert not overdue(replace(state, rounds_seen=1), 2)
        assert overdue(replace(state, rounds_seen=1), 0)


def describe_applied():
    def it_resets_only_that_fragments_counters():
        before = replace(FRESH, local_step=7, steps=(7, 7), tokens=(21, 21))

        after = applied(before, round=1, fragment=1, version=1)

        assert after == replace(
            before, steps=(7, 0), tokens=(21, 0), applied_version=(0, 1), rounds_seen=1
        )

    def when_the_algorithm_mixes():
        def it_keeps_that_share_of_the_steps_the_commit_did_not_incorporate():
            before = replace(FRESH, local_step=7, steps=(7, 7), tokens=(21, 21))
            mine = Participant("learner-a", 5, 5, 15, 45.0)

            took_part = applied(
                before, round=1, fragment=1, version=1, participant=mine, mixing=0.5
            )
            left_out = applied(before, round=1, fragment=1, version=1, participant=None, mixing=0.5)

            assert (took_part.steps, took_part.tokens) == ((7, 1), (21, 3))
            assert (left_out.steps, left_out.tokens) == ((7, 3), (21, 10))


def describe_learner():
    def when_a_round_record_appears():
        def it_overwrites_the_fragment_and_resets_its_counters(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            trained, adapter = started(mailbox, clock)
            try:
                for _ in range(2):
                    trained.step(samples=3)
                commit(exchange, 0, A1)
                applied_at = 2 + step_until(trained, 3, lambda: trained.rounds_seen == 0)
                # Round 1 ("b") goes out at step 4 regardless; round 2 ("a") needs two steps after
                # round 1 and at least one step on "a" since the reset.
                published_at = max(6, applied_at + 1)
                for _ in range(published_at - applied_at):
                    trained.step(samples=3)
            finally:
                trained.close()

            assert dict(adapter.tensors) == {**TENSORS, **A1}
            assert trained.rounds_seen == 0
            untouched = snapshot(exchange, 1)
            assert (
                untouched.steps,
                untouched.tokens,
                untouched.base_version,
                untouched.final,
            ) == (4, 12, 0, False)
            reset = snapshot(exchange, 2)
            since_reset = published_at - applied_at
            assert (reset.steps, reset.tokens, reset.base_version, reset.local_step) == (
                since_reset,
                3 * since_reset,
                1,
                published_at,
            )
            assert reset.rounds_seen == 0

    def when_the_round_is_closing_without_it():
        def it_publishes_the_steps_it_has(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            # A gap of four steps, so nothing is due on schedule before step 4.
            algorithm = DiLoCo(sync_every=8, fragments=2, in_flight=2, quorum_size=2)
            trained, _ = started(mailbox, clock, algorithm=algorithm)
            try:
                asyncio.run(exchange.publish_snapshot(other_snapshot(0, "learner-b"), A1))
                taken = step_until(trained, 3, lambda: len(asyncio.run(exchange.snapshots(0))) == 2)
            finally:
                trained.close()

            [early] = [s for s in asyncio.run(exchange.snapshots(0)) if s.learner == "learner-a"]
            assert 1 <= early.steps <= taken < 4
            assert (early.local_step, early.tokens) == (early.steps, 3 * early.steps)

    def when_fewer_learners_are_live_than_quorum():
        def it_publishes_once_the_other_live_ones_have(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            # Quorum three, but only learner-b holds a lease besides this one: its snapshot
            # alone means the round is about to close.
            algorithm = DiLoCo(sync_every=8, fragments=2, in_flight=2, quorum_size=3)
            asyncio.run(exchange.beat("learner-b"))
            asyncio.run(exchange.publish_snapshot(other_snapshot(0, "learner-b"), A1))
            trained, _ = started(mailbox, clock, algorithm=algorithm)
            try:
                taken = step_until(trained, 3, lambda: len(asyncio.run(exchange.snapshots(0))) == 2)
            finally:
                trained.close()

            assert taken < 4

    def when_the_algorithm_mixes():
        def it_blends_the_commit_with_the_local_parameters_and_keeps_the_commit_as_anchor(
            clock: FakeClock,
        ):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            algorithm = DiLoCo(sync_every=4, fragments=2, in_flight=2, mixing=0.5)
            trained, adapter = started(mailbox, clock, algorithm=algorithm)
            try:
                # Local training moved "a" before any round; round 0 commits it elsewhere.
                adapter.apply({"a": f32(3.0, 5.0)})
                commit(exchange, 0, A1)
                step_until(trained, 1, lambda: len(asyncio.run(exchange.snapshots(2))) == 1)
                blended = adapter.tensors["a"]
            finally:
                trained.close()

            # Aligning set "a" to the midpoint; the next delta is measured from the commit.
            assert blended == f32(6.5, 12.5)
            assert asyncio.run(exchange.fetch(snapshot(exchange, 2).payload)) == {
                "a": f32(3.5, 7.5)
            }

    def when_a_commit_carries_an_update():
        def it_rebuilds_theta_from_the_version_it_holds_instead_of_fetching_it(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            algorithm = DiLoCo(sync_every=4, fragments=2, in_flight=2, commit_dtype="F32")
            trained, adapter = joined(mailbox, clock, algorithm=algorithm)
            commit(exchange, 0, A1, update={"a": f32(9.0, 18.0)})
            # Only the update can produce A1 now.
            [record] = asyncio.run(exchange.rounds_after(-1))
            asyncio.run(mailbox.delete([record.theta.key]))

            asyncio.run(trained.poll())
            trained.step(samples=1)

            assert adapter.tensors["a"] == A1["a"]
            assert trained.rounds_seen == 0

        def when_the_rebuilt_theta_does_not_hash_to_the_record():
            def it_fetches_theta(clock: FakeClock):
                mailbox = MemoryMailbox()
                exchange = Exchange(mailbox, clock=clock)
                algorithm = DiLoCo(sync_every=4, fragments=2, in_flight=2, commit_dtype="F32")
                trained, adapter = joined(mailbox, clock, algorithm=algorithm)
                commit(exchange, 0, A1, update={"a": f32(0.0, 0.0)})

                asyncio.run(trained.poll())
                trained.step(samples=1)

                assert adapter.tensors["a"] == A1["a"]
                assert trained.rounds_seen == 0

    def it_reports_how_many_steps_after_its_snapshot_it_aligned(clock: FakeClock):
        mailbox = MemoryMailbox()
        exchange = Exchange(mailbox, clock=clock)
        seen: list[Event] = []
        trained, _ = joined(mailbox, clock, on_event=seen.append)
        for _ in range(2):
            trained.step(samples=3)
        commit(exchange, 0, A1, participants=(Participant("learner-a", 2, 2, 6, 18.0),))
        commit(exchange, 1, B1)

        asyncio.run(trained.poll())
        trained.step(samples=3)

        aligned = [e for e in seen if isinstance(e, FragmentAligned)]
        assert [(e.fragment, e.local_step, e.lag_steps) for e in aligned] == [
            (0, 3, 1),
            (1, 3, None),
        ]

    def it_publishes_the_delta_against_its_anchor(clock: FakeClock):
        mailbox = MemoryMailbox()
        exchange = Exchange(mailbox, clock=clock)
        algorithm = DiLoCo(sync_every=4, fragments=2, in_flight=2, delta_dtype="BF16")
        trained, adapter = started(mailbox, clock, algorithm=algorithm)
        try:
            # Local training moved "a" away from the (1, 2) that join() set as its anchor.
            adapter.apply({"a": f32(3.0, 5.0)})
            step_until(trained, 1, lambda: len(asyncio.run(exchange.snapshots(0))) == 1)
        finally:
            trained.close()

        [published] = asyncio.run(exchange.snapshots(0))
        minus_two_minus_three = TensorBytes("BF16", (2,), struct.pack("<2H", 0xC000, 0xC040))
        assert asyncio.run(exchange.fetch(published.payload)) == {"a": minus_two_minus_three}

    def it_holds_a_lease_while_running_and_releases_it_on_close(clock: FakeClock):
        mailbox = MemoryMailbox()
        exchange = Exchange(mailbox, clock=clock)
        trained, _ = started(mailbox, clock)
        try:
            step_until(trained, 1, lambda: "learner-a" in asyncio.run(exchange.leases()))
        finally:
            trained.close()

        assert dict(asyncio.run(exchange.leases())) == {}

    def when_a_poll_raises():
        def it_reports_the_failure_and_keeps_renewing_the_lease(clock: FakeClock):
            inner = MemoryMailbox()
            exchange = Exchange(inner, clock=clock)
            # The first lease listing is the first beat's: the renewal itself fails.
            mailbox = TrippedMailbox(
                inner,
                lambda operation, key: operation == "list" and key.startswith("leases/"),
                RuntimeError("boom"),
            )
            seen: list[Event] = []
            trained, _ = started(mailbox, clock, on_event=seen.append)
            try:
                step_until(trained, 1, lambda: "learner-a" in asyncio.run(exchange.leases()))
            finally:
                trained.close()

            [failure] = [e for e in seen if isinstance(e, PollFailed)]
            assert isinstance(failure.error, RuntimeError)

    def when_publishing_a_snapshot_raises():
        def it_reports_the_failure_and_publishes_the_next_round(clock: FakeClock):
            inner = MemoryMailbox()
            exchange = Exchange(inner, clock=clock)
            mailbox = TrippedMailbox(
                inner,
                lambda operation, key: (
                    operation == "put_if_absent" and key.startswith("blobs/snapshots/0000000000/")
                ),
                RuntimeError("boom"),
            )
            seen: list[Event] = []
            trained, _ = started(mailbox, clock, on_event=seen.append)
            try:
                step_until(trained, 1, lambda: len(asyncio.run(exchange.snapshots(1))) == 1)
            finally:
                trained.close()

            assert list(asyncio.run(exchange.snapshots(0))) == []
            assert [e.round for e in seen if isinstance(e, PublishFailed)] == [0]
            assert 1 in {e.round for e in seen if isinstance(e, SnapshotPublished)}

    def when_close_is_cut_short_by_a_slow_upload():
        def it_still_releases_the_lease(clock: FakeClock, monkeypatch: pytest.MonkeyPatch):
            inner = MemoryMailbox()
            exchange = Exchange(inner, clock=clock)
            mailbox = SlowMailbox(inner, lambda key: key.startswith("blobs/snapshots/"), 0.5)
            monkeypatch.setattr("gradientmail.learner._CLOSE_TIMEOUT", 0.05)
            seen: list[Event] = []
            trained, _ = started(mailbox, clock, on_event=seen.append)
            step_until(trained, 1, lambda: "learner-a" in asyncio.run(exchange.leases()))
            # Round 0 is due by the second step, however many the lease took; its upload
            # outlasts the close timeout.
            trained.step(samples=1)
            trained.step(samples=1)

            trained.close()

            assert PublishCancelled("learner-a", 0.05) in seen
            assert dict(asyncio.run(exchange.leases())) == {}

    def when_closed_with_steps_no_round_has_taken():
        def it_publishes_nothing(clock: FakeClock):
            mailbox = MemoryMailbox()
            trained, _ = started(mailbox, clock)
            trained.step(samples=3)

            trained.close()

            assert asyncio.run(mailbox.list("snapshots/")) == []

        def when_the_algorithm_flushes_on_close():
            def it_publishes_every_fragment_and_gives_up_once_nothing_lands(clock: FakeClock):
                mailbox = MemoryMailbox()
                exchange = Exchange(mailbox, clock=clock)
                algorithm = DiLoCo(sync_every=4, fragments=2, in_flight=2, flush_on_close=True)
                trained, _ = started(mailbox, clock, algorithm=algorithm)
                trained.step(samples=3)

                trained.close()

                # One step of a two-step gap: neither round was due. Both fragments go out, and
                # with nobody closing rounds the wait ends after the lease timeout plus the
                # floor of thirty polls.
                assert (snapshot(exchange, 0).steps, snapshot(exchange, 1).steps) == (1, 1)
                assert snapshot(exchange, 0).final and snapshot(exchange, 1).final
                assert clock.now() >= 0.1 + 30 * 0.01

    def when_two_versions_of_a_fragment_arrive_together():
        def it_applies_only_the_newest(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            trained, adapter = joined(mailbox, clock)
            commit(exchange, 0, A1)
            commit(exchange, 1, B1)
            commit(exchange, 2, A2)

            asyncio.run(trained.poll())
            trained.step(samples=1)

            assert dict(adapter.tensors) == {**A2, **B1}
            assert trained.rounds_seen == 2

        def it_reports_every_record_and_aligns_only_with_the_newest(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            seen: list[Event] = []
            trained, _ = joined(mailbox, clock, on_event=seen.append)
            commit(exchange, 0, A1)
            commit(exchange, 1, B1)
            commit(exchange, 2, A2)

            asyncio.run(trained.poll())
            trained.step(samples=1)

            assert [e.record.round for e in seen if isinstance(e, RoundRecorded)] == [0, 1, 2]
            assert sorted(e.round for e in seen if isinstance(e, FragmentAligned)) == [1, 2]

    def when_a_payload_was_garbage_collected():
        def it_applies_the_next_version_instead(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            trained, adapter = joined(mailbox, clock)
            commit(exchange, 0, A1)
            asyncio.run(exchange.sweep_blobs(0))

            asyncio.run(trained.poll())
            trained.step(samples=1)

            assert dict(adapter.tensors) == TENSORS
            assert trained.rounds_seen == -1

            commit(exchange, 1, B1)
            commit(exchange, 2, A2)
            asyncio.run(trained.poll())
            trained.step(samples=1)

            assert dict(adapter.tensors) == {**A2, **B1}
            assert trained.rounds_seen == 2

    def when_restarted():
        def it_reloads_the_newest_fragments_and_resumes_publishing(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            joined(mailbox, clock)
            commit(exchange, 0, A1)
            commit(exchange, 1, B1)
            commit(exchange, 2, A2)

            restarted, adapter = started(mailbox, clock, {"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)})
            try:
                assert dict(adapter.tensors) == {**A2, **B1}
                assert restarted.rounds_seen == 2
                for _ in range(2):
                    restarted.step(samples=5)
            finally:
                restarted.close()

            resumed = snapshot(exchange, 3)
            assert (resumed.base_version, resumed.rounds_seen, resumed.local_step) == (1, 2, 2)
            assert (resumed.steps, resumed.tokens) == (2, 10)

    def when_another_learner_is_creating_the_run():
        def it_waits_for_the_manifest_instead_of_uploading_its_own(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            zeros = {"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)}
            waiter, adapter = learner(mailbox, clock, zeros)
            creator, _ = learner(mailbox, clock, learner_id="creator")

            async def scenario():
                assert await exchange.claim_creation(1, "creator")
                joining = asyncio.create_task(waiter.join())
                for _ in range(5):
                    await asyncio.sleep(0)
                uploaded, still_waiting = await mailbox.list("blobs/init/"), not joining.done()
                await creator.join()
                await spin(joining.done)
                return uploaded, still_waiting, await exchange.manifest()

            uploaded, still_waiting, manifest = asyncio.run(scenario())

            assert not uploaded and still_waiting
            assert manifest is not None and manifest.created_by == "creator"
            assert dict(adapter.tensors) == TENSORS

    def when_the_creator_dies():
        def it_takes_over_once_no_init_payload_lands_for_the_stall_time(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            survivor, _ = learner(mailbox, clock)

            async def scenario():
                assert await exchange.claim_creation(1, "dead")
                joining = asyncio.create_task(survivor.join())
                await spin(joining.done)
                return await exchange.creation_claims(), await exchange.manifest(), clock.now()

            claimants, manifest, now = asyncio.run(scenario())

            assert claimants == ("dead", "learner-a")
            assert manifest is not None and manifest.created_by == "learner-a"
            assert now >= 30 * 0.01

    def when_restarted_while_creating():
        def it_resumes_as_the_creator(clock: FakeClock):
            mailbox = MemoryMailbox()
            exchange = Exchange(mailbox, clock=clock)
            restarted, _ = learner(mailbox, clock)

            async def scenario():
                assert await exchange.claim_creation(1, "learner-a")
                await restarted.join()
                return await exchange.creation_claims(), await exchange.manifest(), clock.now()

            claimants, manifest, now = asyncio.run(scenario())

            assert claimants == ("learner-a",)
            assert manifest is not None and manifest.created_by == "learner-a"
            assert now == 0.0

    def when_manifest_schema_differs():
        def it_fails_before_the_first_step_and_names_the_parameters(clock: FakeClock):
            mailbox = MemoryMailbox()
            joined(mailbox, clock)
            other, _ = learner(
                mailbox, clock, {"a": f32(1.0, 2.0), "b": f32(3.0, 4.0, 5.0), "c": f32(6.0)}
            )

            with pytest.raises(SchemaMismatch) as failure:
                other.start()

            assert failure.value.names == ("b", "c")

    def when_manifest_algorithm_differs():
        def it_raises_run_config_mismatch(clock: FakeClock):
            mailbox = MemoryMailbox()
            joined(mailbox, clock)
            other, _ = learner(
                mailbox, clock, algorithm=DiLoCo(sync_every=8, fragments=2, in_flight=2)
            )

            with pytest.raises(RunConfigMismatch):
                other.start()


def describe_batch_size():
    def when_the_algorithm_sets_no_global_batch():
        def it_is_none(clock: FakeClock):
            trained, _ = joined(MemoryMailbox(), clock)

            assert trained.batch_size is None

    def it_splits_the_global_batch_among_the_live_learners_rounding_up(clock: FakeClock):
        mailbox = MemoryMailbox()
        exchange = Exchange(mailbox, clock=clock)
        algorithm = DiLoCo(sync_every=4, fragments=2, global_batch_size=10)
        trained, _ = learner(mailbox, clock, algorithm=algorithm)
        asyncio.run(trained.join())

        assert trained.batch_size == 10
        asyncio.run(exchange.beat("learner-b"))
        asyncio.run(exchange.beat("learner-c"))
        asyncio.run(trained.poll())
        assert trained.batch_size == 4
        # The others never renew: at the lease timeout this learner is alone again.
        clock.advance(0.1)
        asyncio.run(trained.poll())
        assert trained.batch_size == 10
