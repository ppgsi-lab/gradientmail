"""The syncer over a real Exchange on MemoryMailbox.

Snapshots go in through ``Exchange.publish_snapshot`` and every outcome is read back from the
store: the round record, the committed theta, the claims and keys left behind. ``FakeClock``
advances only when the syncer sleeps, so quorum waits, the grace window, the takeover timeout
and the recorded timings are exact and no test waits on wall time.
"""

import asyncio
import contextlib
import logging
import struct
from array import array
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import pytest
from fakes import FakeClock, RecordingMailbox, TrippedMailbox

from gradientmail.adapters import default_backend
from gradientmail.algorithms import Estimates
from gradientmail.algorithms.diloco import DiLoCo, WeightedMean, grace_window
from gradientmail.events import (
    Callback,
    Event,
    FollowingRound,
    LeaseReleased,
    RoundClaimed,
    RoundCommitted,
    RoundSuperseded,
    TookOverRound,
    log_events,
)
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import (
    Exchange,
    ManifestDraft,
    RoundRecord,
    RunConfigMismatch,
    SnapshotDraft,
    SyncerKind,
    beat_key,
    rank_id,
    round_key,
    syncer_lease_prefix,
)
from gradientmail.serialization import ParameterSpec, Schema, TensorBytes, pack
from gradientmail.syncer import Following, Syncer, SyncerState


def f32(*values: float) -> TensorBytes:
    return TensorBytes("F32", (len(values),), array("f", values).tobytes())


def bf16(*values: float) -> TensorBytes:
    """``values`` as bf16 by truncation, so pass values bf16 represents exactly."""
    patterns = [struct.unpack("<I", struct.pack("<f", value))[0] >> 16 for value in values]
    return TensorBytes("BF16", (len(values),), struct.pack(f"<{len(values)}H", *patterns))


def values(tensor: TensorBytes) -> list[float]:
    return array("f", tensor.data).tolist()


POLL = 0.1
SCHEMA = Schema((ParameterSpec("bias", "F32", (2,)), ParameterSpec("weight", "F32", (2,))))
PLAN = (("bias",), ("weight",))
INIT = ({"bias": f32(1.0, 1.0)}, {"weight": f32(1.0, 1.0)})
# Steps of 0.2 s over the schedule's gap of 5 make a 1 s round; with quorum 0.1 and sync 0.1
# and in_flight 2 the window is 0.9 s.
TIMINGS = Estimates(step=0.2, quorum=0.1, sync=0.1)
WINDOW = grace_window(TIMINGS, 2, 5, None)


def algorithm(*, quorum: int = 1, lr: float = 1.0, momentum: float = 0.0) -> DiLoCo:
    """lr 1 and momentum 0 with the weighted mean make Θ the anchor minus the token-weighted
    mean of the published deltas."""
    return DiLoCo(
        sync_every=10,
        fragments=2,
        in_flight=2,
        quorum_size=quorum,
        outer_lr=lr,
        outer_momentum=momentum,
        merge=WeightedMean(),
    )


@dataclass(frozen=True)
class Rig:
    store: MemoryMailbox
    clock: FakeClock
    config: DiLoCo

    @property
    def exchange(self) -> Exchange:
        return Exchange(self.store, clock=self.clock)

    def syncer(
        self,
        node_id: str = "node-1",
        poll_interval: float = POLL,
        lease_timeout: float = 120.0,
        *,
        exclusive: SyncerKind | None = None,
        lease_every: float = 60.0,
        units: Callable[[], frozenset[str]] = frozenset,
        on_event: Callback = log_events,
    ) -> Syncer:
        return Syncer(
            algorithm=self.config,
            mailbox=self.store,
            node_id=node_id,
            clock=self.clock,
            poll_interval=poll_interval,
            exclusive=exclusive,
            lease_every=lease_every,
            lease_timeout=lease_timeout,
            units=units,
            on_event=on_event,
        )

    async def alive(self, *learners: str) -> None:
        for learner in learners:
            await self.exchange.beat(learner)

    async def training(self, learner: str) -> None:
        """Renew ``learner``'s lease every poll until cancelled, as a learner that trains on."""
        while True:
            await self.exchange.beat(learner)
            await self.clock.sleep(POLL)

    async def claimants(self, round: int) -> Sequence[str]:
        return await self.exchange.claims(round)

    async def create_run(self) -> None:
        draft = ManifestDraft(SCHEMA, PLAN, self.config.to_json(), created_by="learner-a")
        await self.exchange.create_run(draft, INIT)

    async def publish(
        self,
        learner: str,
        round: int,
        delta: Sequence[float],
        *,
        base_version: int = 0,
        tokens: int = 1,
        steps: int = 1,
        digest: str = SCHEMA.digest(),
        dtype: Literal["F32", "BF16"] = "F32",
        tensors: Mapping[str, TensorBytes] | None = None,
        step_seconds: float = 0.5,
    ) -> None:
        draft = SnapshotDraft(
            learner=learner,
            round=round,
            local_step=5 * (round + 1),
            steps=steps,
            tokens=tokens,
            base_version=base_version,
            rounds_seen=round - 1,
            step_seconds=step_seconds,
            schema_digest=digest,
            final=False,
        )
        tensor = f32(*delta) if dtype == "F32" else bf16(*delta)
        if tensors is None:
            tensors = {PLAN[round % 2][0]: tensor}
        await self.exchange.publish_snapshot(draft, tensors)

    async def committed(self) -> Sequence[RoundRecord]:
        return await self.exchange.rounds_after(-1)

    async def theta_of(self, record: RoundRecord) -> Mapping[str, TensorBytes]:
        theta = await self.exchange.fetch(record.theta)
        assert theta is not None
        return theta


async def spin(until: Callable[[], bool]) -> None:
    """Yield to the loop until ``until`` holds; the clock moves only when the syncer sleeps."""
    for _ in range(10_000):
        if until():
            return
        await asyncio.sleep(0)
    raise AssertionError("the syncer never reached the expected point")


async def completed[T](work: Coroutine[object, object, T]) -> T:
    """Run ``work`` as a task and return its result, so a syncer that never returns fails."""
    task = asyncio.create_task(work)
    await spin(task.done)
    return task.result()


def loaded(progress: SyncerState | Following) -> SyncerState:
    assert isinstance(progress, SyncerState), progress
    return progress


def describe_run_round():
    def when_fewer_than_quorum_snapshots_exist():
        def it_waits(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=2))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer(poll_interval=60.0, lease_timeout=600.0)
                state = await syncer.recover()
                await rig.alive("a", "b")
                await rig.publish("a", 0, (0.0, 1.0))
                task = asyncio.create_task(syncer.run_round(state))
                await spin(lambda: clock.now() >= 300.0 or task.done())
                still_waiting = not task.done()
                records_while_waiting = await rig.committed()
                await rig.publish("b", 0, (1.0, 0.0))
                await spin(task.done)
                return (
                    still_waiting,
                    records_while_waiting,
                    loaded(task.result()),
                    await rig.committed(),
                )

            still_waiting, before, state, records = asyncio.run(scenario())

            assert still_waiting
            assert list(before) == []
            assert [record.round for record in records] == [0]
            assert [p.learner for p in records[0].participants] == ["a", "b"]
            assert state.round == 1

    def when_grace_applies():
        def it_closes_with_the_late_learner(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                fresh = await syncer.recover()
                await rig.alive("a", "b")
                state = replace(fresh, estimates=TIMINGS)
                await rig.publish("a", 0, (0.0, 1.0), tokens=2)
                task = asyncio.create_task(syncer.run_round(state))
                await spin(lambda: clock.now() >= 0.3 or task.done())
                await rig.publish("b", 0, (1.0, 0.0), tokens=1)
                await spin(task.done)
                [record] = await rig.committed()
                return record, await rig.theta_of(record)

            record, theta = asyncio.run(scenario())

            assert [p.learner for p in record.participants] == ["a", "b"]
            assert values(theta["bias"]) == pytest.approx([0.8, 0.2])
            assert 0.0 < record.grace_seconds < WINDOW

    def when_a_live_learner_never_reports():
        def it_closes_at_the_end_of_the_window(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                fresh = await syncer.recover()
                await rig.alive("a", "b")
                state = replace(fresh, estimates=TIMINGS)
                await rig.publish("a", 0, (0.0, 1.0))
                await completed(syncer.run_round(state))
                [record] = await rig.committed()
                return record

            record = asyncio.run(scenario())

            assert [p.learner for p in record.participants] == ["a"]
            assert record.grace_seconds == pytest.approx(WINDOW)
            assert clock.now() == pytest.approx(WINDOW)

    def when_all_live_learners_reported():
        def it_closes_before_the_window_ends(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                fresh = await syncer.recover()
                await rig.alive("a", "b")
                state = replace(fresh, estimates=TIMINGS)
                await rig.publish("a", 0, (0.0, 1.0), tokens=2)
                await rig.publish("b", 0, (1.0, 0.0), tokens=1)
                await completed(syncer.run_round(state))
                [record] = await rig.committed()
                return record, await rig.theta_of(record)

            record, theta = asyncio.run(scenario())

            assert [p.learner for p in record.participants] == ["a", "b"]
            assert values(theta["bias"]) == pytest.approx([0.8, 0.2])
            assert clock.now() < WINDOW

    def when_the_grace_window_is_fixed():
        def it_closes_at_the_end_of_the_window_from_the_first_round(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, replace(algorithm(quorum=1), grace_seconds=2.0))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                # No estimates yet: the adaptive window would be zero.
                state = await syncer.recover()
                await rig.alive("a", "b")
                await rig.publish("a", 0, (0.0, 1.0))
                await completed(syncer.run_round(state))
                [record] = await rig.committed()
                return record

            record = asyncio.run(scenario())

            assert [p.learner for p in record.participants] == ["a"]
            assert record.grace_seconds == pytest.approx(2.0)
            assert clock.now() == pytest.approx(2.0)

        def it_closes_early_once_every_live_learner_reported(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, replace(algorithm(quorum=1), grace_seconds=2.0))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                state = await syncer.recover()
                await rig.alive("a", "b")
                await rig.publish("a", 0, (0.0, 1.0), tokens=2)
                task = asyncio.create_task(syncer.run_round(state))
                await spin(lambda: clock.now() >= 0.5 or task.done())
                await rig.publish("b", 0, (1.0, 0.0), tokens=1)
                await spin(task.done)
                [record] = await rig.committed()
                return record, await rig.theta_of(record)

            record, theta = asyncio.run(scenario())

            assert [p.learner for p in record.participants] == ["a", "b"]
            assert values(theta["bias"]) == pytest.approx([0.8, 0.2])
            assert 0.5 <= record.grace_seconds < 2.0

    def when_a_learners_lease_expires():
        def it_closes_without_it(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=2))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer(lease_timeout=5.0)
                state = await syncer.recover()
                await rig.alive("a", "b")
                await rig.publish("a", 0, (0.0, 1.0))
                task = asyncio.create_task(syncer.run_round(state))
                await spin(lambda: clock.now() >= 4.0 or task.done())
                still_waiting = not task.done()
                await spin(task.done)
                [record] = await rig.committed()
                return still_waiting, record, clock.now()

            still_waiting, record, now = asyncio.run(scenario())

            assert still_waiting
            assert now >= 5.0
            assert [p.learner for p in record.participants] == ["a"]

    def it_rejects_stale_base_versions_as_absent(clock: FakeClock):
        rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

        async def scenario():
            await rig.create_run()
            syncer = rig.syncer()
            state = await syncer.recover()
            for round in (0, 1):
                await rig.publish("a", round, (0.0, 0.0))
                state = loaded(await completed(syncer.run_round(state)))
            await rig.publish("a", 2, (9.0, 9.0), base_version=0)
            await rig.publish("b", 2, (0.25, 0.75), base_version=1)
            state = loaded(await completed(syncer.run_round(state)))
            records = await rig.committed()
            return records[2], await rig.theta_of(records[2]), state

        record, theta, state = asyncio.run(scenario())

        assert record.rejected == (("a", "stale"),)
        assert [p.learner for p in record.participants] == ["b"]
        assert values(theta["bias"]) == pytest.approx([0.75, 0.25])
        assert state.fragments[0].version == 2

    def when_the_algorithm_sends_deltas_in_bf16():
        def it_accepts_bf16_deltas_and_rejects_the_schemas_dtype(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, replace(algorithm(quorum=1), delta_dtype="BF16"))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                state = await syncer.recover()
                await rig.publish("f32", 0, (0.5, 0.25))
                await rig.publish("bf16", 0, (0.5, 0.25), dtype="BF16")
                await completed(syncer.run_round(state))
                [record] = await rig.committed()
                return record, await rig.theta_of(record)

            record, theta = asyncio.run(scenario())

            assert record.rejected == (("f32", "schema"),)
            assert [p.learner for p in record.participants] == ["bf16"]
            assert values(theta["bias"]) == [0.5, 0.75]

    def when_the_algorithm_sends_deltas_in_e4m3():
        def it_accepts_a_delta_with_its_scale_and_rejects_one_without(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, replace(algorithm(quorum=1), delta_dtype="F8_E4M3"))
            data, scale = default_backend().load(f32(0.5, 0.25)).quantize("F8_E4M3")

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer()
                state = await syncer.recover()
                await rig.publish("with", 0, (), tensors={"bias": data, "bias.scale": scale})
                await rig.publish("without", 0, (), tensors={"bias": data})
                await completed(syncer.run_round(state))
                [record] = await rig.committed()
                return record, await rig.theta_of(record)

            record, theta = asyncio.run(scenario())

            assert record.rejected == (("without", "schema"),)
            assert [p.learner for p in record.participants] == ["with"]
            assert values(theta["bias"]) == [0.5, 0.75]

    def it_rejects_payloads_whose_sha256_mismatches(clock: FakeClock):
        rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

        async def scenario():
            await rig.create_run()
            syncer = rig.syncer()
            state = await syncer.recover()
            await rig.publish("a", 0, (0.0, 1.0))
            [snapshot] = await rig.exchange.snapshots(0)
            await rig.store.delete([snapshot.payload.key])
            await rig.store.put_if_absent(snapshot.payload.key, pack({"bias": f32(9.0, 9.0)}))
            await rig.publish("b", 0, (0.5, 0.5))
            await completed(syncer.run_round(state))
            [record] = await rig.committed()
            return record, await rig.theta_of(record)

        record, theta = asyncio.run(scenario())

        assert record.rejected == (("a", "corrupt"),)
        assert [p.learner for p in record.participants] == ["b"]
        assert values(theta["bias"]) == pytest.approx([0.5, 0.5])

    def it_rejects_zero_steps_schema_mismatches_and_missing_payloads_as_absent(clock: FakeClock):
        rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

        async def scenario():
            await rig.create_run()
            syncer = rig.syncer()
            state = await syncer.recover()
            await rig.publish("zero", 0, (0.0, 0.0), steps=0)
            await rig.publish("digest", 0, (0.0, 0.0), digest="0" * 64)
            await rig.publish("shape", 0, (0.0, 0.0, 0.0))
            await rig.publish("gone", 0, (0.0, 0.0))
            gone = next(s for s in await rig.exchange.snapshots(0) if s.learner == "gone")
            await rig.store.delete([gone.payload.key])
            await rig.publish("good", 0, (0.5, 0.5))
            await completed(syncer.run_round(state))
            [record] = await rig.committed()
            return record, await rig.theta_of(record)

        record, theta = asyncio.run(scenario())

        assert record.rejected == (
            ("digest", "schema"),
            ("gone", "missing"),
            ("shape", "schema"),
            ("zero", "zero_steps"),
        )
        assert [p.learner for p in record.participants] == ["good"]
        assert values(theta["bias"]) == pytest.approx([0.5, 0.5])

    def it_sweeps_the_rounds_snapshots_after_commit(clock: FakeClock):
        rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

        async def leftovers() -> tuple[Sequence[str], Sequence[str]]:
            return await rig.store.list("snapshots/"), await rig.store.list("blobs/snapshots/")

        async def scenario():
            await rig.create_run()
            syncer = rig.syncer()
            state = await syncer.recover()
            await rig.publish("a", 0, (0.0, 0.0))
            state = loaded(await completed(syncer.run_round(state)))
            after_round_0 = await leftovers()
            # A late snapshot for the closed round is swept with the next one.
            await rig.publish("late", 0, (0.0, 0.0))
            await rig.publish("a", 1, (0.0, 0.0))
            state = loaded(await completed(syncer.run_round(state)))
            after_round_1 = await leftovers()
            records = await rig.committed()
            return after_round_0, after_round_1, records, await rig.theta_of(records[0])

        after_round_0, after_round_1, records, theta = asyncio.run(scenario())

        assert after_round_0 == ([], [])
        assert after_round_1 == ([], [])
        assert [record.round for record in records] == [0, 1]
        assert values(theta["bias"]) == [1.0, 1.0]

    def it_keeps_two_cycles_of_round_payloads(clock: FakeClock):
        rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

        async def scenario():
            await rig.create_run()
            syncer = rig.syncer()
            state = await syncer.recover()
            for _ in range(5):
                version = state.fragments[state.round % 2].version
                await rig.publish("a", state.round, (0.0, 0.0), base_version=version)
                state = loaded(await completed(syncer.run_round(state)))
            swept = await rig.store.list("blobs/rounds/0000000000/")
            kept = [await rig.store.list(f"blobs/rounds/{round:010d}/") for round in (1, 2, 3, 4)]
            return swept, kept, state, await rig.syncer().recover()

        swept, kept, state, recovered = asyncio.run(scenario())

        assert list(swept) == []
        assert all(kept)
        assert recovered.round == state.round == 5
        assert recovered.fragments == state.fragments


def describe_recover():
    def when_the_run_is_fresh():
        def it_starts_at_round_zero_from_the_init_payloads(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm())

            async def scenario():
                await rig.create_run()
                return await rig.syncer().recover()

            state = asyncio.run(scenario())

            assert state.round == 0
            assert [fragment.version for fragment in state.fragments] == [0, 0]
            assert [fragment.theta for fragment in state.fragments] == list(INIT)
            # No outer state yet: the outer step takes a missing momentum as zero.
            assert [fragment.outer for fragment in state.fragments] == [{}, {}]
            assert state.estimates == Estimates()

    def when_restarted_after_a_commit():
        def it_resumes_with_the_same_anchor_and_momentum(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(lr=0.7, momentum=0.9))

            async def scenario():
                await rig.create_run()
                state = await rig.syncer().recover()
                await rig.publish("a", 0, (0.0, 1.0))
                after = loaded(await completed(rig.syncer().run_round(state)))
                [record] = await rig.committed()
                return after, await rig.syncer().recover(), record

            after, restarted, record = asyncio.run(scenario())

            assert values(after.fragments[0].outer["momentum.bias"]) != [0.0, 0.0]
            assert restarted.round == after.round == 1
            assert restarted.fragments == after.fragments
            assert restarted.estimates == record.estimates
            assert record.estimates.quorum == record.quorum_seconds
            assert record.estimates.sync is None

    def when_restarted_mid_round():
        def it_recomputes_the_round_once(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=2))

            async def scenario():
                await rig.create_run()
                crashed = rig.syncer()
                state = await crashed.recover()
                await rig.alive("a", "b")
                await rig.publish("a", 0, (0.0, 1.0))
                task = asyncio.create_task(crashed.run_round(state))
                await spin(lambda: clock.now() >= 2 * POLL or task.done())
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                await rig.publish("b", 0, (1.0, 0.0))

                restarted = rig.syncer()
                state = await restarted.recover()
                resumed_at = state.round
                state = loaded(await completed(restarted.run_round(state)))
                return resumed_at, state, await rig.committed()

            resumed_at, state, records = asyncio.run(scenario())

            assert resumed_at == 0
            assert [record.round for record in records] == [0]
            assert [p.learner for p in records[0].participants] == ["a", "b"]
            assert state.round == 1


def describe_election():
    def when_no_claim_exists():
        def it_claims_the_round_and_the_next_one_on_commit(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                syncer = rig.syncer("node-1")
                state = await syncer.recover()
                await rig.publish("a", 0, (0.0, 1.0))
                state = loaded(await completed(syncer.run_round(state)))
                return state, await rig.claimants(0), await rig.claimants(1)

            state, closed, next_round = asyncio.run(scenario())

            assert state.round == 1
            assert list(closed) == []  # swept with the round's snapshots
            assert list(next_round) == ["node-1"]

    def when_another_node_holds_the_claim():
        def it_follows_and_continues_from_the_record(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                closer, follower = rig.syncer("node-1"), rig.syncer("node-2")
                closer_state = await closer.recover()
                follower_state = await follower.recover()
                await rig.publish("a", 0, (0.0, 1.0))
                closer_state = loaded(await completed(closer.run_round(closer_state)))
                following = await completed(follower.run_round(follower_state))
                await rig.publish("a", 1, (0.0, 0.0))
                closer_state = loaded(await completed(closer.run_round(closer_state)))
                following = await completed(follower.run_round(following))
                return following, closer_state, await rig.committed(), await rig.claimants(2)

            following, closer_state, records, claimants = asyncio.run(scenario())

            predicted = sum(
                seconds
                for seconds in (
                    records[1].estimates.step,
                    records[1].estimates.quorum,
                    records[1].estimates.sync,
                )
                if seconds is not None
            )
            assert following == Following(2, max(30 * POLL, 3 * predicted))
            assert closer_state.round == 2
            assert [record.round for record in records] == [0, 1]
            assert list(claimants) == ["node-1"]

    def when_the_claim_holder_dies():
        def it_takes_over_after_the_timeout(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                # A crashed node's lease stays, unrenewed, until the lease timeout (120 s
                # here): to the survivor it is live, and only the takeover timeout applies.
                await rig.alive("node-1")
                dead, survivor = rig.syncer("node-1"), rig.syncer("node-2")
                dead_state = await dead.recover()
                await rig.publish("a", 0, (0.0, 1.0), tokens=2)
                claiming = asyncio.create_task(dead.run_round(dead_state))
                await spin(lambda: clock.now() >= POLL / 2 or claiming.done())
                # The commit succeeded; the dead node's claim on round 1 now has no owner.
                dead_state = loaded(claiming.result())
                await rig.publish("a", 1, (1.0, 0.0))
                task = asyncio.create_task(survivor.run_round(Following(1, timeout=5.0)))
                await spin(lambda: clock.now() >= 4.0 or task.done())
                still_following = not task.done()
                await spin(task.done)
                return still_following, loaded(task.result()), await rig.committed(), clock.now()

            still_following, state, records, now = asyncio.run(scenario())

            assert still_following
            assert now >= 5.0
            assert [record.round for record in records] == [0, 1]
            assert state.round == 2
            assert [p.learner for p in records[1].participants] == ["a"]

    def when_the_claim_holder_leaves():
        def it_takes_over_as_soon_as_the_lease_is_gone(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                await rig.alive("node-1")
                leaving, survivor = rig.syncer("node-1"), rig.syncer("node-2")
                state = await leaving.recover()
                await rig.publish("a", 0, (0.0, 1.0), tokens=2)
                loaded(await completed(leaving.run_round(state)))
                # Round 0 committed and round 1 claimed; then the node closes and releases
                # its lease, leaving the claim behind.
                await rig.exchange.release("node-1")
                await rig.publish("a", 1, (1.0, 0.0))
                state = loaded(await completed(survivor.run_round(Following(1, timeout=5.0))))
                return state, await rig.committed(), clock.now()

            state, records, now = asyncio.run(scenario())

            assert now < 5.0
            assert [record.round for record in records] == [0, 1]
            assert state.round == 2
            assert [p.learner for p in records[1].participants] == ["a"]

    def when_the_claim_holder_stalls():
        def it_takes_over_once_its_lease_is_stale_and_keeps_the_leases_it_watched(
            clock: FakeClock,
        ):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=2))

            async def scenario():
                await rig.create_run()
                await rig.alive("node-1", "a", "b")
                stuck = rig.syncer("node-1")
                state = await stuck.recover()
                await rig.publish("a", 0, (0.0, 1.0))
                await rig.publish("b", 0, (0.0, 1.0))
                loaded(await completed(stuck.run_round(state)))
                # Round 1 is claimed by node-1, which stalls; nobody renews a lease again, "a"
                # publishes and "b" never does.
                await rig.publish("a", 1, (1.0, 0.0))
                survivor = rig.syncer("node-2", lease_timeout=3.0)
                state = loaded(await completed(survivor.run_round(Following(1, timeout=5.0))))
                return state, await rig.committed(), clock.now()

            state, records, now = asyncio.run(scenario())

            # Takeover at the lease timeout, before the 5 s takeover timeout; and since the
            # survivor had watched "b"'s lease unchanged all along, it did not wait another
            # lease timeout for "b" before closing with "a" alone.
            assert now == pytest.approx(3.0, abs=2 * POLL)
            assert [record.round for record in records] == [0, 1]
            assert [p.learner for p in records[1].participants] == ["a"]
            assert state.round == 2

    def when_two_nodes_commit_the_same_round():
        def it_keeps_the_first_record_and_the_loser_follows(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                first, second = rig.syncer("node-1"), rig.syncer("node-1")
                state = await first.recover()
                stale = await second.recover()
                await rig.publish("a", 0, (0.0, 1.0))
                await completed(first.run_round(state))
                # A learner publishes to the closed round; the second node, still at round 0
                # and holding the same claim, closes it with different bytes and loses.
                await rig.publish("b", 0, (1.0, 0.0))
                outcome = await completed(second.run_round(stale))
                [record] = await rig.committed()
                return outcome, record, await rig.theta_of(record)

            outcome, record, theta = asyncio.run(scenario())

            assert isinstance(outcome, Following)
            assert outcome.round == 1
            assert [p.learner for p in record.participants] == ["a"]
            assert values(theta["bias"]) == pytest.approx([1.0, 0.0])


def describe_commit_check():
    def when_the_store_kept_another_closers_record():
        def it_continues_from_the_surviving_record(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                await rig.create_run()
                closer, other = rig.syncer("node-1", on_event=seen.append), rig.syncer("node-2")
                state = await closer.recover()
                stale = await other.recover()
                await rig.publish("a", 0, (0.0, 1.0))
                state = loaded(await completed(closer.run_round(state)))
                # A store that accepts two writes of one key keeps the second: node-2 closes
                # round 0 too, from a snapshot that landed since, and its record replaces
                # node-1's although node-1's write returned success.
                await rig.store.delete([round_key(0)])
                await rig.publish("b", 0, (1.0, 0.0))
                loaded(await completed(other.close(stale)))
                [survivor] = await rig.committed()
                await rig.publish("a", 1, (0.0, 0.0))
                state = loaded(await completed(closer.run_round(state)))
                return survivor, await rig.theta_of(survivor), state, await rig.committed()

            survivor, theta, state, records = asyncio.run(scenario())

            assert values(theta["bias"]) == pytest.approx([0.0, 1.0])
            assert RoundSuperseded("node-1", survivor) in seen
            assert [record.round for record in records] == [0, 1]
            assert state.round == 2
            assert state.fragments[0].theta == theta


def describe_run():
    def it_waits_for_the_manifest_and_stops_after_until_round(clock: FakeClock):
        rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

        async def scenario():
            task = asyncio.create_task(rig.syncer().run(until_round=1))
            await spin(lambda: clock.now() >= 3 * POLL or task.done())
            still_waiting = not task.done()
            await rig.create_run()
            await rig.publish("a", 0, (0.0, 0.0))
            await rig.publish("a", 1, (0.0, 0.0))
            await spin(task.done)
            task.result()
            return still_waiting, await rig.committed()

        still_waiting, records = asyncio.run(scenario())

        assert still_waiting
        assert [record.round for record in records] == [0, 1]

    def when_a_round_fails():
        def it_logs_the_failure_and_retries_from_the_store(
            clock: FakeClock, caplog: pytest.LogCaptureFixture
        ):
            inner = MemoryMailbox()
            rig = Rig(inner, clock, algorithm(quorum=1))
            # The first listing of the round's snapshots fails, inside close().
            mailbox = TrippedMailbox(
                inner,
                lambda operation, key: (
                    operation == "list" and key.startswith("snapshots/0000000000/")
                ),
                RuntimeError("boom"),
            )
            syncer = Syncer(
                algorithm=rig.config,
                mailbox=mailbox,
                node_id="node-1",
                clock=clock,
                poll_interval=POLL,
            )

            async def scenario():
                await rig.create_run()
                await rig.publish("a", 0, (0.0, 1.0))
                caplog.set_level(logging.INFO, logger="gradientmail")
                await completed(syncer.run(until_round=0))
                return await rig.committed()

            records = asyncio.run(scenario())

            assert [record.round for record in records] == [0]
            assert [p.learner for p in records[0].participants] == ["a"]
            assert "node-1: round 0 failed" in caplog.text
            assert "node-1: round 0 committed" in caplog.text

    def when_manifest_algorithm_differs():
        def it_raises_run_config_mismatch(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            other = Syncer(
                algorithm=algorithm(quorum=2),
                mailbox=rig.store,
                node_id="node-2",
                clock=clock,
                poll_interval=POLL,
            )

            async def scenario() -> None:
                await rig.create_run()
                with pytest.raises(RunConfigMismatch):
                    await completed(other.run(until_round=0))

            asyncio.run(scenario())


def describe_exclusive_syncer():
    def when_it_is_live():
        def it_closes_every_round_and_the_other_nodes_only_follow(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            def following(node: str) -> bool:
                return any(isinstance(e, FollowingRound) and e.node == node for e in seen)

            async def scenario():
                await rig.create_run()
                await rig.alive("both")
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated", on_event=seen.append)
                both = rig.syncer("both", on_event=seen.append)
                closing = asyncio.create_task(dedicated.run(until_round=2))
                # Slow steps make the takeover timeout the records predict far longer than a
                # round takes here, every task's sleep advancing the one fake clock.
                await rig.publish("both", 0, (0.0, 0.0), step_seconds=10.0)
                await spin(lambda: any(isinstance(e, RoundCommitted) for e in seen))
                follower = asyncio.create_task(both.run(until_round=2))
                await spin(lambda: following("both"))
                for round in (1, 2):
                    await rig.publish(
                        "both", round, (0.0, 0.0), base_version=round // 2, step_seconds=10.0
                    )
                await spin(lambda: closing.done() and follower.done())
                closing.result()
                follower.result()
                return await rig.claimants(3), await rig.store.list("syncers/")

            claimants, leases = asyncio.run(scenario())

            committed = [(e.node, e.record.round) for e in seen if isinstance(e, RoundCommitted)]
            assert committed == [("dedicated", 0), ("dedicated", 1), ("dedicated", 2)]
            claimed = [e for e in seen if isinstance(e, RoundClaimed)]
            assert claimed == [RoundClaimed("dedicated", 0)]
            assert [e for e in seen if isinstance(e, TookOverRound)] == []
            # It claimed the round after its last commit, and released its lease on returning.
            assert list(claimants) == ["dedicated"]
            assert list(leases) == []
            assert LeaseReleased("dedicated", "syncer") in seen

        def it_is_left_the_first_claim_on_a_round(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                await rig.alive("both")
                await rig.exchange.beat_syncer(syncer_lease_prefix("0-dedicated", 0, "dedicated"))
                await rig.publish("both", 0, (0.0, 1.0))
                both = rig.syncer("both")
                task = asyncio.create_task(both.run_round(await both.recover()))
                await spin(lambda: clock.now() >= 1.0 or task.done())
                running = not task.done()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return running, await rig.claimants(0), await rig.committed()

            running, claimants, records = asyncio.run(scenario())

            # The snapshot is in, but the round waits for the dedicated syncer to claim it.
            assert running
            assert list(claimants) == []
            assert list(records) == []

    def when_its_lease_expires():
        def it_lets_the_other_nodes_take_over_and_claim_the_next_round(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                await rig.create_run()
                await rig.alive("both")
                # A dedicated syncer crashed holding the claim on round 0; its lease stays behind.
                await rig.exchange.beat_syncer(syncer_lease_prefix("0-dedicated", 0, "dedicated"))
                await rig.exchange.claim(0, 1, "dedicated")
                await rig.publish("both", 0, (0.0, 1.0))
                both = rig.syncer("both", lease_timeout=1.0, on_event=seen.append)
                state = loaded(await completed(both.run_round(await both.recover())))
                return state, clock.now(), await rig.claimants(1), await rig.committed()

            state, now, claimants, records = asyncio.run(scenario())

            [takeover] = [e for e in seen if isinstance(e, TookOverRound)]
            assert (takeover.node, takeover.round, takeover.claimants, takeover.cause) == (
                "both",
                0,
                ("dedicated",),
                "lease_gone",
            )
            # At the lease timeout, before the takeover timeout of thirty polls.
            assert now == pytest.approx(1.0, abs=2 * POLL)
            assert [record.round for record in records] == [0]
            assert state.round == 1
            assert list(claimants) == ["both"]

    def when_it_comes_back():
        def it_claims_the_first_round_the_other_nodes_left_unclaimed(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                await rig.create_run()
                await rig.alive("both")
                both = rig.syncer("both", on_event=seen.append)
                # Slow steps make the takeover timeout the record predicts far longer than round 1
                # takes here, every task's sleep advancing the one fake clock.
                await rig.publish("both", 0, (0.0, 1.0), step_seconds=10.0)
                state = loaded(await completed(both.run_round(await both.recover())))
                # Round 0 is committed and round 1 claimed by "both" when the dedicated syncer
                # comes back: it follows round 1, and "both", seeing it, leaves round 2 unclaimed.
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated", on_event=seen.append)
                closing = asyncio.create_task(dedicated.run(until_round=2))
                await spin(
                    lambda: any(
                        isinstance(e, FollowingRound) and e.node == "dedicated" for e in seen
                    )
                )
                await rig.publish("both", 1, (0.0, 0.0))
                loaded(await completed(both.run_round(state)))
                await rig.publish("both", 2, (0.0, 0.0), base_version=1)
                await spin(closing.done)
                closing.result()

            asyncio.run(scenario())

            committed = [(e.node, e.record.round) for e in seen if isinstance(e, RoundCommitted)]
            assert committed == [("both", 0), ("both", 1), ("dedicated", 2)]
            claimed = [e for e in seen if isinstance(e, RoundClaimed)]
            assert claimed == [RoundClaimed("both", 0), RoundClaimed("dedicated", 2)]

    def when_it_closes_a_round():
        def it_waits_for_no_snapshot_of_its_own(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, replace(algorithm(quorum=1), grace_seconds=2.0))

            async def scenario():
                await rig.create_run()
                await rig.alive("a")
                await rig.publish("a", 0, (0.0, 1.0))
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated")
                await completed(dedicated.run(until_round=0))
                [record] = await rig.committed()
                return record

            record = asyncio.run(scenario())

            # Its lease is not a learner's: with "a" reported, the fixed window ended at once.
            assert [p.learner for p in record.participants] == ["a"]
            assert record.grace_seconds == 0.0

    def when_every_learner_left_and_a_node_is_done():
        def it_returns_and_releases_its_lease(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                await rig.create_run()
                await rig.alive("a", "b")
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated", on_event=seen.append)
                task = asyncio.create_task(dedicated.run())
                await rig.exchange.mark_done("a")
                await rig.exchange.release("a")
                await spin(lambda: clock.now() >= 5.0 or task.done())
                running = not task.done()
                await rig.exchange.release("b")
                await spin(task.done)
                task.result()
                return running, await rig.store.list("syncers/")

            running, leases = asyncio.run(scenario())

            # "b" held a live lease: the run was not finished until it released it.
            assert running
            assert list(leases) == []
            assert LeaseReleased("dedicated", "syncer") in seen

    def when_a_learner_crashed():
        def it_returns_once_the_learners_lease_is_stale(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                await rig.alive("crashed")
                await rig.exchange.mark_done("finished")
                dedicated = rig.syncer(
                    "dedicated", lease_timeout=3.0, exclusive="0-dedicated", lease_every=1.0
                )
                await completed(dedicated.run())
                return clock.now()

            assert asyncio.run(scenario()) >= 3.0

    def when_no_node_is_done():
        def it_runs_until_stopped(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated")
                task = asyncio.create_task(dedicated.run())
                await spin(lambda: clock.now() >= 5.0 or task.done())
                running = not task.done()
                dedicated.stop()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return running, await rig.store.list("syncers/")

            running, leases = asyncio.run(scenario())

            assert running
            assert list(leases) == []

    def when_ranks_were_claimed():
        def it_returns_once_every_rank_ever_claimed_is_done(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                await rig.create_run()
                # Neither holds a lease: one finished its rank, the other crashed in its own.
                await rig.exchange.claim_rank(0, "finished")
                await rig.exchange.claim_rank(1, "crashed")
                await rig.exchange.mark_done(rank_id(0))
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated", on_event=seen.append)
                task = asyncio.create_task(dedicated.run())
                await spin(lambda: clock.now() >= 5.0 or task.done())
                running = not task.done()
                # Another node takes the crashed node's rank over and finishes it.
                await rig.exchange.claim_rank(1, "replacement")
                await rig.exchange.mark_done(rank_id(1))
                await spin(task.done)
                task.result()
                return running, await rig.store.list("syncers/")

            running, leases = asyncio.run(scenario())

            assert running
            assert list(leases) == []
            assert LeaseReleased("dedicated", "syncer") in seen

        def it_waits_for_the_learner_that_marked_its_rank_to_leave(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                await rig.alive("a")
                await rig.exchange.claim_rank(0, "a")
                await rig.exchange.mark_done(rank_id(0))
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated")
                task = asyncio.create_task(dedicated.run())
                await spin(lambda: clock.now() >= 5.0 or task.done())
                running = not task.done()
                await rig.exchange.release("a")
                await spin(task.done)
                task.result()
                return running

            # "a" was still waiting for its last rounds to commit.
            assert asyncio.run(scenario())

    def when_it_arrives_before_the_run():
        def it_holds_its_lease_while_it_waits_and_is_left_round_0(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                dedicated = rig.syncer("dedicated", exclusive="0-dedicated", on_event=seen.append)
                closing = asyncio.create_task(dedicated.run(until_round=0))
                await spin(lambda: clock.now() >= 1.0)
                leases = await rig.store.list("syncers/")
                await rig.create_run()
                await rig.alive("both")
                both = rig.syncer("both", on_event=seen.append)
                follower = asyncio.create_task(both.run(until_round=0))
                await spin(lambda: any(isinstance(e, FollowingRound) for e in seen))
                await rig.publish("both", 0, (0.0, 0.0), step_seconds=10.0)
                await spin(lambda: closing.done() and follower.done())
                closing.result()
                follower.result()
                return leases

            leases = asyncio.run(scenario())

            assert len(leases) == 1 and leases[0].startswith("syncers/0-dedicated/")
            assert [e for e in seen if isinstance(e, RoundClaimed)] == [
                RoundClaimed("dedicated", 0)
            ]
            committed = [(e.node, e.record.round) for e in seen if isinstance(e, RoundCommitted)]
            assert committed == [("dedicated", 0)]

    def when_it_arrives_while_a_round_is_collecting():
        def it_is_left_the_next_round_by_the_node_closing_this_one(clock: FakeClock):
            rig = Rig(MemoryMailbox(), clock, algorithm(quorum=1))
            seen: list[Event] = []

            async def scenario():
                await rig.create_run()
                await rig.alive("both")
                both = rig.syncer("both", on_event=seen.append)
                task = asyncio.create_task(both.run(until_round=0))
                await spin(lambda: RoundClaimed("both", 0) in seen)
                await rig.exchange.beat_syncer(syncer_lease_prefix("0-dedicated", 0, "dedicated"))
                await rig.publish("both", 0, (0.0, 0.0))
                await spin(task.done)
                task.result()
                return await rig.claimants(1)

            claimants = asyncio.run(scenario())

            committed = [(e.node, e.record.round) for e in seen if isinstance(e, RoundCommitted)]
            assert committed == [("both", 0)]
            assert list(claimants) == []


def describe_elected_syncer():
    """``exclusive="1-elected"``: the store's clock is the fake one too, so that a candidate's
    one-second wait is one step of it."""

    def when_no_node_owns_the_role():
        def it_stands_wins_and_holds_its_lease(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm())

            async def scenario():
                await rig.alive("learner")
                candidate = rig.syncer("candidate", exclusive="1-elected")
                won = await candidate.elect()
                return won, candidate.holds_role, await rig.store.list("syncers/")

            won, holds_role, leases = asyncio.run(scenario())

            assert won
            assert holds_role
            assert list(leases) == [beat_key(syncer_lease_prefix("1-elected", 0, "candidate"), 0)]
            # Its one wait: a second of the store's clock.
            assert clock.now() == pytest.approx(1.0)

        def it_loses_to_an_earlier_arrival_whose_lease_lands_while_it_waits(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm())
            clock.advance(100.0)
            early = syncer_lease_prefix("1-elected", 99_999, "early")

            async def scenario():
                await rig.alive("learner")
                late = rig.syncer("late", exclusive="1-elected")
                standing = asyncio.create_task(late.elect())
                await spin(lambda: clock.now() >= 101.0)
                await rig.exchange.beat_syncer(early)
                await spin(standing.done)
                return standing.result(), late.holds_role, await rig.store.list("syncers/")

            won, holds_role, leases = asyncio.run(scenario())

            assert not won
            assert not holds_role
            # It released its own lease; the earlier one, first in byte order, owns the role.
            assert list(leases) == [beat_key(early, 0)]

    def when_a_node_owns_the_role():
        def it_does_not_stand(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm())
            owner = syncer_lease_prefix("1-elected", 0, "owner")

            async def scenario():
                await rig.exchange.beat_syncer(owner)
                late = rig.syncer("late", exclusive="1-elected")
                return await late.elect(), await rig.store.list("syncers/")

            won, leases = asyncio.run(scenario())

            assert not won
            assert list(leases) == [beat_key(owner, 0)]
            assert clock.now() == 0.0

    def when_no_other_node_trains():
        def it_does_not_stand(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm())

            async def scenario():
                # Its own learner's lease is the only one: elected, it would leave none training.
                await rig.alive("alone")
                alone = rig.syncer("alone", exclusive="1-elected")
                return await alone.elect(), await rig.store.list("syncers/")

            won, leases = asyncio.run(scenario())

            assert not won
            assert list(leases) == []
            assert clock.now() == 0.0

    def when_a_dedicated_syncer_arrives():
        def it_gives_the_role_up_and_does_not_stand_while_it_lives(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm(quorum=1))
            recording = RecordingMailbox(rig.store)
            seen: list[Event] = []
            dedicated = syncer_lease_prefix("0-dedicated", 0, "dedicated")
            elected = Syncer(
                algorithm=rig.config,
                mailbox=recording,
                node_id="elected",
                clock=clock,
                poll_interval=POLL,
                exclusive="1-elected",
                on_event=seen.append,
            )

            async def scenario():
                await rig.create_run()
                await rig.alive("learner")
                task = asyncio.create_task(elected.run())
                await spin(lambda: elected.holds_role)
                await rig.exchange.beat_syncer(dedicated)
                await spin(lambda: not elected.holds_role)
                given_up_at = clock.now()
                await spin(lambda: clock.now() >= given_up_at + 10 * POLL or task.done())
                running = not task.done()
                leases = await rig.store.list("syncers/")
                elected.stop()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return running, leases

            running, leases = asyncio.run(scenario())

            assert running
            assert list(leases) == [beat_key(dedicated, 0)]
            assert LeaseReleased("elected", "syncer") in seen
            # One election, won; none since, the dedicated syncer owning the role.
            elections = [key for key in recording.written if key.startswith("syncers/1-elected/")]
            assert len(elections) == 1

    def when_the_elected_node_leaves():
        def it_elects_a_remaining_candidate_on_its_next_poll(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm(quorum=1))

            async def scenario():
                await rig.create_run()
                await rig.alive("learner")
                first = rig.syncer("first", exclusive="1-elected")
                second = rig.syncer("second", exclusive="1-elected")
                leaving = asyncio.create_task(first.run())
                await spin(lambda: first.holds_role)
                staying = asyncio.create_task(second.run())
                await spin(lambda: clock.now() >= 20 * POLL)
                standing_by = not second.holds_role
                first.stop()
                with contextlib.suppress(asyncio.CancelledError):
                    await leaving
                await spin(lambda: second.holds_role)
                leases = await rig.store.list("syncers/")
                second.stop()
                with contextlib.suppress(asyncio.CancelledError):
                    await staying
                return standing_by, leases

            standing_by, leases = asyncio.run(scenario())

            assert standing_by
            [lease] = leases
            assert lease.startswith("syncers/1-elected/")
            assert "-second/" in lease

    def when_the_elected_node_crashed():
        def it_takes_over_its_round_and_elects_itself_once_the_lease_expires(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm(quorum=1))
            seen: list[Event] = []
            crashed = syncer_lease_prefix("1-elected", 0, "crashed")

            async def scenario():
                await rig.create_run()
                # The elected syncer crashed holding the claim on round 0; its lease stays behind.
                await rig.exchange.beat_syncer(crashed)
                await rig.exchange.claim(0, 1, "crashed")
                await rig.publish("both", 0, (0.0, 1.0))
                training = asyncio.create_task(rig.training("both"))
                survivor = rig.syncer(
                    "survivor",
                    lease_timeout=1.0,
                    exclusive="1-elected",
                    lease_every=0.5,
                    on_event=seen.append,
                )
                task = asyncio.create_task(survivor.run())
                await spin(
                    lambda: survivor.holds_role and any(isinstance(e, RoundCommitted) for e in seen)
                )
                leases = await rig.store.list("syncers/")
                survivor.stop()
                training.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                with contextlib.suppress(asyncio.CancelledError):
                    await training
                return leases, await rig.committed()

            leases, records = asyncio.run(scenario())

            [takeover] = [e for e in seen if isinstance(e, TookOverRound)]
            assert (takeover.node, takeover.round, takeover.claimants, takeover.cause) == (
                "survivor",
                0,
                ("crashed",),
                "lease_gone",
            )
            assert [record.round for record in records] == [0]
            # The crashed node's lease stays, stale; the survivor's, after it, owns the role.
            assert [key.split("/")[2].split("-", 1)[1] for key in leases] == [
                "crashed",
                "survivor",
            ]

    def when_no_learner_is_left_while_units_of_its_queue_are_unclaimed():
        def it_gives_the_role_up_and_does_not_stand_again(clock: FakeClock):
            rig = Rig(MemoryMailbox(clock=clock), clock, algorithm(quorum=1))
            seen: list[Event] = []
            elected = rig.syncer(
                "elected",
                exclusive="1-elected",
                units=lambda: frozenset({"unit-0", "unit-1"}),
                on_event=seen.append,
            )

            async def scenario():
                await rig.create_run()
                await rig.alive("learner")
                await rig.exchange.claim_unit("unit-0", "learner")
                await rig.exchange.mark_done("learner")
                task = asyncio.create_task(elected.run())
                await spin(lambda: elected.holds_role)
                await rig.exchange.release("learner")
                await spin(lambda: not elected.holds_role or task.done())
                given_up_at = clock.now()
                await spin(lambda: clock.now() >= given_up_at + 10 * POLL or task.done())
                running = not task.done()
                leases = await rig.store.list("syncers/")
                elected.stop()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return running, leases

            running, leases = asyncio.run(scenario())

            # A node done and none live would have finished the run; "unit-1" is left to train.
            assert running
            assert list(leases) == []
            assert LeaseReleased("elected", "syncer") in seen

        def when_every_unit_is_claimed():
            def it_returns_once_the_run_is_finished(clock: FakeClock):
                rig = Rig(MemoryMailbox(clock=clock), clock, algorithm(quorum=1))
                elected = rig.syncer(
                    "elected", exclusive="1-elected", units=lambda: frozenset({"unit-0", "unit-1"})
                )

                async def scenario():
                    await rig.create_run()
                    await rig.alive("learner")
                    await rig.exchange.claim_unit("unit-0", "learner")
                    await rig.exchange.claim_unit("unit-1", "learner")
                    await rig.exchange.mark_done("learner")
                    task = asyncio.create_task(elected.run())
                    await spin(lambda: elected.holds_role)
                    await rig.exchange.release("learner")
                    await spin(task.done)
                    task.result()
                    return await rig.store.list("syncers/")

                assert list(asyncio.run(scenario())) == []
                assert elected.holds_role
