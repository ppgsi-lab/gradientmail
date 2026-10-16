"""Protocol records and the Exchange over a real mailbox.

The boundary is the bytes left in the store: write ordering, content addressing, fencing and
sweeping are all observed by reading the mailbox back, never by watching calls. ``FaultyMailbox``
loses one response (the write lands, the caller sees ``MailboxUnavailable``) and cancellation
during the retry backoff is the crash.
"""

import asyncio
import contextlib
import hashlib
import json
from array import array
from collections.abc import Callable, Mapping, Sequence

import pytest
from fakes import FakeClock, FaultyMailbox

from gradientmail.algorithms import Estimates, Participant
from gradientmail.algorithms.diloco import DiLoCo
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import (
    Committed,
    Exchange,
    Manifest,
    ManifestDraft,
    PayloadCorrupt,
    PayloadRef,
    ProtocolError,
    RoundRecord,
    RoundRecordDraft,
    Snapshot,
    SnapshotDraft,
    Superseded,
    decode_manifest,
    decode_round_record,
    decode_snapshot,
    encode,
)
from gradientmail.serialization import ParameterSpec, Schema, TensorBytes, pack


def f32(*values: float) -> TensorBytes:
    return TensorBytes("F32", (len(values),), array("f", values).tobytes())


SCHEMA = Schema((ParameterSpec("bias", "F32", (2,)), ParameterSpec("weight", "F32", (4,))))
PLAN = (("bias",), ("weight",))
THETA = {"bias": f32(1.0, 2.0)}
OUTER = {"momentum.bias": f32(0.0, 0.0)}


def manifest_draft(created_by: str) -> ManifestDraft:
    return ManifestDraft(
        schema=SCHEMA,
        plan=PLAN,
        algorithm=DiLoCo(sync_every=100, fragments=2).to_json(),
        created_by=created_by,
    )


def snapshot_draft(learner: str, round: int = 0) -> SnapshotDraft:
    return SnapshotDraft(
        learner=learner,
        round=round,
        local_step=12,
        steps=12,
        tokens=384,
        base_version=0,
        rounds_seen=-1,
        step_seconds=0.25,
        schema_digest=SCHEMA.digest(),
    )


def round_draft(round: int) -> RoundRecordDraft:
    return RoundRecordDraft(
        round=round,
        step=12 * (round + 1),
        fragment=round % 2,
        version=round // 2 + 1,
        participants=(Participant("learner-a", 12, 12, 384, 384 * 384 / 12),),
        rejected=(("learner-b", "stale"), ("learner-c", "zero_steps")),
        quorum_seconds=0.5,
        grace_seconds=0.0,
        estimates=Estimates(quorum=0.5, sync=0.125),
    )


def ref(key: str) -> PayloadRef:
    return PayloadRef(key, hashlib.sha256(key.encode()).hexdigest(), 64)


def describe_records():
    manifest = Manifest(
        schema=SCHEMA,
        plan=PLAN,
        algorithm=DiLoCo(sync_every=100, fragments=2).to_json(),
        created_by="learner-a",
        init=(ref("blobs/init/000/a.safetensors"), ref("blobs/init/001/b.safetensors")),
    )
    snapshot = Snapshot(
        learner="learner-a",
        round=3,
        local_step=37,
        steps=12,
        tokens=4096,
        base_version=1,
        rounds_seen=2,
        step_seconds=0.125,
        schema_digest=SCHEMA.digest(),
        payload=ref("blobs/snapshots/0000000003/learner-a/c.safetensors"),
    )
    record = RoundRecord(
        round=3,
        step=150,
        fragment=1,
        version=2,
        participants=(
            Participant("learner-a", 37, 12, 4096, 4096 * 4096 / 12),
            Participant("learner-b", 40, 10, 2048, 2048 * 2048 / 10),
        ),
        rejected=(("learner-c", "stale"), ("learner-d", "missing")),
        theta=ref("blobs/rounds/0000000003/d.safetensors"),
        outer=ref("blobs/rounds/0000000003/e.safetensors"),
        quorum_seconds=1.5,
        grace_seconds=0.25,
        estimates=Estimates(step=0.125, quorum=1.5, sync=None),
    )

    def it_round_trips_every_record_through_json():
        assert decode_manifest(encode(manifest)) == manifest
        assert decode_snapshot(encode(snapshot)) == snapshot
        assert decode_round_record(encode(record)) == record

    def when_a_record_is_malformed():
        def it_raises_protocol_error():
            fields = json.loads(encode(snapshot))
            without_tokens = {key: value for key, value in fields.items() if key != "tokens"}
            cases: Sequence[tuple[Callable[[bytes], object], bytes]] = (
                (decode_snapshot, b"not json"),
                (decode_snapshot, b"[]"),
                (decode_snapshot, json.dumps(without_tokens).encode()),
                (decode_snapshot, json.dumps({**fields, "round": "3"}).encode()),
                (
                    decode_round_record,
                    json.dumps(
                        {**json.loads(encode(record)), "rejected": [["learner-c", "late"]]}
                    ).encode(),
                ),
                (decode_manifest, json.dumps({**json.loads(encode(manifest)), "plan": 7}).encode()),
            )

            for decode, data in cases:
                with pytest.raises(ProtocolError):
                    decode(data)


def describe_publish_snapshot():
    def when_crashed_between_payload_and_record_and_retried():
        def it_leaves_exactly_one_record_whose_sha_matches_its_payload(clock: FakeClock):
            store = MemoryMailbox()

            async def crash_while_publishing() -> None:
                exchange = Exchange(FaultyMailbox(store, fail_after=1), clock=clock)
                task = asyncio.create_task(exchange.publish_snapshot(snapshot_draft("a"), THETA))
                # The lost response sends the publisher into its backoff sleep, which moves the
                # fake clock; cancelling there is a process dying before it retried.
                while clock.now() == 0.0 and not task.done():
                    await asyncio.sleep(0)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            async def scenario():
                await crash_while_publishing()
                orphans = await store.list("blobs/snapshots/")
                records_after_crash = await store.list("snapshots/")

                exchange = Exchange(store, clock=clock)
                await exchange.publish_snapshot(snapshot_draft("a"), THETA)
                snapshots = await exchange.snapshots(0)
                payloads = await store.list("blobs/snapshots/")
                stored = await store.get(snapshots[0].payload.key)
                return orphans, records_after_crash, snapshots, payloads, stored

            orphans, records_after_crash, snapshots, payloads, stored = asyncio.run(scenario())

            assert len(orphans) == 1
            assert list(records_after_crash) == []
            assert len(snapshots) == 1
            assert list(payloads) == [snapshots[0].payload.key]
            assert stored is not None
            assert hashlib.sha256(stored).hexdigest() == snapshots[0].payload.sha256


def describe_commit():
    def when_two_nodes_commit_the_same_round():
        def it_stores_one_and_hands_its_record_to_the_other(clock: FakeClock):
            store = MemoryMailbox()
            first = Exchange(store, clock=clock)
            second = Exchange(store, clock=clock)
            other_theta = {"bias": f32(3.0, 4.0)}

            async def scenario():
                won = await first.commit(round_draft(0), THETA, OUTER)
                lost = await second.commit(round_draft(0), other_theta, OUTER)
                records = await second.rounds_after(-1)
                return won, lost, records, await second.fetch(records[0].theta)

            won, lost, records, theta = asyncio.run(scenario())

            assert isinstance(won, Committed)
            assert isinstance(lost, Superseded)
            assert lost.record == won.record == records[0]
            assert theta == THETA


def describe_claims():
    def it_grants_each_attempt_once_and_lists_claimants_in_order(clock: FakeClock):
        exchange = Exchange(MemoryMailbox(), clock=clock)

        async def scenario():
            first = await exchange.claim(3, 1, "node-a")
            again = await exchange.claim(3, 1, "node-a")
            taken = await exchange.claim(3, 1, "node-b")
            takeover = await exchange.claim(3, 2, "node-b")
            return first, again, taken, takeover, await exchange.claims(3), await exchange.claims(4)

        first, again, taken, takeover, claimants, none = asyncio.run(scenario())

        assert (first, again, taken, takeover) == (True, True, False, True)
        assert list(claimants) == ["node-a", "node-b"]
        assert list(none) == []

    def it_is_swept_with_the_rounds_snapshots(clock: FakeClock):
        store = MemoryMailbox()
        exchange = Exchange(store, clock=clock)

        async def scenario():
            await exchange.claim(0, 1, "node-a")
            await exchange.publish_snapshot(snapshot_draft("learner-a", 0), THETA)
            await exchange.sweep_snapshots(0)
            return await store.list("")

        assert list(asyncio.run(scenario())) == []

    def when_the_same_syncer_retries_after_a_lost_response():
        def it_succeeds(clock: FakeClock):
            store = MemoryMailbox()
            exchange = Exchange(FaultyMailbox(store, fail_after=3), clock=clock)

            async def scenario():
                await exchange.commit(round_draft(0), THETA, OUTER)
                records = await exchange.rounds_after(-1)
                theta = await exchange.fetch(records[0].theta)
                outer = await exchange.fetch(records[0].outer)
                return records, theta, outer

            records, theta, outer = asyncio.run(scenario())

            assert [record.round for record in records] == [0]
            assert (theta, outer) == (THETA, OUTER)
            assert clock.now() > 0.0


def describe_leases():
    def it_keeps_one_beat_per_learner_and_counts_on_from_a_restart(clock: FakeClock):
        store = MemoryMailbox()
        exchange = Exchange(store, clock=clock)

        async def scenario():
            await exchange.beat("a")
            await exchange.beat("a")
            await exchange.beat("b")
            keys = await store.list("leases/")
            leases = await exchange.leases()
            await exchange.release("a")
            return keys, leases, await exchange.leases()

        keys, leases, released = asyncio.run(scenario())

        assert list(keys) == ["leases/a/0000000001.json", "leases/b/0000000000.json"]
        assert dict(leases) == {"a": 1, "b": 0}
        assert dict(released) == {"b": 0}


def describe_fetch():
    def when_payload_bytes_are_tampered():
        def it_raises_payload_corrupt(clock: FakeClock):
            store = MemoryMailbox()
            exchange = Exchange(store, clock=clock)

            async def scenario() -> None:
                await exchange.publish_snapshot(snapshot_draft("a"), THETA)
                [snapshot] = await exchange.snapshots(0)
                await store.delete([snapshot.payload.key])
                await store.put_if_absent(snapshot.payload.key, pack({"bias": f32(9.0, 9.0)}))
                with pytest.raises(PayloadCorrupt):
                    await exchange.fetch(snapshot.payload)

            asyncio.run(scenario())

    def when_payload_was_swept():
        def it_returns_none(clock: FakeClock):
            store = MemoryMailbox()
            exchange = Exchange(store, clock=clock)

            async def scenario():
                await exchange.publish_snapshot(snapshot_draft("a"), THETA)
                [snapshot] = await exchange.snapshots(0)
                await exchange.sweep_snapshots(0)
                return await exchange.fetch(snapshot.payload), await exchange.snapshots(0)

            fetched, remaining = asyncio.run(scenario())

            assert fetched is None
            assert list(remaining) == []


def describe_compression():
    def when_a_node_writes_zstd():
        def it_names_the_codec_in_the_key_and_any_exchange_reads_it(clock: FakeClock):
            store = MemoryMailbox()
            writer = Exchange(store, clock=clock, compression="zstd")
            reader = Exchange(store, clock=clock)

            async def scenario():
                await writer.publish_snapshot(snapshot_draft("a"), THETA)
                [snapshot] = await reader.snapshots(0)
                stored = await store.get(snapshot.payload.key)
                return snapshot.payload, stored, await reader.fetch(snapshot.payload)

            payload, stored, fetched = asyncio.run(scenario())

            assert payload.key.endswith(".safetensors.zst")
            assert stored is not None and stored != pack(THETA)
            assert (hashlib.sha256(stored).hexdigest(), len(stored)) == (
                payload.sha256,
                payload.size,
            )
            assert fetched == THETA

    def when_a_key_names_no_codec():
        def it_raises_protocol_error(clock: FakeClock):
            store = MemoryMailbox()
            exchange = Exchange(store, clock=clock)
            blob = pack(THETA)
            unknown = PayloadRef(
                "blobs/rounds/0000000000/x.bin", hashlib.sha256(blob).hexdigest(), len(blob)
            )

            async def scenario() -> None:
                await store.put_if_absent(unknown.key, blob)
                with pytest.raises(ProtocolError):
                    await exchange.fetch(unknown)

            asyncio.run(scenario())


def describe_sweep_blobs():
    def it_removes_the_rounds_payloads_but_never_its_record(clock: FakeClock):
        store = MemoryMailbox()
        exchange = Exchange(store, clock=clock)

        async def scenario():
            await exchange.commit(round_draft(0), THETA, OUTER)
            await exchange.sweep_blobs(0)
            records = await exchange.rounds_after(-1)
            theta = await exchange.fetch(records[0].theta)
            outer = await exchange.fetch(records[0].outer)
            return records, theta, outer

        records, theta, outer = asyncio.run(scenario())

        assert [record.round for record in records] == [0]
        assert (theta, outer) == (None, None)


def describe_rounds_after():
    def it_returns_the_records_after_the_cursor_in_round_order(clock: FakeClock):
        exchange = Exchange(MemoryMailbox(), clock=clock)

        async def scenario():
            for round in (0, 1, 2):
                await exchange.commit(round_draft(round), THETA, OUTER)
            return [
                [record.round for record in await exchange.rounds_after(cursor)]
                for cursor in (-1, 0, 2)
            ]

        assert asyncio.run(scenario()) == [[0, 1, 2], [1, 2], []]


def describe_create_run():
    def when_two_learners_initialize_concurrently():
        def it_gives_both_the_same_manifest_and_init(clock: FakeClock):
            store = MemoryMailbox()
            init: Mapping[str, Sequence[Mapping[str, TensorBytes]]] = {
                "learner-a": ({"bias": f32(1.0, 2.0)}, {"weight": f32(1.0, 2.0, 3.0, 4.0)}),
                "learner-b": ({"bias": f32(5.0, 6.0)}, {"weight": f32(5.0, 6.0, 7.0, 8.0)}),
            }

            async def scenario():
                exchange = Exchange(store, clock=clock)
                before = await exchange.manifest()
                async with asyncio.TaskGroup() as group:
                    outcomes = [
                        group.create_task(
                            Exchange(store, clock=clock).create_run(
                                manifest_draft(learner), tensors
                            )
                        )
                        for learner, tensors in init.items()
                    ]
                winner = await exchange.manifest()
                assert winner is not None
                fetched = tuple([await exchange.fetch(ref) for ref in winner.init])
                blobs = await store.list("blobs/init/")
                return before, [task.result() for task in outcomes], winner, fetched, blobs

            before, returned, winner, fetched, blobs = asyncio.run(scenario())

            assert before is None
            assert returned == [winner, winner]
            assert fetched == init[winner.created_by]
            assert len(blobs) == len(PLAN)

    def when_two_learners_initialize_with_identical_parameters():
        def it_keeps_the_payloads_they_share(clock: FakeClock):
            # Same bytes mean the same content-addressed keys, so the loser's cleanup must not
            # take the winner's Θ⁰ with it.
            store = MemoryMailbox()
            init = ({"bias": f32(1.0, 2.0)}, {"weight": f32(1.0, 2.0, 3.0, 4.0)})

            async def scenario():
                async with asyncio.TaskGroup() as group:
                    for learner in ("learner-a", "learner-b"):
                        group.create_task(
                            Exchange(store, clock=clock).create_run(manifest_draft(learner), init)
                        )
                exchange = Exchange(store, clock=clock)
                winner = await exchange.manifest()
                assert winner is not None
                return tuple([await exchange.fetch(ref) for ref in winner.init])

            assert asyncio.run(scenario()) == init
