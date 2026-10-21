"""INTRO.md's example end to end on MemoryMailbox, through the package's public names.

One ``Session`` over a ``MemoryAdapter`` model runs its training loop over a generator; the
session's own syncer closes the rounds, so nothing else runs. What is observed is what a user
would: the round records in the store, the learner they name and the parameters the model holds
once it is aligned with the newest commit.
"""

import asyncio
import itertools
import json
import os
import re
import threading
import time
from array import array
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial

import pytest
from fakes import DrivenClock, FakeClock, RecordingMailbox

import gradientmail as gm
from gradientmail import events
from gradientmail.algorithms import Precision
from gradientmail.clock import MonotonicClock
from gradientmail.protocol import Exchange, ProtocolError, rank_key, syncer_lease_prefix


def f32(*values: float) -> gm.TensorBytes:
    return gm.TensorBytes("F32", (len(values),), array("f", values).tobytes())


def floats(tensor: gm.TensorBytes) -> list[float]:
    return array("f", tensor.data).tolist()


# Eight tensors, so INTRO's ``fragments=8`` is the effective fragment count.
MODEL = {f"block{i}.weight": f32(float(i), float(i)) for i in range(8)}


def train_step(model: gm.MemoryAdapter, batch: Sequence[int]) -> int:
    """Synthetic training: every parameter grows by the batch size, its sample count."""
    model.apply(
        {
            name: f32(*(value + len(batch) for value in array("f", tensor.data)))
            for name, tensor in model.tensors.items()
        }
    )
    return len(batch)


def until(session: gm.Session, rounds: int) -> Iterator[Sequence[int]]:
    """Batches of one sample until the model is aligned with round ``rounds``."""
    deadline = time.monotonic() + 20.0
    while session.rounds_seen < rounds:
        assert time.monotonic() < deadline, f"the model never received round {rounds}"
        yield (1,)
        time.sleep(0.001)


def leases(mailbox: gm.Mailbox) -> set[str]:
    return set(asyncio.run(Exchange(mailbox, clock=MonotonicClock()).leases()))


def waited(condition: Callable[[], bool], what: str) -> None:
    """Poll until ``condition`` holds."""
    deadline = time.monotonic() + 20.0
    while not condition():
        assert time.monotonic() < deadline, what
        time.sleep(0.001)


def holding_leases(mailbox: gm.Mailbox, *nodes: str) -> None:
    """Poll until every one of ``nodes`` holds a lease."""
    waited(lambda: set(nodes) <= leases(mailbox), f"{nodes} never all held a lease")


def threads_of(node: str) -> list[str]:
    return [t.name for t in threading.enumerate() if t.name.endswith(f"-{node}")]


def never_called(batch: Sequence[int]) -> int:
    raise AssertionError("a syncer trains nothing")


def counted(batch: Sequence[int]) -> int:
    """A step that trains nothing: the samples of the batch."""
    return len(batch)


def failing(batch: Sequence[int]) -> int:
    raise RuntimeError("training failed")


def started[T](work: Callable[[], T]) -> Future[T]:
    """``work`` running on a daemon thread, so that a test failing while it blocks still ends."""
    future: Future[T] = Future()

    def run() -> None:
        try:
            future.set_result(work())
        except BaseException as exc:
            future.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return future


def ranked(
    mailbox: gm.Mailbox,
    name: str,
    ranks: int,
    *,
    role: gm.Role = "both",
    lease_every: float = 60.0,
    lease_timeout: float = 120.0,
    callbacks: Sequence[gm.Callback] = (gm.log_events,),
) -> gm.Session:
    return gm.Session(
        gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
        algorithm=gm.DiLoCo(sync_every=4, fragments=2),
        mailbox=mailbox,
        role=role,
        node_id=name,
        ranks=ranks,
        poll_interval=0.01,
        lease_every=lease_every,
        lease_timeout=lease_timeout,
        callbacks=callbacks,
    )


def driven(clock: DrivenClock, *joining: Future[None]) -> None:
    """Advance the store's clock a second at a time until every join has returned."""
    deadline = time.monotonic() + 20.0
    while not all(future.done() for future in joining):
        assert time.monotonic() < deadline, "a node never took a rank"
        clock.advance(1.0)
        time.sleep(0.01)
    for future in joining:
        future.result()


def claimants(mailbox: gm.Mailbox, rank: int) -> list[str]:
    """The nodes whose claims on ``rank`` are in the store, in byte order of their keys."""
    keys = asyncio.run(mailbox.list(f"ranks/{rank:03d}/"))
    return [key.removesuffix(".json").split("-", 1)[1] for key in keys]


def elected_mid_unit() -> tuple[
    dict[str, list[int]], list[str], dict[str, str | None], list[str], list[gm.Event]
]:
    """Two ``either`` nodes on a queue of six four-batch units. A dedicated syncer that never
    renews its lease owns the role while "paused" trains; its lease is deleted while "paused" is
    held in the first batch of its first unit, so that "paused" is elected in the middle of
    that unit, a learner that never trains holding a lease for it to stand. "other" trains once
    "paused" has stopped, the idle learner gone once "other" holds its lease.

    Returns the batches each node trained, the units "paused" claimed, every unit's owner, the
    ``done/`` keys and the events, in order."""
    store = gm.MemoryMailbox(clock=FakeClock())
    exchange = Exchange(store, clock=MonotonicClock())
    ghost = syncer_lease_prefix("0-dedicated", 0, "ghost")
    asyncio.run(exchange.beat_syncer(ghost))
    asyncio.run(exchange.beat("idle"))
    recording = RecordingMailbox(store)
    algorithm = gm.DiLoCo(sync_every=4, fragments=2)
    units = {f"unit-{i}": range(4 * i, 4 * i + 4) for i in range(6)}
    trained: dict[str, list[int]] = {"paused": [], "other": []}
    seen: list[gm.Event] = []
    elected = threading.Event()
    stopped = threading.Event()

    def on_event(event: gm.Event) -> None:
        seen.append(event)
        if isinstance(event, events.BecameSyncer):
            stopped.set()

    def session(name: str, mailbox: gm.Mailbox) -> gm.Session:
        return gm.Session(
            gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
            algorithm=algorithm,
            mailbox=mailbox,
            role="either",
            node_id=name,
            poll_interval=0.01,
            callbacks=(on_event,),
        )

    def step(name: str, batch: int) -> int:
        trained[name].append(batch)
        if name == "paused":
            assert elected.wait(20.0)
        return 1

    def claimed() -> list[str]:
        return [
            key.removeprefix("units/").removesuffix(".json")
            for key in recording.written
            if key.startswith("units/")
        ]

    paused = session("paused", recording)
    paused.join()
    running = started(partial(paused.run, partial(step, "paused"), gm.SharedQueue(units, iter)))
    # Held in its first batch, the node loads ahead: the rest of that unit and two more.
    waited(lambda: len(claimed()) >= 3, "the node never loaded ahead")
    asyncio.run(exchange.release_syncer(ghost))
    waited(lambda: bool(asyncio.run(store.list("syncers/1-elected/"))), "nobody stood")
    elected.set()
    assert stopped.wait(20.0)
    other = session("other", store)
    other.join()
    holding_leases(store, "other")
    asyncio.run(exchange.release("idle"))
    other.run(partial(step, "other"), gm.SharedQueue(units, iter))
    running.result(timeout=20.0)

    owners = {
        unit: json.loads(data) if (data := asyncio.run(store.get(f"units/{unit}.json"))) else None
        for unit in units
    }
    return trained, claimed(), owners, list(asyncio.run(store.list("done/"))), seen


def describe_session():
    def it_runs_intros_example_against_memory_mailbox():
        mailbox = gm.MemoryMailbox()
        algorithm = gm.DiLoCo(sync_every=100, fragments=8)
        model = gm.MemoryAdapter(MODEL)
        session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)

        def dataloader() -> Iterator[Sequence[int]]:
            """INTRO's loop, run until the model is aligned with round 1."""
            deadline = time.monotonic() + 20.0
            while session.rounds_seen < 1:
                assert time.monotonic() < deadline, "the model never received round 1"
                yield (1, 2, 3)
                time.sleep(0.001)

        session.run(partial(train_step, model), dataloader())
        rounds_seen = session.rounds_seen

        exchange = Exchange(mailbox, clock=MonotonicClock())
        records = asyncio.run(exchange.rounds_after(-1))
        assert [record.round for record in records][:2] == [0, 1]
        (node_id,) = {p.learner for record in records for p in record.participants}
        assert re.fullmatch(r"[a-z0-9-]{1,64}", node_id)
        assert node_id.endswith(f"-{os.getpid()}")
        # The last round aligned with is the newest version of its fragment: the model holds it.
        assert rounds_seen >= 1
        theta = asyncio.run(exchange.fetch(records[rounds_seen].theta))
        assert theta is not None
        assert {name: model.tensors[name] for name in theta} == dict(theta)

    def when_constructed():
        def it_neither_writes_to_the_store_nor_starts_a_thread():
            mailbox = gm.MemoryMailbox()
            gm.Session(
                gm.MemoryAdapter({"a": f32(1.0, 2.0)}),
                algorithm=gm.DiLoCo(sync_every=4, fragments=1, in_flight=1),
                mailbox=mailbox,
                node_id="idle",
            )

            assert asyncio.run(mailbox.list("")) == []
            assert threads_of("idle") == []

    def when_joined_before_running():
        def it_holds_a_lease_from_the_join_until_the_run_ends():
            mailbox = gm.MemoryMailbox()
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            seen: list[gm.Event] = []
            session = gm.Session(
                model,
                algorithm=gm.DiLoCo(sync_every=4, fragments=2),
                mailbox=mailbox,
                node_id="node",
                poll_interval=0.01,
                callbacks=(seen.append,),
            )

            session.join()
            session.join()
            holding_leases(mailbox, "node")
            assert session.rank is None
            session.run(partial(train_step, model), [(1,)] * 3)

            assert asyncio.run(Exchange(mailbox, clock=MonotonicClock()).manifest()) is not None
            assert [type(event) for event in seen].count(events.Joined) == 1
            assert leases(mailbox) == set()
            assert threads_of("node") == []

        def it_runs_once():
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0)})
            session = gm.Session(
                model,
                algorithm=gm.DiLoCo(sync_every=4, fragments=1, in_flight=1),
                mailbox=gm.MemoryMailbox(),
                poll_interval=0.01,
            )
            session.run(partial(train_step, model), ())

            with pytest.raises(RuntimeError, match="runs once"):
                session.run(partial(train_step, model), ())

    def when_the_data_is_exhausted():
        def it_marks_the_node_done_before_waiting_for_its_rounds_in_flight():
            mailbox = gm.MemoryMailbox()
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            at_closing: list[bytes | None] = []

            def on_event(event: gm.Event) -> None:
                if isinstance(event, events.Closing):
                    at_closing.append(asyncio.run(mailbox.get("done/node.json")))

            session = gm.Session(
                model,
                algorithm=gm.DiLoCo(sync_every=4, fragments=2),
                mailbox=mailbox,
                node_id="node",
                poll_interval=0.01,
                callbacks=(on_event,),
            )
            session.run(partial(train_step, model), [(1,)] * 3)

            assert at_closing == [b"{}"]

    def when_the_step_raises():
        @pytest.mark.parametrize("error", [RuntimeError("training failed"), KeyboardInterrupt()])
        def it_releases_the_lease_and_stops_the_threads_before_raising(error: BaseException):
            mailbox = gm.MemoryMailbox()
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(
                model,
                algorithm=gm.DiLoCo(sync_every=4, fragments=2),
                mailbox=mailbox,
                node_id="failing",
                poll_interval=0.01,
            )
            steps = itertools.count()

            def step(batch: Sequence[int]) -> int:
                if next(steps) == 3:
                    raise error
                return train_step(model, batch)

            session.join()
            holding_leases(mailbox, "failing")
            with pytest.raises(type(error)):
                session.run(step, itertools.repeat((1,)))

            assert leases(mailbox) == set()
            assert threads_of("failing") == []
            assert asyncio.run(mailbox.list("done/")) == []

    def when_the_data_ends_between_rounds():
        def it_drops_the_steps_trained_since_the_last_round():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
            session.run(partial(train_step, model), [(1,)] * 3)

            exchange = Exchange(mailbox, clock=MonotonicClock())
            [record] = asyncio.run(exchange.rounds_after(-1))
            committed = asyncio.run(exchange.fetch(record.theta))
            assert committed is not None
            # Round 0 ("a") went out on schedule at step 2; the third step reaches neither the
            # run nor the model, which holds round 0's "a" and the initial "b".
            assert (record.round, record.participants[0].steps) == (0, 2)
            assert dict(model.tensors) == {**committed, "b": f32(3.0, 4.0)}
            assert session.rounds_seen == 0

        def when_the_algorithm_flushes_on_close():
            def it_commits_the_steps_trained_since_the_last_round():
                mailbox = gm.MemoryMailbox()
                algorithm = gm.DiLoCo(sync_every=4, fragments=2, flush_on_close=True)
                model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                session = gm.Session(
                    model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01
                )
                session.run(partial(train_step, model), [(1,)] * 3)

                exchange = Exchange(mailbox, clock=MonotonicClock())
                records = asyncio.run(exchange.rounds_after(-1))
                committed = {
                    name: tensor
                    for record in records
                    for name, tensor in (asyncio.run(exchange.fetch(record.theta)) or {}).items()
                }
                # Round 0 ("a") went out on schedule at step 2; round 1 ("b") was a step short
                # of its gap and is what closing published, with every step since the start.
                assert [(r.round, r.participants[0].steps) for r in records] == [(0, 2), (1, 3)]
                assert dict(model.tensors) == committed
                assert session.rounds_seen == 1

            def when_earlier_rounds_built_up_momentum():
                def it_commits_the_parameters_it_trained_without_an_outer_step():
                    mailbox = gm.MemoryMailbox()
                    # No overlap: the model is aligned with every round a step publishes before
                    # the step returns, so closing starts with nothing outstanding.
                    algorithm = gm.DiLoCo(
                        sync_every=4,
                        fragments=2,
                        in_flight=1,
                        overlap_steps=0,
                        outer_momentum=0.9,
                        flush_on_close=True,
                    )
                    model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                    session = gm.Session(
                        model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01
                    )
                    trained: dict[str, list[float]] = {}

                    def batches() -> Iterator[Sequence[int]]:
                        yield from [(1,)] * 9
                        trained.update(
                            {name: floats(tensor) for name, tensor in model.tensors.items()}
                        )

                    session.run(partial(train_step, model), batches())

                    # Rounds 0 to 3 went out on schedule; closing published round 4 ("a", three
                    # steps since round 2) and round 5 ("b", one step since round 3). Nesterov
                    # would have carried both along the momentum of the earlier rounds.
                    assert session.rounds_seen == 5
                    assert {name: floats(tensor) for name, tensor in model.tensors.items()} == {
                        name: pytest.approx(values, rel=1e-6) for name, values in trained.items()
                    }

    def when_overlap_is_zero():
        def it_aligns_with_a_published_round_before_the_next_step():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, in_flight=1, overlap_steps=0)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
            seen: list[int] = []

            def batches() -> Iterator[Sequence[int]]:
                for _ in range(6):
                    yield (1,)
                    seen.append(session.rounds_seen)

            session.run(partial(train_step, model), batches())

            # A round goes out every two steps and the step that publishes it returns only once
            # it is committed and the model aligned with it.
            assert seen == [-1, 0, 0, 1, 1, 2]

    def when_overlap_is_one_step():
        def it_trains_one_step_past_a_publish_and_then_waits():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, overlap_steps=1)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
            seen: list[int] = []

            def batches() -> Iterator[Sequence[int]]:
                for _ in range(7):
                    yield (1,)
                    seen.append(session.rounds_seen)

            session.run(partial(train_step, model), batches())

            # A round goes out every two steps; the step after it returns aligned with it.
            assert seen == [-1, -1, 0, 0, 1, 1, 2]

    def when_a_callback_raises():
        def it_still_hands_the_learners_and_the_syncers_events_to_the_others():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})

            def failing(event: gm.Event) -> None:
                raise RuntimeError("sink down")

            seen: list[gm.Event] = []
            session = gm.Session(
                model,
                algorithm=algorithm,
                mailbox=mailbox,
                node_id="node",
                poll_interval=0.01,
                callbacks=(failing, seen.append),
            )
            session.run(partial(train_step, model), until(session, 1))

            kinds = {type(event) for event in seen}
            assert {
                events.Joined,
                events.SnapshotPublished,
                events.RoundClaimed,
                events.RoundCommitted,
                events.RoundRecorded,
                events.FragmentAligned,
                events.Closed,
                events.LeaseReleased,
            } <= kinds
            assert {event.node for event in seen} == {"node"}

    def when_compression_is_zstd():
        def it_stores_every_payload_compressed_and_aligns_with_the_commits():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(
                model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01, compression="zstd"
            )
            session.run(partial(train_step, model), until(session, 1))
            rounds_seen = session.rounds_seen

            payloads = asyncio.run(mailbox.list("blobs/"))
            assert payloads and all(key.endswith(".safetensors.zst") for key in payloads)
            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            theta = asyncio.run(exchange.fetch(records[rounds_seen].theta))
            assert theta is not None
            assert {name: model.tensors[name] for name in theta} == dict(theta)

    def when_deltas_travel_as_f4():
        def it_commits_from_the_dequantized_deltas():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(
                sync_every=4,
                fragments=2,
                outer_lr=1.0,
                outer_momentum=0.0,
                merge=gm.WeightedMean(),
                delta_dtype="F4",
            )
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
            session.run(partial(train_step, model), until(session, 1))

            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            # Round 0 went out at step 2 with a delta of -2 on "a", round 1 at step 4 with -4 on
            # "b": both exact in E2M1, so with lr 1 the commits hold the trained values.
            assert [record.rejected for record in records] == [()] * len(records)
            assert asyncio.run(exchange.fetch(records[0].theta)) == {"a": f32(3.0, 4.0)}
            assert asyncio.run(exchange.fetch(records[1].theta)) == {"b": f32(7.0, 8.0)}

    def when_commits_carry_f4_updates():
        def it_aligns_the_model_with_every_commit_from_the_updates():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, commit_dtype="F4")
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
            session.run(partial(train_step, model), until(session, 3))
            rounds_seen = session.rounds_seen

            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            assert all(record.update is not None for record in records)
            # Closing set the model to the newest version of every fragment, whether a learner
            # rebuilt it from an update or fetched it.
            newest = {record.fragment: record for record in records[: rounds_seen + 1]}
            for record in newest.values():
                theta = asyncio.run(exchange.fetch(record.theta))
                assert theta is not None
                assert {name: model.tensors[name] for name in theta} == dict(theta)

    def when_the_node_closes_the_rounds_it_trains_for():
        @pytest.mark.parametrize("commit_dtype", [None, "F4"])
        def it_aligns_with_its_own_commits_without_reading_them_back(commit_dtype: Precision):
            class ReadLog(gm.MemoryMailbox):
                def __init__(self) -> None:
                    super().__init__()
                    self.read: list[str] = []

                async def get(self, key: str) -> bytes | None:
                    self.read.append(key)
                    return await super().get(key)

            mailbox = ReadLog()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, commit_dtype=commit_dtype)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
            session.run(partial(train_step, model), until(session, 3))
            rounds_seen = session.rounds_seen

            # The node's learner and syncer share the process: the commits it holds are the ones
            # its own syncer made, never fetched from the store, theta and update alike.
            assert [key for key in mailbox.read if key.startswith("blobs/rounds/")] == []
            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            newest = {record.fragment: record for record in records[: rounds_seen + 1]}
            for record in newest.values():
                theta = asyncio.run(exchange.fetch(record.theta))
                assert theta is not None
                assert {name: model.tensors[name] for name in theta} == dict(theta)

    def when_the_algorithm_sets_a_global_batch():
        def it_gives_each_node_its_share_among_the_live_nodes():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, global_batch_size=8)
            models: dict[str, gm.MemoryAdapter] = {}

            def joined(name: str) -> gm.Session:
                models[name] = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                session = gm.Session(
                    models[name],
                    algorithm=algorithm,
                    mailbox=mailbox,
                    node_id=name,
                    poll_interval=0.01,
                )
                session.join()
                return session

            sessions = {"one": joined("one")}
            try:
                assert sessions["one"].batch_size == 8
                sessions["two"] = joined("two")
                deadline = time.monotonic() + 20.0
                while any(s.batch_size != 4 for s in sessions.values()):
                    assert time.monotonic() < deadline, "the nodes never saw each other's lease"
                    time.sleep(0.001)
            finally:
                # A run with no data closes a joined session.
                for name, session in sessions.items():
                    session.run(partial(train_step, models[name]), ())

    def when_a_node_leaves():
        def it_keeps_closing_rounds_with_the_nodes_that_stay():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, quorum_size=2)
            models = {
                name: gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                for name in ("leaves", "stays")
            }
            sessions = {
                name: gm.Session(
                    model, algorithm=algorithm, mailbox=mailbox, node_id=name, poll_interval=0.01
                )
                for name, model in models.items()
            }
            for session in sessions.values():
                session.join()
            holding_leases(mailbox, "leaves", "stays")

            def train(name: str, rounds: int) -> None:
                session = sessions[name]
                session.run(partial(train_step, models[name]), until(session, rounds))

            # The leaving node's data ends at round 1 and it waits for its rounds in flight while
            # the other trains on, as those rounds need it.
            with ThreadPoolExecutor(max_workers=2) as pool:
                leaving = pool.submit(train, "leaves", 1)
                staying = pool.submit(train, "stays", 6)
                leaving.result()
                staying.result()

            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            participants = [[p.learner for p in record.participants] for record in records]
            # Quorum two held while both were live; alone, the staying node is the quorum.
            assert participants[0] == ["leaves", "stays"]
            assert participants[-1] == ["stays"]

    def when_a_learner_trains_and_a_syncer_closes_the_rounds():
        def it_splits_the_work_and_the_syncer_returns_once_the_learner_is_done():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            trainer_box, closer_box = RecordingMailbox(mailbox), RecordingMailbox(mailbox)
            trainer = gm.Session(
                model,
                algorithm=algorithm,
                mailbox=trainer_box,
                role="learner",
                node_id="trainer",
                poll_interval=0.01,
            )
            closer = gm.Session(
                gm.MemoryAdapter({"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)}),
                algorithm=algorithm,
                mailbox=closer_box,
                role="syncer",
                node_id="closer",
                poll_interval=0.01,
            )

            # The syncer never pulls its data, endless as it is.
            closing = started(partial(closer.run, never_called, itertools.repeat((1,))))
            time.sleep(0.05)
            before_the_learner = asyncio.run(mailbox.list(""))
            trainer.run(partial(train_step, model), until(trainer, 2))
            closing.result(timeout=20.0)

            # The syncer held its lease while it waited for the learner to create the run.
            assert before_the_learner != []
            assert all(key.startswith("syncers/0-dedicated/") for key in before_the_learner)
            exchange = Exchange(mailbox, clock=MonotonicClock())
            manifest = asyncio.run(exchange.manifest())
            records = asyncio.run(exchange.rounds_after(-1))
            assert manifest is not None
            assert manifest.created_by == "trainer"
            assert [record.round for record in records][:3] == [0, 1, 2]
            assert {p.learner for record in records for p in record.participants} == {"trainer"}
            # The learner claimed no round; the syncer wrote commits, claims and its lease only.
            assert [key for key in trainer_box.written if re.match(r"claims/\d", key)] == []
            assert [
                key
                for key in closer_box.written
                if not re.match(r"(rounds|blobs/rounds|claims/\d{10}|syncers)/", key)
            ] == []
            assert asyncio.run(mailbox.list("leases/")) == []
            assert asyncio.run(mailbox.list("syncers/")) == []
            assert threads_of("closer") == []
            assert closer.rounds_seen == -1

    def when_a_syncer_holds_its_lease():
        def it_is_not_counted_among_the_learners():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, global_batch_size=8)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            trainer = gm.Session(
                model, algorithm=algorithm, mailbox=mailbox, node_id="trainer", poll_interval=0.01
            )
            closer = gm.Session(
                gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
                algorithm=algorithm,
                mailbox=mailbox,
                role="syncer",
                node_id="closer",
                poll_interval=0.01,
            )
            trainer.join()
            closer.join()
            deadline = time.monotonic() + 20.0
            while not asyncio.run(mailbox.list("syncers/")):
                assert time.monotonic() < deadline, "the syncer never held its lease"
                time.sleep(0.001)
            # Long enough for the learner to list the leases several times.
            time.sleep(0.1)
            batch_size = trainer.batch_size
            trainer.run(partial(train_step, model), ())
            # The learner is done and gone: the syncer finds the run finished.
            started(partial(closer.run, never_called, ())).result(timeout=20.0)

            assert batch_size == 8
            assert closer.batch_size is None

    def when_a_syncer_joins_a_run_it_does_not_match():
        @pytest.mark.parametrize(
            ("tensors", "sync_every", "error"),
            [
                ({"b": f32(1.0, 2.0)}, 4, gm.SchemaMismatch),
                ({"a": f32(1.0, 2.0)}, 8, gm.RunConfigMismatch),
            ],
        )
        def it_raises_without_taking_a_lease(
            tensors: dict[str, gm.TensorBytes], sync_every: int, error: type[Exception]
        ):
            mailbox = gm.MemoryMailbox()
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0)})
            creator = gm.Session(
                model,
                algorithm=gm.DiLoCo(sync_every=4, fragments=1, in_flight=1),
                mailbox=mailbox,
                node_id="creator",
                poll_interval=0.01,
            )
            closer = gm.Session(
                gm.MemoryAdapter(tensors),
                algorithm=gm.DiLoCo(sync_every=sync_every, fragments=1, in_flight=1),
                mailbox=mailbox,
                role="syncer",
                node_id="closer",
                poll_interval=0.01,
            )
            creator.join()
            try:
                with pytest.raises(error):
                    closer.join()
            finally:
                creator.run(partial(train_step, model), ())

            assert asyncio.run(mailbox.list("syncers/")) == []
            assert threads_of("closer") == []

    def when_the_role_is_either():
        def it_refuses_data_that_is_not_a_shared_queue():
            mailbox = gm.MemoryMailbox()
            session = gm.Session(
                gm.MemoryAdapter({"a": f32(1.0, 2.0)}),
                algorithm=gm.DiLoCo(sync_every=4, fragments=1, in_flight=1),
                mailbox=mailbox,
                role="either",
                node_id="either",
            )

            with pytest.raises(ValueError, match="SharedQueue"):
                session.run(never_called, [(1,)])

            assert asyncio.run(mailbox.list("")) == []
            assert threads_of("either") == []

        def when_several_nodes_join_at_once():
            def it_elects_one_which_closes_the_rounds_the_others_train():
                # The store's clock is monotonic here: each candidate waits a real second.
                mailbox = gm.MemoryMailbox()
                algorithm = gm.DiLoCo(sync_every=4, fragments=2)
                units = {f"unit-{i}": range(4 * i, 4 * i + 4) for i in range(8)}
                names = ("one", "two", "three")
                trained: dict[str, list[int]] = {name: [] for name in names}
                seen: list[gm.Event] = []
                sessions = {
                    name: gm.Session(
                        gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
                        algorithm=algorithm,
                        mailbox=mailbox,
                        role="either",
                        node_id=name,
                        poll_interval=0.01,
                        callbacks=(seen.append,),
                    )
                    for name in names
                }

                def train(name: str) -> None:
                    def step(batch: int) -> int:
                        trained[name].append(batch)
                        time.sleep(0.001)
                        return 1

                    sessions[name].run(step, gm.SharedQueue(units, load=iter))

                # A learner that never trains holds a lease while they join, so that each one
                # finds another node training and stands.
                exchange = Exchange(mailbox, clock=MonotonicClock())
                asyncio.run(exchange.beat("idle"))
                with ThreadPoolExecutor(max_workers=3) as pool:
                    for joining in [pool.submit(session.join) for session in sessions.values()]:
                        joining.result()
                asyncio.run(exchange.release("idle"))
                [elected] = [e.node for e in seen if isinstance(e, events.BecameSyncer)]
                others = sorted(set(names) - {elected})
                holding_leases(mailbox, *others)
                learners_joined = leases(mailbox)
                syncers_joined = asyncio.run(mailbox.list("syncers/"))
                with ThreadPoolExecutor(max_workers=3) as pool:
                    for running in [pool.submit(train, name) for name in names]:
                        running.result()

                # It released its learner lease as it joined, and holds the one syncer lease.
                assert learners_joined == set(others)
                [lease] = syncers_joined
                assert re.fullmatch(rf"syncers/1-elected/\d{{13}}-{elected}/\d{{10}}\.json", lease)
                assert trained[elected] == []
                assert sorted(batch for name in others for batch in trained[name]) == list(
                    range(32)
                )
                published = {e.node for e in seen if isinstance(e, events.SnapshotPublished)}
                assert published == set(others)
                records = asyncio.run(exchange.rounds_after(-1))
                assert records
                assert {p.learner for r in records for p in r.participants} <= set(others)
                # The others made no first claim while it held the role. With polls of 10 ms
                # they may take over round 0 while it still waits out its election.
                released = seen.index(events.LeaseReleased(elected, "syncer"))
                claimed = {e.node for e in seen[:released] if isinstance(e, events.RoundClaimed)}
                assert claimed == {elected}
                assert elected in {e.node for e in seen if isinstance(e, events.RoundCommitted)}
                assert asyncio.run(mailbox.list("done/")) == [
                    f"done/{name}.json" for name in others
                ]
                assert asyncio.run(mailbox.list("syncers/")) == []
                assert not any(isinstance(e, events.ResumedTraining) for e in seen)

        def when_a_dedicated_syncer_arrives():
            def it_releases_its_syncer_lease_and_trains_again():
                mailbox = gm.MemoryMailbox(clock=FakeClock())
                algorithm = gm.DiLoCo(sync_every=4, fragments=2)
                units = {f"unit-{i}": range(4 * i, 4 * i + 4) for i in range(4)}
                seen: list[gm.Event] = []
                resumed = threading.Event()
                checked = threading.Event()
                trained: list[int] = []

                def on_event(event: gm.Event) -> None:
                    seen.append(event)
                    if isinstance(event, events.ResumedTraining):
                        resumed.set()

                def step(batch: int) -> int:
                    assert checked.wait(20.0)
                    trained.append(batch)
                    return 1

                node = gm.Session(
                    gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
                    algorithm=algorithm,
                    mailbox=mailbox,
                    role="either",
                    node_id="either",
                    poll_interval=0.01,
                    callbacks=(on_event,),
                )
                closer = gm.Session(
                    gm.MemoryAdapter({"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)}),
                    algorithm=algorithm,
                    mailbox=mailbox,
                    role="syncer",
                    node_id="dedicated",
                    poll_interval=0.01,
                    callbacks=(seen.append,),
                )

                # A learner that never trains holds a lease: the node finds another training
                # and is elected as it joins.
                exchange = Exchange(mailbox, clock=MonotonicClock())
                asyncio.run(exchange.beat("idle"))
                node.join()
                learners_joined = leases(mailbox)
                syncers_joined = asyncio.run(mailbox.list("syncers/"))
                running = started(partial(node.run, step, gm.SharedQueue(units, load=iter)))
                closing = started(partial(closer.run, never_called, ()))
                assert resumed.wait(20.0)
                asyncio.run(exchange.release("idle"))
                # Its first step waits for these checks, so the new learner is still training.
                holding_leases(mailbox, "either")
                elected_resumed = asyncio.run(mailbox.list("syncers/1-elected/"))
                checked.set()
                running.result(timeout=20.0)
                closing.result(timeout=20.0)

                # It released its own learner lease as it was elected.
                assert learners_joined == {"idle"}
                [lease] = syncers_joined
                assert re.fullmatch(r"syncers/1-elected/\d{13}-either/\d{10}\.json", lease)
                assert elected_resumed == []
                switches = [
                    e
                    for e in seen
                    if isinstance(e, events.BecameSyncer | events.ResumedTraining)
                    or e == events.LeaseReleased("either", "syncer")
                ]
                assert switches[:3] == [
                    events.BecameSyncer("either"),
                    events.LeaseReleased("either", "syncer"),
                    events.ResumedTraining("either", -1),
                ]
                assert sorted(trained) == list(range(16))
                records = asyncio.run(exchange.rounds_after(-1))
                assert "either" in {p.learner for r in records for p in r.participants}
                assert asyncio.run(mailbox.list("done/")) == ["done/either.json"]

        def when_alone():
            def it_is_never_elected_and_trains_every_unit():
                store = gm.MemoryMailbox(clock=FakeClock())
                recording = RecordingMailbox(store)
                units = {f"unit-{i}": range(4 * i, 4 * i + 4) for i in range(4)}
                seen: list[gm.Event] = []
                trained: list[int] = []

                def step(batch: int) -> int:
                    trained.append(batch)
                    return 1

                gm.Session(
                    gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
                    algorithm=gm.DiLoCo(sync_every=4, fragments=2),
                    mailbox=recording,
                    role="either",
                    node_id="alone",
                    poll_interval=0.01,
                    callbacks=(seen.append,),
                ).run(step, gm.SharedQueue(units, load=iter))

                assert sorted(trained) == list(range(16))
                assert [key for key in recording.written if key.startswith("syncers/")] == []
                assert not any(isinstance(e, events.BecameSyncer) for e in seen)
                assert asyncio.run(store.list("done/")) == ["done/alone.json"]

        def when_the_last_other_learner_leaves_with_units_unclaimed():
            def it_gives_the_role_up_and_trains_them():
                store = gm.MemoryMailbox(clock=FakeClock())
                algorithm = gm.DiLoCo(sync_every=4, fragments=2)
                units = {f"unit-{i}": range(4 * i, 4 * i + 4) for i in range(4)}
                seen: list[gm.Event] = []
                trained: list[int] = []

                def step(batch: int) -> int:
                    trained.append(batch)
                    return 1

                model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                other = gm.Session(
                    model,
                    algorithm=algorithm,
                    mailbox=store,
                    role="learner",
                    node_id="other",
                    poll_interval=0.01,
                    callbacks=(seen.append,),
                )
                node = gm.Session(
                    gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}),
                    algorithm=algorithm,
                    mailbox=store,
                    role="either",
                    node_id="either",
                    poll_interval=0.01,
                    callbacks=(seen.append,),
                )

                other.join()
                holding_leases(store, "other")
                # It finds "other" training and is elected as it joins.
                node.join()
                running = started(partial(node.run, step, gm.SharedQueue(units, load=iter)))
                # "other" trains on data of its own, which leaves the queue to the elected node.
                other.run(partial(train_step, model), [(1,)] * 4)
                running.result(timeout=20.0)

                switches = [
                    type(e)
                    for e in seen
                    if isinstance(e, events.BecameSyncer | events.ResumedTraining)
                    or e == events.LeaseReleased("either", "syncer")
                ]
                # Once training again, with no other node training, it never stood again.
                assert switches == [
                    events.BecameSyncer,
                    events.LeaseReleased,
                    events.ResumedTraining,
                ]
                committed = {e.node for e in seen if isinstance(e, events.RoundCommitted)}
                assert "either" in committed
                assert sorted(trained) == list(range(16))
                assert asyncio.run(store.list("done/")) == ["done/either.json", "done/other.json"]
                assert asyncio.run(store.list("syncers/")) == []

        def when_elected_in_the_middle_of_a_unit():
            def it_trains_that_unit_to_its_end_and_then_stops():
                trained, _, _, _, seen = elected_mid_unit()

                # Elected while held in batch 0, it trained batches 1 to 3 before stopping, and
                # never a part of a unit.
                assert trained["paused"][:4] == [0, 1, 2, 3]
                started_units = sorted({batch // 4 for batch in trained["paused"]})
                assert trained["paused"] == [
                    batch for unit in started_units for batch in range(4 * unit, 4 * unit + 4)
                ]
                [stopped] = [i for i, e in enumerate(seen) if isinstance(e, events.BecameSyncer)]
                published = [
                    i
                    for i, e in enumerate(seen)
                    if isinstance(e, events.SnapshotPublished) and e.node == "paused"
                ]
                assert published
                assert max(published) < stopped

            def it_gives_back_the_units_it_claimed_ahead_and_the_other_node_trains_them():
                trained, claimed, owners, done, _ = elected_mid_unit()

                started_units = {f"unit-{batch // 4}" for batch in trained["paused"]}
                given_back = set(claimed) - started_units
                assert given_back
                assert {unit for unit, node in owners.items() if node == "paused"} == started_units
                assert {unit for unit, node in owners.items() if node == "other"} == (
                    set(owners) - started_units
                )
                assert {f"unit-{batch // 4}" for batch in trained["other"]} >= given_back
                assert sorted(trained["paused"] + trained["other"]) == list(range(24))
                # It stopped without its data exhausted.
                assert done == ["done/other.json"]


def describe_align():
    def it_loads_the_newest_commit_of_every_fragment_without_joining():
        mailbox = gm.MemoryMailbox()
        algorithm = gm.DiLoCo(sync_every=4, fragments=2, quorum_size=1)
        model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
        session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
        session.run(partial(train_step, model), until(session, 2))

        exchange = Exchange(mailbox, clock=MonotonicClock())
        records = asyncio.run(exchange.rounds_after(-1))
        newest = {record.fragment: record for record in records}
        expected = {
            name: tensor
            for record in newest.values()
            for name, tensor in (asyncio.run(exchange.fetch(record.theta)) or {}).items()
        }
        reader = gm.MemoryAdapter({"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)})
        keys_before = asyncio.run(mailbox.list(""))

        assert gm.align(reader, mailbox=mailbox) == records[-1].round
        assert dict(reader.tensors) == expected
        # Reading is not joining: no snapshot, no claim, nothing written.
        assert asyncio.run(mailbox.list("")) == keys_before

    def it_refuses_a_mailbox_without_a_run():
        with pytest.raises(ProtocolError):
            gm.align(gm.MemoryAdapter({"a": f32(1.0)}), mailbox=gm.MemoryMailbox())


def describe_ranks():
    """The store's clock is a ``DrivenClock`` where nodes claim at once, so that each one's
    second of waiting ends only once the test has seen their claims land; a ``FakeClock``, whose
    sleep is instant, elsewhere. Leases are judged on each node's own monotonic clock."""

    def it_refuses_a_role_that_takes_no_rank():
        for role in ("syncer", "either"):
            with pytest.raises(ValueError, match="no ranks"):
                ranked(gm.MemoryMailbox(), "node", 2, role=role)

    def when_two_nodes_join_at_once():
        def it_gives_one_rank_0_and_the_other_rank_1():
            clock = DrivenClock()
            store = gm.MemoryMailbox(clock=clock)
            seen: list[gm.Event] = []
            nodes = [ranked(store, name, 2, callbacks=(seen.append,)) for name in ("one", "two")]
            joining = [started(node.join) for node in nodes]
            waited(lambda: len(asyncio.run(store.list("ranks/"))) == 2, "they never both claimed")
            driven(clock, *joining)
            ranks = {node.rank for node in nodes}
            for node in nodes:
                node.run(never_called, ())

            assert ranks == {0, 1}
            assert sorted(e.rank for e in seen if isinstance(e, events.RankClaimed)) == [0, 1]
            # Whichever claimed rank 0 second, it deleted that claim before it took rank 1.
            assert [len(claimants(store, rank)) for rank in (0, 1)] == [1, 1]
            assert asyncio.run(store.list("done/")) == ["done/000.json", "done/001.json"]

    def when_every_rank_is_held():
        def it_waits_and_takes_the_rank_whose_holder_left_deleting_its_claim():
            clock = DrivenClock()
            store = gm.MemoryMailbox(clock=clock)
            first, second, third = (ranked(store, name, 2) for name in ("first", "second", "third"))
            # One after the other: the second finds the first holding rank 0.
            driven(clock, started(first.join))
            driven(clock, started(second.join))
            joining = started(third.join)
            time.sleep(0.2)
            waiting = not joining.done() and claimants(store, 0) == ["first"]
            # The first node fails: its learner stops and releases its lease, its claim stays.
            with pytest.raises(RuntimeError):
                first.run(failing, [(1,)])
            driven(clock, joining)
            claims = [claimants(store, rank) for rank in (0, 1)]
            for node in (second, third):
                node.run(never_called, ())

            assert waiting
            assert (first.rank, second.rank, third.rank) == (0, 1, 0)
            assert claims == [["third"], ["second"]]

    def when_an_earlier_claim_on_its_rank_lands_while_it_waits():
        def it_deletes_its_claim_and_takes_the_next_rank():
            clock = DrivenClock(start=10.0)
            store = gm.MemoryMailbox(clock=clock)
            exchange = Exchange(store, clock=MonotonicClock())
            late = ranked(store, "late", 2)
            joining = started(late.join)
            waited(lambda: claimants(store, 0) == ["late"], "it never claimed rank 0")
            # "early" read the store's clock before it did, and its claim lands now.
            asyncio.run(exchange.beat("early"))
            asyncio.run(store.put_if_absent(rank_key(0, 9_999, "early"), b"{}"))
            driven(clock, joining)
            claims = [claimants(store, rank) for rank in (0, 1)]
            late.run(never_called, ())

            assert late.rank == 1
            assert claims == [["early"], ["late"]]

    def when_the_holder_whose_lease_had_lapsed_renews_it():
        def it_keeps_its_rank_and_the_node_that_took_it_stops_with_rank_lost():
            store = gm.MemoryMailbox(clock=FakeClock(start=10.0))
            exchange = Exchange(store, clock=MonotonicClock())
            asyncio.run(exchange.beat("slow"))
            asyncio.run(store.put_if_absent(rank_key(0, 0, "slow"), b"{}"))
            seen: list[gm.Event] = []
            published = threading.Event()

            def on_event(event: gm.Event) -> None:
                seen.append(event)
                if isinstance(event, events.SnapshotPublished):
                    published.set()

            # The newcomer judges "slow" gone once its lease is 0.2 s old.
            newcomer = ranked(
                store, "newcomer", 1, lease_every=0.05, lease_timeout=0.2, callbacks=(on_event,)
            )
            newcomer.join()
            claims = claimants(store, 0)

            def step(batch: Sequence[int]) -> int:
                time.sleep(0.001)
                return len(batch)

            running = started(partial(newcomer.run, step, itertools.repeat((1,))))
            assert published.wait(20.0)
            asyncio.run(exchange.beat("slow"))
            with pytest.raises(gm.RankLost) as lost:
                running.result(timeout=20.0)

            assert (lost.value.node, lost.value.rank, lost.value.owner) == (
                "newcomer",
                0,
                "slow",
            )
            assert newcomer.rank == 0
            # Its lease was there, unrenewed: the newcomer left its claim for it to come back.
            assert claims == ["slow", "newcomer"]
            assert leases(store) == {"slow"}
            assert threads_of("newcomer") == []
            assert asyncio.run(store.list("done/")) == []
            released = seen.index(events.LeaseReleased("newcomer", "learner"))
            assert not any(isinstance(e, events.SnapshotPublished) for e in seen[released:])

    def when_a_rank_crashes_before_its_data_is_exhausted():
        def it_keeps_the_syncer_up_until_a_node_that_takes_it_over_marks_it_done():
            store = gm.MemoryMailbox(clock=FakeClock())
            closer = gm.Session(
                gm.MemoryAdapter({"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)}),
                algorithm=gm.DiLoCo(sync_every=4, fragments=2),
                mailbox=store,
                role="syncer",
                node_id="closer",
                poll_interval=0.01,
            )
            closing = started(partial(closer.run, never_called, ()))
            one, two, three = (
                ranked(store, name, 2, role="learner") for name in ("one", "two", "three")
            )
            one.join()
            two.join()
            with pytest.raises(RuntimeError):
                two.run(failing, [(1,)])
            one.run(counted, [(1,)] * 3)
            time.sleep(0.2)
            still_running = not closing.done()
            done_then = asyncio.run(store.list("done/"))
            # Rank 0 is done, so the node that joins now takes rank 1 over.
            three.run(counted, [(1,)] * 3)
            closing.result(timeout=20.0)

            assert (one.rank, two.rank, three.rank) == (0, 1, 1)
            assert still_running
            assert done_then == ["done/000.json"]
            assert asyncio.run(store.list("done/")) == ["done/000.json", "done/001.json"]
            assert claimants(store, 1) == ["three"]
            assert asyncio.run(store.list("syncers/")) == []
