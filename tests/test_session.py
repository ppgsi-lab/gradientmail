"""INTRO.md's example end to end on MemoryMailbox, through the package's public names.

One ``Session`` over a ``MemoryAdapter`` model trains on a worker thread; the session's own
syncer closes the rounds, so nothing else runs. What is observed is what a user would: the round
records in the store, the learner they name and the parameters the model holds once it has
installed the newest commit.
"""

import asyncio
import os
import re
import threading
import time
from array import array
from collections.abc import Iterator, Sequence

import pytest

import gradientmail as gm
from gradientmail import events
from gradientmail.clock import MonotonicClock
from gradientmail.protocol import Exchange, ProtocolError


def f32(*values: float) -> gm.TensorBytes:
    return gm.TensorBytes("F32", (len(values),), array("f", values).tobytes())


def floats(tensor: gm.TensorBytes) -> list[float]:
    return array("f", tensor.data).tolist()


# Eight tensors, so INTRO's ``fragments=8`` is the effective fragment count.
MODEL = {f"block{i}.weight": f32(float(i), float(i)) for i in range(8)}


def train_step(model: gm.MemoryAdapter, batch: Sequence[int]) -> None:
    """Synthetic training: every parameter grows by the batch size."""
    model.apply(
        {
            name: f32(*(value + len(batch) for value in array("f", tensor.data)))
            for name, tensor in model.tensors.items()
        }
    )


def describe_session():
    def it_runs_intros_example_against_memory_mailbox():
        mailbox = gm.MemoryMailbox()
        algorithm = gm.DiLoCo(sync_every=100, fragments=8)
        model = gm.MemoryAdapter(MODEL)

        def dataloader() -> Iterator[Sequence[int]]:
            """INTRO's loop, run until the model has installed round 1."""
            deadline = time.monotonic() + 20.0
            while session.rounds_seen < 1:
                assert time.monotonic() < deadline, "the model never received round 1"
                yield (1, 2, 3)
                time.sleep(0.001)

        session = gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01)
        try:
            for batch in dataloader():
                train_step(model, batch)
                session.step(samples=len(batch))
        finally:
            session.close()
        rounds_seen = session.rounds_seen

        exchange = Exchange(mailbox, clock=MonotonicClock())
        records = asyncio.run(exchange.rounds_after(-1))
        assert [record.round for record in records][:2] == [0, 1]
        (learner_id,) = {p.learner for record in records for p in record.participants}
        assert re.fullmatch(r"[a-z0-9-]{1,64}", learner_id)
        assert learner_id.endswith(f"-{os.getpid()}")
        # The last installed round is the newest version of its fragment: the model holds it.
        assert rounds_seen >= 1
        theta = asyncio.run(exchange.fetch(records[rounds_seen].theta))
        assert theta is not None
        assert {name: model.tensors[name] for name in theta} == dict(theta)

    def when_closed_between_rounds():
        def it_drops_the_steps_trained_since_the_last_round():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            with gm.Session(
                model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01
            ) as session:
                for _ in range(3):
                    train_step(model, (1,))
                    session.step(samples=1)

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
                with gm.Session(
                    model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01
                ) as session:
                    for _ in range(3):
                        train_step(model, (1,))
                        session.step(samples=1)

                exchange = Exchange(mailbox, clock=MonotonicClock())
                records = asyncio.run(exchange.rounds_after(-1))
                committed = {
                    name: tensor
                    for record in records
                    for name, tensor in (asyncio.run(exchange.fetch(record.theta)) or {}).items()
                }
                # Round 0 ("a") went out on schedule at step 2; round 1 ("b") was a step short
                # of its gap and is what close() published, with every step since the start.
                assert [(r.round, r.participants[0].steps) for r in records] == [(0, 2), (1, 3)]
                assert dict(model.tensors) == committed
                assert session.rounds_seen == 1

            def when_earlier_rounds_built_up_momentum():
                def it_commits_the_parameters_it_trained_without_an_outer_step():
                    mailbox = gm.MemoryMailbox()
                    # One round in flight, synchronous: every round a step publishes is
                    # installed before the step returns, so close() starts with nothing
                    # outstanding.
                    algorithm = gm.DiLoCo(
                        sync_every=4,
                        fragments=2,
                        in_flight=1,
                        synchronous=True,
                        outer_momentum=0.9,
                        flush_on_close=True,
                    )
                    model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                    with gm.Session(
                        model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01
                    ) as session:
                        for _ in range(9):
                            train_step(model, (1,))
                            session.step(samples=1)
                        trained = {name: floats(tensor) for name, tensor in model.tensors.items()}

                    # Rounds 0 to 3 went out on schedule; close() published round 4 ("a", three
                    # steps since round 2) and round 5 ("b", one step since round 3). Nesterov
                    # would have carried both along the momentum of the earlier rounds.
                    assert session.rounds_seen == 5
                    assert {name: floats(tensor) for name, tensor in model.tensors.items()} == {
                        name: pytest.approx(values, rel=1e-6) for name, values in trained.items()
                    }

    def when_synchronous():
        def it_installs_a_published_round_before_the_next_step():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, in_flight=1, synchronous=True)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            with gm.Session(
                model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01
            ) as session:
                seen = []
                for _ in range(6):
                    train_step(model, (1,))
                    session.step(samples=1)
                    seen.append(session.rounds_seen)

            # A round goes out every two steps and, with one round in flight, the step that
            # publishes it returns only once it is committed and installed.
            assert seen == [-1, 0, 0, 1, 1, 2]

    def when_a_callback_raises():
        def it_still_hands_the_learners_and_the_syncers_events_to_the_others():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})

            def failing(event: gm.Event) -> None:
                raise RuntimeError("sink down")

            seen: list[gm.Event] = []
            with gm.Session(
                model,
                algorithm=algorithm,
                mailbox=mailbox,
                learner_id="node",
                poll_interval=0.01,
                callbacks=(failing, seen.append),
            ) as session:
                deadline = time.monotonic() + 20.0
                while session.rounds_seen < 1:
                    assert time.monotonic() < deadline, "round 1 never arrived"
                    train_step(model, (1,))
                    session.step(samples=1)
                    time.sleep(0.001)

            kinds = {type(event) for event in seen}
            assert {
                events.Joined,
                events.SnapshotPublished,
                events.RoundClaimed,
                events.RoundCommitted,
                events.RoundRecorded,
                events.FragmentInstalled,
                events.Closed,
                events.LeaseReleased,
            } <= kinds
            assert {event.learner for event in seen} == {"node"}

    def when_compression_is_zstd():
        def it_stores_every_payload_compressed_and_installs_the_commits():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2)
            model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
            session = gm.Session(
                model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01, compression="zstd"
            )
            try:
                deadline = time.monotonic() + 20.0
                while session.rounds_seen < 1:
                    assert time.monotonic() < deadline, "the model never received round 1"
                    train_step(model, (1,))
                    session.step(samples=1)
                    time.sleep(0.001)
            finally:
                session.close()
            rounds_seen = session.rounds_seen

            payloads = asyncio.run(mailbox.list("blobs/"))
            assert payloads and all(key.endswith(".safetensors.zst") for key in payloads)
            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            theta = asyncio.run(exchange.fetch(records[rounds_seen].theta))
            assert theta is not None
            assert {name: model.tensors[name] for name in theta} == dict(theta)

    def when_the_algorithm_sets_a_global_batch():
        def it_gives_each_node_its_share_among_the_live_nodes():
            mailbox = gm.MemoryMailbox()
            algorithm = gm.DiLoCo(sync_every=4, fragments=2, global_batch_size=8)

            def session(name: str) -> gm.Session:
                model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
                return gm.Session(
                    model, algorithm=algorithm, mailbox=mailbox, learner_id=name, poll_interval=0.01
                )

            sessions = [session("one")]
            try:
                assert sessions[0].batch_size == 8
                sessions.append(session("two"))
                deadline = time.monotonic() + 20.0
                while any(s.batch_size != 4 for s in sessions):
                    assert time.monotonic() < deadline, "the nodes never saw each other's lease"
                    time.sleep(0.001)
            finally:
                for s in sessions:
                    s.close()

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
                    model, algorithm=algorithm, mailbox=mailbox, learner_id=name, poll_interval=0.01
                )
                for name, model in models.items()
            }

            def train_until(names: Sequence[str], rounds: int) -> None:
                deadline = time.monotonic() + 20.0
                while min(sessions[name].rounds_seen for name in names) < rounds:
                    assert time.monotonic() < deadline, f"round {rounds} never arrived"
                    for name in names:
                        train_step(models[name], (1,))
                        sessions[name].step(samples=1)
                    time.sleep(0.001)

            try:
                train_until(("leaves", "stays"), 1)
                # The leaving node flushes while the other trains on, as its rounds need it.
                leaving = threading.Thread(target=sessions["leaves"].close)
                leaving.start()
                train_until(("stays",), 6)
                leaving.join(20.0)
                assert not leaving.is_alive()
            finally:
                sessions["stays"].close()

            exchange = Exchange(mailbox, clock=MonotonicClock())
            records = asyncio.run(exchange.rounds_after(-1))
            participants = [[p.learner for p in record.participants] for record in records]
            # Quorum two held while both were live; alone, the staying node is the quorum.
            assert participants[0] == ["leaves", "stays"]
            assert participants[-1] == ["stays"]


def describe_install():
    def it_loads_the_newest_commit_of_every_fragment_without_joining():
        mailbox = gm.MemoryMailbox()
        algorithm = gm.DiLoCo(sync_every=4, fragments=2, quorum_size=1)
        model = gm.MemoryAdapter({"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)})
        with gm.Session(model, algorithm=algorithm, mailbox=mailbox, poll_interval=0.01) as session:
            deadline = time.monotonic() + 20.0
            while session.rounds_seen < 2:
                assert time.monotonic() < deadline, "the model never received round 2"
                train_step(model, (1,))
                session.step(samples=1)
                time.sleep(0.001)

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

        assert gm.install(reader, mailbox=mailbox) == records[-1].round
        assert dict(reader.tensors) == expected
        # Reading is not joining: no snapshot, no claim, nothing written.
        assert asyncio.run(mailbox.list("")) == keys_before

    def it_refuses_a_mailbox_without_a_run():
        with pytest.raises(ProtocolError):
            gm.install(gm.MemoryAdapter({"a": f32(1.0)}), mailbox=gm.MemoryMailbox())
