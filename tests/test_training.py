"""Phase 2 end to end: two sessions on MemoryAdapters through a real mailbox, no separate syncer.

Training is synthetic: every step adds a learner-specific constant to every parameter, so the
fragment a learner publishes is its anchor plus ``steps * constant``, and the value
every commit must hold follows from the round record alone (its participants' steps and
weights). Each ``Session`` hosts a learner and a syncer; which node closes a round is settled
through the mailbox. The sessions run on their own threads at different paces and batch sizes,
and the test watches the tape on the wall clock, bounded by a timeout that turns a stall into a
failure. Rejoining uses a bare ``Learner`` so that nothing closes rounds while the tape is read.
"""

import asyncio
import contextlib
import itertools
import math
import os
import threading
import time
from array import array
from collections.abc import Generator, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from gradientmail.adapters import MemoryAdapter
from gradientmail.algorithms.diloco import Carry, DiLoCo, WeightedMean
from gradientmail.clock import MonotonicClock
from gradientmail.learner import Learner
from gradientmail.mailbox import Mailbox
from gradientmail.protocol import Exchange, RoundRecord
from gradientmail.serialization import TensorBytes
from gradientmail.session import Session


def f32(*values: float) -> TensorBytes:
    return TensorBytes("F32", (len(values),), array("f", values).tobytes())


def values(tensor: TensorBytes) -> list[float]:
    return array("f", tensor.data).tolist()


POLL = 0.01
TIMEOUT = float(os.environ.get("GRADIENTMAIL_TEST_TIMEOUT", "30"))
# Two equal-sized tensors: fragment 0 is "a" and fragment 1 is "b".
TENSORS = {"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}
ZEROS = {"a": f32(0.0, 0.0), "b": f32(0.0, 0.0)}


@dataclass(frozen=True)
class Profile:
    """How a synthetic learner trains: what a step adds, how many samples it reports, how often."""

    node_id: str
    delta: float
    samples: int
    pace: float


FAST = Profile("fast", delta=0.25, samples=2, pace=0.001)
SLOW = Profile("slow", delta=-0.5, samples=5, pace=0.004)
DELTAS = {FAST.node_id: FAST.delta, SLOW.node_id: SLOW.delta}


def algorithm(*, quorum: int) -> DiLoCo:
    """lr 1, momentum 0 and the weighted mean: a commit is the weighted average of the fragments."""
    return DiLoCo(
        sync_every=8,
        fragments=2,
        in_flight=2,
        quorum_size=quorum,
        outer_lr=1.0,
        outer_momentum=0.0,
        merge=WeightedMean(),
    )


class Trainer:
    """A session running on its own thread; every step adds ``profile.delta`` to every parameter."""

    def __init__(
        self,
        profile: Profile,
        mailbox: Mailbox,
        config: DiLoCo,
        tensors: Mapping[str, TensorBytes] = TENSORS,
        adapter: MemoryAdapter | None = None,
    ) -> None:
        self.profile = profile
        self.adapter = adapter if adapter is not None else MemoryAdapter(tensors)
        self._session = Session(
            self.adapter,
            algorithm=config,
            mailbox=mailbox,
            node_id=profile.node_id,
            poll_interval=POLL,
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._train, name=f"trainer-{profile.node_id}", daemon=True
        )
        self._failure: Exception | None = None

    @property
    def rounds_seen(self) -> int:
        return self._session.rounds_seen

    def join(self) -> None:
        """Join the run; returns once the adapter holds the newest committed fragments."""
        self._session.join()

    def train(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Stop training and wait for the run to end; raises what the training thread died of.

        A trainer that never trained runs with no data, which ends its session all the same.
        """
        self._stop.set()
        if self._thread.ident is None:
            self._thread.start()
        self._thread.join()
        if self._failure is not None:
            raise self._failure

    def _train(self) -> None:
        def shifted(tensor: TensorBytes) -> TensorBytes:
            return f32(*(value + self.profile.delta for value in values(tensor)))

        def step(_: None) -> int:
            self.adapter.apply(
                {name: shifted(tensor) for name, tensor in self.adapter.tensors.items()}
            )
            return self.profile.samples

        def batches() -> Iterator[None]:
            while not self._stop.is_set():
                yield None
                time.sleep(self.profile.pace)

        try:
            self._session.run(step, batches())
        except Exception as exc:
            self._failure = exc


@dataclass(frozen=True)
class Tape:
    """What the run left in the mailbox, read synchronously."""

    mailbox: Mailbox

    def records(self) -> Sequence[RoundRecord]:
        return asyncio.run(self.exchange.rounds_after(-1))

    def records_at_least(self, count: int) -> Sequence[RoundRecord]:
        """Poll until ``count`` rounds are committed; a stall past ``TIMEOUT`` fails."""
        deadline = time.monotonic() + TIMEOUT
        while len(records := self.records()) < count:
            assert time.monotonic() < deadline, f"only {len(records)} of {count} rounds committed"
            time.sleep(POLL)
        return records

    def live_at_least(self, *learners: str) -> None:
        """Poll until every one of ``learners`` holds a lease; a stall past ``TIMEOUT`` fails."""
        deadline = time.monotonic() + TIMEOUT
        while not set(learners) <= set(asyncio.run(self.exchange.leases())):
            assert time.monotonic() < deadline, f"{learners} never all held a lease"
            time.sleep(POLL)

    def plan(self) -> tuple[tuple[str, ...], ...]:
        manifest = asyncio.run(self.exchange.manifest())
        assert manifest is not None
        return manifest.plan

    def theta(self, record: RoundRecord) -> Mapping[str, Sequence[float]] | None:
        """The fragment ``record`` committed, or None once garbage collection swept it."""
        tensors = asyncio.run(self.exchange.fetch(record.theta))
        return None if tensors is None else {name: values(t) for name, t in tensors.items()}

    def newest(self) -> Mapping[str, TensorBytes]:
        """The global parameters: the newest committed version of every fragment."""

        def fetched(record: RoundRecord) -> Mapping[str, TensorBytes]:
            tensors = asyncio.run(self.exchange.fetch(record.theta))
            assert tensors is not None, f"{record.theta.key} is the newest version but was swept"
            return tensors

        newest = {record.fragment: record for record in self.records()}
        return {
            name: tensor for record in newest.values() for name, tensor in fetched(record).items()
        }

    @property
    def exchange(self) -> Exchange:
        return Exchange(self.mailbox, clock=MonotonicClock())


@contextlib.contextmanager
def training(mailbox: Mailbox, config: DiLoCo) -> Generator[tuple[Trainer, Trainer], None, None]:
    """Both learners joined and training; stopped on the way out, whatever happened.

    Both hold a lease before either trains, so every round waits for both up to quorum.
    """
    fast = Trainer(FAST, mailbox, config)
    fast.join()
    try:
        slow = Trainer(SLOW, mailbox, config)
        slow.join()
        try:
            Tape(mailbox).live_at_least(FAST.node_id, SLOW.node_id)
            fast.train()
            slow.train()
            yield fast, slow
        finally:
            slow.stop()
    finally:
        fast.stop()


def rejoin(adapter: MemoryAdapter, node_id: str, mailbox: Mailbox, config: DiLoCo) -> None:
    """Start and close a bare learner on ``adapter``: it then holds the newest commits."""
    learner = Learner(
        adapter=adapter,
        algorithm=config,
        mailbox=mailbox,
        node_id=node_id,
        poll_interval=POLL,
    )
    learner.start()
    learner.close()


def participants(record: RoundRecord) -> tuple[str, ...]:
    return tuple(participant.learner for participant in record.participants)


def expected_trajectory(
    records: Sequence[RoundRecord], plan: tuple[tuple[str, ...], ...]
) -> Sequence[Mapping[str, Sequence[float]]]:
    """The fragment each record must have committed, from the record and the learners' constants.

    A participant's fragment at publish time was the anchor plus ``steps * delta``; the commit is
    the weighted average of those fragments.
    """

    def after(
        theta: Mapping[str, Sequence[float]], record: RoundRecord
    ) -> Mapping[str, Sequence[float]]:
        total = sum(p.weight for p in record.participants)
        return {
            **theta,
            **{
                name: [
                    sum(p.weight * (x + p.steps * DELTAS[p.learner]) for p in record.participants)
                    / total
                    for x in theta[name]
                ]
                for name in plan[record.fragment]
            },
        }

    start = {name: values(tensor) for name, tensor in TENSORS.items()}
    states = list(itertools.accumulate(records, after, initial=start))[1:]
    return [
        {name: state[name] for name in plan[record.fragment]}
        for record, state in zip(records, states, strict=True)
    ]


def deviations(
    tape: Tape, records: Sequence[RoundRecord]
) -> Sequence[tuple[int, str, Sequence[float], Sequence[float]]]:
    """``(round, name, committed, expected)`` wherever a surviving commit left the trajectory.

    The trajectory itself is computed from the records alone, so a swept payload (None) costs
    only its own comparison; ``surviving`` says which rounds the check actually covered.
    """
    return [
        (record.round, name, committed[name], expected)
        for record, fragment in zip(records, expected_trajectory(records, tape.plan()), strict=True)
        if (committed := tape.theta(record)) is not None
        for name, expected in fragment.items()
        if not all(
            math.isclose(c, e, rel_tol=1e-5, abs_tol=1e-6)
            for c, e in zip(committed[name], expected, strict=True)
        )
    ]


def surviving(tape: Tape, records: Sequence[RoundRecord]) -> Sequence[int]:
    """Rounds whose theta payload is still in the store."""
    return [record.round for record in records if tape.theta(record) is not None]


def rounds(records: Sequence[RoundRecord]) -> list[int]:
    return [record.round for record in records]


def dropped(records: Sequence[RoundRecord]) -> Sequence[tuple[str, int, int]]:
    """``(learner, round, steps)`` wherever a learner's snapshot of a fragment carries fewer
    steps than it took since its previous snapshot of that fragment, or since it joined."""
    previous: dict[tuple[str, int], int] = {}
    lost = []
    for record in records:
        for p in record.participants:
            since = p.local_step - previous.get((p.learner, record.fragment), 0)
            if p.steps != since:
                lost.append((p.learner, record.round, since - p.steps))
            previous[p.learner, record.fragment] = p.local_step
    return lost


def describe_training_run():
    def when_quorum_is_both_and_merge_is_weighted_mean_with_lr_1_and_no_momentum():
        def it_sets_every_fragment_to_the_token_weighted_average(mailbox: Mailbox):
            config = algorithm(quorum=2)
            tape = Tape(mailbox)
            with training(mailbox, config):
                together = tape.records_at_least(6)

            # The rounds closed while both trained have both; the first to stop leaves the
            # other closing its last rounds alone.
            records = tape.records()
            assert list(records[: len(together)]) == list(together)
            assert [participants(record) for record in together] == [("fast", "slow")] * len(
                together
            )
            assert [record.rejected for record in records] == [()] * len(records)
            # Two cycles of payloads are kept, so at least the last four commits are checkable.
            assert set(rounds(records)[-4:]) <= set(surviving(tape, records))
            assert deviations(tape, records) == []

        def it_leaves_both_learners_with_identical_parameters_after_the_run(mailbox: Mailbox):
            # Between syncs a learner's parameters drift by its own training, by design; what
            # the learners share is the global state a learner holds right after (re)joining.
            config = algorithm(quorum=2)
            tape = Tape(mailbox)
            with training(mailbox, config) as (fast, slow):
                tape.records_at_least(6)
                fast.stop()
                slow.stop()
                rejoin(fast.adapter, FAST.node_id, mailbox, config)
                rejoin(slow.adapter, SLOW.node_id, mailbox, config)

            assert dict(fast.adapter.tensors) == dict(slow.adapter.tensors) == tape.newest()

        def it_commits_each_round_exactly_once(mailbox: Mailbox):
            config = algorithm(quorum=2)
            tape = Tape(mailbox)
            with training(mailbox, config):
                tape.records_at_least(6)

            records = tape.records()
            schedule = config.schedule(2)
            assert rounds(records) == list(range(len(records)))
            assert [(record.fragment, record.version, record.step) for record in records] == [
                (schedule.fragment_of(r), schedule.version_after(r), schedule.step_of(r))
                for r in rounds(records)
            ]

    def when_the_alignment_carries():
        def it_drops_no_step_and_stays_on_the_trajectory(mailbox: Mailbox):
            # Between a snapshot and its commit a learner keeps training; overwriting drops
            # those steps, carrying keeps them for the fragment's next snapshot, so every step
            # a learner takes is in exactly one of its snapshots. The trajectory holds because a
            # published fragment is still its anchor plus the counted steps times the constant.
            config = replace(algorithm(quorum=2), alignment=Carry())
            tape = Tape(mailbox)
            with training(mailbox, config):
                tape.records_at_least(8)

            records = tape.records()
            assert dropped(records) == []
            assert [record.rejected for record in records] == [()] * len(records)
            assert deviations(tape, records) == []

    def when_a_learner_dies_mid_run():
        def it_continues_on_quorum_and_the_learner_rejoins_within_one_cycle(mailbox: Mailbox):
            config = algorithm(quorum=1)
            tape = Tape(mailbox)
            with training(mailbox, config) as (fast, slow):
                tape.records_at_least(3)
                slow.stop()
                # The dead learner had published at most two rounds past the ones it had seen,
                # which are at most those committed by now; the survivor takes over closing if
                # it has to and the cycle after that is its alone.
                committed_at_death = len(tape.records())
                records = tape.records_at_least(committed_at_death + 4)
                alone = [participants(r) for r in records[committed_at_death + 2 :]]
                fast.stop()
                global_before_rejoin = tape.newest()
                committed_before_rejoin = len(tape.records())

                rejoined = Trainer(SLOW, mailbox, config, tensors=ZEROS)
                rejoined.join()
                caught_up = dict(rejoined.adapter.tensors), rejoined.rounds_seen
                rejoined.train()
                try:
                    records = tape.records_at_least(committed_before_rejoin + 4)
                finally:
                    rejoined.stop()

            assert alone == [("fast",)] * len(alone) and alone
            assert caught_up == (global_before_rejoin, committed_before_rejoin - 1)
            assert rounds(records) == list(range(len(records)))
            assert [participants(r) for r in records[committed_before_rejoin + 2 :]] == [
                ("slow",)
            ] * (len(records) - committed_before_rejoin - 2)

    def when_a_node_restarts_mid_run():
        def it_produces_the_same_commit_sequence_as_an_uninterrupted_run(mailbox: Mailbox):
            # Whether the restarted node was closing rounds or following is up to timing, so both
            # paths are covered over time. Two threaded runs are never step-for-step equal, so
            # "the same sequence" is the one an uninterrupted run is bound to: contiguous rounds,
            # every participant accepted, every commit on the analytic trajectory, and the
            # records written before the restart left untouched. While the node is away the
            # other closes rounds alone; once it is back, rounds wait for both again.
            config = algorithm(quorum=2)
            tape = Tape(mailbox)
            with training(mailbox, config) as (fast, slow):
                before = tape.records_at_least(2)
                slow.stop()
                restarted = Trainer(SLOW, mailbox, config, adapter=slow.adapter)
                restarted.join()
                tape.live_at_least(FAST.node_id, SLOW.node_id)
                restarted.train()
                try:
                    tape.records_at_least(len(before) + 4)
                    fast.stop()
                finally:
                    restarted.stop()
                rejoin(fast.adapter, FAST.node_id, mailbox, config)
                rejoin(slow.adapter, SLOW.node_id, mailbox, config)

            records = tape.records()
            assert list(records[: len(before)]) == list(before)
            assert rounds(records) == list(range(len(records)))
            assert [participants(record) for record in before] == [("fast", "slow")] * len(before)
            assert ("fast", "slow") in {participants(record) for record in records[len(before) :]}
            assert [record.rejected for record in records] == [()] * len(records)
            assert deviations(tape, records) == []
            assert dict(fast.adapter.tensors) == dict(slow.adapter.tensors) == tape.newest()
