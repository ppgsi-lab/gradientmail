"""The learner: counters and publishing decisions as pure state, plus the two-thread runtime.

The training thread, the one running ``Session.run``, owns the model and every counter:
``step()``, which the session calls after every step of the caller's, advances the state,
applies the fragment versions the receive loop has queued, and hands one fragment snapshot,
with the anchor it was trained from, to the publish loop when a round is due; the publish loop
computes their difference in the backend and publishes that delta. A daemon thread runs the
receive and publish loops on its own event loop; the two sides meet only through two
``queue.SimpleQueue``s, so ``step()`` never awaits and never does I/O. It blocks only when
the schedule bounds ``overlap_steps``, waiting in ``step()`` for the model to align with a
round it published that many steps ago, and ``close()`` waits for the last ones. What aligning
does to the steps trained between a snapshot and its commit is the algorithm's ``alignment``:
``Overwrite`` drops them, ``Blend`` keeps a share, ``Carry`` keeps them all, by holding the
snapshot's bytes until the commit and adding what changed since to it, and ``Adaptive``
carries or overwrites each fragment by the agreement the commit's round records.

A fetched batch is queued only once every payload in it is in hand, so ``rounds_seen`` never
runs ahead of a fragment that is not yet applied: that is what keeps a live learner's
snapshots at the syncer's current version.

The receive loop also renews this learner's lease and lists everyone's (``lease`` module), so
``step()`` knows how many learners are live when it judges whether a round is about to close
and ``batch_size`` can split a global batch among them; ``close()`` releases the lease once the
last snapshot is out, and ``abort()``, when training failed, at once.

With ``ranks``, ``start()`` also takes a rank before it starts the communication thread: the
lowest of them that no node holds and none marked done, through a claim the ``protocol`` module
describes, waiting while every rank is held or done. A rank's holder is its first claimant with
a live lease, so the
lease is written, and renewed, from the first listing on. The receive loop then lists the claims
on that rank on every poll, one more LIST; once a claimant before this learner holds a live
lease again, a node that was slow rather than gone, the rank is that node's: the communication
thread stops at once, releasing the lease and publishing nothing more, and ``step()`` or
``close()`` raises ``RankLost``.
"""

import asyncio
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import assert_never

from gradientmail.adapters import Adapter, Backend, default_backend
from gradientmail.algorithms import (
    Adaptive,
    Algorithm,
    Alignment,
    Blend,
    Carry,
    Estimates,
    Overwrite,
    Participant,
    Precision,
    RoundSchedule,
    decoded,
    encoded,
)
from gradientmail.clock import Clock, MonotonicClock
from gradientmail.compression import Compression
from gradientmail.events import (
    AwaitingCreator,
    Callback,
    Closed,
    Closing,
    CreatingRun,
    FragmentAligned,
    GaveUp,
    Joined,
    LeaseReleased,
    PollFailed,
    PublishCancelled,
    PublishFailed,
    RankClaimed,
    ReleaseFailed,
    RoundRecorded,
    SnapshotPublished,
    TookOverCreation,
    log_events,
)
from gradientmail.lease import Liveness
from gradientmail.mailbox import Mailbox
from gradientmail.protocol import (
    Exchange,
    LocalCommits,
    Manifest,
    ManifestDraft,
    PayloadRef,
    ProtocolError,
    RoundRecord,
    RunConfigMismatch,
    SnapshotDraft,
    rank_id,
    theta_digest,
)
from gradientmail.serialization import DType, Schema, SchemaMismatch, TensorBytes, validate

__all__ = [
    "Aligned",
    "Learner",
    "LearnerState",
    "Publishers",
    "RankLost",
    "advanced",
    "align",
    "applied",
    "check_schema",
    "due_round",
    "overdue",
    "publishable",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnerState:
    """Everything the training thread tracks; one value per fragment where it says so.

    ``rounds_seen`` is the newest round whose fragment the model is aligned with,
    ``last_published`` the newest round this learner has a snapshot queued for and
    ``published`` the ``(round, local_step)`` of its last ``in_flight`` publishes, oldest first.
    """

    local_step: int
    steps: tuple[int, ...]
    tokens: tuple[int, ...]
    applied_version: tuple[int, ...]
    rounds_seen: int
    last_published: int
    published: tuple[tuple[int, int], ...]
    step_seconds: float


@dataclass(frozen=True)
class Publishers:
    """The other learners with a snapshot in one round, as the receive loop last listed it."""

    round: int
    others: frozenset[str]


def advanced(state: LearnerState, samples: int, elapsed: float) -> LearnerState:
    """One training step of ``samples`` tokens that took ``elapsed`` seconds."""
    return replace(
        state,
        local_step=state.local_step + 1,
        steps=tuple(count + 1 for count in state.steps),
        tokens=tuple(count + samples for count in state.tokens),
        step_seconds=(
            elapsed
            if state.local_step == 0
            else state.step_seconds + _EMA_DECAY * (elapsed - state.step_seconds)
        ),
    )


def applied(
    state: LearnerState,
    round: int,
    fragment: int,
    version: int,
    *,
    participant: Participant | None,
    alignment: Overwrite | Blend | Carry,
) -> LearnerState:
    """``fragment`` now holds ``version`` committed by ``round``; its counters start over.

    Of the steps and tokens the commit did not incorporate, those since this learner's
    snapshot in the round (``participant``) or all of them when it took no part, the counters
    keep what ``alignment`` keeps in the parameters: none, ``Blend``'s share rounded down, or
    every one with ``Carry``.
    """

    def with_fragment(values: tuple[int, ...], value: int) -> tuple[int, ...]:
        return (*values[:fragment], value, *values[fragment + 1 :])

    def kept(counted: int, incorporated: int) -> int:
        remaining = counted - incorporated
        match alignment:
            case Overwrite():
                return 0
            case Blend(mixing=mixing):
                return int(mixing * remaining)
            case Carry():
                return remaining
            case _:
                assert_never(alignment)

    steps, tokens = (0, 0) if participant is None else (participant.steps, participant.tokens)
    return replace(
        state,
        steps=with_fragment(state.steps, kept(state.steps[fragment], steps)),
        tokens=with_fragment(state.tokens, kept(state.tokens[fragment], tokens)),
        applied_version=with_fragment(state.applied_version, version),
        rounds_seen=max(state.rounds_seen, round),
    )


def publishable(state: LearnerState, schedule: RoundSchedule) -> int | None:
    """The round this learner publishes next, if it may publish it now.

    The candidate is the round after both the last published and the last seen. It may be
    published while it is within ``in_flight`` rounds of the last seen and its fragment has
    been trained since it was last overwritten.
    """
    round = max(state.last_published + 1, state.rounds_seen + 1)
    allowed = (
        round <= state.rounds_seen + schedule.in_flight
        and state.steps[schedule.fragment_of(round)] >= 1
    )
    return round if allowed else None


def due_round(
    state: LearnerState, schedule: RoundSchedule, *, follow: int, publishers: Publishers
) -> int | None:
    """The round to publish now, if any.

    The publishable round is due on schedule, once the gap of steps has passed since the last
    publish, or by opportunity, once ``follow`` others have a snapshot in it: the round is
    closing, or about to, and the steps trained so far go in rather than waiting for the next
    one. ``follow`` is the effective count, at most the other live learners; ``0`` is on
    schedule only.
    """
    round = publishable(state, schedule)
    if round is None:
        return None
    last = state.published[-1][1] if state.published else 0
    on_schedule = state.local_step - last >= schedule.gap(round)
    closing = follow > 0 and publishers.round == round and len(publishers.others) >= follow
    return round if on_schedule or closing else None


def overdue(state: LearnerState, overlap_steps: int) -> bool:
    """Whether a round this learner published, and is not yet aligned with, went out
    ``overlap_steps`` or more steps ago: training may not run further ahead of it."""
    return any(
        round > state.rounds_seen and state.local_step - at >= overlap_steps
        for round, at in state.published
    )


class RankLost(Exception):
    """``owner``, whose claim on ``rank`` comes before ``node``'s, holds a live lease again: the
    rank is ``owner``'s, and ``node`` stopped training it."""

    node: str
    rank: int
    owner: str

    def __init__(self, node: str, rank: int, owner: str) -> None:
        self.node = node
        self.rank = rank
        self.owner = owner
        super().__init__(f"{node} lost rank {rank} to {owner}, whose claim comes first")


class Learner:
    """Wraps a training loop: ``start()`` once, ``step()`` after every optimizer step, ``close()``.

    ``Session.run`` drives it, and calls ``abort()`` instead of ``close()`` when training fails.
    ``start`` blocks until the model holds the newest committed version of every fragment, and,
    with ``ranks``, until this learner holds one of them, then starts the communication thread.
    An operation that fails on that thread is reported (``PollFailed``, ``PublishFailed``) and
    its loop goes on (``communicate``); only the thread's own death, the loss of the rank
    included, is raised, from the next ``step()`` or ``close()``. When the schedule bounds
    ``overlap_steps``, ``step()`` waits for the model to align with a round published that many
    steps ago before training on, as standard DiLoCo does at zero; otherwise training goes on
    and the algorithm's ``alignment`` says what the commit does to the steps taken meanwhile.
    The lease is renewed every ``lease_every`` seconds and the other learners' leases are
    presumed lost after ``lease_timeout`` seconds without a change.

    What goes out is ``anchor - θ``, computed in ``backend`` (the installed one by default)
    against the version of the fragment this learner aligned with last, of which it keeps one host
    copy. ``compression`` is the codec its payloads are stored with. ``local`` holds the commits
    the syncer in the same process made, taken instead of fetched. ``on_event`` receives every
    event the learner reports (``events`` module), on the thread that produced it.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        algorithm: Algorithm,
        mailbox: Mailbox,
        node_id: str,
        backend: Backend | None = None,
        clock: Clock | None = None,
        poll_interval: float = 1.0,
        lease_every: float = 60.0,
        lease_timeout: float = 120.0,
        compression: Compression = "off",
        local: LocalCommits | None = None,
        ranks: int | None = None,
        on_event: Callback = log_events,
    ) -> None:
        if lease_timeout <= lease_every:
            raise ValueError(
                f"lease_timeout must exceed lease_every, got {lease_timeout} and {lease_every}"
            )
        self._adapter = adapter
        self._algorithm = algorithm
        self._node_id = node_id
        self._backend = backend if backend is not None else default_backend()
        self._clock = clock if clock is not None else MonotonicClock()
        self._store_clock = mailbox.clock()
        self._ranks = ranks
        self._rank: int | None = None
        self._poll_interval = poll_interval
        self._lease_every = lease_every
        self._lease_timeout = lease_timeout
        self._exchange = Exchange(mailbox, clock=self._clock, compression=compression)
        self._local = local
        self._on_event = on_event
        self._inbox: queue.SimpleQueue[tuple[_Update, ...]] = queue.SimpleQueue()
        self._outbox: queue.SimpleQueue[_Outgoing | None] = queue.SimpleQueue()
        self._run: _Run | None = None
        self._anchors: list[Mapping[str, TensorBytes]] = []
        self._snapshots: dict[int, Mapping[str, TensorBytes]] = {}
        self._published_at: dict[int, int] = {}
        self._overwriting: set[int] = set()
        self._mine: dict[int, Participant] = {}
        self._newest: dict[int, tuple[int, Mapping[str, TensorBytes]]] = {}
        self._state = LearnerState(0, (), (), (), -1, -1, (), 0.0)
        self._cursor = -1
        self._publishers = Publishers(-1, frozenset())
        self._liveness = Liveness(lease_timeout)
        self._live: frozenset[str] = frozenset()
        self._estimates = Estimates()
        self._last_step_at = 0.0
        self._thread: threading.Thread | None = None
        self._main: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None

    @property
    def rounds_seen(self) -> int:
        return self._state.rounds_seen

    @property
    def rank(self) -> int | None:
        """The rank this learner holds, from ``start`` on; None without ``ranks``."""
        return self._rank

    @property
    def batch_size(self) -> int | None:
        """This learner's share of the global batch, split among the learners live as of the
        last poll and rounded up; ``None`` when the algorithm sets no global batch."""
        global_batch = self._algorithm.global_batch()
        if global_batch is None:
            return None
        live = len(self._live | {self._node_id})
        return (global_batch + live - 1) // live

    def start(self) -> None:
        """Join the run on the calling thread, take a rank when there are ``ranks``, then start
        the communication thread. Taking the rank writes the lease; a failure, an interrupt
        while waiting for a rank included, releases it before it propagates."""

        def communicate() -> None:
            def failure(exc: BaseException) -> BaseException:
                # TaskGroups wrap a lone failure in one group per nesting level; step()
                # should raise the failure itself.
                match exc:
                    case BaseExceptionGroup(exceptions=(BaseException() as only,)):
                        return failure(only)
                    case _:
                        return exc

            try:
                asyncio.run(self.communicate())
            except BaseException as exc:
                self._failure = failure(exc)

        asyncio.run(self.join())
        if self._ranks is not None:
            try:
                self._rank = asyncio.run(self.claim_rank(self._ranks))
            except BaseException:
                asyncio.run(self._exchange.release(self._node_id))
                raise
        self._last_step_at = self._clock.now()
        self._thread = threading.Thread(
            target=communicate, name=f"gradientmail-{self._node_id}", daemon=True
        )
        self._thread.start()

    def step(self, samples: int) -> None:
        """Account for one training step of ``samples`` tokens; apply and publish as due.

        Queued fragment versions are validated against the schema and applied here, while the
        model is quiescent, and so is the snapshot of a fragment whose round is due. Only bytes
        are copied; the delta, packing, hashing and I/O happen on the communication thread. When
        the schedule bounds ``overlap_steps``, it then waits until no round published that many
        steps ago is still outstanding; the wait does not count as step time.
        """
        if self._failure is not None:
            raise self._failure
        if self._run is None:
            raise RuntimeError("Learner.step() before start()")
        run = self._run
        now = self._clock.now()
        state = advanced(self._state, samples, now - self._last_step_at)
        self._last_step_at = now
        state = self._drained(run, state)
        round = due_round(
            state,
            run.schedule,
            follow=0 if run.follow == 0 else max(min(run.follow, len(self._live) - 1), 1),
            publishers=self._publishers,
        )
        if round is not None:
            state = self._publish(run, state, round, final=False)
        if (limit := run.schedule.overlap_steps) is not None:
            state = self._settled(run, state, lambda s: not overdue(s, limit), patience=None)
            self._last_step_at = self._clock.now()
        self._state = state

    def close(self) -> None:
        """Wait for the commits of this learner's rounds in flight and align with them, then stop.

        The steps trained since a fragment's last round are dropped: the model is set back to
        the newest version of every fragment it aligned with. With the algorithm's
        ``flush_on_close`` they are published instead, as final snapshots, round after round,
        each awaited and aligned with, so the run's parameters end with them and the model holds
        them. The wait gives up once nothing has landed for the lease timeout plus the takeover
        timeout the last record predicts (at least thirty polls): time for crashed learners to
        be presumed lost, then for a round without them to be presumed not to commit, as when
        nobody is left to close one. Then the lease is released, the outbox is drained, bounded
        by ``_CLOSE_TIMEOUT`` seconds, and the communication thread is joined. Closing again
        does nothing. What the communication thread died of, the loss of the rank included, is
        raised once the thread is joined, without waiting for anything.
        """
        if self._thread is None:
            return
        try:
            if self._failure is not None:
                raise self._failure
            if self._run is not None:
                self._state = self._concluded(self._run, self._state)
                if not self._algorithm.flush_on_close:
                    for anchor in self._anchors:
                        self._adapter.apply(anchor)
        finally:
            self._outbox.put(None)
            self._thread.join(_CLOSE_TIMEOUT)
            if self._thread.is_alive() and self._main is not None:
                self._on_event(PublishCancelled(self._node_id, _CLOSE_TIMEOUT))
                self._main.get_loop().call_soon_threadsafe(self._main.cancel)
                self._thread.join()
            self._thread = None

    def abort(self) -> None:
        """Stop at once, for a training loop that failed: nothing more is published or awaited.

        The communication thread is cancelled, which releases the lease; snapshots still queued
        for publishing are dropped and the model keeps what it holds. Does nothing once closed.
        """
        if self._thread is None:
            return
        main = self._main
        if main is None:
            # The thread has not reached its loop yet; the sentinel ends it there.
            self._outbox.put(None)
        elif not main.done() and not main.get_loop().is_closed():
            main.get_loop().call_soon_threadsafe(main.cancel)
        self._thread.join()
        self._thread = None

    async def join(self) -> None:
        """Create the run or adopt it, then load the newest committed version of every fragment.

        Adoption requires the model's schema and the algorithm's configuration to equal the
        manifest's (``SchemaMismatch`` naming the differing parameters, ``RunConfigMismatch``).
        Counters start at zero; a restart takes this same path.
        """
        schema = self._adapter.schema()
        algorithm = self._algorithm.to_json()
        manifest = await self._exchange.manifest()
        if manifest is None:
            manifest = await self.create(schema)
        if manifest.algorithm != algorithm:
            raise RunConfigMismatch(expected=algorithm, found=manifest.algorithm)

        fragments = range(len(manifest.plan))
        aligned = await align(self._adapter, self._exchange, manifest)
        records = aligned.records
        newest = _newest_per_fragment(records)
        self._anchors = list(aligned.fragments)
        self._newest = {
            p: (newest[p].version if p in newest else 0, tensors)
            for p, tensors in enumerate(aligned.fragments)
        }
        follow = self._algorithm.follow
        self._run = _Run(
            schema=schema,
            plan=manifest.plan,
            schedule=self._algorithm.schedule(len(manifest.plan)),
            follow=self._algorithm.quorum() - 1 if follow is None else follow,
            alignment=self._algorithm.alignment,
            commit_dtype=self._algorithm.commit_dtype,
            digest=schema.digest(),
            dtypes={spec.name: spec.dtype for spec in schema.parameters},
            shapes={spec.name: spec.shape for spec in schema.parameters},
        )
        self._state = LearnerState(
            local_step=0,
            steps=(0,) * len(manifest.plan),
            tokens=(0,) * len(manifest.plan),
            applied_version=tuple(newest[p].version if p in newest else 0 for p in fragments),
            rounds_seen=records[-1].round if records else -1,
            last_published=-1,
            published=(),
            step_seconds=0.0,
        )
        self._cursor = self._state.rounds_seen
        if records:
            self._estimates = records[-1].estimates
        self._on_event(
            Joined(self._node_id, manifest.created_by, len(manifest.plan), self._state.rounds_seen)
        )

    async def create(self, schema: Schema[DType]) -> Manifest:
        """Create the run from this model, or wait for the node that is creating it.

        The first claim on the manifest elects the creator, as the first claim on a round elects
        its closer: one node uploads Θ⁰ and writes the manifest while the others poll for it,
        instead of each uploading a Θ⁰ of its own only to lose the race. A waiter counts the
        init payloads as they land; when none lands for as long as the largest fragment would
        take at ``_MIN_UPLOAD_RATE``, and never less than ``_CREATION_FLOOR_POLLS`` polls, it
        takes the next claim and creates the run itself. A creator that was slow rather than
        dead adopts the manifest that won, as ``create_run`` does. A restarted creator finds
        its own claim and resumes.
        """
        plan = self._algorithm.plan(schema)

        async def upload() -> Manifest:
            draft = ManifestDraft(
                schema=schema,
                plan=plan,
                algorithm=self._algorithm.to_json(),
                created_by=self._node_id,
            )
            return await self._exchange.create_run(
                draft, [self._adapter.snapshot(names) for names in plan]
            )

        largest = max(sum(schema.nbytes(name) for name in names) for names in plan)
        stall = max(_CREATION_FLOOR_POLLS * self._poll_interval, largest / _MIN_UPLOAD_RATE)
        claimants = await self._exchange.creation_claims()
        if self._node_id in claimants or (
            not claimants and await self._exchange.claim_creation(1, self._node_id)
        ):
            self._on_event(CreatingRun(self._node_id, len(plan)))
            return await upload()
        claimants = await self._exchange.creation_claims()
        attempts = len(claimants)
        self._on_event(AwaitingCreator(self._node_id, tuple(claimants)))
        uploaded = await self._exchange.init_uploaded()
        seen_at = self._clock.now()
        while True:
            await self._clock.sleep(self._poll_interval)
            manifest = await self._exchange.manifest()
            if manifest is not None:
                return manifest
            count = await self._exchange.init_uploaded()
            if count != uploaded:
                uploaded, seen_at = count, self._clock.now()
            elif self._clock.now() - seen_at >= stall:
                if await self._exchange.claim_creation(attempts + 1, self._node_id):
                    self._on_event(TookOverCreation(self._node_id, stall))
                    return await upload()
                attempts = len(await self._exchange.creation_claims())
                seen_at = self._clock.now()

    async def claim_rank(self, ranks: int) -> int:
        """Take the lowest of ``ranks`` ranks that no other node holds and whose data is not
        marked exhausted, waiting while each one is held or done; the lease is held, and
        renewed, from the first listing on.

        The claim is written and, a second of the store's clock later, the claims are listed
        again (``protocol`` module): the rank is this learner's when its claim is now the first
        whose node holds a live lease, and otherwise the claim is deleted and the next free rank
        tried. A rank this learner already holds, from before a restart, counts as free. Keeping
        a rank deletes the claims before this learner's whose node holds no lease at all.
        """
        renew_at = self._clock.now()
        while True:
            if self._clock.now() >= renew_at:
                await self._exchange.beat(self._node_id)
                renew_at = self._clock.now() + self._lease_every
            await self._observed()
            claims = await self._exchange.rank_claims()
            done = await self._exchange.done()
            free = next(
                (
                    rank
                    for rank in range(ranks)
                    if rank_id(rank) not in done
                    and self._holder(claims.get(rank, ())) in (None, self._node_id)
                ),
                None,
            )
            if free is None:
                await self._clock.sleep(self._poll_interval)
                continue
            await self._exchange.claim_rank(free, self._node_id)
            await self._store_clock.sleep(_RANK_WAIT)
            leases = await self._observed()
            claimants = (await self._exchange.rank_claims()).get(free, ())
            if self._holder(claimants) == self._node_id:
                before = claimants[: claimants.index(self._node_id)]
                for gone in set(before) - leases.keys():
                    await self._exchange.release_rank(free, gone)
                self._on_event(RankClaimed(self._node_id, free))
                return free
            await self._exchange.release_rank(free, self._node_id)

    async def communicate(self) -> None:
        """Receive, publish and renew the lease until ``close()``, then release the lease; what
        the communication thread runs.

        Neither loop dies of a failed operation: the failure is reported and the loop goes on,
        the receive loop with its next poll (so the lease keeps being renewed), the publish loop
        with the next snapshot (the failed round's steps stay counted and go out with the
        fragment's next round). Only cancellation ends them, or the loss of this learner's rank,
        which ends both at once (``RankLost``); the lease is released either way.
        """

        async def receive() -> None:
            renew_at = self._clock.now()
            while True:
                try:
                    if self._clock.now() >= renew_at:
                        await self._exchange.beat(self._node_id)
                        renew_at = self._clock.now() + self._lease_every
                        log.debug("%s: lease renewed", self._node_id)
                    await self.poll()
                except RankLost:
                    raise
                except Exception as exc:
                    self._on_event(PollFailed(self._node_id, exc))
                await self._clock.sleep(self._poll_interval)

        async def publish() -> None:
            while (outgoing := await asyncio.to_thread(self._outbox.get)) is not None:
                draft = outgoing.draft
                try:
                    delta = await asyncio.to_thread(self._delta, run, outgoing)
                    started = time.perf_counter()
                    published = await self._exchange.publish_snapshot(draft, delta)
                    self._on_event(
                        SnapshotPublished(
                            self._node_id,
                            draft.round,
                            run.schedule.fragment_of(draft.round),
                            draft.steps,
                            draft.tokens,
                            published.payload.size,
                            time.perf_counter() - started,
                        )
                    )
                except Exception as exc:
                    self._on_event(PublishFailed(self._node_id, draft.round, exc))

        if self._run is None:
            raise RuntimeError("Learner.communicate() before join()")
        run = self._run
        self._main = asyncio.current_task()
        try:
            async with asyncio.TaskGroup() as group:
                receiving = group.create_task(receive())
                await publish()
                receiving.cancel()
        finally:
            # A cancelled publish leaves a worker thread blocked in outbox.get(), and asyncio.run
            # waits for the executor's threads on shutdown; the sentinel releases it.
            self._outbox.put(None)
            try:
                await self._exchange.release(self._node_id)
                self._on_event(LeaseReleased(self._node_id, "learner"))
            except Exception as exc:
                self._on_event(ReleaseFailed(self._node_id, "learner", exc))

    async def poll(self) -> None:
        """One receive iteration: queue the newest version of each fragment committed meanwhile,
        then list the leases, the claims on this learner's rank when it holds one, and who has
        published the round this learner publishes next. Raises ``RankLost`` once another node
        holds the rank.

        The batch is queued only once every payload in it has been fetched. A payload already
        swept is skipped; the next poll finds that fragment's newer version, and this learner's
        part in the skipped round, kept from its record, goes with that version: what the run
        incorporated of this learner is settled by the records, not by the payloads it reached.
        """
        if self._run is None:
            raise RuntimeError("Learner.poll() before join()")
        run = self._run
        records = await self._exchange.rounds_after(self._cursor)
        if records:
            newest = tuple(_newest_per_fragment(records).values())
            self._mine.update(
                (record.fragment, participant)
                for record in records
                for participant in record.participants
                if participant.learner == self._node_id
            )
            async with asyncio.TaskGroup() as group:
                fetches = [group.create_task(self._fetched(run, record)) for record in newest]
            updates = tuple(
                _Update(
                    record.round,
                    record.fragment,
                    record.version,
                    tensors,
                    self._mine.pop(record.fragment, None),
                    tuple(
                        (r.round, r.cosine)
                        for r in records
                        if r.fragment == record.fragment and r.cosine is not None
                    ),
                )
                for record, fetch in zip(newest, fetches, strict=True)
                if (tensors := fetch.result()) is not None
            )
            if updates:
                self._inbox.put(updates)
            self._cursor = records[-1].round
            self._estimates = records[-1].estimates
            for record in records:
                self._on_event(RoundRecorded(self._node_id, record))
        await self._observed()
        if (rank := self._rank) is not None:
            holder = self._holder((await self._exchange.rank_claims()).get(rank, ()))
            if holder is not None and holder != self._node_id:
                raise RankLost(self._node_id, rank, holder)
        state = self._state
        round = max(state.last_published + 1, state.rounds_seen + 1)
        others = await self._exchange.publishers(round)
        self._publishers = Publishers(round, others - {self._node_id})

    async def _observed(self) -> Mapping[str, int]:
        """Fold one listing of the leases into the live learners, and return it."""
        leases = await self._exchange.leases()
        now = self._clock.now()
        self._liveness = self._liveness.observed(leases, now)
        self._live = self._liveness.live(now)
        return leases

    def _holder(self, claimants: Sequence[str]) -> str | None:
        """The holder of a rank claimed by ``claimants``: the first of them that holds a live
        lease as last listed, this learner always counted live."""
        return next(
            (node for node in claimants if node == self._node_id or node in self._live), None
        )

    def _publish(
        self, run: "_Run", state: LearnerState, round: int, *, final: bool
    ) -> LearnerState:
        """Snapshot ``round``'s fragment and its anchor for the publish loop; the bytes are
        copied here. ``final`` when closing."""
        fragment = run.schedule.fragment_of(round)
        draft = SnapshotDraft(
            learner=self._node_id,
            round=round,
            local_step=state.local_step,
            steps=state.steps[fragment],
            tokens=state.tokens[fragment],
            base_version=state.applied_version[fragment],
            rounds_seen=state.rounds_seen,
            step_seconds=state.step_seconds,
            schema_digest=run.digest,
            final=final,
        )
        snapshot = self._adapter.snapshot(run.plan[fragment])
        if isinstance(run.alignment, Carry | Adaptive):
            self._snapshots[fragment] = snapshot
        self._published_at[fragment] = state.local_step
        self._outbox.put(_Outgoing(draft, self._anchors[fragment], snapshot))
        log.debug(
            "%s: round %d queued: fragment %d, %d steps at local step %d",
            self._node_id,
            round,
            fragment,
            draft.steps,
            state.local_step,
        )
        published = (*state.published, (round, state.local_step))[-run.schedule.in_flight :]
        return replace(state, last_published=round, published=published)

    async def _fetched(self, run: "_Run", record: RoundRecord) -> Mapping[str, TensorBytes] | None:
        """The committed bytes of ``record``'s fragment: taken from ``local`` when this process
        committed them, rebuilt from the commit's update when this learner holds the version
        before it and the result hashes to the record's theta, otherwise fetched; None when
        swept."""
        if self._local is not None and (theta := self._local.take(record)) is not None:
            self._newest[record.fragment] = (record.version, theta)
            return theta
        held = self._newest.get(record.fragment)
        if record.update is not None and held is not None and held[0] == record.version - 1:
            update = await self._exchange.fetch(record.update.payload)
            if update is not None:
                rebuilt = await asyncio.to_thread(
                    self._rebuilt, run, record.fragment, held[1], update
                )
                if theta_digest(rebuilt) == record.update.theta_digest:
                    self._newest[record.fragment] = (record.version, rebuilt)
                    return rebuilt
                log.debug(
                    "%s: round %d: the update does not rebuild theta; fetching it",
                    self._node_id,
                    record.round,
                )
        tensors = await self._exchange.fetch(record.theta)
        if tensors is not None:
            self._newest[record.fragment] = (record.version, tensors)
        return tensors

    def _rebuilt(
        self,
        run: "_Run",
        fragment: int,
        base: Mapping[str, TensorBytes],
        update: Mapping[str, TensorBytes],
    ) -> Mapping[str, TensorBytes]:
        """``base`` plus the commit's update, rounded to the schema's dtypes."""
        load = self._backend.load
        return {
            name: (
                load(base[name])
                + decoded(self._backend, update, name, run.shapes[name], run.commit_dtype)
            ).to_bytes(run.dtypes[name])
            for name in run.plan[fragment]
        }

    def _delta(self, run: "_Run", outgoing: "_Outgoing") -> Mapping[str, TensorBytes]:
        """``anchor - θ`` of every tensor in the snapshot, at the algorithm's delta precision."""
        load = self._backend.load
        return {
            key: tensor
            for name, local in outgoing.snapshot.items()
            for key, tensor in encoded(
                name,
                load(outgoing.anchor[name]) - load(local),
                run.dtypes[name],
                self._algorithm.delta_dtype,
            ).items()
        }

    def _align(self, run: "_Run", state: LearnerState, update: "_Update") -> LearnerState:
        """Set the fragment to the committed version, as the algorithm's alignment has it; the
        committed bytes become the anchor either way."""
        fragment, names = update.fragment, run.plan[update.fragment]
        validate(run.schema, update.tensors, names)
        snapshot = self._snapshots.pop(fragment, None)
        # The participation is this learner's own only if this process took its snapshot: after
        # a restart, the run may still incorporate one from before it, whose steps are in
        # neither the parameters nor the counters.
        mine = update.participant
        published_at = self._published_at.pop(fragment, None)
        own = mine if mine is not None and mine.local_step == published_at else None
        rule = self._rule(run, update)
        match rule:
            case Overwrite():
                aligned = update.tensors
            case Blend(mixing=mixing):
                aligned = self._blended(run, names, update.tensors, mixing)
            case Carry():
                # Everything since the snapshot the commit incorporated, or since the anchor
                # when it incorporated nothing of what the parameters hold.
                since = (
                    snapshot
                    if own is not None and snapshot is not None
                    else self._anchors[fragment]
                )
                aligned = self._carried(run, names, update.tensors, since)
            case _:
                assert_never(rule)
        self._adapter.apply(aligned)
        self._anchors[fragment] = update.tensors
        self._on_event(
            FragmentAligned(
                self._node_id,
                update.round,
                update.fragment,
                update.version,
                state.local_step,
                None if own is None else state.local_step - own.local_step,
            )
        )
        return applied(
            state,
            update.round,
            update.fragment,
            update.version,
            participant=own,
            alignment=rule,
        )

    def _rule(self, run: "_Run", update: "_Update") -> Overwrite | Blend | Carry:
        """What aligning with ``update`` does: the algorithm's rule, or, with ``Adaptive``, the
        one its thresholds select from the cosines of the fragment's rounds since it was last
        aligned, in order, kept between them."""
        match run.alignment:
            case Adaptive(carry_above=above, overwrite_below=below):
                fragment = update.fragment
                for round, cosine in update.cosines:
                    was = fragment in self._overwriting
                    if cosine < below:
                        self._overwriting.add(fragment)
                    elif cosine >= above:
                        self._overwriting.discard(fragment)
                    if (fragment in self._overwriting) != was:
                        log.info(
                            "%s: fragment %d %s from round %d on: cosine %.2f",
                            self._node_id,
                            fragment,
                            "overwriting" if fragment in self._overwriting else "carrying",
                            round,
                            cosine,
                        )
                return Overwrite() if fragment in self._overwriting else Carry()
            case Overwrite() | Blend() | Carry() as fixed:
                return fixed
            case _:
                assert_never(run.alignment)

    def _blended(
        self, run: "_Run", names: Sequence[str], theta: Mapping[str, TensorBytes], mixing: float
    ) -> Mapping[str, TensorBytes]:
        """``mixing·θ + (1 - mixing)·Θ`` over the fragment, rounded to the schema's dtypes."""
        load = self._backend.load
        local = self._adapter.snapshot(names)
        return {
            name: (load(local[name]) * mixing + load(theta[name]) * (1 - mixing)).to_bytes(
                run.dtypes[name]
            )
            for name in names
        }

    def _carried(
        self,
        run: "_Run",
        names: Sequence[str],
        theta: Mapping[str, TensorBytes],
        since: Mapping[str, TensorBytes],
    ) -> Mapping[str, TensorBytes]:
        """``Θ + (θ - since)`` over the fragment, rounded to the schema's dtypes."""
        load = self._backend.load
        local = self._adapter.snapshot(names)
        return {
            name: (load(theta[name]) + (load(local[name]) - load(since[name]))).to_bytes(
                run.dtypes[name]
            )
            for name in names
        }

    def _drained(self, run: "_Run", state: LearnerState) -> LearnerState:
        """Align with every fragment version the receive loop has queued, without waiting."""

        def queued() -> Iterator[_Update]:
            while True:
                try:
                    batch = self._inbox.get_nowait()
                except queue.Empty:
                    return
                yield from batch

        for update in queued():
            state = self._align(run, state, update)
        return state

    def _settled(
        self,
        run: "_Run",
        state: LearnerState,
        done: Callable[[LearnerState], bool],
        *,
        patience: float | None,
    ) -> LearnerState:
        """Align with fragment versions as they arrive until ``done(state)`` holds.

        Blocks the calling thread. With ``patience``, returns once nothing has happened for that
        long: no version aligned with and no new snapshot in the round this learner publishes
        next. Raises what the communication thread died of.
        """
        progress = (state.rounds_seen, self._publishers)
        since = self._clock.now()
        while not done(state):
            try:
                batch = self._inbox.get(timeout=self._poll_interval)
            except queue.Empty:
                batch = ()
            if self._failure is not None:
                raise self._failure
            for update in batch:
                state = self._align(run, state, update)
            now = self._clock.now()
            if (state.rounds_seen, self._publishers) != progress:
                progress, since = (state.rounds_seen, self._publishers), now
            elif patience is not None and now - since >= patience:
                break
        return state

    def _concluded(self, run: "_Run", state: LearnerState) -> LearnerState:
        """Wait for the commits of the rounds in flight; with ``flush_on_close``, first publish
        every fragment trained since its last overwrite as a final snapshot.

        Flushed rounds go out as the schedule allows, ``in_flight`` at a time, each next one
        once the commits make it publishable; done when nothing is publishable and nothing is
        outstanding, or when the run stalls for the lease timeout plus the takeover timeout.
        """

        def flushable(state: LearnerState) -> int | None:
            return publishable(state, run.schedule) if self._algorithm.flush_on_close else None

        def settled(state: LearnerState) -> bool:
            return flushable(state) is not None or state.rounds_seen >= state.last_published

        self._on_event(Closing(self._node_id, self._algorithm.flush_on_close))
        patience = self._lease_timeout + self._estimates.timeout(self._poll_interval)
        state = self._drained(run, state)
        while True:
            while (round := flushable(state)) is not None:
                state = self._publish(run, state, round, final=True)
            if state.rounds_seen >= state.last_published:
                self._on_event(Closed(self._node_id, state.last_published))
                return state
            state = self._settled(run, state, settled, patience=patience)
            if not settled(state):
                self._on_event(
                    GaveUp(self._node_id, state.last_published, patience, state.rounds_seen)
                )
                return state


@dataclass(frozen=True)
class Aligned:
    """What ``align`` loaded: the round records read, in round order, and the tensors the
    adapter now holds, one mapping per fragment."""

    records: Sequence[RoundRecord]
    fragments: tuple[Mapping[str, TensorBytes], ...]


async def align(adapter: Adapter, exchange: Exchange, manifest: Manifest) -> Aligned:
    """Load the newest committed version of every fragment of ``manifest``'s run into ``adapter``.

    A fragment no round has committed yet is loaded from the manifest's initial parameters.
    ``SchemaMismatch`` names the parameters on which the model and the run differ.
    """
    schema = adapter.schema()
    check_schema(schema, manifest)
    fragments = range(len(manifest.plan))
    records = await exchange.rounds_after(-1)
    newest = _newest_per_fragment(records)
    refs = [newest[p].theta if p in newest else manifest.init[p] for p in fragments]
    loaded: list[Mapping[str, TensorBytes]] = []
    for fragment, (ref, tensors) in enumerate(zip(refs, await _fetch(exchange, refs), strict=True)):
        if tensors is None:
            raise ProtocolError(f"{ref.key} was swept while loading; start again")
        validate(schema, tensors, manifest.plan[fragment])
        adapter.apply(tensors)
        loaded.append(tensors)
    return Aligned(records, tuple(loaded))


def check_schema(schema: Schema[DType], manifest: Manifest) -> None:
    """Raise ``SchemaMismatch`` naming the parameters on which ``schema`` and the run differ."""
    if manifest.schema != schema:
        differing = set(manifest.schema.parameters) ^ set(schema.parameters)
        raise SchemaMismatch(
            sorted({spec.name for spec in differing}), "model does not match the run's schema"
        )


@dataclass(frozen=True)
class _Run:
    """The run as joined: fixed for the learner's lifetime."""

    schema: Schema[DType]
    plan: tuple[tuple[str, ...], ...]
    schedule: RoundSchedule
    follow: int
    alignment: Alignment
    commit_dtype: Precision
    digest: str
    dtypes: Mapping[str, DType]
    shapes: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class _Update:
    """A fragment version fetched by the receive loop, waiting for ``step()`` to apply it;
    ``participant`` is this learner's snapshot in the newest of the fragment's rounds it took
    part in since the fragment was last applied, when there is one, and ``cosines`` the mean
    cosine each of the fragment's rounds since then recorded, by round."""

    round: int
    fragment: int
    version: int
    tensors: Mapping[str, TensorBytes]
    participant: Participant | None
    cosines: tuple[tuple[int, float], ...]


@dataclass(frozen=True)
class _Outgoing:
    """A snapshot taken by ``step()`` with the anchor it was trained from, waiting for the
    publish loop to subtract them."""

    draft: SnapshotDraft
    anchor: Mapping[str, TensorBytes]
    snapshot: Mapping[str, TensorBytes]


def _newest_per_fragment(records: Sequence[RoundRecord]) -> Mapping[int, RoundRecord]:
    # Records come in round order, so the last one seen per fragment is its newest version.
    return {record.fragment: record for record in records}


async def _fetch(
    exchange: Exchange, refs: Sequence[PayloadRef]
) -> Sequence[Mapping[str, TensorBytes] | None]:
    async with asyncio.TaskGroup() as group:
        fetches = [group.create_task(exchange.fetch(ref)) for ref in refs]
    return [fetch.result() for fetch in fetches]


_EMA_DECAY = 0.1
_CLOSE_TIMEOUT = 60.0
# Seconds of the store's clock a claimant waits for earlier claims on its rank to land.
_RANK_WAIT = 1.0
_CREATION_FLOOR_POLLS = 30
# A live creator is assumed to upload at least this fast; slower, and a waiter takes over.
_MIN_UPLOAD_RATE = 2**20
