"""The learner: counters and publishing decisions as pure state, plus the two-thread runtime.

The training thread owns the model and every counter: ``step()`` advances the state, applies
the fragment versions the receive loop has queued, and hands one fragment snapshot, with the
anchor it was trained from, to the publish loop when a round is due; the publish loop
computes their difference in the backend and publishes that delta. A daemon thread runs the
receive and publish loops on its own event loop; the two sides meet only through two
``queue.SimpleQueue``s, so ``step()`` never awaits and never does I/O. It blocks only when
the algorithm is synchronous, waiting in ``step()`` for its published rounds to be
installed, and ``close()`` waits for the last ones.

A fetched batch is queued only once every payload in it is in hand, so ``rounds_seen`` never
runs ahead of a fragment that is not yet applied: that is what keeps a live learner's
snapshots at the syncer's current version.

The receive loop also renews this learner's lease and lists everyone's (``lease`` module), so
``step()`` knows how many learners are live when it judges whether a round is about to close
and ``batch_size`` can split a global batch among them; ``close()`` releases the lease once the
last snapshot is out.
"""

import asyncio
import queue
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace

from gradientmail.adapters import Adapter, Backend, default_backend
from gradientmail.algorithms import Algorithm, Estimates, RoundSchedule
from gradientmail.clock import Clock, MonotonicClock
from gradientmail.compression import Compression
from gradientmail.lease import Liveness
from gradientmail.mailbox import Mailbox
from gradientmail.protocol import (
    Exchange,
    Manifest,
    ManifestDraft,
    PayloadRef,
    ProtocolError,
    RoundRecord,
    RunConfigMismatch,
    SnapshotDraft,
)
from gradientmail.serialization import DType, Schema, SchemaMismatch, TensorBytes, validate

__all__ = [
    "Installed",
    "Learner",
    "LearnerState",
    "Publishers",
    "advanced",
    "applied",
    "due_round",
    "install",
    "publishable",
]


@dataclass(frozen=True)
class LearnerState:
    """Everything the training thread tracks; one value per fragment where it says so.

    ``rounds_seen`` is the newest round whose fragment has been installed, ``last_published``
    the newest round this learner has a snapshot queued for.
    """

    local_step: int
    steps: tuple[int, ...]
    tokens: tuple[int, ...]
    applied_version: tuple[int, ...]
    rounds_seen: int
    last_published: int
    step_at_last_publish: int
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


def applied(state: LearnerState, round: int, fragment: int, version: int) -> LearnerState:
    """``fragment`` now holds ``version`` committed by ``round``; its counters start over."""

    def with_fragment(values: tuple[int, ...], value: int) -> tuple[int, ...]:
        return (*values[:fragment], value, *values[fragment + 1 :])

    return replace(
        state,
        steps=with_fragment(state.steps, 0),
        tokens=with_fragment(state.tokens, 0),
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
    state: LearnerState, schedule: RoundSchedule, *, quorum: int, publishers: Publishers
) -> int | None:
    """The round to publish now, if any.

    The publishable round is due on schedule, once the gap of steps has passed since the last
    publish, or by opportunity, once the others' snapshots in it are one short of ``quorum``:
    the round is about to close, and the steps trained so far go in rather than being
    overwritten by its commit. ``quorum`` is the effective one, the configured size or the
    number of live learners, whichever is smaller. Below quorum two, one other snapshot is that
    signal.
    """
    round = publishable(state, schedule)
    if round is None:
        return None
    on_schedule = state.local_step - state.step_at_last_publish >= schedule.gap(round)
    closing = publishers.round == round and len(publishers.others) >= max(quorum - 1, 1)
    return round if on_schedule or closing else None


class Learner:
    """Wraps a training loop: ``start()`` once, ``step()`` after every optimizer step, ``close()``.

    ``start`` blocks until the model holds the newest committed version of every fragment,
    then starts the communication thread. A failure on that thread ends both loops and is
    raised from the next ``step()``. When the algorithm is ``synchronous``, ``step()`` waits for
    a published round to be committed and installed before training on, as standard DiLoCo
    does; otherwise training goes on and the commit overwrites the steps taken meanwhile. The
    lease is renewed every ``lease_every`` seconds and the other learners' leases are presumed lost
    after ``lease_timeout`` seconds without a change.

    What goes out is ``anchor - θ``, computed in ``backend`` (the installed one by default)
    against the version of the fragment this learner installed last, of which it keeps one host
    copy. ``compression`` is the codec its payloads are stored with.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        algorithm: Algorithm,
        mailbox: Mailbox,
        learner_id: str,
        backend: Backend | None = None,
        clock: Clock | None = None,
        poll_interval: float = 1.0,
        lease_every: float = 60.0,
        lease_timeout: float = 120.0,
        compression: Compression = "off",
    ) -> None:
        if lease_timeout <= lease_every:
            raise ValueError(
                f"lease_timeout must exceed lease_every, got {lease_timeout} and {lease_every}"
            )
        self._adapter = adapter
        self._algorithm = algorithm
        self._learner_id = learner_id
        self._backend = backend if backend is not None else default_backend()
        self._clock = clock if clock is not None else MonotonicClock()
        self._poll_interval = poll_interval
        self._lease_every = lease_every
        self._lease_timeout = lease_timeout
        self._exchange = Exchange(mailbox, clock=self._clock, compression=compression)
        self._inbox: queue.SimpleQueue[tuple[_Update, ...]] = queue.SimpleQueue()
        self._outbox: queue.SimpleQueue[_Outgoing | None] = queue.SimpleQueue()
        self._run: _Run | None = None
        self._anchors: list[Mapping[str, TensorBytes]] = []
        self._state = LearnerState(0, (), (), (), -1, -1, 0, 0.0)
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
    def batch_size(self) -> int | None:
        """This learner's share of the global batch, split among the learners live as of the
        last poll and rounded up; ``None`` when the algorithm sets no global batch."""
        global_batch = self._algorithm.global_batch()
        if global_batch is None:
            return None
        live = len(self._live | {self._learner_id})
        return (global_batch + live - 1) // live

    def start(self) -> None:
        """Join the run on the calling thread, then start the communication thread."""

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
        self._last_step_at = self._clock.now()
        self._thread = threading.Thread(
            target=communicate, name=f"gradientmail-{self._learner_id}", daemon=True
        )
        self._thread.start()

    def step(self, samples: int) -> None:
        """Account for one training step of ``samples`` tokens; apply and publish as due.

        Queued fragment versions are validated against the schema and applied here, while the
        model is quiescent, and so is the snapshot of a fragment whose round is due. Only bytes
        are copied; the delta, packing, hashing and I/O happen on the communication thread. When
        the algorithm is synchronous, it then waits until fewer than ``in_flight`` of its rounds
        are outstanding; the wait does not count as step time.
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
            quorum=min(run.quorum, len(self._live)),
            publishers=self._publishers,
        )
        if round is not None:
            state = self._publish(run, state, round)
        if self._algorithm.synchronous:
            state = self._settled(
                run,
                state,
                lambda s: s.last_published - s.rounds_seen < run.schedule.in_flight,
                patience=None,
            )
            self._last_step_at = self._clock.now()
        self._state = state

    def close(self) -> None:
        """Publish the steps trained since the last rounds, wait for their commits, then stop.

        Every fragment trained since it was last overwritten is published, round after round,
        and each round is waited for and installed, so the run's parameters end with this
        learner's last steps and the model holds them. The wait gives up once nothing has
        landed for the lease timeout plus the takeover timeout the last record predicts (at
        least thirty polls): time for crashed learners to be presumed lost, then for a round
        without them to be presumed not to commit, as when nobody is left to close one. Then
        the lease is released, the outbox is drained, bounded by ``_CLOSE_TIMEOUT`` seconds, and
        the communication thread is joined.
        """
        if self._thread is None:
            return
        try:
            if self._run is not None and self._failure is None:
                self._state = self._flushed(self._run, self._state)
        finally:
            self._outbox.put(None)
            self._thread.join(_CLOSE_TIMEOUT)
            if self._thread.is_alive() and self._main is not None:
                self._main.get_loop().call_soon_threadsafe(self._main.cancel)
                self._thread.join()

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
        installed = await install(self._adapter, self._exchange, manifest)
        records = installed.records
        newest = _newest_per_fragment(records)
        self._anchors = list(installed.fragments)
        self._run = _Run(
            schema=schema,
            plan=manifest.plan,
            schedule=self._algorithm.schedule(len(manifest.plan)),
            quorum=self._algorithm.quorum(),
            digest=schema.digest(),
            delta_dtype={
                spec.name: spec.dtype for spec in self._algorithm.delta_schema(schema).parameters
            },
        )
        self._state = LearnerState(
            local_step=0,
            steps=(0,) * len(manifest.plan),
            tokens=(0,) * len(manifest.plan),
            applied_version=tuple(newest[p].version if p in newest else 0 for p in fragments),
            rounds_seen=records[-1].round if records else -1,
            last_published=-1,
            step_at_last_publish=0,
            step_seconds=0.0,
        )
        self._cursor = self._state.rounds_seen
        if records:
            self._estimates = records[-1].estimates

    async def create(self, schema: Schema) -> Manifest:
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
                created_by=self._learner_id,
            )
            return await self._exchange.create_run(
                draft, [self._adapter.snapshot(names) for names in plan]
            )

        largest = max(sum(schema.nbytes(name) for name in names) for names in plan)
        stall = max(_CREATION_FLOOR_POLLS * self._poll_interval, largest / _MIN_UPLOAD_RATE)
        claimants = await self._exchange.creation_claims()
        if self._learner_id in claimants or (
            not claimants and await self._exchange.claim_creation(1, self._learner_id)
        ):
            return await upload()
        attempts = len(await self._exchange.creation_claims())
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
                if await self._exchange.claim_creation(attempts + 1, self._learner_id):
                    return await upload()
                attempts = len(await self._exchange.creation_claims())
                seen_at = self._clock.now()

    async def communicate(self) -> None:
        """Receive, publish and renew the lease until ``close()``, then release the lease; what
        the communication thread runs."""

        async def receive() -> None:
            renew_at = self._clock.now()
            while True:
                if self._clock.now() >= renew_at:
                    await self._exchange.beat(self._learner_id)
                    renew_at = self._clock.now() + self._lease_every
                await self.poll()
                await self._clock.sleep(self._poll_interval)

        async def publish() -> None:
            while (outgoing := await asyncio.to_thread(self._outbox.get)) is not None:
                delta = await asyncio.to_thread(self._delta, run, outgoing)
                await self._exchange.publish_snapshot(outgoing.draft, delta)

        if self._run is None:
            raise RuntimeError("Learner.communicate() before join()")
        run = self._run
        self._main = asyncio.current_task()
        try:
            async with asyncio.TaskGroup() as group:
                receiving = group.create_task(receive())
                await publish()
                receiving.cancel()
            await self._exchange.release(self._learner_id)
        finally:
            # A cancelled publish leaves a worker thread blocked in outbox.get(), and asyncio.run
            # waits for the executor's threads on shutdown; the sentinel releases it.
            self._outbox.put(None)

    async def poll(self) -> None:
        """One receive iteration: queue the newest version of each fragment committed meanwhile,
        then list the leases and who has published the round this learner publishes next.

        The batch is queued only once every payload in it has been fetched. A payload already
        swept is skipped; the next poll finds that fragment's newer version.
        """
        records = await self._exchange.rounds_after(self._cursor)
        if records:
            newest = tuple(_newest_per_fragment(records).values())
            fetched = await _fetch(self._exchange, [record.theta for record in newest])
            updates = tuple(
                _Update(record.round, record.fragment, record.version, tensors)
                for record, tensors in zip(newest, fetched, strict=True)
                if tensors is not None
            )
            if updates:
                self._inbox.put(updates)
            self._cursor = records[-1].round
            self._estimates = records[-1].estimates
        now = self._clock.now()
        self._liveness = self._liveness.observed(await self._exchange.leases(), now)
        self._live = self._liveness.live(now)
        state = self._state
        round = max(state.last_published + 1, state.rounds_seen + 1)
        others = await self._exchange.publishers(round)
        self._publishers = Publishers(round, others - {self._learner_id})

    def _publish(self, run: "_Run", state: LearnerState, round: int) -> LearnerState:
        """Snapshot ``round``'s fragment and its anchor for the publish loop; the bytes are
        copied here."""
        fragment = run.schedule.fragment_of(round)
        draft = SnapshotDraft(
            learner=self._learner_id,
            round=round,
            local_step=state.local_step,
            steps=state.steps[fragment],
            tokens=state.tokens[fragment],
            base_version=state.applied_version[fragment],
            rounds_seen=state.rounds_seen,
            step_seconds=state.step_seconds,
            schema_digest=run.digest,
        )
        self._outbox.put(
            _Outgoing(draft, self._anchors[fragment], self._adapter.snapshot(run.plan[fragment]))
        )
        return replace(state, last_published=round, step_at_last_publish=state.local_step)

    def _delta(self, run: "_Run", outgoing: "_Outgoing") -> Mapping[str, TensorBytes]:
        """``anchor - θ`` of every tensor in the snapshot, in the algorithm's delta dtype."""
        load = self._backend.load
        return {
            name: (load(outgoing.anchor[name]) - load(tensor)).to_bytes(run.delta_dtype[name])
            for name, tensor in outgoing.snapshot.items()
        }

    def _install(self, run: "_Run", state: LearnerState, update: "_Update") -> LearnerState:
        validate(run.schema, update.tensors, run.plan[update.fragment])
        self._adapter.apply(update.tensors)
        self._anchors[update.fragment] = update.tensors
        return applied(state, update.round, update.fragment, update.version)

    def _drained(self, run: "_Run", state: LearnerState) -> LearnerState:
        """Install every fragment version the receive loop has queued, without waiting."""

        def queued() -> Iterator[_Update]:
            while True:
                try:
                    batch = self._inbox.get_nowait()
                except queue.Empty:
                    return
                yield from batch

        for update in queued():
            state = self._install(run, state, update)
        return state

    def _settled(
        self,
        run: "_Run",
        state: LearnerState,
        done: Callable[[LearnerState], bool],
        *,
        patience: float | None,
    ) -> LearnerState:
        """Install fragment versions as they arrive until ``done(state)`` holds.

        Blocks the calling thread. With ``patience``, returns once nothing has happened for that
        long: no version installed and no new snapshot in the round this learner publishes
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
                state = self._install(run, state, update)
            now = self._clock.now()
            if (state.rounds_seen, self._publishers) != progress:
                progress, since = (state.rounds_seen, self._publishers), now
            elif patience is not None and now - since >= patience:
                break
        return state

    def _flushed(self, run: "_Run", state: LearnerState) -> LearnerState:
        """Publish every fragment trained since its last overwrite and wait for the commits.

        Rounds go out as the schedule allows, ``in_flight`` at a time, each next one once the
        commits make it publishable; done when nothing is publishable and nothing is
        outstanding, or when the run stalls for the lease timeout plus the takeover timeout.
        """

        def settled(state: LearnerState) -> bool:
            return (
                publishable(state, run.schedule) is not None
                or state.rounds_seen >= state.last_published
            )

        patience = self._lease_timeout + self._estimates.timeout(self._poll_interval)
        state = self._drained(run, state)
        while True:
            while (round := publishable(state, run.schedule)) is not None:
                state = self._publish(run, state, round)
            if state.rounds_seen >= state.last_published:
                return state
            state = self._settled(run, state, settled, patience=patience)
            if not settled(state):
                return state


@dataclass(frozen=True)
class Installed:
    """What ``install`` loaded: the round records read, in round order, and the tensors the
    adapter now holds, one mapping per fragment."""

    records: Sequence[RoundRecord]
    fragments: tuple[Mapping[str, TensorBytes], ...]


async def install(adapter: Adapter, exchange: Exchange, manifest: Manifest) -> Installed:
    """Load the newest committed version of every fragment of ``manifest``'s run into ``adapter``.

    A fragment no round has committed yet is loaded from the manifest's initial parameters.
    ``SchemaMismatch`` names the parameters on which the model and the run differ.
    """
    schema = adapter.schema()
    if manifest.schema != schema:
        differing = set(manifest.schema.parameters) ^ set(schema.parameters)
        raise SchemaMismatch(
            sorted({spec.name for spec in differing}), "model does not match the run's schema"
        )
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
    return Installed(records, tuple(loaded))


@dataclass(frozen=True)
class _Run:
    """The run as joined: fixed for the learner's lifetime."""

    schema: Schema
    plan: tuple[tuple[str, ...], ...]
    schedule: RoundSchedule
    quorum: int
    digest: str
    delta_dtype: Mapping[str, DType]


@dataclass(frozen=True)
class _Update:
    """A fragment version fetched by the receive loop, waiting for ``step()`` to apply it."""

    round: int
    fragment: int
    version: int
    tensors: Mapping[str, TensorBytes]


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
_CREATION_FLOOR_POLLS = 30
# A live creator is assumed to upload at least this fast; slower, and a waiter takes over.
_MIN_UPLOAD_RATE = 2**20
