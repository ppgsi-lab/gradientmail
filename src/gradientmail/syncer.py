"""The syncer: collects a round's snapshots, closes the fragment and commits the round.

Every node runs one, hosted by ``Session`` next to its learner; the node that closes a round
is elected through the mailbox. Round ``r`` targets fragment ``r mod P`` at global step
``(r div P)·H + ⌊((r mod P) + 1)·H / P⌋``, so every fragment is synchronized once per ``H``
steps. The round record is the commit point: a node rebuilds its state from the newest record
of every fragment, and two nodes that close the same round are resolved at the record write.
Timings are durations on this process's monotonic clock plus the step durations learners
report, never wall clocks compared across machines.

Election
--------
Round ``r`` is closed by the node that created ``claims/{r}/001``, a create-once object naming
the claimant. The node that commits ``r`` claims ``r + 1`` in the same breath, before anyone
else can have seen the record, so the closer stays put while it lives and a fresh run's first
claimant keeps the role. A node that finds a claim by someone else follows: it polls for the
round record and continues from it. It creates the next claim and closes the round itself as
soon as the claimant's lease is gone or stale (``lease`` module: released when the claimant
closed, unrenewed since it crashed), or, for a claimant that is live but silent, once no record
has appeared within a timeout measured on its own clock (three times the round duration the
last record's estimates predict, never less than thirty poll intervals). The leases a follower
observes carry into its ``close``, so a lease first met while following is not trusted afresh
at takeover. A node that restarts finds its own claim and resumes. Claims only avoid redundant
work: whoever commits
first wins, and the other reads the committed record back (``Superseded``) and continues from
it. A follower keeps no tensors in memory; once elected it reloads the fragments from the
records (``recover``), so leadership changes cost one reload and steady state costs followers
one listing per round.

Quorum
------
A round closes once ``quorum_size`` snapshots for it are accepted, or once every live learner
has reported and at least one snapshot is accepted. A snapshot is accepted iff its ``steps``
are positive, its schema digest is the run's, its ``base_version`` is the fragment's current
version, its payload is present and hashes to the record and its tensors match the fragment's
delta schema (the parameters' shapes in the algorithm's delta dtype); otherwise it stays in
the round record as rejected (``zero_steps``, ``schema``,
``stale``, ``missing``, ``corrupt``) with weight 0, the paper's treatment of learners outside
the round. A live learner is never ``stale``: it advances ``rounds_seen`` only after installing
every fragment it fetched, and ``in_flight ≤ P``; a ``stale`` snapshot comes from a learner
that skipped a swept payload and published before its next poll installed the current version.
Liveness is the lease every learner renews while it runs (``lease`` module): a learner whose
lease has not changed for ``lease_timeout`` seconds on this node's clock, or whose lease is
gone, is not waited for. Learners therefore join and leave at will, and a quorum above the
number of live learners closes with the live ones instead of waiting forever; a crashed learner
holds up the rounds it would have taken part in for one timeout, a learner that closes releases
its lease at once. Snapshots and leases are polled every ``poll_interval``, one LIST each, and
payloads are fetched as their snapshots appear, overlapping the wait.

Grace
-----
After quorum the closer keeps collecting for
``min(0.5 · max(τ·ξ_round - (ξ_quorum + ξ_sync), 0), ξ_round)`` seconds, ``ξ_round`` being the
round's gap of steps times the step duration learners report: half the slack the ``in_flight``
rounds leave before learners would stall, capped at one round of training. The three
estimates are EMAs (decay 0.1) of that step duration, taken from the snapshots, of the time
from round start to quorum and of the time from closing to a durable commit. They travel
in every round record, so the node that closes the next round continues them; the sync
estimate lags one round when the closer changes, since a commit's duration is known only to
the node that made it. The window is zero while any estimate is missing (the first round of a
run) and ends early once every live learner has reported. With ``in_flight`` 1 the slack is
negative, since reaching quorum alone takes about a round, and the window stays zero: rounds
close at quorum and a learner late for a round is left out of it, though its next snapshot
carries the steps anyway. ``in_flight`` 2 leaves about half a round. The realized
``quorum_seconds`` and ``grace_seconds`` are in every round record. A snapshot that arrives
after the close is not counted and is swept with the round.

Recovery
--------
``recover`` lists ``rounds/`` and takes the newest record per fragment: its theta payload is
the anchor and its outer payload holds the fp32 momentum and, for F16 and BF16 schemas, the
fp32 master; a fragment never committed starts from the manifest's Θ⁰ with zero momentum. The
next round is the one after the last record and the timing estimates are the last record's.
Nothing else is persisted, so a crash
before the record write recomputes the round from the snapshots still in the store and a crash
after it continues with the next round; the commit sequence is the same either way, except
that a snapshot which landed in between may now be included. A round that fails for any other
reason is reported (``RoundFailed``) and retried the same way, from the store, one poll
later. ``run`` raises ``RunConfigMismatch`` when the manifest was created with another
algorithm configuration. A ``MailboxUnavailable`` is retried indefinitely with backoff. No
``asyncio.timeout`` wraps a store call, because a cancelled ``to_thread`` would leak its
thread; botocore's own timeouts bound them.

Garbage collection
------------------
After committing round ``r`` the closer deletes ``snapshots/{r}/``, ``blobs/snapshots/{r}/``
and ``claims/{r}/``, sweeps round ``r - 1`` again (for snapshots that landed after its close,
for claims created after its sweep, and for a predecessor that crashed between commit and
sweep) and deletes ``blobs/rounds/{r - 2P}/``, which also removes the payloads a superseded
closer left behind. Two cycles of theta and outer payloads are kept so that the newest version
of every fragment, at most ``P`` rounds old, always survives and a learner has a cycle to fetch
a payload whose record it listed. Round records, the manifest and the init payloads are never
deleted; ``rounds/`` is the tape of the run.
"""

import asyncio
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean

from gradientmail.adapters import Backend, default_backend
from gradientmail.algorithms import (
    Aggregator,
    Algorithm,
    Contribution,
    Estimates,
    FragmentState,
    RoundSchedule,
)
from gradientmail.algorithms.diloco import grace_window
from gradientmail.clock import Clock, MonotonicClock
from gradientmail.compression import Compression
from gradientmail.events import (
    Callback,
    FollowingRound,
    Recovered,
    RecoveryFailed,
    RoundClaimed,
    RoundClosed,
    RoundCommitted,
    RoundFailed,
    RoundSuperseded,
    TookOverRound,
    log_events,
)
from gradientmail.lease import Liveness
from gradientmail.mailbox import Mailbox
from gradientmail.protocol import (
    Exchange,
    Manifest,
    PayloadCorrupt,
    PayloadRef,
    ProtocolError,
    Rejection,
    RoundRecord,
    RoundRecordDraft,
    RunConfigMismatch,
    Snapshot,
    Superseded,
)
from gradientmail.serialization import Schema, SchemaMismatch, TensorBytes, validate

__all__ = ["Following", "Progress", "Syncer", "SyncerState"]


@dataclass(frozen=True)
class SyncerState:
    """A closer's knowledge between rounds.

    ``round`` is the next round to run, ``fragments`` the last committed state of every
    fragment and ``estimates`` the timing EMAs behind the grace window.
    """

    round: int
    fragments: tuple[FragmentState, ...]
    estimates: Estimates


@dataclass(frozen=True)
class Following:
    """A follower's knowledge: the next round, and how long to wait for its record before
    claiming it, unless the claimant's lease ends first."""

    round: int
    timeout: float


type Progress = SyncerState | Following


class Syncer:
    """Drives a run's rounds against a mailbox; ``run`` is the whole loop, ``run_round`` one step.

    Everything the manifest fixes (schema, plan, schedule, one aggregator per fragment) is read
    once on first use; the manifest is a create-once object. The outer step runs in ``backend``,
    by default the installed one (torch, then MLX); constructing a syncer with neither installed
    raises ``MissingExtra``. Round records are byte-identical across restarts on one backend;
    another backend may differ in the last bit. ``compression`` is the codec this node's
    commits are stored with. ``on_event`` receives every event the syncer reports (``events``
    module), on the thread running ``run``.
    """

    def __init__(
        self,
        *,
        algorithm: Algorithm,
        mailbox: Mailbox,
        learner_id: str,
        backend: Backend | None = None,
        clock: Clock | None = None,
        poll_interval: float = 0.5,
        lease_timeout: float = 120.0,
        compression: Compression = "off",
        on_event: Callback = log_events,
    ) -> None:
        self._algorithm = algorithm
        self._learner_id = learner_id
        self._backend = backend if backend is not None else default_backend()
        self._clock = clock if clock is not None else MonotonicClock()
        self._exchange = Exchange(mailbox, clock=self._clock, compression=compression)
        self._poll_interval = poll_interval
        self._on_event = on_event
        self._liveness = Liveness(lease_timeout)
        self._run: _Run | None = None
        self._main: asyncio.Task[None] | None = None
        self._stopped = False

    async def run(self, *, until_round: int | None = None) -> None:
        """Wait for the manifest, check the configuration, recover and run round after round.

        Returns once round ``until_round`` is committed; with ``None`` it runs until cancelled
        or ``stop()``. Raises ``RunConfigMismatch`` when the run was created with another
        algorithm configuration. A round that fails is reported and retried as after a restart:
        the claim this node holds, if any, is found again and the round recomputed.
        """
        self._main = asyncio.current_task()
        if self._stopped:
            return
        while (manifest := await self._exchange.manifest()) is None:
            await self._clock.sleep(self._poll_interval)
        expected = self._algorithm.to_json()
        if dict(manifest.algorithm) != dict(expected):
            raise RunConfigMismatch(expected, manifest.algorithm)
        progress: Progress | None = None
        while True:
            try:
                if progress is None:
                    progress = await self.recover()
                if until_round is not None and progress.round > until_round:
                    return
                progress = await self.run_round(progress)
            except Exception as exc:
                self._on_event(
                    RecoveryFailed(self._learner_id, exc, self._poll_interval)
                    if progress is None
                    else RoundFailed(self._learner_id, progress.round, exc, self._poll_interval)
                )
                await self._clock.sleep(self._poll_interval)
                if progress is not None:
                    progress = Following(progress.round, self._timeout_of(progress))

    def stop(self) -> None:
        """Cancel ``run`` from any thread; a ``run`` that has not started yet returns at once."""
        self._stopped = True
        if self._main is not None and not self._main.done():
            loop = self._main.get_loop()
            if not loop.is_closed():
                loop.call_soon_threadsafe(self._main.cancel)

    async def recover(self) -> SyncerState:
        """Rebuild a closer's state from the store.

        Every fragment comes from its newest round record (theta and outer payloads), or from the
        manifest's Θ⁰ with zero momentum when it was never committed. The next round is the one
        after the last record; the timing estimates are that record's.
        """

        async def required(ref: PayloadRef) -> Mapping[str, TensorBytes]:
            tensors = await self._exchange.fetch(ref)
            if tensors is None:
                raise ProtocolError(f"{ref.key} is referenced by a live record but is gone")
            return tensors

        async def fragment_state(fragment: int) -> FragmentState:
            record = newest.get(fragment)
            if record is None:
                momentum = {
                    f"momentum.{name}": TensorBytes(
                        "F32", shapes[name], bytes(4 * math.prod(shapes[name]))
                    )
                    for name in run.manifest.plan[fragment]
                }
                return FragmentState(0, await required(run.manifest.init[fragment]), momentum)
            return FragmentState(
                record.version, await required(record.theta), await required(record.outer)
            )

        run = await self._resolved()
        records = await self._exchange.rounds_after(-1)
        # Records come in round order, so the last one per fragment wins.
        newest = {record.fragment: record for record in records}
        shapes = {spec.name: spec.shape for spec in run.manifest.schema.parameters}
        count = len(run.manifest.plan)
        async with asyncio.TaskGroup() as group:
            loads = [group.create_task(fragment_state(p)) for p in range(count)]
        last = records[-1] if records else None
        state = SyncerState(
            round=0 if last is None else last.round + 1,
            fragments=tuple(load.result() for load in loads),
            estimates=Estimates() if last is None else last.estimates,
        )
        self._on_event(Recovered(self._learner_id, state.round))
        return state

    async def run_round(self, progress: Progress) -> Progress:
        """Advance one round: close it when elected, otherwise follow it to its record."""
        committed = await self._exchange.rounds_after(progress.round - 1)
        if committed:
            # Closed already, its claims swept with it; a node arriving now must not reclaim it.
            return Following(committed[0].round + 1, self._timeout(committed[0]))
        if not await self.elected(progress.round):
            return await self.follow(progress)
        state = progress if isinstance(progress, SyncerState) else await self.recover()
        if state.round != progress.round:
            # Committed meanwhile by an earlier claimant that was slow rather than dead.
            return state
        return await self.close(state)

    async def elected(self, round: int) -> bool:
        """Whether this node closes ``round``: it holds a claim, or it makes the first one."""
        claimants = await self._exchange.claims(round)
        if claimants:
            return self._learner_id in claimants
        claimed = await self._exchange.claim(round, 1, self._learner_id)
        if claimed:
            self._on_event(RoundClaimed(self._learner_id, round))
        return claimed

    async def follow(self, progress: Progress) -> Progress:
        """Wait for the round's record; take the next claim and close the round once the
        claimant's lease is gone or stale, or after the timeout."""
        round = progress.round
        timeout = self._timeout_of(progress)
        claimants = await self._exchange.claims(round)
        self._on_event(FollowingRound(self._learner_id, round, tuple(claimants), timeout))
        seen_at = self._clock.now()
        while True:
            records = await self._exchange.rounds_after(round - 1)
            if records:
                return Following(records[0].round + 1, self._timeout(records[0]))
            leases = await self._exchange.leases()
            now = self._clock.now()
            self._liveness = self._liveness.observed(leases, now)
            gone = bool(claimants) and claimants[-1] not in self._liveness.live(now)
            if gone or now - seen_at >= timeout:
                if await self._exchange.claim(round, len(claimants) + 1, self._learner_id):
                    self._on_event(
                        TookOverRound(
                            self._learner_id,
                            round,
                            tuple(claimants),
                            "lease_gone" if gone else "no_record",
                            timeout,
                        )
                    )
                    state = await self.recover()
                    return state if state.round != round else await self.close(state)
                claimants = await self._exchange.claims(round)
                seen_at = self._clock.now()
            await self._clock.sleep(self._poll_interval)

    async def close(self, state: SyncerState) -> Progress:
        """Close round ``state.round`` through its commit and return the state for the next one.

        Collects snapshots until quorum, or until every live learner has reported, keeps
        collecting through the grace window unless every live learner has reported, closes the
        fragment, commits, claims the next round and sweeps. When another node committed the
        round first, its record is adopted and this node follows from there.
        """

        async def judge(snapshot: Snapshot) -> Contribution | Rejection:
            if snapshot.steps == 0:
                return "zero_steps"
            if snapshot.schema_digest != run.digest:
                return "schema"
            if snapshot.base_version != current.version:
                return "stale"
            try:
                tensors = await self._exchange.fetch(snapshot.payload)
            except PayloadCorrupt:
                return "corrupt"
            if tensors is None:
                return "missing"
            try:
                validate(run.deltas, tensors, names)
            except SchemaMismatch:
                return "schema"
            return Contribution(
                snapshot.learner,
                snapshot.local_step,
                snapshot.steps,
                snapshot.tokens,
                snapshot.step_seconds,
                snapshot.final,
                tensors,
            )

        async def poll() -> None:
            leases = await self._exchange.leases()
            self._liveness = self._liveness.observed(leases, self._clock.now())
            snapshots = await self._exchange.snapshots(round)
            new = tuple(s for s in snapshots if s.learner not in verdicts)
            async with asyncio.TaskGroup() as group:
                judged = [(s.learner, group.create_task(judge(s))) for s in new]
            verdicts.update((learner, task.result()) for learner, task in judged)

        def accepted() -> tuple[Contribution, ...]:
            contributions = (v for v in verdicts.values() if isinstance(v, Contribution))
            return tuple(sorted(contributions, key=lambda c: c.learner))

        def reported() -> bool:
            """Every live learner has a snapshot in the round, accepted or rejected."""
            return self._liveness.live(self._clock.now()) <= verdicts.keys()

        def closable() -> bool:
            return len(accepted()) >= self._algorithm.quorum() or (
                len(accepted()) >= 1 and reported()
            )

        run = await self._resolved()
        round = state.round
        fragment = run.schedule.fragment_of(round)
        current = state.fragments[fragment]
        names = run.manifest.plan[fragment]
        verdicts: dict[str, Contribution | Rejection] = {}

        started = self._clock.now()
        await poll()
        while not closable():
            await self._clock.sleep(self._poll_interval)
            await poll()
        quorum_at = self._clock.now()

        deadline = quorum_at + grace_window(
            state.estimates, run.schedule.in_flight, run.schedule.gap(round)
        )
        while (now := self._clock.now()) < deadline and not reported():
            await self._clock.sleep(min(self._poll_interval, deadline - now))
            await poll()
        grace_ended = self._clock.now()

        contributions = accepted()
        updated, participants = run.aggregators[fragment].close_round(current, contributions)
        estimates = state.estimates.updated(
            step=fmean(c.step_seconds for c in contributions), quorum=quorum_at - started
        )
        draft = RoundRecordDraft(
            round=round,
            step=run.schedule.step_of(round),
            fragment=fragment,
            version=updated.version,
            participants=participants,
            rejected=tuple(
                sorted((learner, why) for learner, why in verdicts.items() if isinstance(why, str))
            ),
            quorum_seconds=quorum_at - started,
            grace_seconds=grace_ended - quorum_at,
            estimates=estimates,
        )
        self._on_event(RoundClosed(self._learner_id, draft))
        committing = time.perf_counter()
        outcome = await self._exchange.commit(draft, updated.theta, updated.outer)
        if isinstance(outcome, Superseded):
            self._on_event(RoundSuperseded(self._learner_id, outcome.record))
            return Following(outcome.record.round + 1, self._timeout(outcome.record))
        sync_seconds = self._clock.now() - grace_ended
        self._on_event(
            RoundCommitted(self._learner_id, outcome.record, time.perf_counter() - committing)
        )

        await self._exchange.claim(round + 1, 1, self._learner_id)
        count = len(state.fragments)
        for swept in (round, round - 1):
            if swept >= 0:
                await self._exchange.sweep_snapshots(swept)
        await self._exchange.sweep_blobs(round - 2 * count)
        return SyncerState(
            round=round + 1,
            fragments=(*state.fragments[:fragment], updated, *state.fragments[fragment + 1 :]),
            estimates=estimates.updated(sync=sync_seconds),
        )

    async def _resolved(self) -> "_Run":
        if self._run is None:
            manifest = await self._exchange.manifest()
            if manifest is None:
                raise ProtocolError("manifest.json does not exist: no learner has created the run")
            schema, plan = manifest.schema, manifest.plan
            self._run = _Run(
                manifest=manifest,
                digest=schema.digest(),
                deltas=self._algorithm.delta_schema(schema),
                schedule=self._algorithm.schedule(len(plan)),
                aggregators=tuple(
                    self._algorithm.aggregator(schema, names, self._backend) for names in plan
                ),
            )
        return self._run

    def _timeout(self, record: RoundRecord) -> float:
        """How long to wait for the round after ``record`` before taking it over."""
        return record.estimates.timeout(self._poll_interval)

    def _timeout_of(self, progress: Progress) -> float:
        return (
            progress.timeout
            if isinstance(progress, Following)
            else progress.estimates.timeout(self._poll_interval)
        )


@dataclass(frozen=True)
class _Run:
    """What the manifest fixes for the life of the run; ``deltas`` is what snapshots must match."""

    manifest: Manifest
    digest: str
    deltas: Schema
    schedule: RoundSchedule
    aggregators: tuple[Aggregator, ...]
