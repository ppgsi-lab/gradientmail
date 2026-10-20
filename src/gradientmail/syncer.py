"""The syncer: collects a round's snapshots, closes the fragment and commits the round.

Every node that closes rounds runs one, hosted by ``Session`` next to its learner or alone; the
node that closes a round is elected through the mailbox. Round ``r`` targets fragment
``r mod P`` at global step ``(r div P)·H + ⌊((r mod P) + 1)·H / P⌋``, so every fragment is
synchronized once per ``H`` steps. The round record is the commit point: a node rebuilds its
state from the newest record of every fragment, and two nodes that close the same round are
resolved at the record write. Timings are durations on this process's monotonic clock plus the
step durations learners report, never wall clocks compared across machines.

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
last record's estimates predict, never less than thirty poll intervals). A claimant is live
while it holds a live learner lease or a live syncer lease. The leases a follower observes
carry into its ``close``, so a lease first met while following is not trusted afresh at
takeover. A node that restarts finds its own claim and resumes. Claims only avoid redundant
work: whoever commits
first wins, and the other reads the committed record back (``Superseded``) and continues from
it. A follower keeps no tensors in memory; once elected it reloads the fragments from the
records (``recover``), so leadership changes cost one reload and steady state costs followers
one listing per round.

A node can run as the run's exclusive syncer (``exclusive``, ``Session``'s ``"syncer"`` role):
it trains nothing and holds a lease of its own, ``syncers/{kind}/{arrival}-{node}/``
(``protocol`` module), written when ``run`` starts, renewed every ``lease_every`` seconds on a
task of its own, so that a long commit cannot let it lapse, and released when ``run`` ends. The
role belongs to the live syncer lease first in byte order, liveness judged as for learners: of
two dedicated syncers the one that arrived first owns it, and the other once the first one's
lease ends. Every node lists the syncer leases at the start of a round and on every poll while
following. While a live owner other than itself exists, a node only follows: it makes no first
claim on a round and, after committing a round it took over, does not claim the next one, which
the owner, following, claims once it sees the record. Takeover is unchanged, so the other nodes
close the rounds while the owner's lease is gone or stale, and the role returns to the owner
at the first round nobody else has claimed. With no live owner, every node behaves as above.
An exclusive syncer, owner or not, returns from ``run`` once the run is finished: no learner
holds a live lease and the data is exhausted, as the ``done/`` markers tell. ``Session.run``
writes one for a node whose data is exhausted, whatever the data, under the node's rank when it
holds one. When any rank was ever claimed (``ranks/``), the data is exhausted once every such
rank is marked, so a rank whose node crashed keeps the syncer up for the node that takes it
over; otherwise once at least one node is marked. A learner that joins after that finds no
syncer.

A node can also stand for the role (``exclusive="1-elected"``, ``Session``'s ``"either"``
role). On every poll of the lease task that finds no live syncer lease while another node
holds a live learner lease, it writes a ``1-elected`` lease, waits a second of the store's
clock, so that the lease of any node that arrived earlier has landed, and lists them again
(``elect``): it wins when its lease is now the first live one, and otherwise releases it. Every
candidate finds the same first lease, so of those that stood at once exactly one wins, and one
that arrives later finds an owner and does not stand; a node alone does not stand, since
elected it would leave nobody training. The winner holds its lease as a dedicated syncer does
and closes the rounds; it releases its lease once another node owns the role, as a dedicated
syncer does on arriving, its lease sorting first, or once no learner holds a live lease while
units of its node's queue are unclaimed (``units``), so that its node trains them instead of the
run finishing without them. Until elected, and after giving the role up, a candidate behaves as
a node without a lease, so when the elected node is gone the others take over its rounds and,
on their next poll, elect another.

Quorum
------
A round closes once ``quorum_size`` snapshots for it are accepted, or once every live learner
has reported and at least one snapshot is accepted. A snapshot is accepted iff its ``steps``
are positive, its schema digest is the run's, its ``base_version`` is the fragment's current
version, its payload is present and hashes to the record and its tensors match the fragment's
delta schema (the parameters in the algorithm's delta dtype, with their scales when it is
quantized); otherwise it stays in
the round record as rejected (``zero_steps``, ``schema``,
``stale``, ``missing``, ``corrupt``) with weight 0, the paper's treatment of learners outside
the round. A live learner is never ``stale``: it advances ``rounds_seen`` only after aligning
with every fragment it fetched, and ``in_flight ≤ P``; a ``stale`` snapshot comes from a learner
that skipped a swept payload and published before aligning with the version its next poll listed.
Liveness is the lease every learner renews while it runs (``lease`` module; a syncer's lease is
not one): a learner whose lease has not changed for ``lease_timeout`` seconds on this node's
clock, or whose lease is gone, is not waited for. Learners therefore join and leave at will,
and a quorum above the number of live learners closes with the live ones instead of waiting
forever; a crashed learner holds up the rounds it would have taken part in for one timeout, a
learner that closes releases its lease at once. Snapshots and leases are polled every
``poll_interval``, one LIST each, and payloads are fetched as their snapshots appear,
overlapping the wait.

Grace
-----
After quorum the closer keeps collecting for the algorithm's ``grace_seconds`` when it fixes
one, otherwise for
``min(0.5 · max(ahead·ξ_step - (ξ_quorum + ξ_sync), 0), ξ_round)`` seconds, ``ξ_round`` being the
round's gap of steps times the step duration learners report and ``ahead`` the steps a learner
trains before it would stall, ``in_flight`` rounds of them or the algorithm's ``overlap_steps``
when that is less: half the slack those leave, capped at one round of training. The three
estimates are EMAs (decay 0.1) of that step duration, taken from the snapshots, of the time
from round start to quorum and of the time from closing to a durable commit. They travel
in every round record, so the node that closes the next round continues them; the sync
estimate lags one round when the closer changes, since a commit's duration is known only to
the node that made it. The window is zero while any estimate is missing (the first round of a
run) and ends early once every live learner has reported. With ``in_flight`` 1 the slack is
negative, since reaching quorum alone takes about a round, and the window stays zero: rounds
close at quorum and a learner late for a round is left out of it; whether its next snapshot
carries the steps is the algorithm's ``alignment``. ``in_flight`` 2 leaves about half a round.
A fixed window ends early the same way and applies from the first round. The realized
``quorum_seconds`` and ``grace_seconds`` are in every round record. A snapshot that arrives
after the close is not counted and is swept with the round.

Recovery
--------
``recover`` lists ``rounds/`` and takes the newest record per fragment: its theta payload is
the anchor and its outer payload holds the momentum and, when the published tensor does not
hold the master (commits that carry an update, a master wider than the parameter), the master
or its residual, in the algorithm's formats; a fragment never committed starts from the
manifest's Θ⁰ with no outer state, which the outer step takes as zero momentum. The
next round is the one after the last record and the timing estimates are the last record's.
Nothing else is persisted, so a crash
before the record write recomputes the round from the snapshots still in the store and a crash
after it continues with the next round; the commit sequence is the same either way, except
that a snapshot which landed in between may now be included. At the start of each round a
closer reads back the record its state continues from (``Exchange.check_commit``, one GET): a
store that accepted two writes of one key may have kept another closer's, and then the state is
rebuilt from the store before the round goes on (``RoundSuperseded``). A round that fails for
any other reason is reported (``RoundFailed``) and retried the same way, from the store, one
poll later. ``run`` raises ``RunConfigMismatch`` when the manifest was created with another
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
import time
from collections.abc import Callable, Mapping
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
    layout,
)
from gradientmail.algorithms.diloco import grace_window
from gradientmail.clock import Clock, MonotonicClock
from gradientmail.compression import Compression
from gradientmail.events import (
    Callback,
    FollowingRound,
    LeaseReleased,
    PollFailed,
    Recovered,
    RecoveryFailed,
    ReleaseFailed,
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
    LocalCommits,
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
    SyncerKind,
    rank_id,
    syncer_lease_prefix,
    syncer_node,
)
from gradientmail.serialization import PayloadDType, Schema, SchemaMismatch, TensorBytes, validate

__all__ = ["Following", "Progress", "Syncer", "SyncerState"]


@dataclass(frozen=True)
class SyncerState:
    """A closer's knowledge between rounds.

    ``round`` is the next round to run, ``fragments`` the last committed state of every
    fragment, ``estimates`` the timing EMAs behind the grace window and ``committed`` the record
    of round ``round - 1`` the state continues from, the one this node committed or the newest
    it recovered; None before the first round.
    """

    round: int
    fragments: tuple[FragmentState, ...]
    estimates: Estimates
    committed: RoundRecord | None


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
    commits are stored with; ``local``, when given, receives every fragment this node commits,
    for the learner in the same process. ``exclusive``, when given, makes this node the run's
    exclusive syncer of that kind (module docstring): ``run`` holds its lease, renewed every
    ``lease_every`` seconds, and returns once the run is finished; of the ``"1-elected"`` kind,
    only while it ``holds_role``. ``units`` returns the ids of the units of the shared queue the
    node trains on, none while it has none, for an elected node to give the role up while some
    are unclaimed and no learner is left to claim them. Liveness, of learners and syncers alike,
    is judged with ``lease_timeout``. ``on_event`` receives every event the syncer reports
    (``events`` module), on the thread running ``run``.
    """

    def __init__(
        self,
        *,
        algorithm: Algorithm,
        mailbox: Mailbox,
        node_id: str,
        backend: Backend | None = None,
        clock: Clock | None = None,
        poll_interval: float = 0.5,
        exclusive: SyncerKind | None = None,
        lease_every: float = 60.0,
        lease_timeout: float = 120.0,
        compression: Compression = "off",
        local: LocalCommits | None = None,
        units: Callable[[], frozenset[str]] = frozenset,
        on_event: Callback = log_events,
    ) -> None:
        if exclusive is not None and lease_timeout <= lease_every:
            raise ValueError(
                f"lease_timeout must exceed lease_every, got {lease_timeout} and {lease_every}"
            )
        self._algorithm = algorithm
        self._node_id = node_id
        self._backend = backend if backend is not None else default_backend()
        self._clock = clock if clock is not None else MonotonicClock()
        self._store_clock = mailbox.clock()
        self._exchange = Exchange(mailbox, clock=self._clock, compression=compression, local=local)
        self._poll_interval = poll_interval
        self._exclusive = exclusive
        self._lease_every = lease_every
        self._units = units
        self._on_event = on_event
        self._liveness = Liveness(lease_timeout)
        self._syncers = Liveness(lease_timeout)
        self._lease: str | None = None
        self._owner: str | None = None
        self._run: _Run | None = None
        self._main: asyncio.Task[None] | None = None
        self._stopped = False

    @property
    def holds_role(self) -> bool:
        """Whether this node holds the role of the run's elected syncer: from the election it
        won until it gives the role up to another owner. Once ``run`` has returned holding it,
        it stays set, the lease released."""
        return self._exclusive == "1-elected" and self._lease is not None

    async def run(self, *, until_round: int | None = None) -> None:
        """Wait for the manifest, check the configuration, recover and run round after round.

        Returns once round ``until_round`` is committed; with ``None`` it runs until cancelled
        or ``stop()``. Raises ``RunConfigMismatch`` when the run was created with another
        algorithm configuration. A round that fails is reported and retried as after a restart:
        the claim this node holds, if any, is found again and the round recomputed.

        An exclusive syncer holds a lease, renewed on a task of its own: a dedicated one writes
        it once the configuration checks out, an elected one wins it (``elect``) on that task or
        before ``run``, and gives it up to another owner, or when units of its node's queue are
        left for nobody to train. A lease held when ``run`` ends is released, however it ends.
        While it holds one, ``run`` also returns once the run is finished: no learner holds a
        live lease and the data is exhausted (module docstring).
        """
        self._main = asyncio.current_task()
        try:
            if self._stopped:
                return
            while (manifest := await self._exchange.manifest()) is None:
                await self._clock.sleep(self._poll_interval)
            expected = self._algorithm.to_json()
            if dict(manifest.algorithm) != dict(expected):
                raise RunConfigMismatch(expected, manifest.algorithm)
            if self._exclusive is None:
                await self._rounds(until_round)
                return
            if self._exclusive == "0-dedicated":
                arrival = int(self._store_clock.now() * 1000)
                lease = syncer_lease_prefix(self._exclusive, arrival, self._node_id)
                self._lease = lease
                await self._exchange.beat_syncer(lease)
            async with asyncio.TaskGroup() as group:
                tasks = (
                    group.create_task(self._hold()),
                    group.create_task(self._rounds(until_round)),
                )
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in tasks:
                    task.cancel()
        finally:
            # The lease stays named once released, so that ``holds_role`` tells a run that
            # finished holding the role from one that gave it up.
            if (lease := self._lease) is not None:
                try:
                    await self._exchange.release_syncer(lease)
                    self._on_event(LeaseReleased(self._node_id, "syncer"))
                except Exception as exc:
                    self._on_event(ReleaseFailed(self._node_id, "syncer", exc))

    async def elect(self) -> bool:
        """Stand for the role of the run's elected syncer unless a node owns it; True when this
        node won it, and holds it from then on.

        Lists the syncer leases: while one of them is live, its node owns the role and this one
        does not stand; nor does it while no other node holds a live learner lease, since
        elected it would leave nobody training. Otherwise it writes a ``1-elected`` lease, waits
        a second of the store's clock for the leases of nodes that arrived earlier to land and
        lists them again: it wins when its own is the first live one in byte order. A lease that
        did not win, or whose election failed, is released.
        """
        self._observed(await self._exchange.syncer_leases(), self._clock.now())
        if self._owner is not None or not await self._learners() - {self._node_id}:
            return False
        arrival = int(self._store_clock.now() * 1000)
        lease = syncer_lease_prefix("1-elected", arrival, self._node_id)
        try:
            await self._exchange.beat_syncer(lease)
            await self._store_clock.sleep(_ELECTION_WAIT)
            self._observed(await self._exchange.syncer_leases(), self._clock.now())
            if self._owner == lease:
                self._lease = lease
        finally:
            if self._lease != lease:
                await self._exchange.release_syncer(lease)
        return self._lease == lease

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
        manifest's Θ⁰ with no outer state when it was never committed. The next round is the one
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
                return FragmentState(0, await required(run.manifest.init[fragment]), {})
            return FragmentState(
                record.version, await required(record.theta), await required(record.outer)
            )

        run = await self._resolved()
        records = await self._exchange.rounds_after(-1)
        # Records come in round order, so the last one per fragment wins.
        newest = {record.fragment: record for record in records}
        count = len(run.manifest.plan)
        async with asyncio.TaskGroup() as group:
            loads = [group.create_task(fragment_state(p)) for p in range(count)]
        last = records[-1] if records else None
        state = SyncerState(
            round=0 if last is None else last.round + 1,
            fragments=tuple(load.result() for load in loads),
            estimates=Estimates() if last is None else last.estimates,
            committed=last,
        )
        self._on_event(Recovered(self._node_id, state.round))
        return state

    async def run_round(self, progress: Progress) -> Progress:
        """Advance one round: close it when elected, otherwise follow it to its record.

        From a ``SyncerState``, the record it continues from is read back first; when the store
        holds another closer's, the state is rebuilt from the store.
        """
        if isinstance(progress, SyncerState) and progress.committed is not None:
            check = await self._exchange.check_commit(progress.committed)
            if isinstance(check, Superseded):
                self._on_event(RoundSuperseded(self._node_id, check.record))
                progress = await self.recover()
        committed = await self._exchange.rounds_after(progress.round - 1)
        if committed:
            # Closed already, its claims swept with it; a node arriving now must not reclaim it.
            return Following(committed[0].round + 1, self._timeout(committed[0]))
        self._observed(await self._exchange.syncer_leases(), self._clock.now())
        if not await self.elected(progress.round):
            return await self.follow(progress)
        state = progress if isinstance(progress, SyncerState) else await self.recover()
        if state.round != progress.round:
            # Committed meanwhile by an earlier claimant that was slow rather than dead.
            return state
        return await self.close(state)

    async def elected(self, round: int) -> bool:
        """Whether this node closes ``round``: it holds a claim, or it makes the first one, which
        it does only while no other node owns the exclusive syncer role."""
        claimants = await self._exchange.claims(round)
        if claimants:
            return self._node_id in claimants
        if self._ceding():
            return False
        claimed = await self._exchange.claim(round, 1, self._node_id)
        if claimed:
            self._on_event(RoundClaimed(self._node_id, round))
        return claimed

    async def follow(self, progress: Progress) -> Progress:
        """Wait for the round's record; take the next claim and close the round once the
        claimant holds neither a live learner lease nor a live syncer lease, or after the
        timeout."""
        round = progress.round
        timeout = self._timeout_of(progress)
        claimants = await self._exchange.claims(round)
        self._on_event(FollowingRound(self._node_id, round, tuple(claimants), timeout))
        seen_at = self._clock.now()
        while True:
            records = await self._exchange.rounds_after(round - 1)
            if records:
                return Following(records[0].round + 1, self._timeout(records[0]))
            leases = await self._exchange.leases()
            syncers = await self._exchange.syncer_leases()
            now = self._clock.now()
            self._liveness = self._liveness.observed(leases, now)
            live = self._liveness.live(now) | {
                syncer_node(lease) for lease in self._observed(syncers, now)
            }
            gone = bool(claimants) and claimants[-1] not in live
            if gone or now - seen_at >= timeout:
                if await self._exchange.claim(round, len(claimants) + 1, self._node_id):
                    self._on_event(
                        TookOverRound(
                            self._node_id,
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
        fragment, commits, claims the next round unless another node owns the exclusive syncer
        role, and sweeps. When another node committed the round first, its record is adopted
        and this node follows from there.
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
                validate(deltas, tensors, [spec.name for spec in deltas.parameters])
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
        deltas = run.deltas[fragment]
        verdicts: dict[str, Contribution | Rejection] = {}

        started = self._clock.now()
        await poll()
        while not closable():
            await self._clock.sleep(self._poll_interval)
            await poll()
        quorum_at = self._clock.now()

        fixed = self._algorithm.grace_seconds
        deadline = quorum_at + (
            grace_window(
                state.estimates,
                run.schedule.in_flight,
                run.schedule.gap(round),
                run.schedule.overlap_steps,
            )
            if fixed is None
            else fixed
        )
        while (now := self._clock.now()) < deadline and not reported():
            await self._clock.sleep(min(self._poll_interval, deadline - now))
            await poll()
        grace_ended = self._clock.now()

        contributions = accepted()
        result = run.aggregators[fragment].close_round(current, contributions)
        updated = result.state
        estimates = state.estimates.updated(
            step=fmean(c.step_seconds for c in contributions), quorum=quorum_at - started
        )
        draft = RoundRecordDraft(
            round=round,
            step=run.schedule.step_of(round),
            fragment=fragment,
            version=updated.version,
            participants=result.participants,
            rejected=tuple(
                sorted((learner, why) for learner, why in verdicts.items() if isinstance(why, str))
            ),
            cosine=result.cosine,
            quorum_seconds=quorum_at - started,
            grace_seconds=grace_ended - quorum_at,
            estimates=estimates,
        )
        self._on_event(RoundClosed(self._node_id, draft))
        committing = time.perf_counter()
        outcome = await self._exchange.commit(draft, updated.theta, updated.outer, result.update)
        if isinstance(outcome, Superseded):
            self._on_event(RoundSuperseded(self._node_id, outcome.record))
            return Following(outcome.record.round + 1, self._timeout(outcome.record))
        sync_seconds = self._clock.now() - grace_ended
        self._on_event(
            RoundCommitted(self._node_id, outcome.record, time.perf_counter() - committing)
        )

        if not self._ceding():
            await self._exchange.claim(round + 1, 1, self._node_id)
        count = len(state.fragments)
        for swept in (round, round - 1):
            if swept >= 0:
                await self._exchange.sweep_snapshots(swept)
        await self._exchange.sweep_blobs(round - 2 * count)
        return SyncerState(
            round=round + 1,
            fragments=(*state.fragments[:fragment], updated, *state.fragments[fragment + 1 :]),
            estimates=estimates.updated(sync=sync_seconds),
            committed=outcome.record,
        )

    async def _rounds(self, until_round: int | None) -> None:
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
                    RecoveryFailed(self._node_id, exc, self._poll_interval)
                    if progress is None
                    else RoundFailed(self._node_id, progress.round, exc, self._poll_interval)
                )
                await self._clock.sleep(self._poll_interval)
                if progress is not None:
                    progress = Following(progress.round, self._timeout_of(progress))

    async def _hold(self) -> None:
        """Renew this node's syncer lease every ``lease_every`` seconds until the run is
        finished: no learner holds a live lease and the data is exhausted (``_exhausted``).

        A node of the elected kind holds one only while it owns the role: without one, it stands
        for election on every poll (``elect``); with one, it releases it once another node owns
        the role, or once no learner holds a live lease while units of its queue are unclaimed,
        and stands again once no node owns the role and another one trains.
        """
        renew_at = self._clock.now() + self._lease_every
        while True:
            try:
                if self._exclusive == "1-elected":
                    if (held := self._lease) is None:
                        if await self.elect():
                            renew_at = self._clock.now() + self._lease_every
                    else:
                        self._observed(await self._exchange.syncer_leases(), self._clock.now())
                        if self._owner != held:
                            await self._give_up(held)
                if (lease := self._lease) is not None:
                    if self._clock.now() >= renew_at:
                        await self._exchange.beat_syncer(lease)
                        renew_at = self._clock.now() + self._lease_every
                    if not await self._learners():
                        if self._exclusive == "1-elected" and await self._unclaimed():
                            await self._give_up(lease)
                        elif await self._exhausted():
                            return
            except Exception as exc:
                self._on_event(PollFailed(self._node_id, exc))
            await self._clock.sleep(self._poll_interval)

    async def _give_up(self, lease: str) -> None:
        await self._exchange.release_syncer(lease)
        self._on_event(LeaseReleased(self._node_id, "syncer"))
        self._lease = None

    async def _learners(self) -> frozenset[str]:
        """Fold one listing of the learner leases and return the live ones."""
        leases = await self._exchange.leases()
        now = self._clock.now()
        self._liveness = self._liveness.observed(leases, now)
        return self._liveness.live(now)

    async def _unclaimed(self) -> bool:
        """Whether units of this node's queue are left that no node has claimed."""
        units = self._units()
        return bool(units) and not units <= await self._exchange.unit_claims()

    async def _exhausted(self) -> bool:
        """Whether the run's data is exhausted, as the ``done/`` markers tell: every rank ever
        claimed is marked or, when none was, at least one node."""
        claimed = await self._exchange.rank_claims()
        done = await self._exchange.done()
        return {rank_id(rank) for rank in claimed} <= done if claimed else bool(done)

    def _observed(self, syncers: Mapping[str, int], now: float) -> frozenset[str]:
        """Fold one listing of the syncer leases and return the live ones. The first of them in
        byte order owns the exclusive role until the next listing."""
        self._syncers = self._syncers.observed(syncers, now)
        live = self._syncers.live(now)
        self._owner = min(live, default=None)
        return live

    def _ceding(self) -> bool:
        """Whether another node owns the exclusive syncer role, as last listed."""
        return self._owner is not None and self._owner != self._lease

    async def _resolved(self) -> "_Run":
        if self._run is None:
            manifest = await self._exchange.manifest()
            if manifest is None:
                raise ProtocolError("manifest.json does not exist: no learner has created the run")
            schema, plan = manifest.schema, manifest.plan
            specs = {spec.name: spec for spec in schema.parameters}
            self._run = _Run(
                manifest=manifest,
                digest=schema.digest(),
                deltas=tuple(
                    layout(
                        Schema(tuple(specs[name] for name in names)), self._algorithm.delta_dtype
                    )
                    for names in plan
                ),
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
    """What the manifest fixes for the life of the run; ``deltas`` is what a snapshot of each
    fragment must hold."""

    manifest: Manifest
    digest: str
    deltas: tuple[Schema[PayloadDType], ...]
    schedule: RoundSchedule
    aggregators: tuple[Aggregator, ...]


# Seconds of the store's clock a candidate waits for earlier candidates' leases to land.
_ELECTION_WAIT = 1.0
