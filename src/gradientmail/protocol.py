"""The learner-syncer exchange: records, their keys, the JSON codec and the operations.

Keys live under the mailbox's prefix; fixed widths make byte order equal numeric order::

    manifest.json                                              Manifest, create-once
    rounds/{r:010d}.json                                       RoundRecord: commit point of r
    snapshots/{r:010d}/{learner}.json                          Snapshot of learner for round r
    claims/{r:010d}/{n:03d}.json                               nth claim to close round r
    claims/manifest/{n:03d}.json                               nth claim to create the run
    leases/{learner}/{n:010d}.json                             nth beat of learner's lease
    syncers/{kind}/{arrival:013d}-{node}/{n:010d}.json         nth beat of a syncer's lease
    blobs/init/{p:03d}/{sha256}{ext}                           Θ⁰ candidates (Manifest.init)
    blobs/rounds/{r:010d}/{sha256}{ext}                        theta, outer and update payloads of r
    blobs/snapshots/{r:010d}/{learner}/{sha256}{ext}           snapshot payload
    units/{unit}.json                                          claim on a shared queue's unit
    ranks/{rank:03d}/{arrival:013d}-{node}.json                the node's claim on a rank
    done/{node}.json                                           the node's data is exhausted
    done/{rank:03d}.json                                       the rank's data is exhausted
    barriers/{name}/{node}                                     the node's arrival at a barrier

Records and payloads live under different prefixes, so ``list("rounds/", after=...)`` returns
exactly the new commits and ``list("snapshots/{r}/")`` exactly the candidates of round ``r``.
``{ext}`` is ``.safetensors`` or ``.safetensors.zst``: the codec the writer stored the payload
with (``compression`` module), which every reader decodes by the extension alone.

Write ordering
--------------
A payload is written before the record that points at it: init payloads then the manifest, a
snapshot payload then its record, theta, outer and the update then the round record. With a
store that
lists and reads its own writes, a record a party can see always has durable payloads, and a
crash between the two writes leaves an orphan payload and no record. Snapshot orphans are
swept with their round and round-payload orphans with their round's payloads; the init
payloads of a party that crashed before the manifest write are never removed.

Idempotency
-----------
Every object is written exactly once with ``put_if_absent``. A payload key contains the sha256
of its bytes, so writing the same tensors again is a no-op and a record can never describe
bytes other than the ones it was written with. Records are keyed by identity (the manifest,
the round, the round and learner), so a retried write that finds its key taken is a success:
``publish_snapshot`` returns, ``commit`` compares payload keys and returns, ``create_run``
adopts the manifest that won and deletes the init payloads it does not reference. Two learners
racing to initialize therefore end up on one Θ⁰, and a learner that restarts and publishes a
second snapshot for a round it had already published changes nothing: the first record stands
and the second payload is swept with the round.

Election
--------
Every node that closes rounds runs the syncer loop; round ``r`` is closed by the node that
created the round's first claim, a create-once object naming the claimant. The others wait for
the round record and, after a timeout measured on their own clock, create the next claim and
take over. While an exclusive syncer holds a live lease, the other nodes make no first claim
(``syncer`` module). Claims only avoid redundant work: the round record is the commit point and
the tape. A node whose record write finds the key taken reads the record back; equal payload
keys mean its own earlier attempt landed (``Committed``), different keys mean another node
closed the round and ``commit`` returns that record as ``Superseded`` for the loser to adopt.
A store that accepts two writes of one key keeps one of them although both returned success;
``check_commit`` reads a record back the same way, for a closer to find out, at its next round,
whose record the store kept. Round records are never deleted, so ``rounds/`` is the full
history of the run and the only input recovery needs.

Creating the run is elected the same way. The node that creates ``claims/manifest/001`` uploads
Θ⁰ and writes the manifest; the others poll for ``manifest.json`` instead of each uploading a
Θ⁰ of its own only to lose the race, which with bytes bodies costs the whole upload before the
412 arrives. A waiter watches the count of init payloads as a sign of life and, when none lands
for long enough (the ``learner`` module decides how long), creates the next claim and creates
the run itself; a creator that was slow rather than dead adopts the manifest that won.

Leases
------
A learner holds a lease while it runs. ``beat`` lists ``leases/{learner}/``, creates the beat
after the newest and deletes the rest, so the listing shows one object per learner (two during
a beat) and a restarted learner continues its own count; ``release`` deletes them all when the
learner closes. ``leases`` is one LIST of the whole prefix: the newest beat per learner. A lease
holds no time; the ``lease`` module judges liveness on the observer's own clock from how long
the newest beat has stayed the same. A learner that crashes leaves its last beat behind.

A node that runs as the run's exclusive syncer (``syncer`` module) holds a lease of the same
form under ``syncers/{kind}/{arrival}-{node}/`` instead: ``kind`` is ``0-dedicated`` for a node
run as a syncer and ``1-elected`` for one the run elected among its ``either`` nodes, and
``arrival`` the milliseconds of the store's clock (``Mailbox.clock``) when the node first wrote
it, so byte order ranks the leases by kind, then by arrival. ``syncer_leases`` is one LIST of
``syncers/``: the newest beat per lease. A syncer holds no learner lease, so it is never counted
among the learners nor waited for in a round.

Shared work
-----------
A ``SharedQueue`` (``queue`` module) hands out units of work by id. A node claims a unit by
writing ``units/{unit}.json``, holding its node id as a JSON string, and reading it back: a
write whose response was lost may have landed, and a store that accepts two writes of one key
keeps one of them. A claim names the unit's owner for the rest of the run unless the owner
releases it unstarted, deleting it so that any node may write it again. A node whose data is
exhausted writes ``done/{node}.json``, or ``done/{rank}.json`` when it holds a rank, before it
waits for its last rounds to commit; one that stops training to serve as the elected syncer
writes none. ``barriers/{name}/{node}`` holds a node's arrival at a ``Barrier`` (``barrier``
module).

Ranks
-----
A run whose nodes each train a fixed slice of the data numbers the slices and hands each to one
node at a time (``Session``'s ``ranks``). A node claims rank ``r`` by writing
``ranks/{r:03d}/{arrival}-{node}.json``, ``arrival`` the milliseconds of the store's clock when
it wrote it, so that byte order ranks a rank's claims by arrival. The rank is held by its first
claimant in byte order that holds a live learner lease (``lease`` module). A node claims the
lowest rank that no node holds and that has no ``done/`` marker, waits a second of the store's
clock, for the claims of nodes that arrived earlier to land, and lists the claims again: it
keeps the rank when its claim is now the first live one and otherwise deletes it. Every claimant
reads the same order, so of those that claim one rank at once exactly one keeps it, and the
claim never needs to be conditional: the key names the node that writes it, so no two nodes
ever write one key, and a store whose conditional write is not atomic, as the HF Storage
Buckets gateway's is not, changes nothing here. A claim with an earlier arrival whose lease
becomes live again, its node slow rather than gone, takes the rank back, and the node that took
the rank meanwhile has lost it; so a node that keeps a rank deletes, of the claims before its
own, only those whose node holds no lease at all. Claims are deleted in these two cases only, so
``ranks/`` names every rank ever held.

Failure modes
-------------
A listing may show a payload before its record (there is no atomicity across keys); every
reader follows records and never lists ``blobs/``. A record listed but gone by the time it is
read was swept meanwhile and is skipped. A snapshot that lands after its round closed is not
in the round record and is swept with the round; its learner's local progress on that fragment
is overwritten by the committed version. ``fetch`` returns None for a payload the sweeper has
removed (the syncer records the snapshot as ``missing``; a learner skips that fragment and
aligns with the newer version its next poll lists) and raises ``PayloadCorrupt`` when the bytes
do not hash to the record (the syncer records ``corrupt``; a learner fails, since a committed
payload that does not match its record is a store fault). Two processes publishing under one
learner id write one snapshot record per round, and the other's contribution is lost without
an error; the default id carries the process id for this reason. A malformed record raises
``ProtocolError``, and so does a manifest or round record that has disappeared, since the
protocol never deletes either. ``MailboxUnavailable`` is retried with exponential backoff
(0.5 s doubling to 30 s) for as long as it lasts; every other error propagates.

A ``*Draft`` carries the fields a party knows before anything is written; the ``Exchange``
completes it with the payload refs it wrote and stores the full record.
"""

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from operator import itemgetter
from typing import Literal, assert_never

from gradientmail.algorithms import Estimates, Participant
from gradientmail.clock import Clock
from gradientmail.compression import EXTENSION, Compression, compression_of
from gradientmail.compression import decode as decode_payload
from gradientmail.compression import encode as encode_payload
from gradientmail.mailbox import Mailbox, retrying
from gradientmail.serialization import DType, ParameterSpec, Schema, TensorBytes, pack

log = logging.getLogger(__name__)

type Rejection = Literal["stale", "zero_steps", "schema", "corrupt", "missing"]


@dataclass(frozen=True)
class PayloadRef:
    key: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ManifestDraft:
    schema: Schema[DType]
    plan: tuple[tuple[str, ...], ...]
    algorithm: Mapping[str, object]
    created_by: str


@dataclass(frozen=True)
class Manifest(ManifestDraft):
    """The run as created: ``init`` holds one Θ⁰ payload per fragment of ``plan``."""

    init: tuple[PayloadRef, ...]


@dataclass(frozen=True)
class SnapshotDraft:
    """``final`` marks a snapshot published by ``close()``: the learner trains no further."""

    learner: str
    round: int
    local_step: int
    steps: int
    tokens: int
    base_version: int
    rounds_seen: int
    step_seconds: float
    schema_digest: str
    final: bool

    def __post_init__(self) -> None:
        if not _LEARNER_ID.fullmatch(self.learner):
            raise ProtocolError(f"learner id {self.learner!r} must match [a-z0-9-]{{1,64}}")


@dataclass(frozen=True)
class Snapshot(SnapshotDraft):
    payload: PayloadRef


@dataclass(frozen=True)
class RoundRecordDraft:
    """``cosine`` is the mean cosine similarity between the participants' deltas over the
    fragment, None with fewer than two; ``quorum_seconds`` and ``grace_seconds`` are this
    round's realized waits; ``estimates`` are the timing EMAs entering the next round, so
    whoever closes it continues them."""

    round: int
    step: int
    fragment: int
    version: int
    participants: tuple[Participant, ...]
    rejected: tuple[tuple[str, Rejection], ...]
    cosine: float | None
    quorum_seconds: float
    grace_seconds: float
    estimates: Estimates


@dataclass(frozen=True)
class CommitUpdate:
    """``Θ_new - Θ_old`` at the algorithm's commit precision: a learner holding the old version
    applies it to its copy instead of fetching theta and checks the result against
    ``theta_digest``, the sha256 of theta packed as safetensors, whatever codec stored it."""

    payload: PayloadRef
    theta_digest: str


@dataclass(frozen=True)
class RoundRecord(RoundRecordDraft):
    theta: PayloadRef
    outer: PayloadRef
    update: CommitUpdate | None


type Record = Manifest | Snapshot | RoundRecord


@dataclass(frozen=True)
class Committed:
    """``commit`` wrote ``record``, now or in an earlier attempt that landed."""

    record: RoundRecord


@dataclass(frozen=True)
class Superseded:
    """Another node had committed the round first; ``record`` is theirs."""

    record: RoundRecord


type CommitOutcome = Committed | Superseded


class LocalCommits:
    """The fragments this process committed, for its own learner to take instead of fetching
    them back from the store.

    ``Exchange.commit`` fills the fragment's slot before it writes the round record, so a learner
    that lists the record finds the slot filled. The slot names the theta payload it holds, so
    a commit that lost the record write to another node's is never taken. One slot per fragment,
    holding the tensors the syncer keeps as its state anyway.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._slots: dict[int, tuple[int, str, Mapping[str, TensorBytes]]] = {}

    def put(self, fragment: int, version: int, key: str, theta: Mapping[str, TensorBytes]) -> None:
        with self._lock:
            self._slots[fragment] = (version, key, theta)

    def take(self, record: RoundRecord) -> Mapping[str, TensorBytes] | None:
        """``record``'s theta when this process committed it. A slot older than ``record``, or
        as old but holding another payload, can no longer be asked for and is dropped."""
        with self._lock:
            slot = self._slots.get(record.fragment)
            if slot is None or slot[0] > record.version:
                return None
            del self._slots[record.fragment]
            version, key, theta = slot
            return theta if (version, key) == (record.version, record.theta.key) else None


class ProtocolError(ValueError):
    """A record that does not follow the protocol: malformed JSON, a bad field, a missing key."""


class RunConfigMismatch(ValueError):
    """The run's manifest was created with a different algorithm configuration."""

    expected: Mapping[str, object]
    found: Mapping[str, object]

    def __init__(self, expected: Mapping[str, object], found: Mapping[str, object]) -> None:
        self.expected = expected
        self.found = found
        super().__init__(
            f"run was created with a different algorithm configuration: {dict(found)!r}, "
            f"this party has {dict(expected)!r}"
        )


class PayloadCorrupt(Exception):
    """The bytes under ``ref.key`` do not hash to ``ref.sha256``."""

    ref: PayloadRef
    actual: str

    def __init__(self, ref: PayloadRef, actual: str) -> None:
        self.ref = ref
        self.actual = actual
        super().__init__(f"{ref.key} hashes to {actual}, record says {ref.sha256}")


MANIFEST_KEY = "manifest.json"


def round_key(round: int) -> str:
    return f"rounds/{round:010d}.json"


def snapshots_prefix(round: int) -> str:
    return f"snapshots/{round:010d}/"


def snapshot_key(round: int, learner: str) -> str:
    return f"{snapshots_prefix(round)}{learner}.json"


def claim_key(round: int, attempt: int) -> str:
    return f"claims/{round:010d}/{attempt:03d}.json"


CREATION_CLAIMS = "claims/manifest/"


def creation_claim_key(attempt: int) -> str:
    return f"{CREATION_CLAIMS}{attempt:03d}.json"


LEASES = "leases/"


def lease_prefix(learner: str) -> str:
    return f"{LEASES}{learner}/"


SYNCERS = "syncers/"

type SyncerKind = Literal["0-dedicated", "1-elected"]


def syncer_lease_prefix(kind: SyncerKind, arrival: int, node: str) -> str:
    """The lease of ``node`` as an exclusive syncer of ``kind``, first written ``arrival``
    milliseconds into the store's clock."""
    return f"{SYNCERS}{kind}/{arrival:013d}-{node}/"


def syncer_node(lease: str) -> str:
    """The node holding ``lease``, a prefix ``syncer_lease_prefix`` made."""
    match = _SYNCER_LEASE.fullmatch(lease)
    if match is None:
        raise ProtocolError(f"{lease} is not a syncer lease")
    return match[1]


def beat_key(lease: str, beat: int) -> str:
    """The ``beat``-th beat of the lease under the prefix ``lease``, a learner's or a syncer's."""
    return f"{lease}{beat:010d}.json"


UNITS = "units/"


def unit_key(unit: str) -> str:
    return f"{UNITS}{unit}.json"


RANKS = "ranks/"


def rank_id(rank: int) -> str:
    """``rank`` as it names its claims and its ``done/`` marker."""
    return f"{rank:03d}"


def rank_prefix(rank: int) -> str:
    return f"{RANKS}{rank_id(rank)}/"


def rank_key(rank: int, arrival: int, node: str) -> str:
    """``node``'s claim on ``rank``, written ``arrival`` milliseconds into the store's clock."""
    return f"{rank_prefix(rank)}{arrival:013d}-{node}.json"


DONE = "done/"


def done_key(name: str) -> str:
    return f"{DONE}{name}.json"


def init_blob_key(fragment: int, sha256: str) -> str:
    """The payload key without its codec extension, as every ``*_blob_key`` is."""
    return f"blobs/init/{fragment:03d}/{sha256}"


def round_blob_key(round: int, sha256: str) -> str:
    return f"blobs/rounds/{round:010d}/{sha256}"


def snapshot_blob_key(round: int, learner: str, sha256: str) -> str:
    return f"blobs/snapshots/{round:010d}/{learner}/{sha256}"


def encode(record: Record) -> bytes:
    """Canonical JSON: sorted keys, no whitespace."""

    def ref(payload: PayloadRef) -> Mapping[str, object]:
        return {"key": payload.key, "sha256": payload.sha256, "size": payload.size}

    def body() -> Mapping[str, object]:
        match record:
            case Manifest():
                return {
                    "schema": [
                        {"name": p.name, "dtype": p.dtype, "shape": p.shape}
                        for p in record.schema.parameters
                    ],
                    "plan": record.plan,
                    "algorithm": dict(record.algorithm),
                    "init": [ref(payload) for payload in record.init],
                    "created_by": record.created_by,
                }
            case Snapshot():
                return {
                    "learner": record.learner,
                    "round": record.round,
                    "local_step": record.local_step,
                    "steps": record.steps,
                    "tokens": record.tokens,
                    "base_version": record.base_version,
                    "rounds_seen": record.rounds_seen,
                    "step_seconds": record.step_seconds,
                    "schema_digest": record.schema_digest,
                    "final": record.final,
                    "payload": ref(record.payload),
                }
            case RoundRecord():
                return {
                    "round": record.round,
                    "step": record.step,
                    "fragment": record.fragment,
                    "version": record.version,
                    "participants": [
                        {
                            "learner": p.learner,
                            "local_step": p.local_step,
                            "steps": p.steps,
                            "tokens": p.tokens,
                            "weight": p.weight,
                        }
                        for p in record.participants
                    ],
                    "rejected": record.rejected,
                    "cosine": record.cosine,
                    "theta": ref(record.theta),
                    "outer": ref(record.outer),
                    "update": None
                    if record.update is None
                    else {
                        "payload": ref(record.update.payload),
                        "theta_digest": record.update.theta_digest,
                    },
                    "quorum_seconds": record.quorum_seconds,
                    "grace_seconds": record.grace_seconds,
                    "estimates": {
                        "step": record.estimates.step,
                        "quorum": record.estimates.quorum,
                        "sync": record.estimates.sync,
                    },
                }
            case _:
                assert_never(record)

    return json.dumps(body(), sort_keys=True, separators=(",", ":")).encode()


def decode_manifest(data: bytes) -> Manifest:
    def parameter(fields: _Fields) -> ParameterSpec[DType]:
        return ParameterSpec(
            fields.string("name"),
            dtype(fields.values.get("dtype")),
            tuple(_integer(size, "shape") for size in fields.items("shape")),
        )

    def dtype(value: _Json) -> DType:
        match value:
            case "F32" | "F16" | "BF16" as code:
                return code
            case other:
                raise ProtocolError(f"dtype: expected F32, F16 or BF16, got {other!r}")

    def schema(fields: _Fields) -> Schema[DType]:
        parameters = tuple(parameter(entry) for entry in fields.each("schema"))
        try:
            return Schema(parameters)
        except ValueError as exc:
            raise ProtocolError(f"schema: {exc}") from exc

    fields = _parse(data)
    return Manifest(
        schema=schema(fields),
        plan=tuple(
            tuple(_string(name, "plan") for name in _items(fragment, "plan"))
            for fragment in fields.items("plan")
        ),
        algorithm=fields.nested("algorithm").values,
        created_by=fields.string("created_by"),
        init=tuple(_ref(entry) for entry in fields.each("init")),
    )


def decode_snapshot(data: bytes) -> Snapshot:
    fields = _parse(data)
    return Snapshot(
        learner=fields.string("learner"),
        round=fields.integer("round"),
        local_step=fields.integer("local_step"),
        steps=fields.integer("steps"),
        tokens=fields.integer("tokens"),
        base_version=fields.integer("base_version"),
        rounds_seen=fields.integer("rounds_seen"),
        step_seconds=fields.number("step_seconds"),
        schema_digest=fields.string("schema_digest"),
        final=fields.boolean("final"),
        payload=_ref(fields.nested("payload")),
    )


def decode_round_record(data: bytes) -> RoundRecord:
    def participant(fields: _Fields) -> Participant:
        return Participant(
            learner=fields.string("learner"),
            local_step=fields.integer("local_step"),
            steps=fields.integer("steps"),
            tokens=fields.integer("tokens"),
            weight=fields.number("weight"),
        )

    def rejection(value: _Json) -> tuple[str, Rejection]:
        match value:
            case [
                str() as learner,
                "stale" | "zero_steps" | "schema" | "corrupt" | "missing" as why,
            ]:
                return learner, why
            case other:
                raise ProtocolError(f"rejected: expected [learner, reason], got {other!r}")

    def estimates(fields: _Fields) -> Estimates:
        return Estimates(
            step=fields.optional_number("step"),
            quorum=fields.optional_number("quorum"),
            sync=fields.optional_number("sync"),
        )

    def update(fields: _Fields) -> CommitUpdate | None:
        if fields.values.get("update") is None:
            return None
        nested = fields.nested("update")
        return CommitUpdate(_ref(nested.nested("payload")), nested.string("theta_digest"))

    fields = _parse(data)
    return RoundRecord(
        round=fields.integer("round"),
        step=fields.integer("step"),
        fragment=fields.integer("fragment"),
        version=fields.integer("version"),
        participants=tuple(participant(entry) for entry in fields.each("participants")),
        rejected=tuple(rejection(entry) for entry in fields.items("rejected")),
        cosine=fields.optional_number("cosine"),
        theta=_ref(fields.nested("theta")),
        outer=_ref(fields.nested("outer")),
        update=update(fields),
        quorum_seconds=fields.number("quorum_seconds"),
        grace_seconds=fields.number("grace_seconds"),
        estimates=estimates(fields.nested("estimates")),
    )


class Exchange:
    """The protocol operations over a ``Mailbox``.

    Every operation runs through ``retrying``: a ``MailboxUnavailable`` is retried with
    backoff on ``clock``; anything else propagates. ``compression`` is the codec this exchange
    writes payloads with; it reads payloads in any. ``local`` receives every fragment it
    commits.
    """

    def __init__(
        self,
        mailbox: Mailbox,
        *,
        clock: Clock,
        compression: Compression = "off",
        local: LocalCommits | None = None,
    ) -> None:
        self._mailbox = mailbox
        self._clock = clock
        self._compression = compression
        self._local = local

    async def manifest(self) -> Manifest | None:
        data = await self._retry(partial(self._mailbox.get, MANIFEST_KEY))
        return None if data is None else decode_manifest(data)

    async def create_run(
        self, draft: ManifestDraft, init: Sequence[Mapping[str, TensorBytes]]
    ) -> Manifest:
        """Create the run from ``draft`` with ``init[p]`` as Θ⁰ of fragment ``p``.

        Init payloads are written first, the manifest last. When another party created the run
        first, this caller's payloads that the winner does not reference are deleted and the
        winner's manifest is returned, so racing learners always start from one Θ⁰.
        """
        if len(init) != len(draft.plan):
            raise ValueError(f"{len(draft.plan)} fragments planned but {len(init)} init payloads")
        refs = tuple(
            [
                await self._publish(partial(init_blob_key, fragment), tensors)
                for fragment, tensors in enumerate(init)
            ]
        )
        manifest = Manifest(
            schema=draft.schema,
            plan=draft.plan,
            algorithm=draft.algorithm,
            created_by=draft.created_by,
            init=refs,
        )
        created = await self._retry(
            partial(self._mailbox.put_if_absent, MANIFEST_KEY, encode(manifest))
        )
        if created:
            return manifest
        winner = decode_manifest(await self._require(MANIFEST_KEY))
        kept = {ref.key for ref in winner.init}
        orphans = tuple(ref.key for ref in refs if ref.key not in kept)
        await self._retry(partial(self._mailbox.delete, orphans))
        return winner

    async def publish_snapshot(
        self, draft: SnapshotDraft, tensors: Mapping[str, TensorBytes]
    ) -> Snapshot:
        """Publish ``tensors`` as ``draft``'s snapshot; idempotent for one (round, learner).

        Returns the snapshot as published, its payload reference included.
        """
        payload = await self._publish(
            partial(snapshot_blob_key, draft.round, draft.learner), tensors
        )
        snapshot = Snapshot(
            learner=draft.learner,
            round=draft.round,
            local_step=draft.local_step,
            steps=draft.steps,
            tokens=draft.tokens,
            base_version=draft.base_version,
            rounds_seen=draft.rounds_seen,
            step_seconds=draft.step_seconds,
            schema_digest=draft.schema_digest,
            final=draft.final,
            payload=payload,
        )
        # False means this (round, learner) already has a record: an earlier attempt landed.
        await self._retry(
            partial(
                self._mailbox.put_if_absent,
                snapshot_key(draft.round, draft.learner),
                encode(snapshot),
            )
        )
        return snapshot

    async def snapshots(self, round: int) -> Sequence[Snapshot]:
        return await self._records(snapshots_prefix(round), decode_snapshot)

    async def publishers(self, round: int) -> frozenset[str]:
        """The learners with a snapshot record in ``round``, from the listing alone."""
        prefix = snapshots_prefix(round)
        keys = await self._retry(partial(self._mailbox.list, prefix))
        return frozenset(key[len(prefix) : -len(".json")] for key in keys)

    async def claim(self, round: int, attempt: int, node: str) -> bool:
        """Create the ``attempt``-th claim on round ``round``; True when it is ``node``'s."""
        return await self._claim(claim_key(round, attempt), node)

    async def claims(self, round: int) -> Sequence[str]:
        """The claimants of round ``round`` in attempt order."""
        return await self._claimants(f"claims/{round:010d}/")

    async def claim_creation(self, attempt: int, node: str) -> bool:
        """Create the ``attempt``-th claim on creating the run; True when it is ``node``'s."""
        return await self._claim(creation_claim_key(attempt), node)

    async def creation_claims(self) -> Sequence[str]:
        """The claimants on creating the run, in attempt order."""
        return await self._claimants(CREATION_CLAIMS)

    async def init_uploaded(self) -> int:
        """How many Θ⁰ payloads exist so far: a creator's sign of life, counted and never read."""
        return len(await self._retry(partial(self._mailbox.list, "blobs/init/")))

    async def beat(self, node: str) -> None:
        """Renew ``node``'s lease: create the beat after its newest and delete the older ones."""
        await self._beat(lease_prefix(node))

    async def release(self, node: str) -> None:
        """Delete ``node``'s lease: it is leaving the run."""
        await self._release(lease_prefix(node))

    async def leases(self) -> Mapping[str, int]:
        """The newest beat of every learner holding a lease, from one listing."""
        keys = await self._retry(partial(self._mailbox.list, LEASES))
        # Keys come in byte order and beats have a fixed width, so a learner's last key is its
        # newest beat.
        return dict(_lease_of(key) for key in keys)

    async def beat_syncer(self, lease: str) -> None:
        """Renew the syncer lease ``lease`` (``syncer_lease_prefix``) as ``beat`` renews a
        learner's."""
        await self._beat(lease)

    async def release_syncer(self, lease: str) -> None:
        """Delete the syncer lease ``lease``: its syncer is stopping."""
        await self._release(lease)

    async def syncer_leases(self) -> Mapping[str, int]:
        """The newest beat of every syncer lease, keyed by the lease's prefix, from one listing."""
        keys = await self._retry(partial(self._mailbox.list, SYNCERS))
        return dict(_syncer_lease_of(key) for key in keys)

    async def claim_unit(self, unit: str, node: str) -> bool:
        """Claim ``unit`` of the shared queue for ``node``; True when the claim is ``node``'s.

        The claim is read back whatever the write returned: a write whose response was lost may
        have landed, and a store that accepted two writes of the key keeps one of them.
        """
        key, body = unit_key(unit), json.dumps(node).encode()
        await self._retry(partial(self._mailbox.put_if_absent, key, body))
        return await self._retry(partial(self._mailbox.get, key)) == body

    async def unit_claims(self) -> frozenset[str]:
        """The units of the shared queue claimed so far, by any node, from one listing."""
        keys = await self._retry(partial(self._mailbox.list, UNITS))
        return frozenset(key[len(UNITS) : -len(".json")] for key in keys)

    async def release_units(self, units: Iterable[str]) -> None:
        """Delete the claims on ``units``, so that any node may claim them again."""
        await self._retry(partial(self._mailbox.delete, tuple(unit_key(unit) for unit in units)))

    async def claim_rank(self, rank: int, node: str) -> None:
        """Write ``node``'s claim on ``rank``, timed by the store's clock."""
        arrival = int(self._mailbox.clock().now() * 1000)
        await self._retry(
            partial(self._mailbox.put_if_absent, rank_key(rank, arrival, node), b"{}")
        )

    async def rank_claims(self) -> Mapping[int, Sequence[str]]:
        """The claimants of every claimed rank, each rank's in byte order of their claims, from
        one listing."""
        keys = await self._retry(partial(self._mailbox.list, RANKS))
        # Keys come in byte order and the rank has a fixed width, so each rank's are adjacent.
        claims = groupby((_rank_claim_of(key) for key in keys), key=itemgetter(0))
        return {rank: tuple(node for _, node in claim) for rank, claim in claims}

    async def release_rank(self, rank: int, node: str) -> None:
        """Delete ``node``'s claims on ``rank``."""
        keys = await self._retry(partial(self._mailbox.list, rank_prefix(rank)))
        mine = tuple(key for key in keys if _rank_claim_of(key)[1] == node)
        await self._retry(partial(self._mailbox.delete, mine))

    async def mark_done(self, name: str) -> None:
        """Record that the data of ``name``, a node or a rank (``rank_id``), is exhausted;
        idempotent."""
        await self._retry(partial(self._mailbox.put_if_absent, done_key(name), b"{}"))

    async def done(self) -> frozenset[str]:
        """The nodes and ranks whose data is exhausted so far, from one listing."""
        keys = await self._retry(partial(self._mailbox.list, DONE))
        return frozenset(key[len(DONE) : -len(".json")] for key in keys)

    async def rounds_after(self, round: int) -> Sequence[RoundRecord]:
        """Round records newer than ``round`` in round order; ``-1`` means all of them."""
        after = round_key(round) if round >= 0 else ""
        return await self._records("rounds/", decode_round_record, after=after)

    async def fetch(self, ref: PayloadRef) -> Mapping[str, TensorBytes] | None:
        """The tensors behind ``ref``; None once swept, ``PayloadCorrupt`` on a sha mismatch."""
        compression = compression_of(ref.key)
        if compression is None:
            raise ProtocolError(f"{ref.key} names no payload codec")
        started = time.perf_counter()
        blob = await self._retry(partial(self._mailbox.get, ref.key))
        if blob is None:
            return None
        log.debug("got %s: %d bytes in %.1f s", ref.key, len(blob), time.perf_counter() - started)
        actual = _sha256(blob)
        if actual != ref.sha256:
            raise PayloadCorrupt(ref, actual)
        return decode_payload(blob, compression)

    async def commit(
        self,
        draft: RoundRecordDraft,
        theta: Mapping[str, TensorBytes],
        outer: Mapping[str, TensorBytes],
        update: Mapping[str, TensorBytes] | None = None,
    ) -> CommitOutcome:
        """Commit round ``draft.round``: theta, outer and the update, if any, then the record.

        A record already present with the same payload keys is this node's own earlier attempt
        and counts as ``Committed``; different keys mean another node closed the round, and its
        record comes back as ``Superseded``.
        """
        blob_key = partial(round_blob_key, draft.round)
        theta_ref = await self._publish(blob_key, theta)
        outer_ref = await self._publish(blob_key, outer)
        update_ref = (
            None
            if update is None
            else CommitUpdate(await self._publish(blob_key, update), theta_digest(theta))
        )
        if self._local is not None:
            self._local.put(draft.fragment, draft.version, theta_ref.key, theta)
        record = RoundRecord(
            round=draft.round,
            step=draft.step,
            fragment=draft.fragment,
            version=draft.version,
            participants=draft.participants,
            rejected=draft.rejected,
            cosine=draft.cosine,
            quorum_seconds=draft.quorum_seconds,
            grace_seconds=draft.grace_seconds,
            estimates=draft.estimates,
            theta=theta_ref,
            outer=outer_ref,
            update=update_ref,
        )
        key = round_key(draft.round)
        created = await self._retry(partial(self._mailbox.put_if_absent, key, encode(record)))
        return Committed(record) if created else await self.check_commit(record)

    async def check_commit(self, record: RoundRecord) -> CommitOutcome:
        """Read ``record``'s round back: ``Committed`` when the stored record has ``record``'s
        payload keys, ``Superseded`` with the stored record when it is another node's."""
        stored = decode_round_record(await self._require(round_key(record.round)))
        if _payload_keys(stored) == _payload_keys(record):
            return Committed(stored)
        return Superseded(stored)

    async def sweep_snapshots(self, round: int) -> None:
        """Delete round ``round``'s snapshot records, payloads and claims, orphans included."""
        records = await self._retry(partial(self._mailbox.list, f"snapshots/{round:010d}/"))
        payloads = await self._retry(partial(self._mailbox.list, f"blobs/snapshots/{round:010d}/"))
        claims = await self._retry(partial(self._mailbox.list, f"claims/{round:010d}/"))
        await self._retry(partial(self._mailbox.delete, (*records, *payloads, *claims)))

    async def sweep_blobs(self, round: int) -> None:
        """Delete round ``round``'s theta and outer payloads; its record is never deleted."""
        if round < 0:
            return
        payloads = await self._retry(partial(self._mailbox.list, f"blobs/rounds/{round:010d}/"))
        await self._retry(partial(self._mailbox.delete, payloads))

    async def _publish(
        self, key_of: Callable[[str], str], tensors: Mapping[str, TensorBytes]
    ) -> PayloadRef:
        blob = encode_payload(tensors, self._compression)
        sha256 = _sha256(blob)
        ref = PayloadRef(key_of(sha256) + EXTENSION[self._compression], sha256, len(blob))
        started = time.perf_counter()
        # False means identical bytes are already there: the key contains their hash.
        await self._retry(partial(self._mailbox.put_if_absent, ref.key, blob))
        log.debug("put %s: %d bytes in %.1f s", ref.key, len(blob), time.perf_counter() - started)
        return ref

    async def _beat(self, lease: str) -> None:
        older = await self._retry(partial(self._mailbox.list, lease))
        beat = max((_beat_of(lease, key) for key in older), default=-1) + 1
        # False means this beat landed in an earlier attempt whose response was lost.
        await self._retry(partial(self._mailbox.put_if_absent, beat_key(lease, beat), b"{}"))
        await self._retry(partial(self._mailbox.delete, older))

    async def _release(self, lease: str) -> None:
        keys = await self._retry(partial(self._mailbox.list, lease))
        await self._retry(partial(self._mailbox.delete, keys))

    async def _claim(self, key: str, node: str) -> bool:
        body = json.dumps({"learner": node}, sort_keys=True, separators=(",", ":")).encode()
        if await self._retry(partial(self._mailbox.put_if_absent, key, body)):
            return True
        return _parse(await self._require(key)).string("learner") == node

    async def _claimants(self, prefix: str) -> Sequence[str]:
        return await self._records(prefix, lambda data: _parse(data).string("learner"))

    async def _records[T](
        self, prefix: str, decode: Callable[[bytes], T], *, after: str = ""
    ) -> Sequence[T]:
        keys = await self._retry(partial(self._mailbox.list, prefix, after=after))
        async with asyncio.TaskGroup() as group:
            reads = [
                group.create_task(self._retry(partial(self._mailbox.get, key))) for key in keys
            ]
        # A record listed but gone by the time it is read was swept meanwhile; skip it.
        return tuple(decode(data) for read in reads if (data := read.result()) is not None)

    async def _require(self, key: str) -> bytes:
        data = await self._retry(partial(self._mailbox.get, key))
        if data is None:
            raise ProtocolError(f"{key} exists but could not be read back")
        return data

    async def _retry[T](self, op: Callable[[], Awaitable[T]]) -> T:
        return await retrying(op, clock=self._clock)


type _Json = bool | int | float | str | list[_Json] | dict[str, _Json] | None

_LEARNER_ID = re.compile(r"[a-z0-9-]{1,64}")
_LEASE_KEY = re.compile(rf"{LEASES}({_LEARNER_ID.pattern})/(\d{{10}})\.json")
_SYNCER_LEASE = re.compile(rf"{SYNCERS}\d-[a-z]+/\d{{13}}-({_LEARNER_ID.pattern})/")
_SYNCER_LEASE_KEY = re.compile(rf"({_SYNCER_LEASE.pattern})(\d{{10}})\.json")
_RANK_KEY = re.compile(rf"{RANKS}(\d{{3}})/\d{{13}}-({_LEARNER_ID.pattern})\.json")
_BEAT = re.compile(r"(\d{10})\.json")


def _lease_of(key: str) -> tuple[str, int]:
    match = _LEASE_KEY.fullmatch(key)
    if match is None:
        raise ProtocolError(f"{key} is not a lease key")
    return match[1], int(match[2])


def _syncer_lease_of(key: str) -> tuple[str, int]:
    match = _SYNCER_LEASE_KEY.fullmatch(key)
    if match is None:
        raise ProtocolError(f"{key} is not a syncer lease key")
    return match[1], int(match[3])


def _rank_claim_of(key: str) -> tuple[int, str]:
    match = _RANK_KEY.fullmatch(key)
    if match is None:
        raise ProtocolError(f"{key} is not a rank claim")
    return int(match[1]), match[2]


def _beat_of(lease: str, key: str) -> int:
    match = _BEAT.fullmatch(key, len(lease))
    if match is None:
        raise ProtocolError(f"{key} is not a beat of {lease}")
    return int(match[1])


@dataclass(frozen=True)
class _Fields:
    """Typed access to one decoded JSON object; every mismatch is a ``ProtocolError``."""

    values: Mapping[str, _Json]

    def string(self, key: str) -> str:
        return _string(self.values.get(key), key)

    def integer(self, key: str) -> int:
        return _integer(self.values.get(key), key)

    def boolean(self, key: str) -> bool:
        match self.values.get(key):
            case bool() as value:
                return value
            case other:
                raise ProtocolError(f"{key}: expected a boolean, got {other!r}")

    def number(self, key: str) -> float:
        match self.values.get(key):
            case int() | float() as value:
                return float(value)
            case other:
                raise ProtocolError(f"{key}: expected a number, got {other!r}")

    def optional_number(self, key: str) -> float | None:
        return None if self.values.get(key) is None else self.number(key)

    def items(self, key: str) -> Sequence[_Json]:
        return _items(self.values.get(key), key)

    def nested(self, key: str) -> "_Fields":
        return _object(self.values.get(key), key)

    def each(self, key: str) -> Sequence["_Fields"]:
        return tuple(_object(item, key) for item in self.items(key))


def _parse(data: bytes) -> _Fields:
    try:
        parsed: _Json = json.loads(data)
    except ValueError as exc:
        raise ProtocolError("record is not JSON") from exc
    return _object(parsed, "record")


def _object(value: _Json, where: str) -> _Fields:
    match value:
        case dict() as values:
            return _Fields(values)
        case other:
            raise ProtocolError(f"{where}: expected an object, got {other!r}")


def _items(value: _Json, where: str) -> Sequence[_Json]:
    match value:
        case list() as items:
            return items
        case other:
            raise ProtocolError(f"{where}: expected an array, got {other!r}")


def _string(value: _Json, where: str) -> str:
    match value:
        case str() as text:
            return text
        case other:
            raise ProtocolError(f"{where}: expected a string, got {other!r}")


def _integer(value: _Json, where: str) -> int:
    match value:
        case int() as number:
            return number
        case other:
            raise ProtocolError(f"{where}: expected an integer, got {other!r}")


def _ref(fields: _Fields) -> PayloadRef:
    return PayloadRef(fields.string("key"), fields.string("sha256"), fields.integer("size"))


def _payload_keys(record: RoundRecord) -> tuple[str, str, str | None]:
    return (
        record.theta.key,
        record.outer.key,
        None if record.update is None else record.update.payload.key,
    )


def theta_digest(theta: Mapping[str, TensorBytes]) -> str:
    """The sha256 of ``theta`` packed as safetensors: the same whatever codec stores it."""
    return _sha256(pack(theta))


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
