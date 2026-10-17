"""The learner-syncer exchange: records, their keys, the JSON codec and the operations.

Keys live under the mailbox's prefix; fixed widths make byte order equal numeric order::

    manifest.json                                              Manifest, create-once
    rounds/{r:010d}.json                                       RoundRecord: commit point of r
    snapshots/{r:010d}/{learner}.json                          Snapshot of learner for round r
    claims/{r:010d}/{n:03d}.json                               nth claim to close round r
    claims/manifest/{n:03d}.json                               nth claim to create the run
    leases/{learner}/{n:010d}.json                             nth beat of learner's lease
    blobs/init/{p:03d}/{sha256}{ext}                           Θ⁰ candidates (Manifest.init)
    blobs/rounds/{r:010d}/{sha256}{ext}                        theta and outer payloads of r
    blobs/snapshots/{r:010d}/{learner}/{sha256}{ext}           snapshot payload

Records and payloads live under different prefixes, so ``list("rounds/", after=...)`` returns
exactly the new commits and ``list("snapshots/{r}/")`` exactly the candidates of round ``r``.
``{ext}`` is ``.safetensors`` or ``.safetensors.zst``: the codec the writer stored the payload
with (``compression`` module), which every reader decodes by the extension alone.

Write ordering
--------------
A payload is written before the record that points at it: init payloads then the manifest, a
snapshot payload then its record, theta then outer then the round record. With a store that
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
Every node runs the syncer loop; round ``r`` is closed by the node that created the round's
first claim, a create-once object naming the claimant. The others wait for the round record
and, after a timeout measured on their own clock, create the next claim and take over. Claims
only avoid redundant work: the round record is the commit point and the tape. A node whose
record write finds the key taken reads the record back; equal payload keys mean its own
earlier attempt landed (``Committed``), different keys mean another node closed the round and
``commit`` returns that record as ``Superseded`` for the loser to adopt. Round records are
never deleted, so ``rounds/`` is the full history of the run and the only input recovery needs.

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

Failure modes
-------------
A listing may show a payload before its record (there is no atomicity across keys); every
reader follows records and never lists ``blobs/``. A record listed but gone by the time it is
read was swept meanwhile and is skipped. A snapshot that lands after its round closed is not
in the round record and is swept with the round; its learner's local progress on that fragment
is overwritten by the committed version. ``fetch`` returns None for a payload the sweeper has
removed (the syncer records the snapshot as ``missing``; a learner skips that fragment and
installs the newer version its next poll lists) and raises ``PayloadCorrupt`` when the bytes
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
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Literal, assert_never

from gradientmail.algorithms import Estimates, Participant
from gradientmail.clock import Clock
from gradientmail.compression import EXTENSION, Compression, compression_of
from gradientmail.compression import decode as decode_payload
from gradientmail.compression import encode as encode_payload
from gradientmail.mailbox import Mailbox, retrying
from gradientmail.serialization import DType, ParameterSpec, Schema, TensorBytes

log = logging.getLogger(__name__)

type Rejection = Literal["stale", "zero_steps", "schema", "corrupt", "missing"]


@dataclass(frozen=True)
class PayloadRef:
    key: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ManifestDraft:
    schema: Schema
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
    """``quorum_seconds`` and ``grace_seconds`` are this round's realized waits; ``estimates``
    are the timing EMAs entering the next round, so whoever closes it continues them."""

    round: int
    step: int
    fragment: int
    version: int
    participants: tuple[Participant, ...]
    rejected: tuple[tuple[str, Rejection], ...]
    quorum_seconds: float
    grace_seconds: float
    estimates: Estimates


@dataclass(frozen=True)
class RoundRecord(RoundRecordDraft):
    theta: PayloadRef
    outer: PayloadRef


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


def lease_key(learner: str, beat: int) -> str:
    return f"{lease_prefix(learner)}{beat:010d}.json"


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
                    "theta": ref(record.theta),
                    "outer": ref(record.outer),
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
    def parameter(fields: _Fields) -> ParameterSpec:
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

    def schema(fields: _Fields) -> Schema:
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

    fields = _parse(data)
    return RoundRecord(
        round=fields.integer("round"),
        step=fields.integer("step"),
        fragment=fields.integer("fragment"),
        version=fields.integer("version"),
        participants=tuple(participant(entry) for entry in fields.each("participants")),
        rejected=tuple(rejection(entry) for entry in fields.items("rejected")),
        theta=_ref(fields.nested("theta")),
        outer=_ref(fields.nested("outer")),
        quorum_seconds=fields.number("quorum_seconds"),
        grace_seconds=fields.number("grace_seconds"),
        estimates=estimates(fields.nested("estimates")),
    )


class Exchange:
    """The protocol operations over a ``Mailbox``.

    Every operation runs through ``retrying``: a ``MailboxUnavailable`` is retried with
    backoff on ``clock``; anything else propagates. ``compression`` is the codec this exchange
    writes payloads with; it reads payloads in any.
    """

    def __init__(self, mailbox: Mailbox, *, clock: Clock, compression: Compression = "off") -> None:
        self._mailbox = mailbox
        self._clock = clock
        self._compression = compression

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

    async def claim(self, round: int, attempt: int, learner: str) -> bool:
        """Create the ``attempt``-th claim on round ``round``; True when it is ``learner``'s."""
        return await self._claim(claim_key(round, attempt), learner)

    async def claims(self, round: int) -> Sequence[str]:
        """The claimants of round ``round`` in attempt order."""
        return await self._claimants(f"claims/{round:010d}/")

    async def claim_creation(self, attempt: int, learner: str) -> bool:
        """Create the ``attempt``-th claim on creating the run; True when it is ``learner``'s."""
        return await self._claim(creation_claim_key(attempt), learner)

    async def creation_claims(self) -> Sequence[str]:
        """The claimants on creating the run, in attempt order."""
        return await self._claimants(CREATION_CLAIMS)

    async def init_uploaded(self) -> int:
        """How many Θ⁰ payloads exist so far: a creator's sign of life, counted and never read."""
        return len(await self._retry(partial(self._mailbox.list, "blobs/init/")))

    async def beat(self, learner: str) -> None:
        """Renew ``learner``'s lease: create the beat after its newest and delete the older ones."""
        older = await self._retry(partial(self._mailbox.list, lease_prefix(learner)))
        beat = max((_lease_of(key)[1] for key in older), default=-1) + 1
        # False means this beat landed in an earlier attempt whose response was lost.
        await self._retry(partial(self._mailbox.put_if_absent, lease_key(learner, beat), b"{}"))
        await self._retry(partial(self._mailbox.delete, older))

    async def release(self, learner: str) -> None:
        """Delete ``learner``'s lease: it is leaving the run."""
        keys = await self._retry(partial(self._mailbox.list, lease_prefix(learner)))
        await self._retry(partial(self._mailbox.delete, keys))

    async def leases(self) -> Mapping[str, int]:
        """The newest beat of every learner holding a lease, from one listing."""
        keys = await self._retry(partial(self._mailbox.list, LEASES))
        # Keys come in byte order and beats have a fixed width, so a learner's last key is its
        # newest beat.
        return dict(_lease_of(key) for key in keys)

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
    ) -> CommitOutcome:
        """Commit round ``draft.round``: theta, then outer, then the record.

        A record already present with the same payload keys is this node's own earlier attempt
        and counts as ``Committed``; different keys mean another node closed the round, and its
        record comes back as ``Superseded``.
        """
        theta_ref = await self._publish(partial(round_blob_key, draft.round), theta)
        outer_ref = await self._publish(partial(round_blob_key, draft.round), outer)
        record = RoundRecord(
            round=draft.round,
            step=draft.step,
            fragment=draft.fragment,
            version=draft.version,
            participants=draft.participants,
            rejected=draft.rejected,
            quorum_seconds=draft.quorum_seconds,
            grace_seconds=draft.grace_seconds,
            estimates=draft.estimates,
            theta=theta_ref,
            outer=outer_ref,
        )
        key = round_key(draft.round)
        created = await self._retry(partial(self._mailbox.put_if_absent, key, encode(record)))
        if created:
            return Committed(record)
        existing = decode_round_record(await self._require(key))
        if (existing.theta.key, existing.outer.key) == (theta_ref.key, outer_ref.key):
            return Committed(existing)
        return Superseded(existing)

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

    async def _claim(self, key: str, learner: str) -> bool:
        body = json.dumps({"learner": learner}, sort_keys=True, separators=(",", ":")).encode()
        if await self._retry(partial(self._mailbox.put_if_absent, key, body)):
            return True
        return _parse(await self._require(key)).string("learner") == learner

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


def _lease_of(key: str) -> tuple[str, int]:
    match = _LEASE_KEY.fullmatch(key)
    if match is None:
        raise ProtocolError(f"{key} is not a lease key")
    return match[1], int(match[2])


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


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
