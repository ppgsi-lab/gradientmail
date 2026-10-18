"""S3-compatible object storage as a ``Mailbox``: AWS S3, Cloudflare R2, Hugging Face Storage
Buckets, MinIO, moto.

Consistency assumptions
-----------------------
Three properties of the store carry the whole protocol: strong read-after-write and
list-after-write (AWS general-purpose buckets since 2020-12, R2, MinIO on local xfs, zfs or
btrfs, moto), an atomic ``If-None-Match: *`` on PUT (AWS since 2024-08, R2, MinIO since
2024-04, moto since 5.0.15) and ``ListObjectsV2`` in UTF-8 byte order. ``StartAfter`` is sent
but not relied on: Hugging Face's gateway accepts and ignores it, so ``list`` cuts at ``after``
itself and a poll there lists the whole prefix. Nothing else is needed because every object is
written once, payload keys carry the sha256 of their bytes and a record is written after the
payloads it points at.

Hugging Face Storage Buckets are reached at ``https://s3.hf.co/<namespace>`` with
``request_checksums="when_required"`` and S3 credentials generated from a Hub token; a PUT there
takes about half a second. Verified with the contract and training suites in October 2026.

Not assumed: FIFO delivery, exactly-once, atomicity across keys or ETag-as-MD5. Without
multi-key atomicity a listing may show a payload before its record, so readers follow records
and never list ``blobs/``.

The server's clock is read but nothing depends on it for correctness: ``clock`` is a
``StoreClock`` fed by botocore's ``before-send`` and ``before-parse`` events, which fire on the
calling thread around every HTTP exchange, retries and listing pages included, and carry the
response's ``Date`` header. It starts at the local wall clock.

Unsupported: S3 Express directory buckets (unordered listing, no ``StartAfter``), R2 behind a
caching custom domain (a cached read breaks read-after-write), MinIO on NFS (not a supported
MinIO backend).

Failure modes
-------------
A refused connection, a connect or read timeout and a connection closed mid-request become
``MailboxUnavailable`` once botocore's standard retry mode has made its five attempts; the
protocol layer then retries with backoff for as long as the outage lasts. A 412
``PreconditionFailed`` is the create-once contract (``put_if_absent`` returns False). A 409
``ConditionalRequestConflict`` means another conditional write on the same key was in flight
and the outcome is unknown; the put is retried three times, then raised as
``MailboxUnavailable``. Every other ``ClientError`` (a missing bucket, access denied,
throttling that outlasts botocore's retries) propagates unchanged: on a learner it is raised
from the ``Session`` constructor while joining and from the next ``step()`` afterwards, on the
syncer from ``run()``. A payload above 5 GiB raises
``ValueError`` instead of switching to multipart uploads; choose more fragments. A bulk delete
reports per-key failures in a 200 response; the first one is raised as a ``ClientError``. On
AWS and R2 a LIST costs 12.5 times a GET, and every learner and the syncer issue one LIST per
poll, so ``poll_interval`` is the cost lever.

Every call runs in ``asyncio.to_thread``; a ``threading.BoundedSemaphore`` acquired inside the
thread keeps the connection pool from being oversubscribed. No asyncio primitive lives on the
instance, so one mailbox can be driven from several event loops in turn.
"""

import asyncio
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from email.utils import parsedate_to_datetime
from functools import partial
from itertools import batched
from typing import TYPE_CHECKING, Literal

import boto3.session
from botocore.config import Config
from botocore.exceptions import (
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from gradientmail.clock import Clock, MonotonicClock, StoreClock
from gradientmail.mailbox import Mailbox, MailboxUnavailable

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

MAX_OBJECT_BYTES = 5 * 2**30
"""Largest single PUT S3 accepts; larger payloads need more fragments, not multipart uploads."""


class _SentAt(threading.local):
    at: float | None = None


class S3Mailbox(Mailbox):
    """Keys live under ``prefix`` in ``bucket``; credentials come from the boto3 default chain.

    Parameters
    ----------
    endpoint_url
        None for AWS; the service URL for R2, MinIO or any other S3-compatible store.
    region
        Signing region; None lets boto3 resolve it from the environment.
    max_concurrency
        Requests in flight at once, which is also the size of the connection pool.
    request_checksums
        ``"when_supported"`` asks the store to verify a SHA-256 of every uploaded object;
        ``"when_required"`` sends only the checksums the protocol mandates, for stores that
        reject flexible checksums (Hugging Face Storage Buckets).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        *,
        endpoint_url: str | None = None,
        region: str | None = None,
        max_concurrency: int = 8,
        request_checksums: Literal["when_supported", "when_required"] = "when_supported",
    ) -> None:
        self._bucket = bucket
        self._prefix = f"{prefix.strip('/')}/" if prefix.strip("/") else ""
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._client: S3Client = boto3.session.Session().client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region,
            config=Config(
                retries={"mode": "standard", "total_max_attempts": 5},
                max_pool_connections=max_concurrency,
                connect_timeout=10,
                read_timeout=120,
                s3={"addressing_style": "path"},
                request_checksum_calculation=request_checksums,
                response_checksum_validation=request_checksums,
            ),
        )
        put_object = partial(self._client.put_object, Bucket=bucket, IfNoneMatch="*")
        self._put_object = (
            partial(put_object, ChecksumAlgorithm="SHA256")
            if request_checksums == "when_supported"
            else put_object
        )
        self._local = MonotonicClock()
        self._clock = StoreClock(self._local, time.time() - self._local.now())
        self._sent = _SentAt()
        self._client.meta.events.register("before-send.s3", self._mark_sent)
        self._client.meta.events.register("before-parse.s3", self._observe_date)

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        if len(data) > MAX_OBJECT_BYTES:
            raise ValueError(
                f"{key}: {len(data)} bytes exceeds the 5 GiB single-object limit; "
                "use more fragments"
            )

        def put() -> bool:
            # A 409 ConditionalRequestConflict means another conditional write on the same
            # key was in progress; the outcome is unknown until one of them has landed.
            for _ in range(3):
                try:
                    self._put_object(Key=self._prefix + key, Body=data)
                except ClientError as exc:
                    code = exc.response.get("Error", {}).get("Code")
                    if code == "PreconditionFailed":
                        return False
                    if code != "ConditionalRequestConflict":
                        raise
                else:
                    return True
            raise MailboxUnavailable(f"{key}: conditional write conflicted three times")

        return await self._call(put)

    async def get(self, key: str) -> bytes | None:
        def get() -> bytes | None:
            try:
                response = self._client.get_object(Bucket=self._bucket, Key=self._prefix + key)
            except self._client.exceptions.NoSuchKey:
                return None
            return response["Body"].read()

        return await self._call(get)

    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]:
        def list_keys() -> Sequence[str]:
            paginator = self._client.get_paginator("list_objects_v2")
            full_prefix = self._prefix + prefix
            pages = (
                paginator.paginate(
                    Bucket=self._bucket, Prefix=full_prefix, StartAfter=self._prefix + after
                )
                if after
                else paginator.paginate(Bucket=self._bucket, Prefix=full_prefix)
            )
            # Hugging Face's gateway accepts StartAfter and ignores it, so the cut is made here
            # as well; on stores that honour it the comparison never drops anything.
            return tuple(
                stripped
                for page in pages
                for obj in page.get("Contents", ())
                if (key := obj.get("Key")) is not None
                and (stripped := key[len(self._prefix) :]) > after
            )

        return await self._call(list_keys)

    async def delete(self, keys: Iterable[str]) -> None:
        def delete_chunk(chunk: Sequence[str]) -> None:
            response = self._client.delete_objects(
                Bucket=self._bucket,
                Delete={"Objects": [{"Key": self._prefix + key} for key in chunk], "Quiet": True},
            )
            # Per-key failures arrive in a 200 response, not as an exception.
            if errors := response.get("Errors"):
                raise ClientError({"Error": errors[0]}, "DeleteObjects")

        for chunk in batched(keys, 1000):
            await self._call(partial(delete_chunk, chunk))

    def clock(self) -> Clock:
        return self._clock

    def _mark_sent(self, **_: object) -> None:
        self._sent.at = self._local.now()

    def _observe_date(self, response_dict: Mapping[str, object], **_: object) -> None:
        received = self._local.now()
        headers = response_dict.get("headers")
        date = headers.get("Date") if isinstance(headers, Mapping) else None
        sent = self._sent.at
        if sent is None or not isinstance(date, str):
            return
        # An exception raised here would fail the request being observed.
        try:
            stamped = parsedate_to_datetime(date).timestamp()
        except ValueError:
            return
        self._clock.observe(sent, received, stamped)

    async def _call[T](self, op: Callable[[], T]) -> T:
        def guarded() -> T:
            with self._slots:
                try:
                    return op()
                except (
                    EndpointConnectionError,
                    ConnectTimeoutError,
                    ReadTimeoutError,
                    ConnectionClosedError,
                ) as exc:
                    raise MailboxUnavailable(str(exc)) from exc

        return await asyncio.to_thread(guarded)
