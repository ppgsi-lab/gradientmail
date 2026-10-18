# Implementing a mailbox backend

*October 2026.*

GradientMail nodes do not connect to each other. They read and write objects in a shared store, the mailbox, and the run is coordinated there: who creates it, which snapshots a round contains, who closes the round and which nodes are still alive. [`S3Mailbox`](../../src/gradientmail/mailbox/s3.py) covers AWS S3, Cloudflare R2, MinIO and Hugging Face Storage Buckets; [`MemoryMailbox`](../../src/gradientmail/mailbox/memory.py) keeps a dict for tests.

Another store needs a class with four methods and a clock. This post describes what they must guarantee, where GradientMail relies on each guarantee and how to test a new backend, using one that keeps objects as files on one machine. For a service with an S3-compatible API, `S3Mailbox` with `endpoint_url=` is the first thing to try; the test suite runs against it when `GRADIENTMAIL_S3_ENDPOINT` is set.

## The interface

The protocol is defined in [`gradientmail/mailbox/__init__.py`](../../src/gradientmail/mailbox/__init__.py):

```python
class Mailbox(Protocol):
    async def put_if_absent(self, key: str, data: bytes) -> bool: ...
    async def get(self, key: str) -> bytes | None: ...
    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]: ...
    async def delete(self, keys: Iterable[str]) -> None: ...
    def clock(self) -> Clock: ...
```

`put_if_absent` creates an object and returns whether this call created it; an existing object is left untouched. `get` returns `None` for a missing key. `list` returns every key under `prefix` greater than `after`, in UTF-8 byte order, across all pages. `delete` removes any number of keys, and a missing key is not an error. `clock` returns the store's time as this node estimates it, and the [Barrier](../patterns/01-barrier.md) releases nodes at one instant of it. `S3Mailbox` estimates it from the `Date` header of its responses; a store whose nodes all run on one machine can return `MonotonicClock()`, which every process there reads alike.

Keys are relative to the run: `S3Mailbox` prepends its `prefix`, and each run uses a new one. [`protocol.py`](../../src/gradientmail/protocol.py) documents the layout (`manifest.json`, `rounds/`, `snapshots/`, `claims/`, `leases/`, `blobs/`), and the [Shared Queue](../patterns/00-shared-queue.md) adds `units/`. Numbers in keys have a fixed width, so byte order is numeric order.

The methods are coroutines, and one instance serves several event loops on different threads: a `Session` joins on the calling thread and then runs the learner's communication loop and the syncer on threads of their own. A backend therefore holds no asyncio primitive tied to one loop and runs blocking I/O in `asyncio.to_thread`; a learner publishes and polls on one loop, so a blocking upload would delay its lease renewals. No `asyncio.timeout` wraps a store call, because a cancelled `to_thread` call leaves its thread running, so a backend bounds its own requests, as `S3Mailbox` does with botocore's timeouts.

## What the store must guarantee

**Atomic create-once writes.** For each key, exactly one `put_if_absent` returns `True`, and no later call replaces the object. Every object in a run is written this way. Claims elect the run's creator and each round's closer; the manifest, snapshot records, lease beats and Shared Queue units are also keyed by identity, and the first writer wins. When two nodes close the same round, the one whose record write fails adopts the other's record. Payload keys contain the sha256 of their bytes, so writing the same tensors again changes nothing.

**Whole objects.** A reader sees all of an object or none of it. After a `put_if_absent` that returns `False`, the protocol reads the existing object and acts on it: a claim compares the learner named in its JSON, a commit compares the payload keys. A truncated record raises `ProtocolError`, and a truncated payload fails its hash check with `PayloadCorrupt`.

**Read-after-write and list-after-write.** Once a write returns, every client sees the object through `get` and `list`. Payloads are written before the records that point at them, so a node that sees a record can fetch its payloads until they are swept. When `put_if_absent` reports an existing key and `get` then returns `None`, the protocol raises `ProtocolError`.

**Complete, ordered listings.** Learners and syncers poll `list("rounds/", after=...)` for new commits, read a round's claims in attempt order and take a learner's last key under `leases/` as its newest beat. All three need every key, in byte order. Hugging Face's S3 gateway ignores `StartAfter`, so `S3Mailbox` drops the keys up to `after` itself; a backend should do the same when its store's cursor is unreliable.

**Idempotent deletes.** A sweep lists keys and then deletes them, and another node or a retried request may have removed some of them in between.

The protocol does not assume FIFO delivery, exactly-once delivery or atomicity across keys. Nothing depends on the clock for correctness: only the Barrier reads it, to time its release.

## Errors and retries

`MailboxUnavailable` marks a transient failure. Every protocol operation retries it with backoff, from 0.5 s doubling to 30 s, for as long as the failure lasts. A backend should also raise it when the outcome of a write is unknown, as after a timeout: the retried `put_if_absent` finds the key taken and returns `False`, and the protocol accepts that as its own earlier write, because a key names either an identity or the bytes it holds. `FaultyMailbox` in [`tests/fakes.py`](../../tests/fakes.py) tests this by applying an operation and then losing its response. `S3Mailbox` handles a 409 `ConditionalRequestConflict` the same way, retrying three times before raising `MailboxUnavailable`.

The protocol does not retry other exceptions. While a node joins, they are raised from the `Session` constructor; in the background loops they are reported as events (`PollFailed`, `PublishFailed`, `RoundFailed`), and the loop goes on. `shared_queue` calls the mailbox directly, so any error, `MailboxUnavailable` included, ends the queue and reaches the training loop after the buffered batches.

## Why conditional creation must be atomic

Checking for a key and then writing it does not replace a conditional write. Two nodes claiming the same Shared Queue unit can interleave as follows: node B checks and finds no object; node A checks, writes its learner ID, reads it back and starts loading; node B writes its own ID, reads it back and starts loading as well. Reading back shows only the newest write and cannot detect a write that arrives later. A `MemoryMailbox` subclass that pauses between its check and its write reproduces this with the claim expression `shared_queue` uses: both claims return `True`.

Round records fail in a similar way. `commit` returns `Committed` as soon as its write reports creation, without reading the record back. On such a store, two closers can each commit round `r` with different records, and learners that read the round before and after the second write hold different parameters for the same version.

## A filesystem mailbox

```python
import asyncio
import os
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

import gradientmail as gm
from gradientmail.clock import Clock, MonotonicClock


class FileMailbox(gm.Mailbox):
    """Objects as files under ``root/objects``; the slashes of a key are directories."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._objects = Path(root) / "objects"
        self._staging = Path(root) / "staging"
        self._objects.mkdir(parents=True, exist_ok=True)
        self._staging.mkdir(parents=True, exist_ok=True)

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        return await asyncio.to_thread(self._put_if_absent, key, data)

    async def get(self, key: str) -> bytes | None:
        return await asyncio.to_thread(self._get, key)

    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]:
        return await asyncio.to_thread(self._list, prefix, after)

    async def delete(self, keys: Iterable[str]) -> None:
        await asyncio.to_thread(self._delete, tuple(keys))

    def clock(self) -> Clock:
        return MonotonicClock()

    def _put_if_absent(self, key: str, data: bytes) -> bool:
        target = self._objects / key
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = self._staging / uuid.uuid4().hex
        staged.write_bytes(data)
        try:
            # link() refuses an existing target, and the file appears complete or not at all.
            os.link(staged, target)
        except FileExistsError:
            return False
        finally:
            staged.unlink()
        return True

    def _get(self, key: str) -> bytes | None:
        try:
            return (self._objects / key).read_bytes()
        except FileNotFoundError:
            return None

    def _list(self, prefix: str, after: str) -> Sequence[str]:
        start = self._objects / prefix.rpartition("/")[0]
        keys = (
            (Path(directory) / name).relative_to(self._objects).as_posix()
            for directory, _, names in os.walk(start)
            for name in names
        )
        return sorted(key for key in keys if key.startswith(prefix) and key > after)

    def _delete(self, keys: Sequence[str]) -> None:
        for key in keys:
            (self._objects / key).unlink(missing_ok=True)
```

`put_if_absent` writes the bytes to a uniquely named file under `staging/` and hard-links it to the key's path under `objects/`. `link()` fails with `FileExistsError` when the target exists, and the linked file already holds every byte, so writes are create-once and objects are whole.

Opening the target with `O_CREAT | O_EXCL` and then writing would also create once, but the file is visible, empty, before its bytes are written. In a test where eight processes raced to create the same 500 keys with 64 KiB bodies, each reading the key back after its attempt, a reader saw an incomplete body in four of six runs of the `O_EXCL` version. In eight runs of the version above, no reader did, and every key had exactly one winner, whose body was stored.

`list` walks only the deepest directory the prefix names, then sorts whole keys rather than relying on the walk's order: in bytes, `a-c` comes before `a/b` and `a/b` before `a0`.

## Using it in a run

A backend is passed wherever a mailbox is expected:

```python
box = FileMailbox("/var/tmp/gradientmail/run-1")

with gm.Session(
    model,
    algorithm=gm.DiLoCo(sync_every=100, fragments=8),
    mailbox=box,
) as session:
    for batch in dataloader:
        train_step(model, optimizer, batch)
        session.step(samples=len(batch))
```

Every process of the run opens a `FileMailbox` on the same directory, and a new run uses a new directory. `gm.align(model, mailbox=box)` loads the newest committed parameters into a model without joining the run, for evaluation.

As an end-to-end check, two processes trained on one directory with the setup of [`examples/synthetic.py`](../../examples/synthetic.py), in which each commit must equal a token-weighted average computed from its round record. In three runs of 400 and 100 steps, they committed 34 to 37 rounds, 30 to 33 of them with both nodes. A third process checked every commit as it landed, and each matched; `gm.align` in that process then loaded the newest commit of each fragment.

## Running the test suite

[`tests/conftest.py`](../../tests/conftest.py) parametrizes the `mailbox` fixture over `MAILBOXES`, a tuple of named generators that each yield a fresh store, so every test that requests the fixture runs once per store. Those are the six contract tests in [`test_mailbox.py`](../../tests/test_mailbox.py), four Shared Queue tests and five two-learner training runs. To include the filesystem backend, place `file_mailbox.py` in `tests/`, add this generator to `conftest.py` with its imports and append `("filesystem", filesystem)` to `MAILBOXES`:

```python
def filesystem() -> Iterator[Mailbox]:
    with tempfile.TemporaryDirectory() as root:
        yield FileMailbox(root)
```

`uv run pytest -k filesystem` then selects those 15 tests, and all of them pass with the backend above. The contract tests write 1,001 keys, one more than an S3 listing page or bulk delete holds, and check byte order with keys whose case-folded order differs. The training tests fail when the expected rounds have not committed within 30 seconds; `GRADIENTMAIL_TEST_TIMEOUT` raises that limit for a slower store. A store that outlives the test should give each test its own prefix and remove it afterwards, as `s3_endpoint` does.

## Limits of the example

The backend was tested on APFS on macOS, with every process on one machine. It was not tested on network filesystems such as NFS or SMB, whose caching and `link()` behavior may not provide read-after-write visibility or create-once writes across machines.

It does not call `fsync`. A process that crashes leaves each object whole or absent, but nothing is promised about what survives a power loss or a kernel crash. A process killed between staging and linking leaves a file in `staging/` that is never listed or removed.

`delete` removes files but not the directories that held them; two-node runs of 37 rounds left about 180 empty directories each. A listing walks only the directory its prefix names, but `list("")` walks all of them.

A key cannot also be a directory of another key: once `a` exists, writing `a/b` raises `FileExistsError`. GradientMail's own keys never overlap this way; Shared Queue identifiers, which become keys under `units/`, should not either.
