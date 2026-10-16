"""Test doubles that are real implementations of the core protocols."""

import asyncio
from collections.abc import Iterable, Sequence

from gradientmail.clock import Clock
from gradientmail.mailbox import Mailbox, MailboxUnavailable


class FakeClock(Clock):
    """A clock that moves only when asked: ``sleep`` advances it and yields once to the loop."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self._now += seconds
        await asyncio.sleep(0)

    def advance(self, seconds: float) -> None:
        self._now += seconds


class FaultyMailbox(Mailbox):
    """Loses the response of exactly one operation.

    The first ``fail_after`` operations reach ``inner``; the last of them raises
    ``MailboxUnavailable`` after taking effect, as a timed-out request whose write
    landed. Every later operation is delegated normally, so a retry observes the store
    as the crashed caller left it.
    """

    def __init__(self, inner: Mailbox, fail_after: int) -> None:
        self._inner = inner
        self._fail_after = fail_after
        self._operations = 0

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        created = await self._inner.put_if_absent(key, data)
        self._tick()
        return created

    async def get(self, key: str) -> bytes | None:
        data = await self._inner.get(key)
        self._tick()
        return data

    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]:
        keys = await self._inner.list(prefix, after=after)
        self._tick()
        return keys

    async def delete(self, keys: Iterable[str]) -> None:
        await self._inner.delete(keys)
        self._tick()

    def _tick(self) -> None:
        self._operations += 1
        if self._operations == self._fail_after:
            raise MailboxUnavailable(f"response of operation {self._operations} lost")
