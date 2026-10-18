"""Transport: four operations over immutable, create-once objects, and the store's clock.

An object is written at most once (``put_if_absent``), read back by key (``get``), discovered
by prefix in UTF-8 byte order (``list``) and removed idempotently (``delete``). ``clock`` is the
store's time as this node estimates it, the time base in which nodes agree on an instant.
Nothing here knows about rounds or snapshots; ``gradientmail.protocol`` builds the exchange on
top.
"""

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Protocol, runtime_checkable

from gradientmail.clock import Clock

log = logging.getLogger(__name__)


class MailboxUnavailable(Exception):
    """Transient transport failure (timeouts, dropped connections); callers retry."""


@runtime_checkable
class Mailbox(Protocol):
    async def put_if_absent(self, key: str, data: bytes) -> bool:
        """Create ``key`` holding ``data``; True iff this call created it.

        An existing object is left untouched and False is returned.
        """
        ...

    async def get(self, key: str) -> bytes | None:
        """The bytes under ``key``, or None when no such object exists."""
        ...

    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]:
        """Every key under ``prefix`` greater than ``after``, in UTF-8 byte order, all pages."""
        ...

    async def delete(self, keys: Iterable[str]) -> None:
        """Remove ``keys`` in any number; missing keys are not an error."""
        ...

    def clock(self) -> Clock:
        """The store's time, as closely as this node can tell; every node reads it alike."""
        ...


async def retrying[T](
    op: Callable[[], Awaitable[T]], *, clock: Clock, max_delay: float = 30.0
) -> T:
    """Run ``op`` until it returns, sleeping through ``clock`` after each ``MailboxUnavailable``.

    Delays start at 0.5 s and double up to ``max_delay``. Any other exception propagates at once.
    """
    delay = min(0.5, max_delay)
    while True:
        try:
            return await op()
        except MailboxUnavailable as exc:
            log.warning("mailbox unavailable: %s; retrying in %.1f s", exc, delay)
            await clock.sleep(delay)
            delay = min(delay * 2, max_delay)
