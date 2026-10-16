"""In-memory Mailbox for development, tests and the Phase 2 example."""

import threading
from collections.abc import Iterable, Sequence

from gradientmail.mailbox import Mailbox


class MemoryMailbox(Mailbox):
    """A dict behind a ``threading.Lock``.

    The lock is not an asyncio primitive because one instance is shared between event loops
    running on different threads (a ``Session``'s background loop and a ``Syncer``).
    """

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}
        self._lock = threading.Lock()

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        with self._lock:
            if key in self._store:
                return False
            self._store[key] = data
            return True

    async def get(self, key: str) -> bytes | None:
        with self._lock:
            return self._store.get(key)

    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]:
        # Code point order is UTF-8 byte order, so str comparison matches S3's listing order.
        with self._lock:
            return sorted(k for k in self._store if k.startswith(prefix) and k > after)

    async def delete(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                self._store.pop(key, None)
