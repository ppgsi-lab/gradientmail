"""Mailbox contract: one suite, run against every store through the ``mailbox`` fixture.

The boundary under test is the store's semantics (create-once writes, byte-ordered paging,
idempotent bulk delete), so nothing here is mocked.
"""

import asyncio
import itertools
import threading
import time
from collections.abc import Sequence
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fakes import FakeClock, FaultyMailbox

from gradientmail.mailbox import Mailbox, MailboxUnavailable, retrying
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.mailbox.s3 import S3Mailbox


def describe_mailbox():
    def describe_put_if_absent():
        def it_keeps_the_first_body_and_returns_false_for_the_second(mailbox: Mailbox):
            async def scenario() -> tuple[bool, bool, bytes | None]:
                first = await mailbox.put_if_absent("k", b"first")
                second = await mailbox.put_if_absent("k", b"second")
                return first, second, await mailbox.get("k")

            assert asyncio.run(scenario()) == (True, False, b"first")

    def describe_list():
        def it_returns_keys_in_byte_order_across_pages(mailbox: Mailbox):
            # 1,001 keys cross S3's page size; "0" < "Z" < "_x" < "a" holds in bytes but not
            # under case folding, and zero padding makes numeric order equal byte order.
            groups = ("0", "Z", "_x", "a")
            keys = [f"order/{groups[i % len(groups)]}{i:04d}" for i in range(1001)]

            async def scenario() -> Sequence[str]:
                async with asyncio.TaskGroup() as group:
                    for key in keys:
                        group.create_task(mailbox.put_if_absent(key, key.encode()))
                return await mailbox.list("order/")

            assert list(asyncio.run(scenario())) == sorted(keys, key=str.encode)

        def it_excludes_the_after_key(mailbox: Mailbox):
            async def scenario() -> Sequence[str]:
                for key in (
                    "rounds/0000000001.json",
                    "rounds/0000000002.json",
                    "rounds/0000000003.json",
                ):
                    await mailbox.put_if_absent(key, b"{}")
                return await mailbox.list("rounds/", after="rounds/0000000002.json")

            assert list(asyncio.run(scenario())) == ["rounds/0000000003.json"]

        def it_is_empty_for_a_missing_prefix(mailbox: Mailbox):
            async def scenario() -> Sequence[str]:
                await mailbox.put_if_absent("present/1", b"x")
                return await mailbox.list("absent/")

            assert list(asyncio.run(scenario())) == []

    def describe_get():
        def it_returns_none_for_a_missing_key(mailbox: Mailbox):
            assert asyncio.run(mailbox.get("missing")) is None

    def describe_delete():
        def it_is_idempotent_and_accepts_more_than_1000_keys(mailbox: Mailbox):
            keys = [f"blobs/{i:04d}" for i in range(1001)]

            async def scenario() -> tuple[Sequence[str], Sequence[str]]:
                async with asyncio.TaskGroup() as group:
                    for key in keys:
                        group.create_task(mailbox.put_if_absent(key, b"x"))
                await mailbox.delete([*keys, "blobs/never-written"])
                after_first = await mailbox.list("blobs/")
                await mailbox.delete(keys)
                return after_first, await mailbox.list("blobs/")

            after_first, after_second = asyncio.run(scenario())

            assert list(after_first) == []
            assert list(after_second) == []


def describe_s3_mailbox():
    def it_raises_mailbox_unavailable_when_the_endpoint_is_unreachable(
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Credentials are needed to sign the request, which happens before the connection
        # attempt; port 1 on loopback refuses the connection at once. The test still takes
        # several seconds: botocore's standard retry mode makes five attempts with random
        # exponential backoff before the error reaches the mailbox.
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unused")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unused")
        mailbox = S3Mailbox("bucket", "run", endpoint_url="http://127.0.0.1:1", region="us-east-1")

        with pytest.raises(MailboxUnavailable):
            asyncio.run(mailbox.get("rounds/0000000000.json"))

    def it_reads_the_store_clock_from_the_date_of_its_responses(monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unused")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unused")
        ahead = 3600.0

        class AheadStore(BaseHTTPRequestHandler):
            """Answers every GET with NoSuchKey, dated an hour ahead of this machine."""

            def do_GET(self) -> None:
                body = b"<Error><Code>NoSuchKey</Code><Message>absent</Message></Error>"
                self.send_response(404)
                self.send_header("Content-Type", "application/xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def date_time_string(self, timestamp: float | None = None) -> str:
                return formatdate(time.time() + ahead, usegmt=True)

            def log_message(self, format: str, *args: object) -> None:
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), AheadStore)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            mailbox = S3Mailbox(
                "bucket",
                "run",
                endpoint_url=f"http://127.0.0.1:{server.server_port}",
                region="us-east-1",
            )
            before = mailbox.clock().now() - time.time()
            assert asyncio.run(mailbox.get("k")) is None
            after = mailbox.clock().now() - time.time()
        finally:
            server.shutdown()
            server.server_close()

        assert abs(before) < 0.1
        assert abs(after - ahead) < 1.0


def describe_retrying():
    def it_retries_a_lost_response_through_the_clock_and_returns_the_retried_result(
        clock: FakeClock,
    ):
        inner = MemoryMailbox()
        faulty = FaultyMailbox(inner, fail_after=1)

        async def scenario() -> tuple[bool, bytes | None]:
            created = await retrying(lambda: faulty.put_if_absent("k", b"v"), clock=clock)
            return created, await inner.get("k")

        assert asyncio.run(scenario()) == (False, b"v")
        assert clock.now() > 0

    def it_backs_off_exponentially_up_to_max_delay(clock: FakeClock):
        attempts: list[float] = []

        async def flaky() -> str:
            attempts.append(clock.now())
            if len(attempts) <= 6:
                raise MailboxUnavailable()
            return "ok"

        result = asyncio.run(retrying(flaky, clock=clock, max_delay=4.0))
        delays = [later - earlier for earlier, later in itertools.pairwise(attempts)]

        assert result == "ok"
        assert all(0 < delay <= 4.0 for delay in delays)
        assert delays == sorted(delays)
        assert delays[0] < delays[-1] == 4.0

    def it_propagates_any_other_error_without_retrying(clock: FakeClock):
        attempts: list[float] = []

        async def broken() -> None:
            attempts.append(clock.now())
            raise ValueError("not transient")

        with pytest.raises(ValueError, match="not transient"):
            asyncio.run(retrying(broken, clock=clock))

        assert len(attempts) == 1
