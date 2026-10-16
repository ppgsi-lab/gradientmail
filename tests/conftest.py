"""Shared fixtures.

``mailbox`` runs every test that requests it once per store in ``MAILBOXES``; a store is a
generator that yields one fresh mailbox and may ``pytest.skip`` when its backend is absent.
"""

import asyncio
import contextlib
import os
import uuid
from collections.abc import Callable, Iterator
from typing import Literal

import boto3.session
import pytest
from botocore.config import Config
from fakes import FakeClock
from moto import mock_aws

from gradientmail.mailbox import Mailbox
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.mailbox.s3 import S3Mailbox


def memory() -> Iterator[Mailbox]:
    yield MemoryMailbox()


def s3_moto() -> Iterator[Mailbox]:
    # moto intercepts only clients created while the mock is active, so both the bucket
    # setup client and the mailbox's own client are built inside the context.
    with mock_aws():
        boto3.session.Session().client("s3", region_name="us-east-1").create_bucket(
            Bucket="gradientmail"
        )
        yield S3Mailbox("gradientmail", "run", region="us-east-1")


def s3_endpoint() -> Iterator[Mailbox]:
    """The S3-compatible store at ``GRADIENTMAIL_S3_ENDPOINT``: MinIO, R2, HF Storage Buckets.

    Credentials come from boto3's default chain. ``GRADIENTMAIL_S3_CHECKSUMS=when_required``
    selects the checksum mode for stores that reject flexible checksums.
    """
    endpoint = os.environ.get("GRADIENTMAIL_S3_ENDPOINT")
    if not endpoint:
        pytest.skip("GRADIENTMAIL_S3_ENDPOINT is not set")
    checksums: Literal["when_supported", "when_required"] = (
        "when_required"
        if os.environ.get("GRADIENTMAIL_S3_CHECKSUMS") == "when_required"
        else "when_supported"
    )
    setup = boto3.session.Session().client(
        "s3",
        endpoint_url=endpoint,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, request_checksum_calculation=checksums),
    )
    with contextlib.suppress(
        setup.exceptions.BucketAlreadyOwnedByYou, setup.exceptions.BucketAlreadyExists
    ):
        setup.create_bucket(Bucket="gradientmail-test")
    # The store outlives the test, so every test gets its own prefix and sweeps it.
    mailbox = S3Mailbox(
        "gradientmail-test",
        f"test-{uuid.uuid4().hex}",
        endpoint_url=endpoint,
        region="us-east-1",
        request_checksums=checksums,
    )
    yield mailbox

    async def sweep() -> None:
        await mailbox.delete(await mailbox.list(""))

    asyncio.run(sweep())


MAILBOXES: tuple[tuple[str, Callable[[], Iterator[Mailbox]]], ...] = (
    ("memory", memory),
    ("s3-moto", s3_moto),
    ("s3-endpoint", s3_endpoint),
)


@pytest.fixture(params=[factory for _, factory in MAILBOXES], ids=[name for name, _ in MAILBOXES])
def mailbox(request: pytest.FixtureRequest) -> Iterator[Mailbox]:
    yield from request.param()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
