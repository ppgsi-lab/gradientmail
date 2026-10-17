"""GradientMail: distributed training through asynchronous message exchange.

``import gradientmail`` needs only the standard library and safetensors. ``Session`` wraps a
training loop; every session also takes part in closing rounds, so a run has no separate
process to operate. What a node reports while it runs is in ``gradientmail.events``, handed
to the session's ``callbacks``. ``S3Mailbox`` is resolved on first access and raises
``MissingExtra`` naming the ``s3`` extra when boto3 is absent. The framework backends live in
``gradientmail.adapters.torch`` and ``gradientmail.adapters.mlx``.
"""

from typing import TYPE_CHECKING

from gradientmail.adapters import Adapter, Backend, MemoryAdapter, MissingExtra
from gradientmail.algorithms import Algorithm
from gradientmail.algorithms.diloco import DiLoCo, Rda, WeightedMean
from gradientmail.events import Callback, Event, log_events
from gradientmail.mailbox import Mailbox, MailboxUnavailable
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import RunConfigMismatch
from gradientmail.serialization import Schema, SchemaMismatch, TensorBytes
from gradientmail.session import Session, align

if TYPE_CHECKING:
    from gradientmail.mailbox.s3 import S3Mailbox as S3Mailbox

__all__ = [
    "Adapter",
    "Algorithm",
    "Backend",
    "Callback",
    "DiLoCo",
    "Event",
    "Mailbox",
    "MailboxUnavailable",
    "MemoryAdapter",
    "MemoryMailbox",
    "MissingExtra",
    "Rda",
    "RunConfigMismatch",
    "Schema",
    "SchemaMismatch",
    "Session",
    "TensorBytes",
    "WeightedMean",
    "align",
    "log_events",
]


def __getattr__(name: str) -> object:
    if name == "S3Mailbox":
        try:
            from gradientmail.mailbox.s3 import S3Mailbox
        except ImportError as exc:
            raise MissingExtra("s3") from exc
        return S3Mailbox
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
