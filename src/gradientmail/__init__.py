"""GradientMail: distributed training through asynchronous message exchange.

``import gradientmail`` needs only the standard library and safetensors. ``Session`` runs a
node's training loop: ``join`` joins the run, ``run(step, data)`` trains ``step`` over the data
and returns once the node's last rounds are committed. By default every session also takes
part in closing rounds, so a run has no separate process to operate; a session's ``role`` can
instead make it a learner that only trains, or the run's exclusive syncer, which only closes
rounds and to which the other nodes leave every round while it lives, or one that trains until
the run elects it its syncer, so that a pool of two or more like nodes keeps one. The data is
any iterable or a ``SharedQueue``, units of work the run's nodes claim from one another. With
``ranks`` the nodes split fixed slices of the data, each holding one rank at a time, and a node
that finds its rank held by another raises ``RankLost``. ``Barrier`` holds nodes at a named
point until enough of them arrive. What a node reports while it runs is in
``gradientmail.events``, handed to the session's ``callbacks``. ``S3Mailbox`` is resolved on
first access and raises ``MissingExtra`` naming the ``s3`` extra when boto3 is absent. The
framework backends live in ``gradientmail.adapters.torch`` and ``gradientmail.adapters.mlx``.
"""

from typing import TYPE_CHECKING

from gradientmail.adapters import Adapter, Backend, MemoryAdapter, MissingExtra
from gradientmail.algorithms import Adaptive, Algorithm, Blend, Carry, Overwrite
from gradientmail.algorithms.diloco import ByBytes, DiLoCo, Rda, Strided, WeightedMean
from gradientmail.barrier import Barrier
from gradientmail.events import Callback, Event, log_events
from gradientmail.learner import RankLost
from gradientmail.mailbox import Mailbox, MailboxUnavailable
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import RunConfigMismatch
from gradientmail.queue import SharedQueue
from gradientmail.serialization import Schema, SchemaMismatch, TensorBytes
from gradientmail.session import Role, Session, align

if TYPE_CHECKING:
    from gradientmail.mailbox.s3 import S3Mailbox as S3Mailbox

__all__ = [
    "Adapter",
    "Adaptive",
    "Algorithm",
    "Backend",
    "Barrier",
    "Blend",
    "ByBytes",
    "Callback",
    "Carry",
    "DiLoCo",
    "Event",
    "Mailbox",
    "MailboxUnavailable",
    "MemoryAdapter",
    "MemoryMailbox",
    "MissingExtra",
    "Overwrite",
    "RankLost",
    "Rda",
    "Role",
    "RunConfigMismatch",
    "Schema",
    "SchemaMismatch",
    "Session",
    "SharedQueue",
    "Strided",
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
