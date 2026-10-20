"""What a node reports while it runs, as values, and the callbacks that receive them.

Every report the learner and the syncer make is an ``Event``: one frozen dataclass per kind,
each naming the ``node`` it happened on, since one process may host several nodes. A
``Callback`` receives every event of its session on the thread that produced it: the caller's,
which joins and runs the training loop, for the join and the rank, aligning and closing and an
``either`` node's switches between training and closing rounds, the communication thread for
publishing, polling and the lease, the syncer thread for rounds and an exclusive syncer's
lease. One
callback may therefore run on several threads at once, and one that keeps state guards it. The
last two threads run event loops, so a callback that blocks stalls the node's polling or its
rounds, as a slow logging handler does; one that does I/O hands the event to a thread of its
own. ``combined`` makes one callback of several; ``log_events`` is the one a session runs by
default.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, assert_never

from gradientmail.protocol import RoundRecord, RoundRecordDraft

__all__ = [
    "AwaitingCreator",
    "BecameSyncer",
    "Callback",
    "Closed",
    "Closing",
    "CreatingRun",
    "Event",
    "FollowingRound",
    "FragmentAligned",
    "GaveUp",
    "Joined",
    "LearnerEvent",
    "Lease",
    "LeaseReleased",
    "PollFailed",
    "PublishCancelled",
    "PublishFailed",
    "RankClaimed",
    "Recovered",
    "RecoveryFailed",
    "ReleaseFailed",
    "ResumedTraining",
    "RoundClaimed",
    "RoundClosed",
    "RoundCommitted",
    "RoundFailed",
    "RoundRecorded",
    "RoundSuperseded",
    "SnapshotPublished",
    "SyncerEvent",
    "SyncerStopped",
    "Takeover",
    "TookOverCreation",
    "TookOverRound",
    "combined",
    "log_events",
]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CreatingRun:
    """This node holds the creation claim and is uploading Θ⁰ of ``fragments`` fragments."""

    node: str
    fragments: int


@dataclass(frozen=True)
class AwaitingCreator:
    """``claimants`` hold the creation claim; this node polls for the manifest."""

    node: str
    claimants: tuple[str, ...]


@dataclass(frozen=True)
class TookOverCreation:
    """No init payload landed for ``stalled`` seconds; this node took the next creation claim."""

    node: str
    stalled: float


@dataclass(frozen=True)
class Joined:
    """The model holds the newest committed version of every fragment of the run ``created_by``
    created; ``round`` is the newest committed round, -1 on a fresh run."""

    node: str
    created_by: str
    fragments: int
    round: int


@dataclass(frozen=True)
class RankClaimed:
    """This node holds ``rank``: its claim is the first of the rank's whose node holds a live
    lease."""

    node: str
    rank: int


@dataclass(frozen=True)
class FragmentAligned:
    """``fragment`` now holds ``version``, committed by ``round``, applied at ``local_step``;
    ``lag_steps`` is how many steps that is after this learner's snapshot in the round, None
    when it took no part in it."""

    node: str
    round: int
    fragment: int
    version: int
    local_step: int
    lag_steps: int | None


@dataclass(frozen=True)
class SnapshotPublished:
    """The delta of ``fragment`` for ``round``, ``steps`` steps and ``tokens`` tokens, is in the
    store: ``size`` bytes in ``seconds``."""

    node: str
    round: int
    fragment: int
    steps: int
    tokens: int
    size: int
    seconds: float


@dataclass(frozen=True)
class PublishFailed:
    """Publishing ``round`` failed; its steps stay counted and go out with the fragment's next
    round."""

    node: str
    round: int
    error: Exception


@dataclass(frozen=True)
class PollFailed:
    """A learner's receive iteration, or an exclusive syncer's election, lease renewal or check
    for the end of the run, failed; the next poll retries it."""

    node: str
    error: Exception


@dataclass(frozen=True)
class RoundRecorded:
    """The receive loop listed ``record``, whichever node committed it. Every node reports every
    round committed after it joined, whether or not it aligns with that version: a newer one of
    the same fragment in the same listing replaces it."""

    node: str
    record: RoundRecord


@dataclass(frozen=True)
class Closing:
    """``close()`` began: publishing the steps since each fragment's last round when
    ``flushing``, otherwise waiting for the rounds in flight."""

    node: str
    flushing: bool


@dataclass(frozen=True)
class Closed:
    """Every round this learner published, up to ``round``, is committed and the model is
    aligned with it."""

    node: str
    round: int


@dataclass(frozen=True)
class GaveUp:
    """``close()`` stopped waiting for ``round`` after nothing landed for ``waited`` seconds;
    the model is aligned up to round ``aligned``."""

    node: str
    round: int
    waited: float
    aligned: int


@dataclass(frozen=True)
class PublishCancelled:
    """The communication thread was still publishing ``waited`` seconds after the last rounds
    and was cancelled."""

    node: str
    waited: float


type Lease = Literal["learner", "syncer"]


@dataclass(frozen=True)
class LeaseReleased:
    """The node deleted its ``lease``: the learner's, or the exclusive syncer's."""

    node: str
    lease: Lease


@dataclass(frozen=True)
class ReleaseFailed:
    node: str
    lease: Lease
    error: Exception


@dataclass(frozen=True)
class Recovered:
    """The syncer rebuilt a closer's state from the store; ``round`` is the next to run."""

    node: str
    round: int


@dataclass(frozen=True)
class RecoveryFailed:
    """Rebuilding a closer's state failed; it is retried in ``retry_in`` seconds."""

    node: str
    error: Exception
    retry_in: float


@dataclass(frozen=True)
class RoundClaimed:
    """This node made the first claim on ``round`` and closes it."""

    node: str
    round: int


@dataclass(frozen=True)
class FollowingRound:
    """``claimants`` hold ``round``; this node waits for its record and takes over once the last
    claimant's lease ends or no record lands within ``timeout`` seconds."""

    node: str
    round: int
    claimants: tuple[str, ...]
    timeout: float


type Takeover = Literal["lease_gone", "no_record"]


@dataclass(frozen=True)
class TookOverRound:
    """This node took the next claim on ``round``: the last of ``claimants`` lost its lease, or
    no record landed within ``timeout`` seconds."""

    node: str
    round: int
    claimants: tuple[str, ...]
    cause: Takeover
    timeout: float


@dataclass(frozen=True)
class RoundClosed:
    """This node closed ``draft.round`` and commits it next; the participants with their weights,
    the rejected snapshots and the realized waits are in ``draft``."""

    node: str
    draft: RoundRecordDraft


@dataclass(frozen=True)
class RoundCommitted:
    """This node wrote ``record``, its payloads and the record itself in ``seconds``."""

    node: str
    record: RoundRecord
    seconds: float


@dataclass(frozen=True)
class RoundSuperseded:
    """The store holds another node's record of a round this node committed or tried to: the
    other committed first, or a store that accepted both writes kept the other's. ``record``
    is theirs and this node continues from it."""

    node: str
    record: RoundRecord


@dataclass(frozen=True)
class RoundFailed:
    """Closing or following ``round`` failed; it is retried as after a restart in ``retry_in``
    seconds."""

    node: str
    round: int
    error: Exception
    retry_in: float


@dataclass(frozen=True)
class BecameSyncer:
    """The run elected this ``either`` node its syncer: it trained the unit in progress to its
    end, gave back the units it had claimed and not started, closed its learner and now only
    closes rounds."""

    node: str


@dataclass(frozen=True)
class ResumedTraining:
    """Another node owns the syncer role, as a dedicated syncer that arrives does: this node,
    elected before, released its syncer lease and trains again, its model aligned with
    ``round``, the newest committed round."""

    node: str
    round: int


@dataclass(frozen=True)
class SyncerStopped:
    """The syncer thread died of ``error``; ``Session.run`` raises it after the next step."""

    node: str
    error: BaseException


type LearnerEvent = (
    CreatingRun
    | AwaitingCreator
    | TookOverCreation
    | Joined
    | RankClaimed
    | FragmentAligned
    | SnapshotPublished
    | PublishFailed
    | PollFailed
    | RoundRecorded
    | Closing
    | Closed
    | GaveUp
    | PublishCancelled
    | LeaseReleased
    | ReleaseFailed
    | ResumedTraining
)
type SyncerEvent = (
    Recovered
    | RecoveryFailed
    | RoundClaimed
    | FollowingRound
    | TookOverRound
    | RoundClosed
    | RoundCommitted
    | RoundSuperseded
    | RoundFailed
    | BecameSyncer
    | SyncerStopped
    | PollFailed
    | LeaseReleased
    | ReleaseFailed
)
type Event = LearnerEvent | SyncerEvent
type Callback = Callable[[Event], None]


def combined(*callbacks: Callback) -> Callback:
    """One callback that hands every event to each of ``callbacks``, in order.

    A callback that raises is logged and the rest still receive the event: observing a run
    never changes it, so a failing sink cannot fail a publish or retry a round.
    """

    def notify(event: Event) -> None:
        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                log.exception("%s: a callback failed on %s", event.node, type(event).__name__)

    return notify


def log_events(event: Event) -> None:
    """One line per event on the ``gradientmail.events`` logger: failures at ERROR with their
    traceback, takeovers and give-ups at WARNING, the rest at INFO. ``RoundRecorded`` is not
    logged; the alignment or the commit it leads to is."""
    match event:
        case CreatingRun(node, fragments):
            log.info("%s: creating the run: uploading Θ⁰ of %d fragments", node, fragments)
        case AwaitingCreator(node, claimants):
            log.info("%s: waiting for %s to create the run", node, ", ".join(claimants))
        case TookOverCreation(node, stalled):
            log.warning("%s: no init payload landed for %.0f s; creating the run", node, stalled)
        case Joined(node, created_by, fragments, round):
            log.info(
                "%s: joined the run created by %s: %d fragments, newest round %d",
                node,
                created_by,
                fragments,
                round,
            )
        case RankClaimed(node, rank):
            log.info("%s: holds rank %d", node, rank)
        case FragmentAligned(node, round, fragment, version, local_step, lag_steps):
            log.info(
                "%s: fragment %d aligned with version %d (round %d), at local step %d%s",
                node,
                fragment,
                version,
                round,
                local_step,
                "" if lag_steps is None else f", {lag_steps} steps after its snapshot",
            )
        case SnapshotPublished(node, round, fragment, steps, tokens, size, seconds):
            log.info(
                "%s: round %d published: fragment %d, %d steps, %d tokens, %d bytes, %.1f s",
                node,
                round,
                fragment,
                steps,
                tokens,
                size,
                seconds,
            )
        case PublishFailed(node, round, error):
            log.error(
                "%s: round %d: publishing failed; its steps go out with the fragment's next round",
                node,
                round,
                exc_info=error,
            )
        case PollFailed(node, error):
            log.error("%s: poll failed; retrying", node, exc_info=error)
        case RoundRecorded():
            pass
        case Closing(node, flushing):
            log.info(
                "%s: closing: %s",
                node,
                "publishing the steps since the last rounds"
                if flushing
                else "waiting for the rounds in flight",
            )
        case Closed(node, round):
            log.info("%s: closed: aligned with every round up to %d", node, round)
        case GaveUp(node, round, waited, aligned):
            log.warning(
                "%s: giving up on round %d: nothing landed for %.0f s; aligned up to round %d",
                node,
                round,
                waited,
                aligned,
            )
        case PublishCancelled(node, waited):
            log.warning(
                "%s: still publishing %.0f s after the last rounds; cancelling", node, waited
            )
        case LeaseReleased(node, lease):
            log.info("%s: %s lease released", node, lease)
        case ReleaseFailed(node, lease, error):
            log.error("%s: releasing the %s lease failed", node, lease, exc_info=error)
        case Recovered(node, round):
            log.info("%s: recovered: next round %d", node, round)
        case RecoveryFailed(node, error, retry_in):
            log.error("%s: recovery failed; retrying in %.1f s", node, retry_in, exc_info=error)
        case RoundClaimed(node, round):
            log.info("%s: claimed round %d", node, round)
        case FollowingRound(node, round, claimants, timeout):
            log.info(
                "%s: following round %d claimed by %s; will take over if no record lands in "
                "%.0f s or the claimant's lease ends",
                node,
                round,
                ", ".join(claimants),
                timeout,
            )
        case TookOverRound(node, round, claimants, cause, timeout):
            why = (
                f"{claimants[-1]}'s lease is gone"
                if cause == "lease_gone"
                else f"no record after {timeout:.0f} s"
            )
            log.warning("%s: taking over round %d: %s", node, round, why)
        case RoundClosed(node, draft):
            log.info(
                "%s: round %d closed: fragment %d, participants %s, rejected %s, quorum %.1f s, "
                "grace %.1f s%s",
                node,
                draft.round,
                draft.fragment,
                [p.learner for p in draft.participants],
                [f"{rejected} ({why})" for rejected, why in draft.rejected],
                draft.quorum_seconds,
                draft.grace_seconds,
                "" if draft.cosine is None else f", cosine {draft.cosine:.3f}",
            )
        case RoundCommitted(node, record, seconds):
            log.info(
                "%s: round %d committed: version %d, %d bytes, %.1f s",
                node,
                record.round,
                record.version,
                record.theta.size
                + record.outer.size
                + (0 if record.update is None else record.update.payload.size),
                seconds,
            )
        case RoundSuperseded(node, record):
            log.info("%s: round %d was committed by another node", node, record.round)
        case RoundFailed(node, round, error, retry_in):
            log.error(
                "%s: round %d failed; retrying in %.1f s", node, round, retry_in, exc_info=error
            )
        case BecameSyncer(node):
            log.info("%s: elected the run's syncer; training stopped", node)
        case ResumedTraining(node, round):
            log.info(
                "%s: another node holds the syncer role; training resumed at round %d", node, round
            )
        case SyncerStopped(node, error):
            log.error("%s: the syncer stopped; the run raises it", node, exc_info=error)
        case _:
            assert_never(event)
