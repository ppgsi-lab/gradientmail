"""What a node reports while it runs, as values, and the callbacks that receive them.

Every report the learner and the syncer make is an ``Event``: one frozen dataclass per kind,
each naming the ``learner`` it happened on, since one process may host several nodes. A
``Callback`` receives every event of its session on the thread that produced it: the caller's
for the join, the training thread for installs and closing, the communication thread for
publishing, polling and the lease, the syncer thread for rounds. One callback may therefore run
on several threads at once, and one that keeps state guards it. The last two threads run event
loops, so a callback that blocks stalls the node's polling or its rounds, as a slow logging
handler does; one that does I/O hands the event to a thread of its own. ``combined`` makes one
callback of several; ``log_events`` is the one a session runs by default.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, assert_never

from gradientmail.protocol import RoundRecord, RoundRecordDraft

__all__ = [
    "AwaitingCreator",
    "Callback",
    "Closed",
    "Closing",
    "CreatingRun",
    "Event",
    "FollowingRound",
    "FragmentInstalled",
    "GaveUp",
    "Joined",
    "LearnerEvent",
    "LeaseReleased",
    "PollFailed",
    "PublishCancelled",
    "PublishFailed",
    "Recovered",
    "RecoveryFailed",
    "ReleaseFailed",
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

    learner: str
    fragments: int


@dataclass(frozen=True)
class AwaitingCreator:
    """``claimants`` hold the creation claim; this node polls for the manifest."""

    learner: str
    claimants: tuple[str, ...]


@dataclass(frozen=True)
class TookOverCreation:
    """No init payload landed for ``stalled`` seconds; this node took the next creation claim."""

    learner: str
    stalled: float


@dataclass(frozen=True)
class Joined:
    """The model holds the newest committed version of every fragment of the run ``created_by``
    created; ``round`` is the newest committed round, -1 on a fresh run."""

    learner: str
    created_by: str
    fragments: int
    round: int


@dataclass(frozen=True)
class FragmentInstalled:
    """``fragment`` now holds ``version``, committed by ``round``, applied at ``local_step``."""

    learner: str
    round: int
    fragment: int
    version: int
    local_step: int


@dataclass(frozen=True)
class SnapshotPublished:
    """The delta of ``fragment`` for ``round``, ``steps`` steps and ``tokens`` tokens, is in the
    store: ``size`` bytes in ``seconds``."""

    learner: str
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

    learner: str
    round: int
    error: Exception


@dataclass(frozen=True)
class PollFailed:
    """A receive iteration failed; the next poll retries it."""

    learner: str
    error: Exception


@dataclass(frozen=True)
class RoundRecorded:
    """The receive loop listed ``record``, whichever node committed it. Every node reports every
    round committed after it joined, whether or not it installs that version: a newer one of
    the same fragment in the same listing replaces it."""

    learner: str
    record: RoundRecord


@dataclass(frozen=True)
class Closing:
    """``close()`` began: publishing the steps since each fragment's last round when
    ``flushing``, otherwise waiting for the rounds in flight."""

    learner: str
    flushing: bool


@dataclass(frozen=True)
class Closed:
    """Every round this learner published, up to ``round``, is committed and installed."""

    learner: str
    round: int


@dataclass(frozen=True)
class GaveUp:
    """``close()`` stopped waiting for ``round`` after nothing landed for ``waited`` seconds;
    the newest installed round is ``installed``."""

    learner: str
    round: int
    waited: float
    installed: int


@dataclass(frozen=True)
class PublishCancelled:
    """The communication thread was still publishing ``waited`` seconds after the last rounds
    and was cancelled."""

    learner: str
    waited: float


@dataclass(frozen=True)
class LeaseReleased:
    learner: str


@dataclass(frozen=True)
class ReleaseFailed:
    learner: str
    error: Exception


@dataclass(frozen=True)
class Recovered:
    """The syncer rebuilt a closer's state from the store; ``round`` is the next to run."""

    learner: str
    round: int


@dataclass(frozen=True)
class RecoveryFailed:
    """Rebuilding a closer's state failed; it is retried in ``retry_in`` seconds."""

    learner: str
    error: Exception
    retry_in: float


@dataclass(frozen=True)
class RoundClaimed:
    """This node made the first claim on ``round`` and closes it."""

    learner: str
    round: int


@dataclass(frozen=True)
class FollowingRound:
    """``claimants`` hold ``round``; this node waits for its record and takes over once the last
    claimant's lease ends or no record lands within ``timeout`` seconds."""

    learner: str
    round: int
    claimants: tuple[str, ...]
    timeout: float


type Takeover = Literal["lease_gone", "no_record"]


@dataclass(frozen=True)
class TookOverRound:
    """This node took the next claim on ``round``: the last of ``claimants`` lost its lease, or
    no record landed within ``timeout`` seconds."""

    learner: str
    round: int
    claimants: tuple[str, ...]
    cause: Takeover
    timeout: float


@dataclass(frozen=True)
class RoundClosed:
    """This node closed ``draft.round`` and commits it next; the participants with their weights,
    the rejected snapshots and the realized waits are in ``draft``."""

    learner: str
    draft: RoundRecordDraft


@dataclass(frozen=True)
class RoundCommitted:
    """This node wrote ``record``, its payloads and the record itself in ``seconds``."""

    learner: str
    record: RoundRecord
    seconds: float


@dataclass(frozen=True)
class RoundSuperseded:
    """Another node committed the round first; ``record`` is theirs and this node follows on
    from it."""

    learner: str
    record: RoundRecord


@dataclass(frozen=True)
class RoundFailed:
    """Closing or following ``round`` failed; it is retried as after a restart in ``retry_in``
    seconds."""

    learner: str
    round: int
    error: Exception
    retry_in: float


@dataclass(frozen=True)
class SyncerStopped:
    """The syncer thread died of ``error``; the next ``Session.step()`` raises it."""

    learner: str
    error: BaseException


type LearnerEvent = (
    CreatingRun
    | AwaitingCreator
    | TookOverCreation
    | Joined
    | FragmentInstalled
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
    | SyncerStopped
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
                log.exception("%s: a callback failed on %s", event.learner, type(event).__name__)

    return notify


def log_events(event: Event) -> None:
    """One line per event on the ``gradientmail.events`` logger: failures at ERROR with their
    traceback, takeovers and give-ups at WARNING, the rest at INFO. ``RoundRecorded`` is not
    logged; the install or the commit it leads to is."""
    match event:
        case CreatingRun(learner, fragments):
            log.info("%s: creating the run: uploading Θ⁰ of %d fragments", learner, fragments)
        case AwaitingCreator(learner, claimants):
            log.info("%s: waiting for %s to create the run", learner, ", ".join(claimants))
        case TookOverCreation(learner, stalled):
            log.warning("%s: no init payload landed for %.0f s; creating the run", learner, stalled)
        case Joined(learner, created_by, fragments, round):
            log.info(
                "%s: joined the run created by %s: %d fragments, newest round %d",
                learner,
                created_by,
                fragments,
                round,
            )
        case FragmentInstalled(learner, round, fragment, version, local_step):
            log.info(
                "%s: round %d installed: fragment %d version %d, at local step %d",
                learner,
                round,
                fragment,
                version,
                local_step,
            )
        case SnapshotPublished(learner, round, fragment, steps, tokens, size, seconds):
            log.info(
                "%s: round %d published: fragment %d, %d steps, %d tokens, %d bytes, %.1f s",
                learner,
                round,
                fragment,
                steps,
                tokens,
                size,
                seconds,
            )
        case PublishFailed(learner, round, error):
            log.error(
                "%s: round %d: publishing failed; its steps go out with the fragment's next round",
                learner,
                round,
                exc_info=error,
            )
        case PollFailed(learner, error):
            log.error("%s: poll failed; retrying", learner, exc_info=error)
        case RoundRecorded():
            pass
        case Closing(learner, flushing):
            log.info(
                "%s: closing: %s",
                learner,
                "publishing the steps since the last rounds"
                if flushing
                else "waiting for the rounds in flight",
            )
        case Closed(learner, round):
            log.info("%s: closed: every round up to %d is installed", learner, round)
        case GaveUp(learner, round, waited, installed):
            log.warning(
                "%s: giving up on round %d: nothing landed for %.0f s; newest installed is %d",
                learner,
                round,
                waited,
                installed,
            )
        case PublishCancelled(learner, waited):
            log.warning(
                "%s: still publishing %.0f s after the last rounds; cancelling", learner, waited
            )
        case LeaseReleased(learner):
            log.info("%s: lease released", learner)
        case ReleaseFailed(learner, error):
            log.error("%s: releasing the lease failed", learner, exc_info=error)
        case Recovered(learner, round):
            log.info("%s: recovered: next round %d", learner, round)
        case RecoveryFailed(learner, error, retry_in):
            log.error("%s: recovery failed; retrying in %.1f s", learner, retry_in, exc_info=error)
        case RoundClaimed(learner, round):
            log.info("%s: claimed round %d", learner, round)
        case FollowingRound(learner, round, claimants, timeout):
            log.info(
                "%s: following round %d claimed by %s; will take over if no record lands in "
                "%.0f s or the claimant's lease ends",
                learner,
                round,
                ", ".join(claimants),
                timeout,
            )
        case TookOverRound(learner, round, claimants, cause, timeout):
            why = (
                f"{claimants[-1]}'s lease is gone"
                if cause == "lease_gone"
                else f"no record after {timeout:.0f} s"
            )
            log.warning("%s: taking over round %d: %s", learner, round, why)
        case RoundClosed(learner, draft):
            log.info(
                "%s: round %d closed: fragment %d, participants %s, rejected %s, quorum %.1f s, "
                "grace %.1f s",
                learner,
                draft.round,
                draft.fragment,
                [p.learner for p in draft.participants],
                [f"{rejected} ({why})" for rejected, why in draft.rejected],
                draft.quorum_seconds,
                draft.grace_seconds,
            )
        case RoundCommitted(learner, record, seconds):
            log.info(
                "%s: round %d committed: version %d, %d bytes, %.1f s",
                learner,
                record.round,
                record.version,
                record.theta.size + record.outer.size,
                seconds,
            )
        case RoundSuperseded(learner, record):
            log.info("%s: round %d was committed by another node", learner, record.round)
        case RoundFailed(learner, round, error, retry_in):
            log.error(
                "%s: round %d failed; retrying in %.1f s", learner, round, retry_in, exc_info=error
            )
        case SyncerStopped(learner, error):
            log.error("%s: the syncer stopped; the next step() raises", learner, exc_info=error)
        case _:
            assert_never(event)
