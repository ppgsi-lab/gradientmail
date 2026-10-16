"""Pluggable synchronization algorithms: the protocols runtimes drive and the records they share."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, Self, runtime_checkable

from gradientmail.adapters import Backend
from gradientmail.serialization import Schema, TensorBytes

__all__ = [
    "Aggregator",
    "Algorithm",
    "Contribution",
    "Estimates",
    "FragmentState",
    "Participant",
    "RoundSchedule",
]


@dataclass(frozen=True)
class RoundSchedule:
    """Which fragment each round synchronizes and at which global step.

    Round ``r`` targets fragment ``r mod P`` at step ``(r div P)·H + ⌊((r mod P) + 1)·H / P⌋``,
    so every fragment is synchronized exactly once per ``H`` global steps and the gaps between
    consecutive rounds sum to ``H``. The only requirement is ``P ≤ H``.
    """

    sync_every: int
    fragments: int
    in_flight: int

    def fragment_of(self, round: int) -> int:
        return round % self.fragments

    def step_of(self, round: int) -> int:
        cycle, position = divmod(round, self.fragments)
        return cycle * self.sync_every + (position + 1) * self.sync_every // self.fragments

    def version_after(self, round: int) -> int:
        return round // self.fragments + 1

    def gap(self, round: int) -> int:
        return self.step_of(round) - self.step_of(round - 1)


@dataclass(frozen=True)
class FragmentState:
    """A fragment as last committed: its version, the published anchor and the outer state.

    ``outer`` holds ``momentum.<name>`` in F32 and ``master.<name>`` in F32 when the schema
    dtype is not F32.
    """

    version: int
    theta: Mapping[str, TensorBytes]
    outer: Mapping[str, TensorBytes]


@dataclass(frozen=True)
class Contribution:
    """One learner's accepted delta of a fragment for one round: ``anchor - θ`` as the learner
    computed it against the version it had installed, in the algorithm's delta schema."""

    learner: str
    local_step: int
    steps: int
    tokens: int
    step_seconds: float
    tensors: Mapping[str, TensorBytes]


@dataclass(frozen=True)
class Participant:
    """A contribution as recorded in the round record, with the weight it received."""

    learner: str
    local_step: int
    steps: int
    tokens: int
    weight: float


@dataclass(frozen=True)
class Estimates:
    """Exponential moving averages of the round timings, carried from record to record.

    ``step`` is the learners' reported seconds per training step, ``quorum`` the seconds from
    round start to quorum and ``sync`` the seconds from closing to a durable commit. Each is
    ``None`` until its first observation, which is taken as is.
    """

    step: float | None = None
    quorum: float | None = None
    sync: float | None = None

    def updated(
        self,
        *,
        step: float | None = None,
        quorum: float | None = None,
        sync: float | None = None,
    ) -> Self:
        def ema(old: float | None, new: float | None) -> float | None:
            if new is None:
                return old
            if old is None:
                return new
            return old + _EMA_DECAY * (new - old)

        return replace(
            self,
            step=ema(self.step, step),
            quorum=ema(self.quorum, quorum),
            sync=ema(self.sync, sync),
        )

    def timeout(self, poll_interval: float) -> float:
        """Seconds after which a round these estimates describe is presumed not to commit.

        Three times the round duration they predict (the known estimates summed), never less
        than thirty polls: a follower takes over the closer after it, and a closing learner
        stops waiting for its last rounds after it.
        """
        expected = sum(s for s in (self.step, self.quorum, self.sync) if s is not None)
        return max(_TIMEOUT_FLOOR_POLLS * poll_interval, _TIMEOUT_FACTOR * expected)


@runtime_checkable
class Aggregator(Protocol):
    """Closes one round of one fragment; the only place tensor arithmetic happens."""

    def close_round(
        self, state: FragmentState, contributions: Sequence[Contribution]
    ) -> tuple[FragmentState, tuple[Participant, ...]]: ...


@runtime_checkable
class Algorithm(Protocol):
    def schedule(self, fragments: int) -> RoundSchedule:
        """The round schedule for the effective fragment count produced by ``plan``."""
        ...

    def plan(self, schema: Schema) -> tuple[tuple[str, ...], ...]: ...

    def delta_schema(self, schema: Schema) -> Schema:
        """The schema of the deltas learners publish: ``schema``'s parameters in the dtype the
        deltas travel in."""
        ...

    def aggregator(self, schema: Schema, names: Sequence[str], backend: Backend) -> Aggregator:
        """An aggregator for the fragment holding ``names``, computing in ``backend``."""
        ...

    def quorum(self) -> int: ...

    def global_batch(self) -> int | None:
        """The batch the live learners share between them; ``None`` when each sets its own."""
        ...

    def to_json(self) -> Mapping[str, object]:
        """A JSON-serializable record of the configuration, stored in the manifest."""
        ...


_EMA_DECAY = 0.1
_TIMEOUT_FACTOR = 3.0
_TIMEOUT_FLOOR_POLLS = 30
