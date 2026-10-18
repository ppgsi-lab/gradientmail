"""Pluggable synchronization algorithms: the protocols runtimes drive and the records they share."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, Self, assert_never, runtime_checkable

from gradientmail.adapters import Backend, Value
from gradientmail.serialization import (
    DType,
    ParameterSpec,
    PayloadDType,
    Quantized,
    Schema,
    TensorBytes,
    quantized_specs,
    scale_name,
)

__all__ = [
    "Aggregator",
    "Algorithm",
    "Contribution",
    "Estimates",
    "FragmentState",
    "Participant",
    "Precision",
    "RoundResult",
    "RoundSchedule",
    "decoded",
    "encoded",
    "layout",
]

type Precision = DType | Quantized | None
"""How a tensor travels in a payload: in a parameter dtype, quantized with its scales beside
it, or, with ``None``, in its own dtype."""


@dataclass(frozen=True)
class RoundSchedule:
    """Which fragment each round synchronizes and at which global step.

    Round ``r`` targets fragment ``r mod P`` at step ``(r div P)·H + ⌊((r mod P) + 1)·H / P⌋``,
    so every fragment is synchronized exactly once per ``H`` global steps and the gaps between
    consecutive rounds sum to ``H``. The only requirement is ``P ≤ H``. A learner keeps at most
    ``in_flight`` rounds outstanding and, with ``overlap_steps``, trains at most that many steps
    past publishing a round before it waits for the round's commit.
    """

    sync_every: int
    fragments: int
    in_flight: int
    overlap_steps: int | None = None

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

    ``outer`` holds ``momentum.<name>`` in F32 and ``master.<name>`` in F32 when the published
    tensor is not the master: the schema dtype is not F32, or commits carry an update.
    """

    version: int
    theta: Mapping[str, TensorBytes]
    outer: Mapping[str, TensorBytes]


@dataclass(frozen=True)
class Contribution:
    """One learner's accepted delta of a fragment for one round: ``anchor - θ`` as the learner
    computed it against the version it had aligned with, in the algorithm's delta schema.
    ``final`` when the learner published it while closing: it trains no further."""

    learner: str
    local_step: int
    steps: int
    tokens: int
    step_seconds: float
    final: bool
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


@dataclass(frozen=True)
class RoundResult:
    """What closing a round produced: the fragment's next state, the contributions as the
    round record lists them, ``cosine``, the mean cosine similarity between the contributions'
    deltas taken over the whole fragment (None with fewer than two), and ``update``, the new
    theta minus the old at the algorithm's commit precision, for learners holding the old one
    to rebuild the new from (None when commits carry none)."""

    state: FragmentState
    participants: tuple[Participant, ...]
    cosine: float | None
    update: Mapping[str, TensorBytes] | None


@runtime_checkable
class Aggregator(Protocol):
    """Closes one round of one fragment; the only place tensor arithmetic happens."""

    def close_round(
        self, state: FragmentState, contributions: Sequence[Contribution]
    ) -> RoundResult: ...


@runtime_checkable
class Algorithm(Protocol):
    def schedule(self, fragments: int) -> RoundSchedule:
        """The round schedule for the effective fragment count produced by ``plan``."""
        ...

    def plan(self, schema: Schema[DType]) -> tuple[tuple[str, ...], ...]: ...

    @property
    def delta_dtype(self) -> Precision:
        """The precision learners publish their deltas (``anchor - θ``) at; a snapshot of a
        fragment must hold the ``layout`` of the fragment's schema at it."""
        ...

    @property
    def commit_dtype(self) -> Precision:
        """The precision a commit publishes its update (``Θ_new - Θ_old``) at, for a learner
        holding the old version to rebuild the new one from; ``None`` publishes theta alone."""
        ...

    def aggregator(
        self, schema: Schema[DType], names: Sequence[str], backend: Backend
    ) -> Aggregator:
        """An aggregator for the fragment holding ``names``, computing in ``backend``."""
        ...

    def quorum(self) -> int: ...

    @property
    def mixing(self) -> float:
        """The share of its local parameters a learner keeps when it aligns a fragment with a
        commit: ``mixing·θ + (1 - mixing)·Θ``, in ``[0, 1)``."""
        ...

    @property
    def flush_on_close(self) -> bool:
        """Whether a closing learner publishes the steps trained since each fragment's last
        round, as final snapshots, instead of only waiting for its rounds in flight."""
        ...

    def global_batch(self) -> int | None:
        """The batch the live learners share between them; ``None`` when each sets its own."""
        ...

    def to_json(self) -> Mapping[str, object]:
        """A JSON-serializable record of the configuration, stored in the manifest."""
        ...


_EMA_DECAY = 0.1
_TIMEOUT_FACTOR = 3.0
_TIMEOUT_FLOOR_POLLS = 30


def layout(schema: Schema[DType], precision: Precision) -> Schema[PayloadDType]:
    """What ``schema``'s tensors travel as at ``precision``: each in that dtype, a quantized
    one with its scales beside it (``quantized_specs``), or as it is."""

    def entries(spec: ParameterSpec[DType]) -> Iterable[ParameterSpec[PayloadDType]]:
        match precision:
            case None:
                return (spec,)
            case "F32" | "F16" | "BF16" as dtype:
                return (ParameterSpec(spec.name, dtype, spec.shape),)
            case "F8_E4M3" | "F4" as dtype:
                return quantized_specs(spec.name, dtype, spec.shape)
            case _:
                assert_never(precision)

    specs = (entry for spec in schema.parameters for entry in entries(spec))
    return Schema(tuple(sorted(specs, key=lambda spec: spec.name)))


def encoded(
    name: str, value: Value, dtype: DType, precision: Precision
) -> Mapping[str, TensorBytes]:
    """The payload entries the fp32 ``value`` of ``name``, a parameter of ``dtype``, travels as
    at ``precision``."""
    match precision:
        case None:
            return {name: value.to_bytes(dtype)}
        case "F32" | "F16" | "BF16" as narrowed:
            return {name: value.to_bytes(narrowed)}
        case "F8_E4M3" | "F4" as quantized:
            data, scale = value.quantize(quantized)
            return {name: data, scale_name(name): scale}
        case _:
            assert_never(precision)


def decoded(
    backend: Backend,
    tensors: Mapping[str, TensorBytes],
    name: str,
    shape: tuple[int, ...],
    precision: Precision,
) -> Value:
    """The fp32 value of ``name``, of ``shape``, from the entries ``encoded`` produced at
    ``precision``."""
    match precision:
        case "F8_E4M3" | "F4":
            return backend.dequantize(tensors[name], tensors[scale_name(name)], shape)
        case _:
            return backend.load(tensors[name])
