"""Decoupled DiLoCo: its configuration, the pure pieces of its round logic and the outer step.

The outer step (``DiLoCoAggregator.close_round``) is written once, over the ``Value``
arithmetic of whatever ``Backend`` the syncer runs on, so this module imports no framework.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Self, assert_never

from gradientmail.adapters import Backend, Value
from gradientmail.algorithms import (
    Aggregator,
    Algorithm,
    Contribution,
    Estimates,
    FragmentState,
    Participant,
    RoundSchedule,
)
from gradientmail.serialization import DType, Schema

__all__ = [
    "DecoupledDiLoCo",
    "DiLoCoAggregator",
    "Merge",
    "Rda",
    "WeightedMean",
    "grace_window",
    "nesterov",
    "plan",
    "rda",
    "weight",
    "weighted_mean",
]


@dataclass(frozen=True)
class WeightedMean:
    """Merge every tensor's deltas by their weighted mean."""


@dataclass(frozen=True)
class Rda:
    """Weighted RDA for every tensor except those whose lowercased name contains
    ``direct_pattern``, which use the weighted mean (the paper's treatment of embeddings)."""

    direct_pattern: str = "embed"


type Merge = WeightedMean | Rda


@dataclass(frozen=True)
class DecoupledDiLoCo(Algorithm):
    """Decoupled DiLoCo configuration: ``sync_every`` is H, ``fragments`` is P, ``in_flight`` is τ.

    Standard DiLoCo is ``fragments=1`` with ``quorum_size`` equal to the number of learners;
    Local SGD is ``outer_lr=1, outer_momentum=0, merge=WeightedMean()``.
    """

    sync_every: int
    fragments: int
    in_flight: int = 2
    quorum_size: int = 1
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    merge: Merge = Rda()

    def __post_init__(self) -> None:
        if not 1 <= self.in_flight <= self.fragments <= self.sync_every:
            raise ValueError(
                "need 1 <= in_flight <= fragments <= sync_every, got "
                f"in_flight={self.in_flight}, fragments={self.fragments}, "
                f"sync_every={self.sync_every}"
            )
        if self.quorum_size < 1:
            raise ValueError(f"quorum_size must be at least 1, got {self.quorum_size}")

    def schedule(self, fragments: int) -> RoundSchedule:
        if fragments < self.in_flight:
            raise ValueError(
                f"in_flight={self.in_flight} exceeds the {fragments} fragment(s) the model allows"
            )
        return RoundSchedule(self.sync_every, fragments, self.in_flight)

    def plan(self, schema: Schema) -> tuple[tuple[str, ...], ...]:
        return plan(schema, self.fragments)

    def aggregator(self, schema: Schema, names: Sequence[str], backend: Backend) -> Aggregator:
        return DiLoCoAggregator(config=self, schema=schema, names=names, backend=backend)

    def quorum(self) -> int:
        return self.quorum_size

    def to_json(self) -> Mapping[str, object]:
        def merge() -> Mapping[str, object]:
            match self.merge:
                case WeightedMean():
                    return {"kind": "weighted_mean"}
                case Rda(direct_pattern=pattern):
                    return {"kind": "rda", "direct_pattern": pattern}
                case _:
                    assert_never(self.merge)

        return {
            "sync_every": self.sync_every,
            "fragments": self.fragments,
            "in_flight": self.in_flight,
            "quorum_size": self.quorum_size,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "merge": merge(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> Self:
        """Inverse of ``to_json``; raises ``ValueError`` on a record of another shape."""

        def integer(key: str) -> int:
            match data.get(key):
                case int() as value:
                    return value
                case other:
                    raise ValueError(f"{key!r} must be an integer, got {other!r}")

        def number(key: str) -> float:
            match data.get(key):
                case int() | float() as value:
                    return float(value)
                case other:
                    raise ValueError(f"{key!r} must be a number, got {other!r}")

        def merge() -> Merge:
            found = data.get("merge")
            if not isinstance(found, Mapping):
                raise ValueError(f"'merge' must be an object, got {found!r}")
            match found.get("kind"), found.get("direct_pattern"):
                case "weighted_mean", None:
                    return WeightedMean()
                case "rda", str() as pattern:
                    return Rda(pattern)
                case _:
                    raise ValueError(f"unrecognized merge {dict(found)!r}")

        return cls(
            sync_every=integer("sync_every"),
            fragments=integer("fragments"),
            in_flight=integer("in_flight"),
            quorum_size=integer("quorum_size"),
            outer_lr=number("outer_lr"),
            outer_momentum=number("outer_momentum"),
            merge=merge(),
        )


@dataclass(frozen=True)
class DiLoCoAggregator(Aggregator):
    """Closes one round of the fragment holding ``names`` with Decoupled DiLoCo's outer step.

    The anchor is the published ``theta`` as is, so a learner whose bytes did not change
    contributes an exactly zero delta. The fp32 master lives in ``outer["master.<name>"]``
    when the schema dtype is not F32 (otherwise the published tensor is the master) and the
    fp32 momentum in ``outer["momentum.<name>"]``. Pure for one backend: equal inputs give
    equal bytes.
    """

    config: DecoupledDiLoCo
    schema: Schema
    names: Sequence[str]
    backend: Backend

    def close_round(
        self, state: FragmentState, contributions: Sequence[Contribution]
    ) -> tuple[FragmentState, tuple[Participant, ...]]:
        def merged(name: str, deltas: Sequence[Value]) -> Value:
            match self.config.merge:
                case WeightedMean():
                    return weighted_mean(deltas, weights)
                case Rda(direct_pattern=pattern):
                    if pattern in name.lower():
                        return weighted_mean(deltas, weights)
                    return rda(deltas, weights)
                case _:
                    assert_never(self.config.merge)

        def stepped(name: str) -> tuple[Value, Value]:
            anchor = load(state.theta[name])
            delta = merged(name, [anchor - load(c.tensors[name]) for c in contributions])
            master = state.outer.get(f"master.{name}")
            return nesterov(
                anchor if master is None else load(master),
                load(state.outer[f"momentum.{name}"]),
                delta,
                lr=self.config.outer_lr,
                beta=self.config.outer_momentum,
            )

        load = self.backend.load
        weights = [weight(c.tokens, c.steps) for c in contributions]
        dtypes: Mapping[str, DType] = {spec.name: spec.dtype for spec in self.schema.parameters}
        stepped_by_name = {name: stepped(name) for name in self.names}
        theta = {
            name: master.to_bytes(dtypes[name]) for name, (master, _) in stepped_by_name.items()
        }
        momenta = {
            f"momentum.{name}": momentum.to_bytes("F32")
            for name, (_, momentum) in stepped_by_name.items()
        }
        masters = {
            f"master.{name}": master.to_bytes("F32")
            for name, (master, _) in stepped_by_name.items()
            if dtypes[name] != "F32"
        }
        participants = tuple(
            Participant(c.learner, c.local_step, c.steps, c.tokens, w)
            for c, w in zip(contributions, weights, strict=True)
        )
        return FragmentState(state.version + 1, theta, {**momenta, **masters}), participants


def plan(schema: Schema, fragments: int) -> tuple[tuple[str, ...], ...]:
    """Split the schema's tensors into at most ``fragments`` fragments of balanced byte size.

    Greedy bin-packing: tensors are taken largest first (ties by name) and each goes to the
    lightest fragment so far. The effective count is ``min(fragments, len(parameters))``, and
    the result is a function of the schema alone, so every party computes the same plan.
    """
    count = min(fragments, len(schema.parameters))
    names: list[list[str]] = [[] for _ in range(count)]
    loads = [0] * count
    by_size = sorted(schema.parameters, key=lambda spec: (-schema.nbytes(spec.name), spec.name))
    for spec in by_size:
        lightest = min(range(count), key=loads.__getitem__)
        names[lightest].append(spec.name)
        loads[lightest] += schema.nbytes(spec.name)
    return tuple(tuple(sorted(fragment)) for fragment in names)


def weight(tokens: int, steps: int) -> float:
    """The paper's contribution weight ``tokens² / steps``, squared in exact integers first."""
    return tokens * tokens / steps


def weighted_mean[V: Value](values: Sequence[V], weights: Sequence[float]) -> V:
    """``Σ wᵢ·vᵢ / Σ wᵢ``, with the weights normalized in Python floats first."""
    total = sum(weights)
    scaled = (value * (w / total) for value, w in zip(values, weights, strict=True))
    return reduce(lambda acc, value: acc + value, scaled)


def rda[V: Value](deltas: Sequence[V], weights: Sequence[float]) -> V:
    """Weighted RDA: the weighted mean of the norms along the weighted mean direction.

    ``(Σ wᵢ‖vᵢ‖ / Σ wᵢ) · φ(Σ wᵢ φ(vᵢ) / Σ wᵢ)`` with ``φ(x) = x / max(‖x‖, 1e-12)``, so the
    result is zero when the directions cancel. Norms are taken per tensor.
    """
    norms = [delta.norm() for delta in deltas]
    directions = [
        delta * (1 / max(norm, _NORM_FLOOR)) for delta, norm in zip(deltas, norms, strict=True)
    ]
    mean_norm = sum(w * norm for w, norm in zip(weights, norms, strict=True)) / sum(weights)
    mean_direction = weighted_mean(directions, weights)
    return mean_direction * (mean_norm / max(mean_direction.norm(), _NORM_FLOOR))


def nesterov[V: Value](master: V, momentum: V, delta: V, *, lr: float, beta: float) -> tuple[V, V]:
    """One Nesterov SGD step on the outer gradient ``delta``: returns ``(master, momentum)``.

    ``m ← β·m + Δ`` and ``Θ ← Θ - η·(Δ + β·m)``.
    """
    updated = momentum * beta + delta
    return master - (updated * beta + delta) * lr, updated


def grace_window(estimates: Estimates, in_flight: int, gap: int) -> float:
    """Seconds to keep collecting after quorum.

    ``min(0.5 · max(τ·ξ_round - (ξ_quorum + ξ_sync), 0), ξ_round)`` with ``ξ_round = gap·ξ_step``:
    half the slack the in-flight rounds leave before learners would stall, never more than one
    round of training, and zero while any estimate is still missing. The paper states the
    slack in compute steps because it synchronizes a fragment every step (H = P); a round here
    spans ``gap`` steps of the step time learners report.
    """
    if estimates.step is None or estimates.quorum is None or estimates.sync is None:
        return 0.0
    round_seconds = gap * estimates.step
    slack = max(in_flight * round_seconds - (estimates.quorum + estimates.sync), 0.0)
    return min(0.5 * slack, round_seconds)


_NORM_FLOOR = 1e-12
