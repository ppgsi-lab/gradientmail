"""DiLoCo: its configuration, the pure pieces of its round logic and the outer step.

The outer step (``DiLoCoAggregator.close_round``) is written once, over the ``Value``
arithmetic of whatever ``Backend`` the syncer runs on, so this module imports no framework.
"""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from statistics import fmean
from typing import Self, assert_never

from gradientmail.adapters import Backend, Value
from gradientmail.algorithms import (
    Adaptive,
    Aggregator,
    Algorithm,
    Alignment,
    Blend,
    Carry,
    Contribution,
    Estimates,
    FragmentState,
    Overwrite,
    Participant,
    RoundResult,
    RoundSchedule,
    decoded,
    encoded,
)
from gradientmail.serialization import DType, ParameterSpec, Quantized, Schema, TensorBytes

__all__ = [
    "Adaptive",
    "Blend",
    "ByBytes",
    "Carry",
    "DiLoCo",
    "DiLoCoAggregator",
    "Merge",
    "Overwrite",
    "Partition",
    "Rda",
    "Strided",
    "WeightedMean",
    "grace_window",
    "mean_cosine",
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
class ByBytes:
    """Fragments of balanced byte size: tensors largest first, each into the lightest so far."""


@dataclass(frozen=True)
class Strided:
    """Streaming DiLoCo's strided pattern: the transformer blocks, in index order, go round the
    fragments, so fragment ``p`` holds every ``P``-th block from the ``p``-th. A block is the
    tensors sharing the integer the first group of ``block_pattern`` captures from their names;
    tensors it does not match (embeddings, the final norm) are bin-packed by bytes onto the
    fragments the blocks made."""

    block_pattern: str = r"\.(\d+)\."

    def __post_init__(self) -> None:
        if re.compile(self.block_pattern).groups < 1:
            raise ValueError(
                f"block_pattern needs a group for the block index: {self.block_pattern!r}"
            )


type Partition = ByBytes | Strided


@dataclass(frozen=True)
class DiLoCo(Algorithm):
    """DiLoCo configuration: ``sync_every`` is H, ``fragments`` is P, ``in_flight`` is τ.

    The defaults are Decoupled DiLoCo's. Standard DiLoCo is ``fragments=1, in_flight=1,
    overlap_steps=0, merge=WeightedMean()`` with ``quorum_size`` equal to the number of
    learners; Local SGD is ``outer_lr=1, outer_momentum=0, merge=WeightedMean()``.

    ``overlap_steps`` bounds how far training runs ahead of a published round: once a round a
    learner published that many steps ago is still not aligned with, its ``step()`` waits for
    the round's commit. ``0`` is standard DiLoCo's barrier. ``None``, the default, never waits:
    training goes on, and what the commit does to the steps taken meanwhile is ``alignment``.

    ``alignment`` is what aligning a fragment with its commit does to the steps trained since
    the snapshot the commit incorporated. ``Overwrite()``, the default and the papers' rule,
    sets the fragment to Θ and drops them; a learner trains on between its snapshot and the
    commit for as long as the round takes to close, a dozen steps of every fifty in the
    Pythia-14M run of four nodes. ``Blend(mixing)`` is Streaming DiLoCo's alpha: the fragment
    becomes ``mixing·θ + (1 - mixing)·Θ`` and the counters keep that share of those steps; in
    the paper's Figure 8, at ten overlapped steps, 0.5 loses a third of what 0 loses.
    ``Carry()`` sets the fragment to ``Θ + (θ - θ_snapshot)`` and keeps every one of those
    steps for the next snapshot, so nothing is dropped. ``Adaptive()`` carries while the
    learners agree and overwrites while they do not, fragment by fragment, judged by the mean
    cosine between the round's deltas that every commit records: carrying from a round at
    ``carry_above`` (0.6) or more, overwriting from one under ``overwrite_below`` (0.4), or
    both at ``Adaptive(threshold)``. The published Θ is the anchor either way.

    ``follow`` is how many other learners' snapshots in a round make a learner publish its own
    at once, ahead of its schedule, with the steps it has, so that the round does not close
    without it: ``1`` makes the first learner to reach a round set the pace for all; ``None``,
    the default, is one short of ``quorum_size``, the round being about to close; ``0`` never
    publishes ahead of schedule, standard DiLoCo's pace, every learner training its full
    ``sync_every`` steps while the round waits. A positive value is never more than the other
    live learners, nor less than one.

    ``quorum_size`` is how many accepted snapshots close a round short of every live learner
    reporting, and ``grace_seconds`` how long the closer keeps collecting after that, unless
    every live learner has reported: ``None``, the default, estimates the window from the
    round timings (``grace_window``). ``quorum_size=1`` with a fixed window closes every round
    with whoever reports within it of the first snapshot.

    ``global_batch_size`` fixes the batch the live learners share: each trains on its share,
    so a learner joining shrinks the others' batches instead of growing the global batch.
    ``None`` leaves every learner's batch its own.

    ``delta_dtype`` is the format learners publish their deltas (``anchor - θ``) in. ``None``
    keeps each parameter's own dtype, the same bytes as publishing θ; ``"BF16"`` halves an fp32
    model's uploads, rounding each element to 8 significant bits; ``"F8_E4M3"`` quarters them,
    at 4 significant bits with one power-of-two scale per tensor; ``"F4"`` takes an eighth, at 2
    significant bits with a scale per 32 elements (the MX layout), which the paper found to cost
    nothing in loss. Scales travel as ``<name>.scale`` entries and dequantize exactly.

    ``commit_dtype`` makes every commit publish, beside theta, ``Θ_new - Θ_old`` in that
    format: a learner holding the old version downloads the update instead of theta and
    rebuilds theta from it, checking the result against the record. The published theta is
    ``Θ_old`` plus the dequantized update, so the rounding stays in the master and the next
    round corrects it. ``None`` publishes theta alone.

    ``master_dtype`` is the dtype the outer step accumulates the parameters in, and
    ``momentum_dtype`` the format its momentum is kept in; ``None``, the default for both, is
    each parameter's own dtype, so a bf16 model's outer state is bf16. A bf16 master rounds
    away any step under half an ulp of the parameter, 2**-9 to 2**-8 of it, as training the
    model in bf16 does each inner step; ``"F32"`` keeps it. When the published theta does not
    hold the master (a commit update, or a master wider than the parameter), the commit stores
    the master in ``master_dtype``, or, with ``residual_dtype``, only the master minus the
    published theta in that format: the residual is what the update and the rounding left
    out, a fraction of a step, so ``"BF16"`` halves an fp32 master's bytes at a rounding of at
    most 2**-8 of the residual. The outer step continues from the state as stored, so a node
    that takes over a round continues from the same state.

    ``flush_on_close`` makes a closing learner publish the steps trained since each fragment's
    last round as final snapshots, and a round of final snapshots alone commits the closing
    learners' merged parameters without the outer step. Off, as in the DiLoCo papers, closing
    waits only for the rounds in flight and those steps are dropped: in a run of many learners
    the parameters of the few that close last are worse than the merged ones they replace.

    ``partition`` is how the model's tensors are split into fragments: ``ByBytes`` balances
    their sizes, ``Strided`` deals the transformer blocks round the fragments as Streaming
    DiLoCo does.
    """

    sync_every: int
    fragments: int
    in_flight: int = 2
    overlap_steps: int | None = None
    alignment: Alignment = Overwrite()
    follow: int | None = None
    quorum_size: int = 1
    grace_seconds: float | None = None
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    merge: Merge = Rda()
    partition: Partition = ByBytes()
    global_batch_size: int | None = None
    delta_dtype: DType | Quantized | None = None
    commit_dtype: DType | Quantized | None = None
    master_dtype: DType | None = None
    momentum_dtype: DType | Quantized | None = None
    residual_dtype: DType | Quantized | None = None
    flush_on_close: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.in_flight <= self.fragments <= self.sync_every:
            raise ValueError(
                "need 1 <= in_flight <= fragments <= sync_every, got "
                f"in_flight={self.in_flight}, fragments={self.fragments}, "
                f"sync_every={self.sync_every}"
            )
        if self.overlap_steps is not None and self.overlap_steps < 0:
            raise ValueError(f"overlap_steps must be at least 0, got {self.overlap_steps}")
        if self.follow is not None and self.follow < 0:
            raise ValueError(f"follow must be at least 0, got {self.follow}")
        if self.quorum_size < 1:
            raise ValueError(f"quorum_size must be at least 1, got {self.quorum_size}")
        if self.grace_seconds is not None and self.grace_seconds < 0.0:
            raise ValueError(f"grace_seconds must be at least 0, got {self.grace_seconds}")
        if self.global_batch_size is not None and self.global_batch_size < 1:
            raise ValueError(f"global_batch_size must be at least 1, got {self.global_batch_size}")

    def schedule(self, fragments: int) -> RoundSchedule:
        if fragments < self.in_flight:
            raise ValueError(
                f"in_flight={self.in_flight} exceeds the {fragments} fragment(s) the model allows"
            )
        return RoundSchedule(self.sync_every, fragments, self.in_flight, self.overlap_steps)

    def plan(self, schema: Schema[DType]) -> tuple[tuple[str, ...], ...]:
        return plan(schema, self.fragments, self.partition)

    def aggregator(
        self, schema: Schema[DType], names: Sequence[str], backend: Backend
    ) -> Aggregator:
        return DiLoCoAggregator(config=self, schema=schema, names=names, backend=backend)

    def quorum(self) -> int:
        return self.quorum_size

    def global_batch(self) -> int | None:
        return self.global_batch_size

    def to_json(self) -> Mapping[str, object]:
        def merge() -> Mapping[str, object]:
            match self.merge:
                case WeightedMean():
                    return {"kind": "weighted_mean"}
                case Rda(direct_pattern=pattern):
                    return {"kind": "rda", "direct_pattern": pattern}
                case _:
                    assert_never(self.merge)

        def partition() -> Mapping[str, object]:
            match self.partition:
                case ByBytes():
                    return {"kind": "by_bytes"}
                case Strided(block_pattern=pattern):
                    return {"kind": "strided", "block_pattern": pattern}
                case _:
                    assert_never(self.partition)

        def alignment() -> Mapping[str, object]:
            match self.alignment:
                case Overwrite():
                    return {"kind": "overwrite"}
                case Blend(mixing=mixing):
                    return {"kind": "blend", "mixing": mixing}
                case Carry():
                    return {"kind": "carry"}
                case Adaptive(carry_above=above, overwrite_below=below):
                    return {"kind": "adaptive", "carry_above": above, "overwrite_below": below}
                case _:
                    assert_never(self.alignment)

        return {
            "sync_every": self.sync_every,
            "fragments": self.fragments,
            "in_flight": self.in_flight,
            "overlap_steps": self.overlap_steps,
            "alignment": alignment(),
            "follow": self.follow,
            "quorum_size": self.quorum_size,
            "grace_seconds": self.grace_seconds,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "merge": merge(),
            "partition": partition(),
            "global_batch_size": self.global_batch_size,
            "delta_dtype": self.delta_dtype,
            "commit_dtype": self.commit_dtype,
            "master_dtype": self.master_dtype,
            "momentum_dtype": self.momentum_dtype,
            "residual_dtype": self.residual_dtype,
            "flush_on_close": self.flush_on_close,
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

        def boolean(key: str) -> bool:
            match data.get(key):
                case bool() as value:
                    return value
                case other:
                    raise ValueError(f"{key!r} must be a boolean, got {other!r}")

        def optional_integer(key: str) -> int | None:
            match data.get(key):
                case None:
                    return None
                case int() as value:
                    return value
                case other:
                    raise ValueError(f"{key!r} must be an integer or null, got {other!r}")

        def optional_number(key: str) -> float | None:
            match data.get(key):
                case None:
                    return None
                case int() | float() as value:
                    return float(value)
                case other:
                    raise ValueError(f"{key!r} must be a number or null, got {other!r}")

        def optional_dtype(key: str) -> DType | Quantized | None:
            match data.get(key):
                case None:
                    return None
                case "F32" | "F16" | "BF16" | "F8_E4M3" | "F4" as value:
                    return value
                case other:
                    raise ValueError(
                        f"{key!r} must be F32, F16, BF16, F8_E4M3, F4 or null, got {other!r}"
                    )

        def optional_parameter_dtype(key: str) -> DType | None:
            match data.get(key):
                case None:
                    return None
                case "F32" | "F16" | "BF16" as value:
                    return value
                case other:
                    raise ValueError(f"{key!r} must be F32, F16, BF16 or null, got {other!r}")

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

        def partition() -> Partition:
            found = data.get("partition")
            if not isinstance(found, Mapping):
                raise ValueError(f"'partition' must be an object, got {found!r}")
            match found.get("kind"), found.get("block_pattern"):
                case "by_bytes", None:
                    return ByBytes()
                case "strided", str() as pattern:
                    return Strided(pattern)
                case _:
                    raise ValueError(f"unrecognized partition {dict(found)!r}")

        def alignment() -> Alignment:
            found = data.get("alignment")
            if not isinstance(found, Mapping):
                raise ValueError(f"'alignment' must be an object, got {found!r}")
            fields = ("kind", "mixing", "carry_above", "overwrite_below")
            match tuple(found.get(field) for field in fields):
                case "overwrite", None, None, None:
                    return Overwrite()
                case "blend", int() | float() as mixing, None, None:
                    return Blend(float(mixing))
                case "carry", None, None, None:
                    return Carry()
                case "adaptive", None, int() | float() as above, int() | float() as below:
                    return Adaptive(carry_above=float(above), overwrite_below=float(below))
                case _:
                    raise ValueError(f"unrecognized alignment {dict(found)!r}")

        return cls(
            sync_every=integer("sync_every"),
            fragments=integer("fragments"),
            in_flight=integer("in_flight"),
            overlap_steps=optional_integer("overlap_steps"),
            alignment=alignment(),
            follow=optional_integer("follow"),
            quorum_size=integer("quorum_size"),
            grace_seconds=optional_number("grace_seconds"),
            outer_lr=number("outer_lr"),
            outer_momentum=number("outer_momentum"),
            merge=merge(),
            partition=partition(),
            global_batch_size=optional_integer("global_batch_size"),
            delta_dtype=optional_dtype("delta_dtype"),
            commit_dtype=optional_dtype("commit_dtype"),
            master_dtype=optional_parameter_dtype("master_dtype"),
            momentum_dtype=optional_dtype("momentum_dtype"),
            residual_dtype=optional_dtype("residual_dtype"),
            flush_on_close=boolean("flush_on_close"),
        )


@dataclass(frozen=True)
class DiLoCoAggregator(Aggregator):
    """Closes one round of the fragment holding ``names`` with DiLoCo's outer step.

    The contributions are the deltas the learners computed against the published ``theta``,
    which is the anchor as is, so a learner whose bytes did not change contributes an exactly
    zero delta. ``outer["momentum.<name>"]`` holds the momentum at ``momentum_dtype``, and is
    absent, zero, for a fragment never committed. When the published tensor does not hold the
    master (commits carry an update, or ``master_dtype`` is wider than the parameter's),
    ``outer`` also holds ``master.<name>`` in ``master_dtype`` or, with ``residual_dtype``,
    ``residual.<name>``, the master minus the published tensor. With a ``commit_dtype`` the
    published theta is the anchor plus the update as it dequantizes, so what the update lost
    stays in the master. Pure for one backend: equal inputs give equal bytes.

    A round whose contributions are all final (``flush_on_close``) moves the master by the
    merged delta alone and
    keeps the momentum: the commit holds the closing learners' parameters. Nesterov's step
    would also move it by ``η·β²·m``, about as far as a full round moves it however few steps
    the deltas hold, and no training follows to correct that.
    """

    config: DiLoCo
    schema: Schema[DType]
    names: Sequence[str]
    backend: Backend

    def close_round(
        self, state: FragmentState, contributions: Sequence[Contribution]
    ) -> RoundResult:
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

        def stepped(name: str) -> tuple[Value, Value, Value]:
            """The anchor, the new master and the new momentum of ``name``."""
            deltas = [
                decoded(self.backend, c.tensors, name, shapes[name], self.config.delta_dtype)
                for c in contributions
            ]
            for i, j in pairs:
                gram[i][j] += deltas[i].dot(deltas[j])
            anchor = load(state.theta[name])
            delta = merged(name, deltas)
            master = held(name, anchor)
            momentum = (
                decoded(
                    self.backend,
                    state.outer,
                    f"momentum.{name}",
                    shapes[name],
                    self.config.momentum_dtype,
                )
                if f"momentum.{name}" in state.outer
                else anchor * 0.0
            )
            if final:
                master = master - delta
            else:
                master, momentum = nesterov(
                    master,
                    momentum,
                    delta,
                    lr=self.config.outer_lr,
                    beta=self.config.outer_momentum,
                )
            # Values compute in fp32: only a narrower master is rounded to its dtype.
            dtype = master_dtypes[name]
            return anchor, master if dtype == "F32" else load(master.to_bytes(dtype)), momentum

        def held(name: str, anchor: Value) -> Value:
            """The master the last commit left: stored as is, rebuilt from its residual over the
            anchor, or the anchor itself when that holds it."""
            if (stored := state.outer.get(f"master.{name}")) is not None:
                return load(stored)
            if f"residual.{name}" not in state.outer:
                return anchor
            residual = decoded(
                self.backend,
                state.outer,
                f"residual.{name}",
                shapes[name],
                self.config.residual_dtype,
            )
            return anchor + residual

        def published(name: str, anchor: Value, master: Value) -> TensorBytes:
            if self.config.commit_dtype is None:
                return master.to_bytes(dtypes[name])
            entries = encoded(name, master - anchor, dtypes[name], self.config.commit_dtype)
            update.update(entries)
            moved = decoded(self.backend, entries, name, shapes[name], self.config.commit_dtype)
            return (anchor + moved).to_bytes(dtypes[name])

        def kept(
            name: str, master: Value, momentum: Value, theta: TensorBytes
        ) -> Mapping[str, TensorBytes]:
            """The outer state of ``name``: the momentum, and the master unless ``theta``
            holds it, as F32 holds F16 and BF16 exactly."""
            entries = encoded(
                f"momentum.{name}", momentum, dtypes[name], self.config.momentum_dtype
            )
            if self.config.commit_dtype is None and dtypes[name] in (master_dtypes[name], "F32"):
                return entries
            if self.config.residual_dtype is None:
                return {**entries, f"master.{name}": master.to_bytes(master_dtypes[name])}
            residual = encoded(
                f"residual.{name}", master - load(theta), dtypes[name], self.config.residual_dtype
            )
            return {**entries, **residual}

        load = self.backend.load
        final = all(c.final for c in contributions)
        weights = [weight(c.tokens, c.steps) for c in contributions]
        dtypes: Mapping[str, DType] = {spec.name: spec.dtype for spec in self.schema.parameters}
        master_dtypes: Mapping[str, DType] = {
            name: dtype if self.config.master_dtype is None else self.config.master_dtype
            for name, dtype in dtypes.items()
        }
        shapes = {spec.name: spec.shape for spec in self.schema.parameters}
        count = len(contributions)
        pairs = [(i, j) for i in range(count) for j in range(i, count)]
        gram = [[0.0] * count for _ in range(count)]
        update: dict[str, TensorBytes] = {}
        stepped_by_name = {name: stepped(name) for name in self.names}
        theta = {
            name: published(name, anchor, master)
            for name, (anchor, master, _) in stepped_by_name.items()
        }
        outer = {
            key: tensor
            for name, (_, master, momentum) in stepped_by_name.items()
            for key, tensor in kept(name, master, momentum, theta[name]).items()
        }
        participants = tuple(
            Participant(c.learner, c.local_step, c.steps, c.tokens, w)
            for c, w in zip(contributions, weights, strict=True)
        )
        return RoundResult(
            FragmentState(state.version + 1, theta, outer),
            participants,
            mean_cosine(gram),
            update if self.config.commit_dtype is not None else None,
        )


def plan(
    schema: Schema[DType], fragments: int, partition: Partition
) -> tuple[tuple[str, ...], ...]:
    """Split the schema's tensors into at most ``fragments`` fragments.

    With ``ByBytes`` every tensor is bin-packed: taken largest first (ties by name), each goes
    to the lightest fragment so far. With ``Strided`` the blocks are dealt round the fragments
    first and only the tensors outside any block are bin-packed. The count is at most
    ``min(fragments, len(parameters))``, fragments left empty are dropped, and the result is a
    function of the schema alone, so every party computes the same plan.
    """
    count = min(fragments, len(schema.parameters))
    names: list[list[str]] = [[] for _ in range(count)]
    loads = [0] * count

    def place(spec: ParameterSpec[DType], fragment: int) -> None:
        names[fragment].append(spec.name)
        loads[fragment] += schema.nbytes(spec.name)

    loose: list[ParameterSpec[DType]] = []
    match partition:
        case ByBytes():
            loose.extend(schema.parameters)
        case Strided(block_pattern=pattern):
            blocks: dict[int, list[ParameterSpec[DType]]] = {}
            for spec in schema.parameters:
                found = re.search(pattern, spec.name)
                if found is None:
                    loose.append(spec)
                else:
                    blocks.setdefault(int(found[1]), []).append(spec)
            for rank, index in enumerate(sorted(blocks)):
                for spec in blocks[index]:
                    place(spec, rank % count)
        case _:
            assert_never(partition)
    for spec in sorted(loose, key=lambda spec: (-schema.nbytes(spec.name), spec.name)):
        place(spec, min(range(count), key=loads.__getitem__))
    return tuple(tuple(sorted(fragment)) for fragment in names if fragment)


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


def mean_cosine(gram: Sequence[Sequence[float]]) -> float | None:
    """The mean cosine similarity between the vectors of a Gram matrix (its upper triangle
    filled); None with fewer than two vectors, and a pair with a zero vector counts as 0."""

    def cosine(i: int, j: int) -> float:
        norms = math.sqrt(gram[i][i] * gram[j][j])
        return gram[i][j] / norms if norms > 0.0 else 0.0

    pairs = [(i, j) for i in range(len(gram)) for j in range(i + 1, len(gram))]
    return fmean(cosine(i, j) for i, j in pairs) if pairs else None


def nesterov[V: Value](master: V, momentum: V, delta: V, *, lr: float, beta: float) -> tuple[V, V]:
    """One Nesterov SGD step on the outer gradient ``delta``: returns ``(master, momentum)``.

    ``m ← β·m + Δ`` and ``Θ ← Θ - η·(Δ + β·m)``.
    """
    updated = momentum * beta + delta
    return master - (updated * beta + delta) * lr, updated


def grace_window(
    estimates: Estimates, in_flight: int, gap: int, overlap_steps: int | None
) -> float:
    """Seconds to keep collecting after quorum.

    ``min(0.5 · max(ahead·ξ_step - (ξ_quorum + ξ_sync), 0), ξ_round)`` with ``ξ_round =
    gap·ξ_step`` and ``ahead`` the steps a learner trains past a publish before it would stall:
    ``in_flight`` rounds of ``gap`` steps, or ``overlap_steps`` when that is less. Half the slack
    those steps leave, never more than one round of training, and zero while any estimate is
    still missing. The paper states the slack in compute steps because it synchronizes a
    fragment every step (H = P); a round here spans ``gap`` steps of the step time learners
    report.
    """
    if estimates.step is None or estimates.quorum is None or estimates.sync is None:
        return 0.0
    ahead = in_flight * gap if overlap_steps is None else min(in_flight * gap, overlap_steps)
    slack = max(ahead * estimates.step - (estimates.quorum + estimates.sync), 0.0)
    return min(0.5 * slack, gap * estimates.step)


_NORM_FLOOR = 1e-12
