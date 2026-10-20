"""Alignment, follow and grace compared on a toy that runs in seconds: four learners fit a
small MLP to a random teacher through a mailbox with artificial latency.

The teacher is a fixed two-layer tanh MLP (16 inputs, 32 hidden, one output); targets are its
outputs plus Gaussian noise of variance ``NOISE²``, so a held-out mean squared error reads as
excess over that floor. Every learner starts from the same student (seeded by ``--seed``) and
trains AdamW. The data is ``--steps`` global batches of four slices, each slice a function of
the seed, the step and the slice, and the slices are claimed in units of ``--unit`` steps
through the mailbox (the Shared Queue pattern, as ``pythia.py`` does), so a fast node takes
more of them and every slice is trained once whatever the paces. The merged model is scored
on one fixed held-out set. What varies between runs is the algorithm and the mailbox.

Locally a round closes in less than a step, so nothing is trained between a snapshot and its
commit and every alignment is the same. What made the Pythia-14M run on Salad drop 22% of its
steps was the bucket: about 25 s from a snapshot to its commit against steps of 2 s. ``Latent``
puts ``--latency`` seconds on every mailbox operation and ``--step-seconds`` paces the steps,
so their ratio sets how many steps a round spans; each result reports the share of steps the
rounds incorporated, to be read next to the loss. ``--latency-window`` confines the latency to
a share of the run, to price the dropped steps early against late. ``--paces`` slows nodes
down relative to each other for the heterogeneous case that ``follow`` and ``grace`` address.

    uv run python examples/teacher_student.py --alignment overwrite carry --latency 0 0.005
    uv run python examples/teacher_student.py --alignment carry --follow 1 --grace 0.1 \\
        --paces 1 0.95 0.75 0.4 --seeds 5

``--solo`` adds the reference: one learner on all four slices of every batch, no mailbox.
"""

import argparse
import asyncio
import copy
import statistics
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

import gradientmail as gm
from gradientmail.adapters.torch import TorchBackend
from gradientmail.algorithms import Alignment
from gradientmail.clock import Clock, MonotonicClock
from gradientmail.mailbox import Mailbox
from gradientmail.patterns.shared_queue import shared_queue
from gradientmail.protocol import Exchange

INPUTS = 16
HIDDEN = 32
NOISE = 0.1
NODES = 4
BATCH = 64
HELD_OUT = 4096
TEACHER_SEED = 12345
HELD_OUT_SEED = 54321
POLL = 0.01


class Latent(Mailbox):
    """``inner`` with ``seconds`` of latency on every operation, as a remote store has; the
    run changes ``seconds`` as it goes."""

    def __init__(self, inner: Mailbox, seconds: float) -> None:
        self._inner = inner
        self.seconds = seconds

    async def put_if_absent(self, key: str, data: bytes) -> bool:
        await asyncio.sleep(self.seconds)
        return await self._inner.put_if_absent(key, data)

    async def get(self, key: str) -> bytes | None:
        await asyncio.sleep(self.seconds)
        return await self._inner.get(key)

    async def list(self, prefix: str, *, after: str = "") -> Sequence[str]:
        await asyncio.sleep(self.seconds)
        return await self._inner.list(prefix, after=after)

    async def delete(self, keys: Iterable[str]) -> None:
        await asyncio.sleep(self.seconds)
        await self._inner.delete(keys)

    def clock(self) -> Clock:
        return self._inner.clock()


@dataclass(frozen=True)
class Setup:
    """One run: the algorithm (None trains one learner on every slice), the data seed, the
    global steps, the unit of steps a claim takes, and how the mailbox and the nodes are
    slowed down."""

    algorithm: gm.DiLoCo | None
    seed: int
    steps: int
    unit: int
    lr: float
    step_seconds: float
    latency: float
    window: tuple[float, float]  # the share of the run, by steps, in which the latency applies
    paces: tuple[float, ...]


@dataclass(frozen=True)
class Unit:
    """``count`` global steps from ``first``, slice ``lane`` of each one's batch."""

    first: int
    count: int
    lane: int


@dataclass(frozen=True)
class Result:
    loss: float
    incorporated: float | None
    rounds: int
    cosines: tuple[float, ...]  # mean cosine between a round's deltas, by tenth of the run


def mlp(seed: int) -> nn.Sequential:
    generator = torch.Generator().manual_seed(seed)
    layers = [nn.Linear(INPUTS, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh()]
    layers.append(nn.Linear(HIDDEN, 1))
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer, nn.Linear):
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) * 0.3)
                layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator) * 0.1)
    return nn.Sequential(*layers)


def labelled(teacher: nn.Module, inputs: Tensor, generator: torch.Generator) -> Tensor:
    with torch.no_grad():
        return teacher(inputs) + NOISE * torch.randn(len(inputs), 1, generator=generator)


def batch(teacher: nn.Module, seed: int, step: int, lane: int) -> tuple[Tensor, Tensor]:
    """Slice ``lane`` of global step ``step``'s batch: the same whoever trains on it."""
    generator = torch.Generator().manual_seed((seed * 1_000_003 + step) * NODES + lane)
    inputs = torch.randn(BATCH, INPUTS, generator=generator)
    return inputs, labelled(teacher, inputs, generator)


def units(setup: Setup, rank: int) -> dict[str, Unit]:
    """Every unit of the run, this node's own lane first within each block of steps."""
    lanes = tuple((rank + i) % NODES for i in range(NODES))
    return {
        f"{first // setup.unit:06d}-{lane}": Unit(first, min(setup.unit, setup.steps - first), lane)
        for first in range(0, setup.steps, setup.unit)
        for lane in lanes
    }


def held_out(teacher: nn.Module) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(HELD_OUT_SEED)
    inputs = torch.randn(HELD_OUT, INPUTS, generator=generator)
    return inputs, labelled(teacher, inputs, generator)


def mse(model: nn.Module, inputs: Tensor, targets: Tensor) -> float:
    with torch.no_grad():
        return nn.functional.mse_loss(model(inputs), targets).item()


def train(
    model: nn.Module,
    stream: Iterable[tuple[Tensor, Tensor]],
    lr: float,
    pace: float,
    session: gm.Session | None,
) -> int:
    """AdamW over ``stream``; returns the steps taken."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    taken = 0
    for inputs, targets in stream:
        loss = nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        taken += 1
        if session is not None:
            session.step(samples=len(inputs))
        if pace > 0:
            time.sleep(pace)
    return taken


def solo(setup: Setup, teacher: nn.Module) -> Result:
    """One learner over all four slices of every batch, no mailbox: the reference."""

    def merged() -> Iterator[tuple[Tensor, Tensor]]:
        for step in range(setup.steps):
            slices = [batch(teacher, setup.seed, step, lane) for lane in range(NODES)]
            yield torch.cat([x for x, _ in slices]), torch.cat([y for _, y in slices])

    model = mlp(setup.seed)
    train(model, merged(), setup.lr, 0.0, None)
    return Result(mse(model, *held_out(teacher)), None, 0, ())


def distributed(setup: Setup, algorithm: gm.DiLoCo, teacher: nn.Module) -> Result:
    """Four learners on threads through one latent mailbox; scores the merged model."""

    def load(unit: Unit) -> Iterator[tuple[Tensor, Tensor]]:
        low, high = setup.window
        mailbox.seconds = setup.latency if low <= unit.first / setup.steps < high else 0.0
        for step in range(unit.first, unit.first + unit.count):
            yield batch(teacher, setup.seed, step, unit.lane)

    def node(rank: int) -> None:
        model = copy.deepcopy(student)
        session = gm.Session(
            model,
            algorithm=algorithm,
            mailbox=mailbox,
            learner_id=f"node-{rank}",
            poll_interval=POLL,
            backend=TorchBackend("cpu"),
            lease_every=1.0,
            lease_timeout=3.0,
            callbacks=(),
        )
        with (
            session,
            shared_queue(
                units(setup, rank), mailbox=mailbox, learner_id=f"node-{rank}", load=load
            ) as stream,
        ):
            trained[rank] = train(
                model, stream, setup.lr, setup.step_seconds * setup.paces[rank], session
            )

    async def incorporated() -> tuple[float, int, tuple[float, ...]]:
        exchange = Exchange(mailbox, clock=MonotonicClock())
        manifest = await exchange.manifest()
        assert manifest is not None
        records = await exchange.rounds_after(-1)
        taken = sum(p.steps for record in records for p in record.participants)
        cosines = [record.cosine for record in records]
        tenths = tuple(
            statistics.mean(c for c in chunk if c is not None)
            for t in range(10)
            if (chunk := cosines[t * len(cosines) // 10 : (t + 1) * len(cosines) // 10])
            and any(c is not None for c in chunk)
        )
        return taken / (sum(trained) * len(manifest.plan)), len(records), tenths

    student = mlp(setup.seed)
    mailbox = Latent(gm.MemoryMailbox(), setup.latency)
    trained = [0] * NODES
    threads = [threading.Thread(target=node, args=(rank,)) for rank in range(NODES)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(trained) == NODES * setup.steps, trained
    model = copy.deepcopy(student)
    gm.align(model, mailbox=mailbox)
    share, rounds, cosines = asyncio.run(incorporated())
    return Result(mse(model, *held_out(teacher)), share, rounds, cosines)


def run(setup: Setup) -> Result:
    teacher = mlp(TEACHER_SEED)
    if setup.algorithm is None:
        return solo(setup, teacher)
    return distributed(setup, setup.algorithm, teacher)


def aligned_by(name: str) -> Alignment:
    """``overwrite``, ``carry``, ``blend:<mixing>``, ``adaptive``, ``adaptive:<threshold>`` or
    ``adaptive:<overwrite_below>:<carry_above>``."""
    match name.split(":"):
        case ["overwrite"]:
            return gm.Overwrite()
        case ["carry"]:
            return gm.Carry()
        case ["blend", mixing]:
            return gm.Blend(float(mixing))
        case ["adaptive"]:
            return gm.Adaptive()
        case ["adaptive", threshold]:
            return gm.Adaptive(float(threshold))
        case ["adaptive", below, above]:
            return gm.Adaptive(carry_above=float(above), overwrite_below=float(below))
        case _:
            raise ValueError(
                "--alignment takes overwrite, carry, blend:<mixing> or "
                f"adaptive[:<below>[:<above>]], not {name!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--alignment", nargs="+", default=["overwrite", "carry"])
    parser.add_argument("--latency", nargs="+", type=float, default=[0.0, 0.005])
    parser.add_argument("--latency-window", nargs=2, type=float, default=[0.0, 1.0])
    parser.add_argument("--follow", type=int, default=None)
    parser.add_argument("--grace", type=float, default=None)
    parser.add_argument("--quorum", type=int, default=3)
    parser.add_argument("--sync-every", type=int, default=50)
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument("--paces", nargs=NODES, type=float, default=[1.0] * NODES)
    parser.add_argument("--step-seconds", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--unit", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--solo", action="store_true", help="add the one-learner reference")
    args = parser.parse_args()
    torch.set_num_threads(1)

    def algorithm(alignment: str) -> gm.DiLoCo:
        return gm.DiLoCo(
            sync_every=args.sync_every,
            fragments=args.fragments,
            quorum_size=args.quorum,
            follow=args.follow,
            grace_seconds=args.grace,
            alignment=aligned_by(alignment),
            merge=gm.WeightedMean(),
        )

    low, high = args.latency_window
    window = "" if (low, high) == (0.0, 1.0) else f" in {low:g}-{high:g}"
    configurations: list[tuple[str, gm.DiLoCo | None, float]] = [
        (f"{alignment} latency={latency:g}{window}", algorithm(alignment), latency)
        for alignment in args.alignment
        for latency in args.latency
    ]
    if args.solo:
        configurations.insert(0, ("solo", None, 0.0))
    print(f"{NODES} nodes, {args.steps} steps of {BATCH} each, noise floor {NOISE**2:.4f}")
    for label, config, latency in configurations:
        results = [
            run(
                Setup(
                    config,
                    seed,
                    args.steps,
                    args.unit,
                    args.lr,
                    args.step_seconds,
                    latency,
                    (low, high),
                    tuple(args.paces),
                )
            )
            for seed in range(args.seeds)
        ]
        losses = [r.loss for r in results]
        spread = f" ± {statistics.stdev(losses):.4f}" if len(losses) > 1 else ""
        incorporated = [r.incorporated for r in results if r.incorporated is not None]
        detail = (
            f"  incorporated {statistics.mean(incorporated):.0%}"
            f"  rounds {statistics.mean(r.rounds for r in results):.0f}"
            if incorporated
            else ""
        )
        # The seeds vary the student and the data alike for every configuration, so the
        # comparison between configurations is seed by seed.
        by_seed = " ".join(f"{loss:.4f}" for loss in losses)
        print(f"{label:<28} loss {statistics.mean(losses):.4f}{spread}{detail}", flush=True)
        print(f"{'':<28} seeds {by_seed}", flush=True)
        if tenths := [r.cosines for r in results if r.cosines]:
            by_tenth = " ".join(f"{statistics.mean(c):.2f}" for c in zip(*tenths, strict=True))
            print(f"{'':<28} cosine by tenth {by_tenth}", flush=True)


if __name__ == "__main__":
    main()
