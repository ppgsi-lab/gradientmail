"""Phase 2 demo: two synthetic sessions sharing a MemoryMailbox, no separate syncer.

Each learner is a MemoryAdapter over two small tensors. Its "training" adds a constant of its
own to every parameter per step, at its own pace and batch size, so the value every committed
fragment must hold is known from the round record alone: with quorum 2, lr 1, no momentum and
the weighted mean, a commit is the weighted average of the fragments the learners published,
each being the learner's anchor plus ``steps * constant``. Which session closes a round is
settled through the mailbox; the outer step runs on the installed backend (torch or MLX). The
script prints every commit next to that expectation, then aligns each learner with the newest
commit and shows its parameters.

    uv run python examples/synthetic.py --rounds 6
"""

import argparse
import asyncio
import threading
import time
from array import array
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from gradientmail.adapters import MemoryAdapter
from gradientmail.algorithms.diloco import DiLoCo, WeightedMean
from gradientmail.clock import MonotonicClock
from gradientmail.mailbox.memory import MemoryMailbox
from gradientmail.protocol import Exchange, RoundRecord
from gradientmail.serialization import TensorBytes
from gradientmail.session import Session, align

POLL = 0.01


@dataclass(frozen=True)
class Profile:
    """How a synthetic learner trains: what a step adds, how many samples it reports, how often."""

    node_id: str
    delta: float
    samples: int
    pace: float


FAST = Profile("fast", delta=0.25, samples=2, pace=0.001)
SLOW = Profile("slow", delta=-0.5, samples=5, pace=0.004)
DELTAS = {FAST.node_id: FAST.delta, SLOW.node_id: SLOW.delta}
ALGORITHM = DiLoCo(
    sync_every=8,
    fragments=2,
    in_flight=2,
    quorum_size=2,
    outer_lr=1.0,
    outer_momentum=0.0,
    merge=WeightedMean(),
)


def f32(*values: float) -> TensorBytes:
    return TensorBytes("F32", (len(values),), array("f", values).tobytes())


def values(tensor: TensorBytes) -> list[float]:
    return array("f", tensor.data).tolist()


def initial_tensors() -> Mapping[str, TensorBytes]:
    return {"a": f32(1.0, 2.0), "b": f32(3.0, 4.0)}


def train(
    profile: Profile, adapter: MemoryAdapter, session: Session, stop: threading.Event
) -> None:
    """The training loop of one learner: a step shifts every parameter, until ``stop`` is set."""

    def shifted(tensor: TensorBytes) -> TensorBytes:
        return f32(*(value + profile.delta for value in values(tensor)))

    def step(_: None) -> int:
        adapter.apply({name: shifted(tensor) for name, tensor in adapter.tensors.items()})
        time.sleep(profile.pace)
        return profile.samples

    def steps() -> Iterator[None]:
        while not stop.is_set():
            yield None

    session.run(step, steps())


def expected(
    theta: Mapping[str, Sequence[float]], record: RoundRecord, names: Sequence[str]
) -> Mapping[str, Sequence[float]]:
    """The fragment ``record`` must have committed, given the anchor ``theta`` it started from."""
    total = sum(p.weight for p in record.participants)
    return {
        name: [
            sum(p.weight * (x + p.steps * DELTAS[p.learner]) for p in record.participants) / total
            for x in theta[name]
        ]
        for name in names
    }


async def report(exchange: Exchange, rounds: int) -> None:
    """Print each commit as it lands, next to its analytic expectation, through ``rounds``."""

    def rounded(vector: Sequence[float]) -> str:
        return "[" + ", ".join(f"{x:.4f}" for x in vector) + "]"

    manifest = await exchange.manifest()
    assert manifest is not None
    theta: dict[str, Sequence[float]] = {
        name: values(tensor) for name, tensor in initial_tensors().items()
    }
    cursor = -1
    while cursor < rounds - 1:
        for record in await exchange.rounds_after(cursor):
            names = manifest.plan[record.fragment]
            committed = await exchange.fetch(record.theta)
            assert committed is not None
            wanted = expected(theta, record, names)
            theta.update(wanted)
            cursor = record.round
            who = "  ".join(
                f"{p.learner}: steps={p.steps} tokens={p.tokens} weight={p.weight:g}"
                for p in record.participants
            )
            print(
                f"round {record.round}  fragment {record.fragment} -> version {record.version}"
                f"  {who}"
            )
            for name in names:
                print(
                    f"    {name}: committed {rounded(values(committed[name]))}"
                    f"  expected {rounded(wanted[name])}"
                )
        await asyncio.sleep(POLL)


def newest(exchange: Exchange) -> Mapping[str, TensorBytes]:
    """The newest committed version of every fragment."""

    async def fetch() -> Mapping[str, TensorBytes]:
        records = await exchange.rounds_after(-1)
        by_fragment = {record.fragment: record for record in records}
        tensors: dict[str, TensorBytes] = {}
        for record in by_fragment.values():
            fetched = await exchange.fetch(record.theta)
            assert fetched is not None
            tensors.update(fetched)
        return tensors

    return asyncio.run(fetch())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="two synthetic sessions sharing a MemoryMailbox, no separate syncer"
    )
    parser.add_argument("--rounds", type=int, default=6, help="rounds to commit (default: 6)")
    rounds: int = parser.parse_args().rounds

    mailbox = MemoryMailbox()
    exchange = Exchange(mailbox, clock=MonotonicClock())
    stop = threading.Event()
    adapters = {profile: MemoryAdapter(initial_tensors()) for profile in (FAST, SLOW)}

    gap = ALGORITHM.sync_every // ALGORITHM.fragments
    print(
        f"{ALGORITHM.fragments} fragments, a round every {gap} steps, quorum"
        f" {ALGORITHM.quorum_size}, weighted mean, lr {ALGORITHM.outer_lr},"
        f" momentum {ALGORITHM.outer_momentum}"
    )
    for profile in (FAST, SLOW):
        print(
            f"{profile.node_id}: adds {profile.delta:+g} per step, {profile.samples} samples"
            f" per step, one step every {profile.pace * 1000:g} ms"
        )
    sessions = {
        profile: Session(
            adapter,
            algorithm=ALGORITHM,
            mailbox=mailbox,
            node_id=profile.node_id,
            poll_interval=POLL,
        )
        for profile, adapter in adapters.items()
    }
    for session in sessions.values():
        session.join()
    threads = [
        threading.Thread(
            target=train, args=(profile, adapters[profile], session, stop), daemon=True
        )
        for profile, session in sessions.items()
    ]
    for thread in threads:
        thread.start()
    try:
        asyncio.run(report(exchange, rounds))
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    # A learner's parameters drift by its own training between syncs; aligning loads the newest
    # committed version of every fragment at that moment, without joining the run again.
    for profile, adapter in adapters.items():
        committed = newest(exchange)
        align(adapter, mailbox=mailbox)
        held = "  ".join(f"{name}={values(t)}" for name, t in sorted(adapter.tensors.items()))
        print(f"{profile.node_id} after aligning: {held}")
        print(f"{profile.node_id} holds the newest commit: {dict(adapter.tensors) == committed}")


if __name__ == "__main__":
    main()
