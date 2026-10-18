import math
import random
from dataclasses import dataclass

import pytest
from fakes import FakeClock

from gradientmail.clock import StoreClock

EPOCH = 1_792_000_000.0
"""The store's clock reads true time plus this."""


@dataclass(frozen=True)
class Node:
    origin: float
    """What the local monotonic clock read at true time zero."""
    wall_error: float
    """How far the node's wall clock is from the store's, the offset it starts from."""
    latency: tuple[float, float]
    """Bounds of each one-way trip, in seconds."""


def node_clock(node: Node) -> tuple[FakeClock, StoreClock]:
    local = FakeClock(start=node.origin)
    return local, StoreClock(local, offset=EPOCH - node.origin + node.wall_error)


def observe(clock: StoreClock, node: Node, rng: random.Random, sent_at: float) -> None:
    """One request sent at true time ``sent_at``, stamped by the store on arrival."""
    forward, back = rng.uniform(*node.latency), rng.uniform(*node.latency)
    date = math.floor(sent_at + forward + EPOCH)
    clock.observe(node.origin + sent_at, node.origin + sent_at + forward + back, date)


def describe_store_clock():
    def it_reads_local_time_plus_the_initial_offset_before_any_observation():
        local = FakeClock(start=50.0)
        clock = StoreClock(local, offset=1000.0)
        local.advance(2.5)

        assert clock.now() == 1052.5

    def it_keeps_an_offset_every_observation_allows():
        node = Node(origin=7.0, wall_error=0.004, latency=(0.02, 0.15))
        _, clock = node_clock(node)
        rng = random.Random(0)
        for second in range(60):
            observe(clock, node, rng, sent_at=second + rng.random())

        assert clock.now() == pytest.approx(EPOCH + 0.004)

    def it_moves_an_offset_the_observations_rule_out_into_their_window():
        node = Node(origin=7.0, wall_error=-2.7, latency=(0.02, 0.15))
        _, clock = node_clock(node)
        rng = random.Random(0)
        observe(clock, node, rng, sent_at=0.0)

        assert abs(clock.now() - EPOCH) < 1.3

        for second in range(1, 60):
            observe(clock, node, rng, sent_at=second + rng.random())

        assert abs(clock.now() - EPOCH) < 0.15

    def it_starts_the_window_over_when_the_local_clock_steps():
        node = Node(origin=7.0, wall_error=0.0, latency=(0.02, 0.15))
        local, clock = node_clock(node)
        rng = random.Random(0)
        for second in range(30):
            observe(clock, node, rng, sent_at=second + rng.random())
        stepped = Node(origin=node.origin + 5.0, wall_error=0.0, latency=node.latency)
        local.advance(5.0)
        for second in range(30, 60):
            observe(clock, stepped, rng, sent_at=second + rng.random())

        assert abs(clock.now() - EPOCH) < 0.15

    @pytest.mark.parametrize("seed", range(5))
    def it_brings_nodes_whose_clocks_disagree_to_one_store_time(seed: int):
        # Wall clocks between 2.7 s behind and 41 s ahead; one-way trips from 5 ms to 300 ms,
        # each node sending one request a second, at any fraction of it, for a minute.
        nodes = (
            Node(origin=12.5, wall_error=0.0, latency=(0.005, 0.03)),
            Node(origin=86_400.0, wall_error=-2.7, latency=(0.04, 0.12)),
            Node(origin=3.25, wall_error=0.6, latency=(0.02, 0.3)),
            Node(origin=999.0, wall_error=41.3, latency=(0.06, 0.08)),
        )
        rng = random.Random(seed)
        clocks = [node_clock(node) for node in nodes]
        for node, (_, clock) in zip(nodes, clocks, strict=True):
            for second in range(60):
                observe(clock, node, rng, sent_at=second + rng.random())

        for local, _ in clocks:
            local.advance(61.0)
        readings = [clock.now() for _, clock in clocks]

        assert max(readings) - min(readings) < 0.15
        for node, reading in zip(nodes, readings, strict=True):
            assert abs(reading - (EPOCH + 61.0)) < node.latency[1]
