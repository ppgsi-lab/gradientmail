"""Learner liveness: a lease per learner, renewed by beats and judged on the observer's clock.

A learner's lease is the newest of the create-once beat objects under ``leases/{learner}/``
(``protocol`` module). The learner renews it every ``lease_every`` seconds while it runs and
deletes it when it closes. An observer lists the leases on every poll and remembers, per
learner, the newest beat and the moment on its own clock when it first saw that beat: a
learner is live while its beat has changed within ``lease_timeout`` seconds, and gone the
moment its lease leaves the listing. No wall clock crosses machines, so a lease written by a
node with a skewed clock reads like any other; the cost is that an observer trusts a lease it
has just met for one timeout before doubting it.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

__all__ = ["Liveness"]


@dataclass(frozen=True)
class Liveness:
    """Which learners hold a live lease, as one observer has seen it.

    ``seen`` maps a learner to its newest beat and when this observer first saw that beat.
    """

    timeout: float
    seen: Mapping[str, tuple[int, float]] = field(default_factory=dict)

    def observed(self, leases: Mapping[str, int], now: float) -> "Liveness":
        """Fold one listing of the leases, taken at ``now``."""

        def since(learner: str, beat: int) -> float:
            known = self.seen.get(learner)
            return known[1] if known is not None and known[0] == beat else now

        return replace(
            self, seen={learner: (beat, since(learner, beat)) for learner, beat in leases.items()}
        )

    def live(self, now: float) -> frozenset[str]:
        """The learners whose beat has changed within ``timeout`` seconds of ``now``."""
        return frozenset(
            learner for learner, (_, since) in self.seen.items() if now - since < self.timeout
        )
