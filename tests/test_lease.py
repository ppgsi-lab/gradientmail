"""Liveness judged from lease listings on the observer's own clock."""

from gradientmail.lease import Liveness


def describe_liveness():
    def it_keeps_a_learner_live_while_its_beat_changes():
        liveness = Liveness(timeout=2.0).observed({"a": 0}, now=0.0).observed({"a": 1}, now=1.5)

        assert liveness.live(3.4) == {"a"}
        assert liveness.live(3.5) == frozenset()

    def when_a_lease_leaves_the_listing():
        def it_drops_the_learner_at_once():
            liveness = Liveness(timeout=2.0).observed({"a": 0, "b": 0}, now=0.0)

            assert liveness.observed({"a": 0}, now=0.1).live(0.1) == {"a"}
