"""Pure pieces of Decoupled DiLoCo: fragment planning, the round schedule, weights, grace."""

import json
from collections.abc import Mapping
from itertools import pairwise

import pytest

from gradientmail.algorithms.diloco import (
    DecoupledDiLoCo,
    Estimates,
    Merge,
    Rda,
    WeightedMean,
    grace_window,
    plan,
    weight,
)
from gradientmail.serialization import DType, ParameterSpec, Schema


def schema_of(numel: Mapping[str, int]) -> Schema:
    return Schema(tuple(ParameterSpec(name, "F32", (n,)) for name, n in sorted(numel.items())))


def describe_plan():
    def it_assigns_every_tensor_exactly_once():
        schema = schema_of({"a": 7, "b": 3, "c": 5, "d": 1, "e": 2})

        fragments = plan(schema, 2)

        assert len(fragments) == 2
        assert sorted(name for fragment in fragments for name in fragment) == list("abcde")

    def it_places_each_tensor_in_the_lightest_fragment():
        schema = schema_of({"a": 6, "b": 5, "c": 4, "d": 3, "e": 2, "f": 1})

        fragments = plan(schema, 2)

        loads = sorted(sum(schema.nbytes(name) for name in fragment) for fragment in fragments)
        assert loads == [4 * 10, 4 * 11]

    def it_is_deterministic_for_equal_sizes():
        names = [f"block.{i}.weight" for i in range(10)]

        first = plan(schema_of({name: 4 for name in names}), 3)
        second = plan(schema_of({name: 4 for name in reversed(names)}), 3)

        assert first == second
        assert len(first) == 3

    def when_fewer_tensors_than_fragments():
        def it_uses_one_fragment_per_tensor():
            schema = schema_of({"a": 1, "b": 2, "c": 3})

            fragments = plan(schema, 8)

            assert sorted(fragments) == [("a",), ("b",), ("c",)]


def describe_schedule():
    def when_sync_every_is_100_and_fragments_8():
        schedule = DecoupledDiLoCo(sync_every=100, fragments=8).schedule(8)
        rounds = range(8 * 4)

        def it_synchronizes_each_fragment_every_100_steps():
            for cycle in range(4):
                window = range(100 * cycle + 1, 100 * (cycle + 1) + 1)
                synced = [schedule.fragment_of(r) for r in rounds if schedule.step_of(r) in window]
                assert sorted(synced) == list(range(8))

            for fragment in range(8):
                own = [r for r in rounds if schedule.fragment_of(r) == fragment]
                steps = [schedule.step_of(r) for r in own]
                assert [b - a for a, b in pairwise(steps)] == [100, 100, 100]
                assert [schedule.version_after(r) for r in own] == [1, 2, 3, 4]

        def it_has_gaps_summing_to_sync_every():
            gaps = [schedule.gap(r) for r in rounds]

            assert sum(gaps[:8]) == 100
            assert sum(gaps[8:16]) == 100
            assert set(gaps) == {12, 13}
            assert [sum(gaps[: r + 1]) for r in rounds] == [schedule.step_of(r) for r in rounds]


def describe_weight():
    def it_grows_quadratically_with_tokens_and_inversely_with_steps():
        base = weight(tokens=200, steps=4)

        assert base == 10_000.0
        assert weight(tokens=400, steps=4) == 4 * base
        assert weight(tokens=200, steps=8) == base / 2


def describe_estimates():
    def it_takes_the_first_observation_as_is_and_keeps_step_when_none_is_reported():
        seeded = Estimates().updated(step=1.0, quorum=2.0, sync=3.0)

        assert seeded == Estimates(step=1.0, quorum=2.0, sync=3.0)
        assert seeded.updated(step=None, quorum=2.0, sync=3.0) == seeded


def describe_grace_window():
    def when_estimates_are_missing():
        @pytest.mark.parametrize(
            "estimates",
            [Estimates(), Estimates(step=1.0, quorum=0.5), Estimates(quorum=0.5, sync=0.5)],
            ids=["all", "sync", "step"],
        )
        def it_is_zero(estimates: Estimates):
            assert grace_window(estimates, in_flight=2, gap=1) == 0.0

    def when_slack_is_negative():
        def it_is_zero():
            estimates = Estimates(step=0.1, quorum=1.0, sync=1.0)

            assert grace_window(estimates, in_flight=2, gap=1) == 0.0

    def it_waits_half_the_slack():
        estimates = Estimates(step=2.0, quorum=0.5, sync=0.5)

        assert grace_window(estimates, in_flight=2, gap=1) == 1.5

    def it_measures_a_round_as_the_gap_of_steps():
        # Ten steps of one second per round: two rounds in flight leave 20 - 6 of slack.
        estimates = Estimates(step=1.0, quorum=5.0, sync=1.0)

        assert grace_window(estimates, in_flight=2, gap=10) == 7.0

    def it_never_exceeds_one_round():
        estimates = Estimates(step=10.0, quorum=0.0, sync=0.0)

        assert grace_window(estimates, in_flight=4, gap=1) == 10.0


def describe_decoupled_diloco():
    def it_rejects_in_flight_above_fragments():
        with pytest.raises(ValueError):
            DecoupledDiLoCo(sync_every=100, fragments=2, in_flight=3)

    def when_the_model_yields_fewer_fragments_than_in_flight():
        def it_rejects_the_schedule():
            algorithm = DecoupledDiLoCo(sync_every=100, fragments=8, in_flight=4)

            with pytest.raises(ValueError):
                algorithm.schedule(3)

    def it_rejects_a_global_batch_below_one():
        with pytest.raises(ValueError):
            DecoupledDiLoCo(sync_every=100, fragments=2, global_batch_size=0)

    @pytest.mark.parametrize(
        "merge", [WeightedMean(), Rda("tok_embeddings")], ids=["weighted_mean", "rda"]
    )
    @pytest.mark.parametrize("global_batch_size", [None, 512], ids=["own_batches", "shared_batch"])
    @pytest.mark.parametrize("delta_dtype", [None, "BF16"], ids=["own_dtype", "bf16_deltas"])
    def it_round_trips_through_json(
        merge: Merge, global_batch_size: int | None, delta_dtype: DType | None
    ):
        algorithm = DecoupledDiLoCo(
            sync_every=100,
            fragments=8,
            in_flight=3,
            quorum_size=2,
            outer_lr=1.0,
            outer_momentum=0.0,
            merge=merge,
            global_batch_size=global_batch_size,
            delta_dtype=delta_dtype,
        )

        encoded = json.loads(json.dumps(algorithm.to_json(), sort_keys=True))

        assert DecoupledDiLoCo.from_json(encoded) == algorithm

    def it_rejects_an_unknown_delta_dtype_in_json():
        encoded = {**DecoupledDiLoCo(sync_every=100, fragments=2).to_json(), "delta_dtype": "INT8"}

        with pytest.raises(ValueError):
            DecoupledDiLoCo.from_json(encoded)

    def it_describes_deltas_in_its_dtype_or_each_parameters_own():
        schema = schema_of({"a": 2, "b": 3})
        in_bf16 = Schema(tuple(ParameterSpec(p.name, "BF16", p.shape) for p in schema.parameters))

        assert DecoupledDiLoCo(sync_every=100, fragments=2).delta_schema(schema) == schema
        assert (
            DecoupledDiLoCo(sync_every=100, fragments=2, delta_dtype="BF16").delta_schema(schema)
            == in_bf16
        )
