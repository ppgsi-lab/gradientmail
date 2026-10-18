"""Every installed backend against the outer step's contract: bytes in and out, the Value
arithmetic through weighted mean, RDA and Nesterov, and round closure. Values are read back
through F32 bytes, so no array library is needed to check them."""

import math
import random
import struct
from array import array
from collections.abc import Mapping, Sequence
from importlib.util import find_spec

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from gradientmail.adapters import Backend, Value
from gradientmail.algorithms import Aggregator, Contribution, FragmentState, Participant
from gradientmail.algorithms.diloco import (
    DiLoCo,
    DiLoCoAggregator,
    Merge,
    Rda,
    WeightedMean,
    nesterov,
    rda,
    weighted_mean,
)
from gradientmail.serialization import DType, ParameterSpec, Schema, TensorBytes

BACKENDS = ("torch", "mlx")


def make_backend(name: str) -> Backend:
    match name:
        case "torch":
            pytest.importorskip("torch")
            from gradientmail.adapters.torch import TorchBackend

            return TorchBackend()
        case "mlx":
            pytest.importorskip("mlx.core")
            from gradientmail.adapters.mlx import MlxBackend

            return MlxBackend()
        case _:
            raise ValueError(name)


def installed_backends() -> list[Backend]:
    return [make_backend(name) for name in BACKENDS if find_spec(name) is not None]


@pytest.fixture(params=BACKENDS)
def backend(request: pytest.FixtureRequest) -> Backend:
    return make_backend(request.param)


def f32(*values: float, shape: tuple[int, ...] | None = None) -> TensorBytes:
    data = struct.pack(f"<{len(values)}f", *values)
    return TensorBytes("F32", (len(values),) if shape is None else shape, data)


def bf16(*patterns: int) -> TensorBytes:
    return TensorBytes("BF16", (len(patterns),), struct.pack(f"<{len(patterns)}H", *patterns))


def floats(tensor: TensorBytes) -> list[float]:
    assert tensor.dtype == "F32"
    return array("f", tensor.data).tolist()


def as_list(value: Value) -> list[float]:
    return floats(value.to_bytes("F32"))


def read(backend: Backend, tensor: TensorBytes) -> list[float]:
    return as_list(backend.load(tensor))


def contribution(
    learner: str,
    tensors: Mapping[str, TensorBytes],
    *,
    tokens: int,
    steps: int = 1,
    final: bool = False,
) -> Contribution:
    return Contribution(
        learner, steps, steps, tokens, step_seconds=0.1, final=final, tensors=tensors
    )


def aggregator(
    backend: Backend,
    dtype: DType,
    names: Sequence[str],
    *,
    merge: Merge,
    lr: float,
    momentum: float,
) -> Aggregator:
    schema = Schema(tuple(ParameterSpec(name, dtype, (2,)) for name in sorted(names)))
    config = DiLoCo(
        sync_every=10, fragments=1, in_flight=1, outer_lr=lr, outer_momentum=momentum, merge=merge
    )
    return config.aggregator(schema, names, backend)


def same(actual: Sequence[float], expected: Sequence[float]) -> bool:
    return len(actual) == len(expected) and all(
        (math.isnan(a) and math.isnan(e)) or a == e for a, e in zip(actual, expected, strict=True)
    )


def describe_load_and_to_bytes():
    @pytest.mark.parametrize("dtype", ["F32", "F16", "BF16"])
    def it_round_trips_scalars_matrices_and_empty_tensors(backend: Backend, dtype: DType):
        scalar = f32(math.nan, shape=())
        matrix = f32(math.nan, 1.0, -2.0, math.inf, shape=(2, 2))
        empty = f32(shape=(0, 3))

        def round_trip(tensor: TensorBytes) -> tuple[TensorBytes, list[float]]:
            narrowed = backend.load(tensor).to_bytes(dtype)
            again = backend.load(narrowed).to_bytes(dtype)
            assert again == narrowed
            return narrowed, read(backend, narrowed)

        scalar_back, scalar_values = round_trip(scalar)
        matrix_back, matrix_values = round_trip(matrix)
        empty_back, empty_values = round_trip(empty)

        assert (scalar_back.dtype, scalar_back.shape) == (dtype, ())
        assert math.isnan(scalar_values[0])
        assert (matrix_back.dtype, matrix_back.shape) == (dtype, (2, 2))
        assert same(matrix_values, [math.nan, 1.0, -2.0, math.inf])
        assert (empty_back.dtype, empty_back.shape, empty_back.data) == (dtype, (0, 3), b"")
        assert empty_values == []

    def it_rounds_to_nearest_even_on_ties_in_bf16(backend: Backend):
        halfway_below_odd = 1 + 2**-8
        halfway_above_odd = 1 + 3 * 2**-8

        rounded = backend.load(f32(halfway_below_odd, halfway_above_odd)).to_bytes("BF16")

        assert rounded == bf16(0x3F80, 0x3F82)
        assert read(backend, rounded) == [1.0, 1.015625]

    def it_encodes_bf16_like_torch_on_every_normal_pattern(backend: Backend):
        from gradientmail.adapters.torch import TorchBackend

        reference = make_backend("torch")
        assert isinstance(reference, TorchBackend)
        lows = (0x0000, 0x0001, 0x7FFF, 0x8000, 0x8001, 0xFFFF)
        bits = [(high << 16) | low for high in range(0x10000) for low in lows]
        values = TensorBytes("F32", (len(bits),), struct.pack(f"<{len(bits)}I", *bits))

        theirs = array("H", backend.load(values).to_bytes("BF16").data)
        ours = array("H", reference.load(values).to_bytes("BF16").data)

        for pattern, mine, other in zip(bits, theirs, ours, strict=True):
            exponent, mantissa = (pattern >> 23) & 0xFF, pattern & 0x7FFFFF
            if exponent == 0xFF and mantissa:
                assert (mine & 0x7FFF) > 0x7F80, f"{pattern:08x}: NaN did not stay NaN"
            elif exponent == 0 and mantissa:
                # A backend may flush subnormals to zero (MLX does), nothing else.
                assert mine in (other, other & 0x8000), f"{pattern:08x}: {mine:04x} vs {other:04x}"
            else:
                assert mine == other, f"{pattern:08x}: {mine:04x} vs {other:04x}"


def describe_dot():
    def it_sums_the_elementwise_products(backend: Backend):
        assert backend.load(f32(1, 2, 3)).dot(backend.load(f32(4, -5, 6))) == pytest.approx(12.0)


def describe_quantize():
    def when_e4m3():
        def it_scales_the_largest_magnitude_under_256_and_keeps_4_significant_bits(
            backend: Backend,
        ):
            data, scale = backend.load(f32(1.0, -0.5, 0.001, 300.0, 0.0)).quantize("F8_E4M3")

            assert (data.dtype, data.shape, scale.dtype, scale.shape) == (
                "F8_E4M3",
                (5,),
                "F32",
                (),
            )
            assert floats(scale) == [2.0]
            # 150 lies between 144 and 160 on the E4M3 grid; 0.0005 is below half its smallest
            # subnormal.
            assert as_list(backend.dequantize(data, scale, (5,))) == [1.0, -0.5, 0.0, 288.0, 0.0]

    def when_e2m1():
        def it_packs_the_grid_low_nibble_first_with_one_e8m0_scale_per_block(backend: Backend):
            values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -6.0] + [0.0] * 22

            data, scale = backend.load(f32(*values)).quantize("F4")

            assert (data.dtype, data.shape, scale.dtype, scale.shape) == (
                "F4",
                (32,),
                "F8_E8M0",
                (1,),
            )
            assert data.data == bytes([0x10, 0x32, 0x54, 0x76, 0xF9]) + bytes(11)
            assert scale.data == bytes([127])
            assert as_list(backend.dequantize(data, scale, (32,))) == values

        def it_pads_to_whole_blocks_and_scales_each_block_by_its_largest_magnitude(
            backend: Backend,
        ):
            values = [8.0] * 32 + [0.75]

            data, scale = backend.load(f32(*values, shape=(3, 11))).quantize("F4")
            back = backend.dequantize(data, scale, (3, 11)).to_bytes("F32")

            assert (data.shape, scale.shape) == ((64,), (2,))
            # 8 sits in [4, 8) at a scale of 2; 0.75 alone in its block at a scale of 2^-3.
            assert scale.data == bytes([127 + 1, 127 - 3])
            assert (back.shape, floats(back)) == ((3, 11), values)

    def it_keeps_the_error_within_the_formats_precision(backend: Backend):
        generator = random.Random(5)
        values = [generator.gauss(0.0, 1.0) * 10 ** generator.uniform(-3.0, 3.0) for _ in range(70)]
        largest = max(abs(v) for v in values)

        e4m3 = as_list(backend.dequantize(*backend.load(f32(*values)).quantize("F8_E4M3"), (70,)))
        e2m1 = as_list(backend.dequantize(*backend.load(f32(*values)).quantize("F4"), (70,)))

        assert max(abs(a - b) for a, b in zip(values, e4m3, strict=True)) <= largest / 16
        for start in range(0, 70, 32):
            block = values[start : start + 32]
            block_largest = max(abs(v) for v in block)
            assert all(
                abs(a - b) <= block_largest / 4
                for a, b in zip(block, e2m1[start : start + 32], strict=True)
            )

    def it_produces_the_same_bytes_and_floats_on_every_installed_backend():
        backends = installed_backends()
        if len(backends) < 2:
            pytest.skip("needs two backends")
        generator = random.Random(9)
        values = f32(
            *(generator.gauss(0.0, 1.0) * 2.0 ** generator.randint(-20, 20) for _ in range(100))
        )

        for dtype in ("F8_E4M3", "F4"):
            quantized = [backend.load(values).quantize(dtype) for backend in backends]
            assert all(q == quantized[0] for q in quantized)
            data, scale = quantized[0]
            back = [as_list(backend.dequantize(data, scale, (100,))) for backend in backends]
            assert all(b == back[0] for b in back)

    def when_the_tensor_is_empty():
        def it_round_trips_as_empty(backend: Backend):
            for dtype in ("F8_E4M3", "F4"):
                data, scale = backend.load(f32(shape=(0, 3))).quantize(dtype)
                back = backend.dequantize(data, scale, (0, 3)).to_bytes("F32")
                assert (back.shape, back.data) == ((0, 3), b"")

    def it_refuses_to_load_quantized_bytes_without_their_scale(backend: Backend):
        with pytest.raises(ValueError):
            backend.load(TensorBytes("F8_E4M3", (2,), bytes(2)))


def describe_merge():
    def when_rule_is_weighted_mean():
        def it_matches_the_analytic_mean(backend: Backend):
            merged = weighted_mean([backend.load(f32(1, 0)), backend.load(f32(0, 1))], [4.0, 1.0])

            assert as_list(merged) == pytest.approx([0.8, 0.2], rel=1e-6)

    def when_rule_is_rda():
        direction = st.lists(st.floats(-1.0, 1.0), min_size=3, max_size=3).filter(
            lambda d: math.hypot(*d) > 0.1
        )

        @pytest.mark.parametrize("name", BACKENDS)
        @given(
            st.lists(st.tuples(direction, st.floats(1e-3, 1e3)), min_size=1, max_size=4),
            st.floats(1e-3, 1e3),
        )
        def it_preserves_the_common_norm(
            name: str, contributions: list[tuple[list[float], float]], radius: float
        ):
            backend = make_backend(name)
            units = [[x / math.hypot(*d) for x in d] for d, _ in contributions]
            weights = [w for _, w in contributions]
            total = sum(weights)
            mean_direction = [
                sum(w * u[i] for w, u in zip(weights, units, strict=True)) / total for i in range(3)
            ]
            assume(math.hypot(*mean_direction) > 1e-3)

            merged = rda([backend.load(f32(*(radius * x for x in u))) for u in units], weights)

            assert merged.norm() == pytest.approx(radius, rel=1e-5)

        def it_equals_weighted_mean_for_identical_contributions(backend: Backend):
            delta = backend.load(f32(3.0, -4.0))

            merged = rda([delta, delta, delta], [1.0, 7.0, 2.0])

            assert as_list(merged) == pytest.approx([3.0, -4.0], rel=1e-6)

        def when_directions_cancel():
            def it_returns_zeros(backend: Backend):
                merged = rda([backend.load(f32(1, 0)), backend.load(f32(-1, 0))], [1.0, 1.0])

                assert as_list(merged) == [0.0, 0.0]

    def it_routes_embedding_tensors_to_the_weighted_mean(backend: Backend):
        names = ["mlp.weight", "tok_embed.weight"]
        state = FragmentState(
            version=0,
            theta={name: f32(1, 1) for name in names},
            outer={f"momentum.{name}": f32(0, 0) for name in names},
        )
        contributions = [
            contribution("a", {name: f32(1, 0) for name in names}, tokens=2),
            contribution("b", {name: f32(0, 1) for name in names}, tokens=1),
        ]
        close = aggregator(backend, "F32", names, merge=Rda(), lr=1.0, momentum=0.0)

        new_state = close.close_round(state, contributions).state

        assert floats(new_state.theta["tok_embed.weight"]) == pytest.approx(
            [1 - 0.8, 1 - 0.2], rel=1e-6
        )
        assert floats(new_state.theta["mlp.weight"]) == pytest.approx(
            [1 - 0.9701, 1 - 0.2425], abs=1e-4
        )


def describe_nesterov():
    def when_lr_is_one_and_momentum_zero():
        def it_subtracts_the_weighted_mean_delta(backend: Backend):
            state = FragmentState(
                version=3, theta={"w": f32(1, 1)}, outer={"momentum.w": f32(0, 0)}
            )
            contributions = [
                contribution("a", {"w": f32(1, 0)}, tokens=2),
                contribution("b", {"w": f32(0, 1)}, tokens=1),
            ]
            close = aggregator(backend, "F32", ["w"], merge=WeightedMean(), lr=1.0, momentum=0.0)

            result = close.close_round(state, contributions)
            new_state, participants = result.state, result.participants

            assert floats(new_state.theta["w"]) == pytest.approx([0.2, 0.8], rel=1e-6)
            assert new_state.version == 4
            assert set(new_state.outer) == {"momentum.w"}
            assert participants == (
                Participant("a", 1, 1, 2, 4.0),
                Participant("b", 1, 1, 1, 1.0),
            )

    def it_accumulates_momentum_across_two_hand_computed_rounds(backend: Backend):
        zero = backend.load(f32(0, 0))

        first, momentum = nesterov(zero, zero, backend.load(f32(1, 0)), lr=0.7, beta=0.9)
        second, momentum = nesterov(first, momentum, backend.load(f32(0, 1)), lr=0.7, beta=0.9)

        assert as_list(first) == pytest.approx([-1.33, 0.0], rel=1e-6)
        assert as_list(momentum) == pytest.approx([0.9, 1.0], rel=1e-6)
        assert as_list(second) == pytest.approx([-1.897, -1.33], rel=1e-6)


def normal_tensor(
    generator: random.Random, size: int, dtype: DType, backend: Backend
) -> TensorBytes:
    values = f32(*(generator.gauss(0.0, 1.0) for _ in range(size)))
    return backend.load(values).to_bytes(dtype)


def describe_close_round():
    def it_is_bitwise_deterministic_on_one_backend(backend: Backend):
        generator = random.Random(7)
        names = ["a", "b"]
        schema = Schema(tuple(ParameterSpec(name, "BF16", (64,)) for name in names))
        config = DiLoCo(sync_every=10, fragments=1, in_flight=1)
        close = config.aggregator(schema, names, backend)
        state = FragmentState(
            version=0,
            theta={n: normal_tensor(generator, 64, "BF16", backend) for n in names},
            outer={f"momentum.{n}": f32(*([0.0] * 64)) for n in names},
        )
        contributions = [
            contribution(
                learner,
                {n: normal_tensor(generator, 64, "BF16", backend) for n in names},
                tokens=tokens,
            )
            for learner, tokens in [("x", 5), ("y", 3), ("z", 8)]
        ]

        first = close.close_round(state, contributions)
        second = close.close_round(state, contributions)

        assert first == second

    def it_agrees_across_installed_backends_to_float_tolerance():
        generator = random.Random(11)
        names = ["a", "b"]
        schema = Schema(tuple(ParameterSpec(name, "F32", (64,)) for name in names))
        config = DiLoCo(sync_every=10, fragments=1, in_flight=1)
        state = FragmentState(
            version=0,
            theta={n: f32(*(generator.gauss(0.0, 1.0) for _ in range(64))) for n in names},
            outer={f"momentum.{n}": f32(*([0.0] * 64)) for n in names},
        )
        contributions = [
            contribution(
                learner,
                {n: f32(*(generator.gauss(0.0, 1.0) for _ in range(64))) for n in names},
                tokens=tokens,
            )
            for learner, tokens in [("x", 5), ("y", 3), ("z", 8)]
        ]

        results = [
            config.aggregator(schema, names, backend).close_round(state, contributions).state
            for backend in installed_backends()
        ]

        assert results
        for other in results[1:]:
            for name in names:
                assert floats(other.theta[name]) == pytest.approx(
                    floats(results[0].theta[name]), rel=1e-6
                )

    def it_keeps_an_fp32_master_for_bf16_schemas(backend: Backend):
        anchor = bf16(0x3F80, 0x3F80)  # 1.0
        one_ulp_below = bf16(0x3F7F, 0x3F7F)  # 1 - 2**-8
        one_ulp = bf16(0x3B80, 0x3B80)  # 2**-8: the delta of a learner at one_ulp_below
        contributions = [contribution("a", {"w": one_ulp}, tokens=1)]
        close = aggregator(backend, "BF16", ["w"], merge=WeightedMean(), lr=0.25, momentum=0.0)
        state = FragmentState(0, {"w": anchor}, {"momentum.w": f32(0, 0)})

        after_one = close.close_round(state, contributions).state
        after_two = close.close_round(after_one, contributions).state
        after_three = close.close_round(after_two, contributions).state

        assert set(after_one.outer) == {"master.w", "momentum.w"}
        assert after_one.theta["w"] == anchor
        assert floats(after_one.outer["master.w"]) == [1 - 2**-10, 1 - 2**-10]
        assert after_two.theta["w"] == anchor
        assert after_three.theta["w"] == one_ulp_below
        assert floats(after_three.outer["master.w"]) == [1 - 3 * 2**-10] * 2

    def it_records_the_mean_cosine_between_the_deltas_over_the_fragment(backend: Backend):
        names = ["u", "v"]
        state = FragmentState(
            version=0,
            theta={name: f32(0, 0) for name in names},
            outer={f"momentum.{name}": f32(0, 0) for name in names},
        )
        # Over both tensors: a = (1, 0, 0, 1), b = (1, 0, 0, 0), c = (0, 1, 0, 0); a·b = 1/√2,
        # a·c = 0, b·c = 0.
        contributions = [
            contribution("a", {"u": f32(1, 0), "v": f32(0, 1)}, tokens=1),
            contribution("b", {"u": f32(1, 0), "v": f32(0, 0)}, tokens=1),
            contribution("c", {"u": f32(0, 1), "v": f32(0, 0)}, tokens=1),
        ]
        close = aggregator(backend, "F32", names, merge=WeightedMean(), lr=1.0, momentum=0.0)

        result = close.close_round(state, contributions)

        assert result.cosine == pytest.approx(math.sqrt(0.5) / 3)
        assert close.close_round(state, contributions[:1]).cosine is None

    def when_commits_carry_an_update():
        def it_publishes_the_anchor_plus_the_update_as_it_dequantizes_and_keeps_the_master(
            backend: Backend,
        ):
            state = FragmentState(
                version=0, theta={"w": f32(0, 0)}, outer={"momentum.w": f32(0, 0)}
            )
            config = DiLoCo(
                sync_every=10,
                fragments=1,
                in_flight=1,
                outer_lr=1.0,
                outer_momentum=0.0,
                merge=WeightedMean(),
                commit_dtype="F4",
            )
            close = config.aggregator(Schema((ParameterSpec("w", "F32", (2,)),)), ["w"], backend)

            # The master moves to (5, 7); on the E2M1 grid at a scale of one that is (4, 6).
            first = close.close_round(state, [contribution("a", {"w": f32(-5, -7)}, tokens=1)])
            # Nobody moves: the update carries what the first commit left in the master.
            second = close.close_round(first.state, [contribution("a", {"w": f32(0, 0)}, tokens=1)])

            assert floats(first.state.theta["w"]) == [4.0, 6.0]
            assert floats(first.state.outer["master.w"]) == [5.0, 7.0]
            assert first.update is not None and set(first.update) == {"w", "w.scale"}
            assert floats(second.state.theta["w"]) == [5.0, 7.0]

    def when_a_learner_did_not_move():
        def it_contributes_a_zero_delta(backend: Backend):
            published = bf16(0x3F80, 0x3F80)
            master = f32(1 - 2**-10, 1 - 2**-10)
            state = FragmentState(
                version=2,
                theta={"w": published},
                outer={"master.w": master, "momentum.w": f32(0, 0)},
            )
            close = aggregator(backend, "BF16", ["w"], merge=Rda(), lr=0.7, momentum=0.9)

            new_state = close.close_round(
                state, [contribution("idle", {"w": bf16(0x0000, 0x0000)}, tokens=1)]
            ).state

            assert new_state.theta["w"] == published
            assert new_state.outer["master.w"] == master
            assert floats(new_state.outer["momentum.w"]) == [0.0, 0.0]

    def when_every_contribution_is_final():
        def it_moves_by_the_merged_delta_and_keeps_the_momentum(backend: Backend):
            state = FragmentState(
                version=3, theta={"w": f32(1, 1)}, outer={"momentum.w": f32(4, -2)}
            )
            contributions = [
                contribution("a", {"w": f32(1, 0)}, tokens=2, final=True),
                contribution("b", {"w": f32(0, 1)}, tokens=1, final=True),
            ]
            close = aggregator(backend, "F32", ["w"], merge=WeightedMean(), lr=0.7, momentum=0.9)

            new_state = close.close_round(state, contributions).state

            assert floats(new_state.theta["w"]) == pytest.approx([0.2, 0.8], rel=1e-6)
            assert new_state.outer == {"momentum.w": f32(4, -2)}
            assert new_state.version == 4

    def when_a_contribution_is_not_final():
        def it_takes_the_nesterov_step(backend: Backend):
            state = FragmentState(
                version=3, theta={"w": f32(1, 1)}, outer={"momentum.w": f32(4, -2)}
            )
            contributions = [
                contribution("a", {"w": f32(1, 0)}, tokens=2, final=True),
                contribution("b", {"w": f32(0, 1)}, tokens=1, final=False),
            ]
            close = aggregator(backend, "F32", ["w"], merge=WeightedMean(), lr=0.7, momentum=0.9)

            new_state = close.close_round(state, contributions).state

            # Δ = (0.8, 0.2), m = 0.9·(4, -2) + Δ, Θ = (1, 1) - 0.7·(Δ + 0.9·m).
            assert floats(new_state.theta["w"]) == pytest.approx([-2.332, 1.868], rel=1e-6)
            assert floats(new_state.outer["momentum.w"]) == pytest.approx([4.4, -1.6], rel=1e-6)


def describe_decoupled_diloco():
    def it_builds_a_diloco_aggregator_on_the_given_backend(backend: Backend):
        schema = Schema((ParameterSpec("w", "F32", (2,)),))
        algorithm = DiLoCo(sync_every=10, fragments=1, in_flight=1)

        built = algorithm.aggregator(schema, ["w"], backend)

        assert isinstance(built, Aggregator)
        assert built == DiLoCoAggregator(
            config=algorithm, schema=schema, names=["w"], backend=backend
        )
