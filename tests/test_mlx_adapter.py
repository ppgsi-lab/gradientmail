"""MlxAdapter against real MLX modules: schema, byte round trip, optimizer state, ties."""

import struct

import pytest

from gradientmail.serialization import ParameterSpec, Schema, SchemaMismatch, TensorBytes

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
optim = pytest.importorskip("mlx.optimizers")

from gradientmail.adapters.mlx import MlxAdapter  # noqa: E402

# The interoperability claim: the same literal must come out of the torch demo model.
EXPECTED_MLP_SCHEMA = Schema(
    (
        ParameterSpec("fc1.bias", "F32", (32,)),
        ParameterSpec("fc1.weight", "F32", (32, 16)),
        ParameterSpec("fc2.bias", "F32", (4,)),
        ParameterSpec("fc2.weight", "F32", (4, 32)),
        ParameterSpec("norm.bias", "F32", (32,)),
        ParameterSpec("norm.weight", "F32", (32,)),
    )
)

MLP_NAMES = tuple(spec.name for spec in EXPECTED_MLP_SCHEMA.parameters)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 4)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.norm(self.fc1(x)))


def describe_mlx_adapter():
    def it_produces_the_expected_mlp_schema():
        assert MlxAdapter(MLP()).schema() == EXPECTED_MLP_SCHEMA

    @pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16], ids=str)
    def it_round_trips_snapshot_apply_byte_exactly(dtype: mx.Dtype):
        source, target = MLP(), MLP()
        source.set_dtype(dtype)
        target.set_dtype(dtype)
        published = MlxAdapter(source).snapshot(MLP_NAMES)
        adapter = MlxAdapter(target)

        adapter.apply(published)

        assert adapter.snapshot(MLP_NAMES) == published
        assert mx.array_equal(target.fc1.weight, source.fc1.weight)
        assert target.fc1.weight.dtype == dtype

    def it_keeps_adamw_state_after_apply():
        model = MLP()
        optimizer = optim.AdamW(learning_rate=1e-3)
        _, grads = nn.value_and_grad(model, lambda m, x: m(x).sum())(model, mx.ones((2, 16)))
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        first_moment = optimizer.state["fc1"]["weight"]["m"]
        adapter = MlxAdapter(model)

        adapter.apply(MlxAdapter(MLP()).snapshot(["fc1.weight"]))

        assert mx.array_equal(optimizer.state["fc1"]["weight"]["m"], first_moment)
        assert optimizer.state["step"].item() == 1

    def it_rejects_identity_tied_parameters():
        class Tied(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(8, 4)
                self.head_weight = self.embed.weight

        with pytest.raises(SchemaMismatch) as excinfo:
            MlxAdapter(Tied()).schema()

        assert set(excinfo.value.names) == {"embed.weight", "head_weight"}

    def when_payload_has_wrong_shape():
        def it_raises_before_mutating():
            adapter = MlxAdapter(MLP())
            before = adapter.snapshot(["fc1.bias", "fc1.weight"])
            payload = {
                "fc1.bias": TensorBytes("F32", (32,), bytes(32 * 4)),
                "fc1.weight": TensorBytes("F32", (16, 32), bytes(16 * 32 * 4)),
            }

            with pytest.raises(SchemaMismatch) as excinfo:
                adapter.apply(payload)

            assert excinfo.value.names == ("fc1.weight",)
            assert adapter.snapshot(["fc1.bias", "fc1.weight"]) == before

    def when_a_parameter_is_a_scalar():
        def it_round_trips_its_bytes():
            class Scaled(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.scale = mx.array(2.0)

            model = Scaled()
            adapter = MlxAdapter(model)
            half = {"scale": TensorBytes("F32", (), struct.pack("<f", 0.5))}

            adapter.apply(half)

            assert adapter.snapshot(["scale"]) == half
            assert model.scale.item() == 0.5


def describe_mlx_backend():
    def it_adapts_modules_and_is_what_backend_for_resolves():
        from gradientmail.adapters import backend_for
        from gradientmail.adapters.mlx import MlxBackend

        backend = backend_for(MLP())

        assert isinstance(backend, MlxBackend)
        assert isinstance(backend.adapter(MLP()), MlxAdapter)
        with pytest.raises(TypeError):
            backend.adapter(object())

    def it_loads_bf16_natively_and_writes_it_back():
        from gradientmail.adapters.mlx import MlxBackend, MlxValue

        published = TensorBytes("BF16", (2,), struct.pack("<2H", 0x3F80, 0xC020))

        value = MlxBackend().load(published)

        assert isinstance(value, MlxValue)
        assert value.data.dtype == mx.float32
        assert value.data.tolist() == [1.0, -2.5]
        assert (value * 2.0).to_bytes("BF16") == TensorBytes(
            "BF16", (2,), struct.pack("<2H", 0x4000, 0xC0A0)
        )
