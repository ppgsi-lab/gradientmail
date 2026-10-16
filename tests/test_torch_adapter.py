"""TorchAdapter against real ``nn.Module``s: bytes out, bytes in, optimizer state untouched."""

import struct

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from gradientmail.adapters.torch import TorchAdapter
from gradientmail.serialization import ParameterSpec, Schema, SchemaMismatch, TensorBytes

# The interoperability claim: the MLX adapter tests and test_interop.py hold the same literal.
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

NAMES = tuple(spec.name for spec in EXPECTED_MLP_SCHEMA.parameters)

DEVICES = (
    "cpu",
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable"),
    ),
)


class MLP(nn.Module):
    """The demo model: layers as attributes, not ``Sequential``, so names match MLX."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(torch.relu(self.fc1(x))))


class TiedLM(nn.Module):
    """Input embedding and output head share one ``Parameter``, as weight-tied LMs do."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.embed.weight


def describe_torch_adapter():
    def it_produces_the_expected_mlp_schema():
        assert TorchAdapter(MLP()).schema() == EXPECTED_MLP_SCHEMA

    @pytest.mark.parametrize("device", DEVICES)
    def it_round_trips_snapshot_apply_byte_exactly(device: str):
        source = MLP().to(device)
        target = MLP().to(device)

        published = TorchAdapter(source).snapshot(NAMES)
        TorchAdapter(target).apply(published)

        assert TorchAdapter(target).snapshot(NAMES) == published
        assert struct.unpack("<512f", published["fc1.weight"].data) == tuple(
            source.fc1.weight.flatten().tolist()
        )
        assert target.fc1.weight.device.type == device

    def it_keeps_adamw_state_after_apply():
        model = MLP()
        optimizer = torch.optim.AdamW(model.parameters())
        model(torch.randn(8, 16)).sum().backward()
        optimizer.step()
        weight = model.fc1.weight
        values = weight.detach().clone()
        exp_avg = optimizer.state[weight]["exp_avg"].clone()

        TorchAdapter(model).apply(TorchAdapter(MLP()).snapshot(["fc1.weight"]))

        assert model.fc1.weight is weight
        assert not torch.equal(weight, values)
        assert torch.equal(optimizer.state[weight]["exp_avg"], exp_avg)

    def it_writes_through_to_tied_aliases():
        model = TiedLM()
        adapter = TorchAdapter(model)
        payload = TensorBytes("F32", (10, 4), struct.pack("<40f", *range(40)))

        adapter.apply({"embed.weight": payload})

        assert [spec.name for spec in adapter.schema().parameters] == ["embed.weight"]
        assert torch.equal(model.head.weight, torch.arange(40, dtype=torch.float32).reshape(10, 4))
        assert model.head.weight is model.embed.weight

    def when_payload_has_wrong_shape():
        def it_raises_before_mutating():
            adapter = TorchAdapter(MLP())
            before = adapter.snapshot(NAMES)
            payload = {
                "fc1.bias": TensorBytes("F32", (32,), bytes(128)),
                # (1, 16) broadcasts into (32, 16): Tensor.copy_ alone would accept it.
                "fc1.weight": TensorBytes("F32", (1, 16), bytes(64)),
            }

            with pytest.raises(SchemaMismatch) as excinfo:
                adapter.apply(payload)

            assert excinfo.value.names == ("fc1.weight",)
            assert adapter.snapshot(NAMES) == before


def describe_torch_backend():
    def it_adapts_modules_and_is_what_backend_for_resolves():
        from gradientmail.adapters import backend_for
        from gradientmail.adapters.torch import TorchBackend

        backend = backend_for(MLP())

        assert isinstance(backend, TorchBackend)
        assert isinstance(backend.adapter(MLP()), TorchAdapter)
        with pytest.raises(TypeError):
            backend.adapter(object())

    @pytest.mark.parametrize("device", DEVICES)
    def it_loads_bf16_natively_and_writes_back_on_the_host(device: str):
        from gradientmail.adapters.torch import TorchBackend, TorchValue

        published = TensorBytes("BF16", (2,), struct.pack("<2H", 0x3F80, 0xC020))

        value = TorchBackend(device).load(published)

        assert isinstance(value, TorchValue)
        assert value.data.dtype == torch.float32
        assert value.data.device.type == device
        assert value.data.tolist() == [1.0, -2.5]
        assert (value * 2.0).to_bytes("BF16") == TensorBytes(
            "BF16", (2,), struct.pack("<2H", 0x4000, 0xC0A0)
        )
