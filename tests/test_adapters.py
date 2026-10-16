"""MemoryAdapter as a real Adapter; backend resolution at its cheap edges."""

import subprocess
import sys

import pytest

from gradientmail.adapters import MemoryAdapter, adapter_for, backend_for, default_backend
from gradientmail.serialization import ParameterSpec, Schema, TensorBytes, validate

TENSORS = {
    "fc1.weight": TensorBytes("F32", (2, 3), bytes(range(24))),
    "fc1.bias": TensorBytes("F16", (4,), bytes(range(100, 108))),
    "norm.weight": TensorBytes("BF16", (2, 2), bytes(range(200, 208))),
}


def describe_memory_adapter():
    def it_round_trips_snapshot_and_apply():
        adapter = MemoryAdapter(TENSORS)
        replacement = {"fc1.weight": TensorBytes("F32", (2, 3), bytes(24))}

        adapter.apply(replacement)

        assert dict(adapter.snapshot(["fc1.weight"])) == replacement
        assert dict(adapter.snapshot(["fc1.bias"])) == {"fc1.bias": TENSORS["fc1.bias"]}
        assert dict(adapter.tensors) == {**TENSORS, **replacement}

    def it_reports_a_schema_its_snapshots_satisfy():
        adapter = MemoryAdapter(dict(reversed(list(TENSORS.items()))))

        schema = adapter.schema()

        assert schema == Schema(
            (
                ParameterSpec("fc1.bias", "F16", (4,)),
                ParameterSpec("fc1.weight", "F32", (2, 3)),
                ParameterSpec("norm.weight", "BF16", (2, 2)),
            )
        )
        validate(schema, adapter.snapshot(TENSORS), TENSORS)


def describe_backend_for():
    def when_given_an_adapter():
        def it_returns_it_unchanged():
            adapter = MemoryAdapter(TENSORS)

            assert adapter_for(adapter) is adapter

    def when_given_an_unsupported_object():
        def it_raises_type_error():
            with pytest.raises(TypeError):
                backend_for(object())
            with pytest.raises(TypeError):
                adapter_for({"w": TENSORS["fc1.weight"]})


def describe_default_backend():
    def it_prefers_torch_when_installed():
        pytest.importorskip("torch")
        from gradientmail.adapters.torch import TorchBackend

        assert isinstance(default_backend(), TorchBackend)

    def when_no_framework_is_installed():
        def it_raises_missing_extra_naming_torch():
            script = "\n".join(
                [
                    "import sys",
                    "for name in ('torch', 'mlx'):",
                    "    sys.modules[name] = None",
                    "from gradientmail.adapters import MissingExtra, default_backend",
                    "try:",
                    "    default_backend()",
                    "except MissingExtra as exc:",
                    "    assert exc.extra == 'torch', exc",
                    "else:",
                    "    raise SystemExit('default_backend() succeeded without any framework')",
                ]
            )

            result = subprocess.run(
                [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=False
            )

            assert result.returncode == 0, result.stderr
