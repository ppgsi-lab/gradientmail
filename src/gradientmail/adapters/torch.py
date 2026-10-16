"""PyTorch backend: ``nn.Module`` parameters in and out as bytes, and the outer step on tensors.

``TorchAdapter.snapshot`` reads host memory through ``ctypes`` because ``memoryview(tensor)``
is not supported. ``apply`` writes with ``Tensor.copy_`` under ``no_grad``: it moves the data
to the parameter's device, writes through shared storage (tied weights) and never rebinds the
``nn.Parameter``, so inner-optimizer state keyed by it survives. bf16 and f16 are native, so
``TorchValue`` converts with ``Tensor.to``.
"""

import ctypes
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Self

import torch
from torch import nn

from gradientmail.adapters import Adapter, Backend, Value
from gradientmail.serialization import (
    DType,
    ParameterSpec,
    Schema,
    SchemaMismatch,
    TensorBytes,
    validate,
)

__all__ = ["TORCH_DTYPE", "TorchAdapter", "TorchBackend", "TorchValue"]

TORCH_DTYPE: Mapping[DType, torch.dtype] = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
}


class TorchBackend(Backend):
    """Adapts ``nn.Module``s; loads values on ``device`` (``"cpu"`` unless told otherwise)."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = device

    def adapter(self, model: object) -> Adapter:
        if not isinstance(model, nn.Module):
            raise TypeError(f"not a torch.nn.Module: {type(model).__qualname__}")
        return TorchAdapter(model)

    def load(self, tensor: TensorBytes) -> Value:
        return TorchValue(_as_tensor(tensor).to(self._device, torch.float32))


class TorchAdapter(Adapter):
    """Adapter over ``model.named_parameters()``, where a tied parameter appears once."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def schema(self) -> Schema:
        def spec(name: str) -> ParameterSpec:
            p = parameters[name]
            return ParameterSpec(name, _DTYPE_CODE[p.dtype], tuple(p.shape))

        parameters = dict(self._model.named_parameters())
        unsupported = tuple(name for name, p in parameters.items() if p.dtype not in _DTYPE_CODE)
        if unsupported:
            raise SchemaMismatch(unsupported, "parameter dtype is not float32, float16 or bfloat16")
        return Schema(tuple(spec(name) for name in sorted(parameters)))

    def snapshot(self, names: Iterable[str]) -> Mapping[str, TensorBytes]:
        parameters = dict(self._model.named_parameters())
        return {name: _tensor_bytes(parameters[name].detach()) for name in names}

    def apply(self, tensors: Mapping[str, TensorBytes]) -> None:
        # copy_ would cast a wrong dtype and broadcast a compatible wrong shape silently.
        validate(self.schema(), tensors, tensors)
        parameters = dict(self._model.named_parameters())
        with torch.no_grad():
            for name, t in tensors.items():
                parameters[name].copy_(_as_tensor(t))


@dataclass(frozen=True, eq=False)
class TorchValue(Value):
    data: torch.Tensor

    def __add__(self, other: Self) -> Self:
        return replace(self, data=self.data + other.data)

    def __sub__(self, other: Self) -> Self:
        return replace(self, data=self.data - other.data)

    def __mul__(self, scalar: float) -> Self:
        return replace(self, data=self.data * scalar)

    def norm(self) -> float:
        return float(self.data.square().sum().sqrt())

    def to_bytes(self, dtype: DType) -> TensorBytes:
        return _tensor_bytes(self.data.to(TORCH_DTYPE[dtype]))


_DTYPE_CODE: Mapping[torch.dtype, DType] = {dtype: code for code, dtype in TORCH_DTYPE.items()}


def _as_tensor(t: TensorBytes) -> torch.Tensor:
    # frombuffer rejects an empty buffer and warns on a read-only one.
    if not t.data:
        return torch.empty(t.shape, dtype=TORCH_DTYPE[t.dtype])
    return torch.frombuffer(bytearray(t.data), dtype=TORCH_DTYPE[t.dtype]).reshape(t.shape)


def _tensor_bytes(tensor: torch.Tensor) -> TensorBytes:
    t = tensor.to("cpu").contiguous()
    data = ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())
    return TensorBytes(_DTYPE_CODE[t.dtype], tuple(t.shape), data)
