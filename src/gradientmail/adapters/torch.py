"""PyTorch backend: ``nn.Module`` parameters in and out as bytes, and the outer step on tensors.

``TorchAdapter.snapshot`` reads host memory through ``ctypes`` because ``memoryview(tensor)``
is not supported. ``apply`` writes with ``Tensor.copy_`` under ``no_grad``: it moves the data
to the parameter's device, writes through shared storage (tied weights) and never rebinds the
``nn.Parameter``, so inner-optimizer state keyed by it survives. bf16, f16 and E4M3 are
native, so ``TorchValue`` converts with ``Tensor.to``; E2M1 nibbles and E8M0 scales are
packed and unpacked by hand.
"""

import ctypes
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Self

import torch
from torch import nn

from gradientmail.adapters import Adapter, Backend, Value
from gradientmail.serialization import (
    F4_BLOCK,
    DType,
    ParameterSpec,
    Quantized,
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

    def dequantize(self, data: TensorBytes, scale: TensorBytes, shape: tuple[int, ...]) -> Value:
        match data.dtype:
            case "F8_E4M3":
                values = _raw(data).view(torch.float8_e4m3fn).to(torch.float32) * _as_tensor(scale)
            case "F4":
                codes = _raw(data)
                nibbles = torch.stack((codes & 0xF, codes >> 4), dim=1).reshape(-1, F4_BLOCK)
                magnitudes = _F4_VALUES[(nibbles & 7).long()]
                signed = torch.where((nibbles & 8) != 0, -magnitudes, magnitudes)
                scales = _pow2(_raw(scale).to(torch.int32) - 127)
                values = (signed * scales[:, None]).reshape(-1)[: math.prod(shape)]
            case other:
                raise ValueError(f"{other} is not a quantized dtype")
        return TorchValue(values.reshape(shape).to(self._device))


class TorchAdapter(Adapter):
    """Adapter over ``model.named_parameters()``, where a tied parameter appears once."""

    def __init__(self, model: nn.Module) -> None:
        self._model = model

    def schema(self) -> Schema[DType]:
        def spec(name: str) -> ParameterSpec[DType]:
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

    def dot(self, other: Self) -> float:
        return float((self.data * other.data).sum())

    def to_bytes(self, dtype: DType) -> TensorBytes:
        return _tensor_bytes(self.data.to(TORCH_DTYPE[dtype]))

    def quantize(self, dtype: Quantized) -> tuple[TensorBytes, TensorBytes]:
        flat = self.data.reshape(-1)
        match dtype:
            case "F8_E4M3":
                largest = flat.abs().max().reshape(1) if flat.numel() else torch.zeros(1)
                _, scale = _scales(largest, _E4M3_HEADROOM)
                codes = (self.data / scale.reshape(())).to(torch.float8_e4m3fn)
                return (
                    TensorBytes(dtype, tuple(self.data.shape), _bytes(codes)),
                    TensorBytes("F32", (), _bytes(scale.reshape(()))),
                )
            case "F4":
                blocks = -(-flat.numel() // F4_BLOCK)
                padding = torch.zeros(blocks * F4_BLOCK - flat.numel(), dtype=flat.dtype)
                blocked = torch.cat((flat, padding)).reshape(blocks, F4_BLOCK)
                exponents, scales = _scales(blocked.abs().amax(dim=1), _E2M1_HEADROOM)
                q = (blocked.abs() / scales[:, None]).clamp(max=6.0)
                # Round to nearest even on the E2M1 grid 0, .5, 1, 1.5, 2, 3, 4, 6, whose codes
                # are 0 to 7 in that order.
                codes = torch.where(
                    q < 2,
                    torch.round(q * 2),
                    torch.where(q < 4, torch.round(q) + 2, torch.round(q / 2) + 4),
                ).to(torch.uint8)
                codes = codes | torch.where(blocked < 0, 8, 0).to(torch.uint8)
                pairs = codes.reshape(-1, 2)
                packed = pairs[:, 0] | (pairs[:, 1] << 4)
                return (
                    TensorBytes(dtype, (blocks * F4_BLOCK,), _bytes(packed)),
                    TensorBytes("F8_E8M0", (blocks,), _bytes((exponents + 127).to(torch.uint8))),
                )
            case _:
                raise ValueError(f"{dtype} is not a quantized dtype")


_DTYPE_CODE: Mapping[torch.dtype, DType] = {dtype: code for code, dtype in TORCH_DTYPE.items()}
_F4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Scales put the largest magnitude of a tensor or block in the binade just under the format's
# top: below 2^8 for E4M3 (whose maximum is 448), in [4, 8) for E2M1 (whose maximum is 6).
_E4M3_HEADROOM = 7
_E2M1_HEADROOM = 2


def _scales(largest: torch.Tensor, headroom: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per entry of ``largest``: the exponent ``k`` with ``largest < 2^(k + headroom + 1)``
    (zero gets ``k = -headroom``, a scale of one), clamped to what E8M0 holds, and ``2^k``."""
    exponents = (
        torch.where(largest > 0, (largest.view(torch.int32) >> 23) - 127, headroom) - headroom
    )
    exponents = exponents.clamp(-126, 127)
    return exponents, _pow2(exponents)


def _pow2(exponents: torch.Tensor) -> torch.Tensor:
    """``2^k`` for int32 ``k`` in ``[-126, 127]``, built from the fp32 exponent bits."""
    return ((exponents + 127) << 23).view(torch.float32)


def _as_tensor(t: TensorBytes) -> torch.Tensor:
    match t.dtype:
        case "F32" | "F16" | "BF16" as dtype:
            # frombuffer rejects an empty buffer and warns on a read-only one.
            if not t.data:
                return torch.empty(t.shape, dtype=TORCH_DTYPE[dtype])
            return torch.frombuffer(bytearray(t.data), dtype=TORCH_DTYPE[dtype]).reshape(t.shape)
        case other:
            raise ValueError(f"{other} bytes need their scale: use dequantize")


def _raw(t: TensorBytes) -> torch.Tensor:
    """``t``'s bytes as a flat uint8 tensor."""
    if not t.data:
        return torch.empty(0, dtype=torch.uint8)
    return torch.frombuffer(bytearray(t.data), dtype=torch.uint8)


def _bytes(tensor: torch.Tensor) -> bytes:
    t = tensor.to("cpu").contiguous()
    return ctypes.string_at(t.data_ptr(), t.numel() * t.element_size())


def _tensor_bytes(tensor: torch.Tensor) -> TensorBytes:
    return TensorBytes(_DTYPE_CODE[tensor.dtype], tuple(tensor.shape), _bytes(tensor))
