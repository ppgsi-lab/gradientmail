"""MLX backend: the trainable arrays of an ``mlx.nn.Module`` as ``TensorBytes`` and back, and
the outer step on ``mx.array``.

Parameter names are the dotted paths of ``model.trainable_parameters()`` (``fc1.weight``,
``layers.0.bias``), the names ``mlx.utils.tree_flatten`` and ``load_weights`` use, so a
torch model with the same attribute structure produces the same schema. ``apply`` hands
the payload to MLX's own safetensors reader and sets the arrays with ``Module.update``;
MLX optimizer state is keyed by path rather than by array identity, so it survives. bf16 and
f16 are native, so ``MlxValue`` converts with ``astype``; MLX flushes subnormals to zero on the
way, where NumPy and torch round them. E4M3 goes through ``mx.to_fp8`` and ``mx.from_fp8``,
which agree with torch's casts bit for bit; E2M1 nibbles and E8M0 scales are packed and
unpacked by hand, as in the torch backend, so both produce the same bytes.

Only ``mlx.core`` ships type information a checker accepts, so this module imports nothing
from ``mlx.nn`` and types the model structurally through ``Module``.
"""

import io
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, Self, assert_never, runtime_checkable

import mlx.core as mx
from mlx.utils import tree_unflatten

from gradientmail.adapters import Adapter, Backend, Value
from gradientmail.serialization import (
    F4_BLOCK,
    DType,
    ParameterSpec,
    Quantized,
    Schema,
    SchemaMismatch,
    TensorBytes,
    pack,
    validate,
)

__all__ = ["MLX_DTYPE", "MlxAdapter", "MlxBackend", "MlxValue", "Module", "ParameterTree"]

type ParameterTree = mx.array | Mapping[str, ParameterTree] | Sequence[ParameterTree]

MLX_DTYPE: Mapping[DType, mx.Dtype] = {
    "F32": mx.float32,
    "F16": mx.float16,
    "BF16": mx.bfloat16,
}


@runtime_checkable
class Module(Protocol):
    """What the adapter needs from a model; every ``mlx.nn.Module`` satisfies it."""

    def trainable_parameters(self) -> Mapping[str, ParameterTree]: ...

    def update(self, parameters: dict[str, ParameterTree], strict: bool = True) -> object: ...


class MlxBackend(Backend):
    def adapter(self, model: object) -> Adapter:
        if not isinstance(model, Module):
            raise TypeError(f"not an mlx.nn.Module: {type(model).__qualname__}")
        return MlxAdapter(model)

    def load(self, tensor: TensorBytes) -> Value:
        match tensor.dtype:
            case "F32" | "F16" | "BF16":
                return MlxValue(_as_array(tensor).astype(mx.float32))
            case other:
                raise ValueError(f"{other} bytes need their scale: use dequantize")

    def dequantize(self, data: TensorBytes, scale: TensorBytes, shape: tuple[int, ...]) -> Value:
        match data.dtype:
            case "F8_E4M3":
                values = mx.from_fp8(_as_uint8(data.data), mx.float32) * _as_array(scale)
            case "F4":
                codes = _as_uint8(data.data)
                nibbles = mx.stack([codes & 0xF, codes >> 4], axis=1).reshape(-1, F4_BLOCK)
                magnitudes = mx.take(_F4_VALUES, nibbles & 7)
                signed = mx.where((nibbles & 8) != 0, -magnitudes, magnitudes)
                scales = _pow2(_as_uint8(scale.data).astype(mx.int32) - 127)
                values = (signed * scales[:, None]).reshape(-1)[: math.prod(shape)]
            case other:
                raise ValueError(f"{other} is not a quantized dtype")
        return MlxValue(values.reshape(shape))


class MlxAdapter(Adapter):
    """Adapter over an MLX module.

    ``schema`` rejects two paths that hold the same array object: MLX ties are
    identity-only and break at the first ``update``, so weights must be tied
    structurally (``embed.as_linear``), which lists them once.
    """

    def __init__(self, model: Module) -> None:
        self._model = model

    def schema(self) -> Schema[DType]:
        parameters = self._parameters()
        holders = Counter(id(array) for array in parameters.values())
        tied = tuple(name for name, array in parameters.items() if holders[id(array)] > 1)
        if tied:
            raise SchemaMismatch(tied, "parameters tied by identity; tie them structurally")
        unsupported = tuple(
            name for name, array in parameters.items() if array.dtype not in _DTYPE_CODE
        )
        if unsupported:
            raise SchemaMismatch(unsupported, "dtype is not float32, float16 or bfloat16")
        return Schema(
            tuple(
                ParameterSpec(name, _DTYPE_CODE[parameters[name].dtype], parameters[name].shape)
                for name in sorted(parameters)
            )
        )

    def snapshot(self, names: Iterable[str]) -> Mapping[str, TensorBytes]:
        parameters = self._parameters()
        return {name: _tensor_bytes(parameters[name]) for name in names}

    def apply(self, tensors: Mapping[str, TensorBytes]) -> None:
        validate(self.schema(), tensors, tensors)
        loaded = mx.load(io.BytesIO(pack(tensors)), format="safetensors")
        assert isinstance(loaded, dict)  # the stub's union; safetensors always decodes to a dict
        self._model.update(tree_unflatten(list(loaded.items())))

    def _parameters(self) -> Mapping[str, mx.array]:
        def flattened(tree: ParameterTree, prefix: str) -> Iterator[tuple[str, mx.array]]:
            match tree:
                case mx.array():
                    yield prefix.removeprefix("."), tree
                case Mapping():
                    for key, child in tree.items():
                        yield from flattened(child, f"{prefix}.{key}")
                case Sequence():
                    for index, child in enumerate(tree):
                        yield from flattened(child, f"{prefix}.{index}")
                case _:
                    assert_never(tree)

        return dict(flattened(self._model.trainable_parameters(), ""))


@dataclass(frozen=True, eq=False)
class MlxValue(Value):
    data: mx.array

    def __add__(self, other: Self) -> Self:
        return replace(self, data=self.data + other.data)

    def __sub__(self, other: Self) -> Self:
        return replace(self, data=self.data - other.data)

    def __mul__(self, scalar: float) -> Self:
        return replace(self, data=self.data * scalar)

    def norm(self) -> float:
        return float(mx.linalg.norm(mx.reshape(self.data, (-1,))))

    def dot(self, other: Self) -> float:
        return float(mx.sum(self.data * other.data))

    def to_bytes(self, dtype: DType) -> TensorBytes:
        return _tensor_bytes(self.data.astype(MLX_DTYPE[dtype]))

    def quantize(self, dtype: Quantized) -> tuple[TensorBytes, TensorBytes]:
        flat = mx.reshape(self.data, (-1,))
        match dtype:
            case "F8_E4M3":
                largest = mx.max(mx.abs(flat)).reshape(1) if flat.size else mx.zeros((1,))
                _, scale = _scales(largest, _E4M3_HEADROOM)
                codes = mx.to_fp8(self.data / scale.reshape(()))
                return (
                    TensorBytes(dtype, self.data.shape, _bytes(codes)),
                    TensorBytes("F32", (), _bytes(scale.reshape(()))),
                )
            case "F4":
                blocks = -(-flat.size // F4_BLOCK)
                blocked = mx.pad(flat, (0, blocks * F4_BLOCK - flat.size)).reshape(blocks, F4_BLOCK)
                exponents, scales = _scales(mx.max(mx.abs(blocked), axis=1), _E2M1_HEADROOM)
                q = mx.minimum(mx.abs(blocked) / scales[:, None], 6.0)
                # Round to nearest even on the E2M1 grid 0, .5, 1, 1.5, 2, 3, 4, 6, whose codes
                # are 0 to 7 in that order.
                codes = mx.where(
                    q < 2, mx.round(q * 2), mx.where(q < 4, mx.round(q) + 2, mx.round(q / 2) + 4)
                ).astype(mx.uint8)
                codes = codes | mx.where(blocked < 0, 8, 0).astype(mx.uint8)
                pairs = codes.reshape(-1, 2)
                packed = pairs[:, 0] | (pairs[:, 1] << 4)
                return (
                    TensorBytes(dtype, (blocks * F4_BLOCK,), _bytes(packed)),
                    TensorBytes("F8_E8M0", (blocks,), _bytes((exponents + 127).astype(mx.uint8))),
                )
            case _:
                raise ValueError(f"{dtype} is not a quantized dtype")


_DTYPE_CODE: Mapping[mx.Dtype, DType] = {dtype: code for code, dtype in MLX_DTYPE.items()}
_F4_VALUES = mx.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Scales put the largest magnitude of a tensor or block in the binade just under the format's
# top: below 2^8 for E4M3 (whose maximum is 448), in [4, 8) for E2M1 (whose maximum is 6).
_E4M3_HEADROOM = 7
_E2M1_HEADROOM = 2


def _scales(largest: mx.array, headroom: int) -> tuple[mx.array, mx.array]:
    """Per entry of ``largest``: the exponent ``k`` with ``largest < 2^(k + headroom + 1)``
    (zero gets ``k = -headroom``, a scale of one), clamped to what E8M0 holds, and ``2^k``."""
    exponents = (
        mx.where(largest > 0, (largest.view(mx.uint32) >> 23).astype(mx.int32) - 127, headroom)
        - headroom
    )
    exponents = mx.clip(exponents, -126, 127)
    return exponents, _pow2(exponents)


def _pow2(exponents: mx.array) -> mx.array:
    """``2^k`` for int32 ``k`` in ``[-126, 127]``, built from the fp32 exponent bits."""
    return ((exponents + 127).astype(mx.uint32) << 23).view(mx.float32)


def _as_array(t: TensorBytes) -> mx.array:
    # The only typed way from bytes to an array is MLX's own safetensors reader.
    return _loaded(pack({"tensor": t}))


def _as_uint8(data: bytes) -> mx.array:
    """``data`` as a flat uint8 array, through a safetensors blob written here: the reader is the
    typed way in, and ``U8`` is not a payload dtype ``pack`` knows."""
    header = json.dumps(
        {"tensor": {"dtype": "U8", "shape": [len(data)], "data_offsets": [0, len(data)]}}
    ).encode()
    return _loaded(len(header).to_bytes(8, "little") + header + data)


def _loaded(blob: bytes) -> mx.array:
    loaded = mx.load(io.BytesIO(blob), format="safetensors")
    assert isinstance(loaded, dict)  # the stub's union; safetensors always decodes to a dict
    return loaded["tensor"]


def _bytes(array: mx.array) -> bytes:
    # Flattening first makes a 0-d array viewable as uint8 and gives a transposed array its
    # logical row-major bytes; the buffer protocol evaluates the array.
    return memoryview(mx.reshape(array, (-1,)).view(mx.uint8)).tobytes()


def _tensor_bytes(array: mx.array) -> TensorBytes:
    return TensorBytes(_DTYPE_CODE[array.dtype], array.shape, _bytes(array))
