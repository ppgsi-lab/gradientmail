"""MLX backend: the trainable arrays of an ``mlx.nn.Module`` as ``TensorBytes`` and back, and
the outer step on ``mx.array``.

Parameter names are the dotted paths of ``model.trainable_parameters()`` (``fc1.weight``,
``layers.0.bias``), the names ``mlx.utils.tree_flatten`` and ``load_weights`` use, so a
torch model with the same attribute structure produces the same schema. ``apply`` hands
the payload to MLX's own safetensors reader and installs the arrays with ``Module.update``;
MLX optimizer state is keyed by path rather than by array identity, so it survives. bf16 and
f16 are native, so ``MlxValue`` converts with ``astype``; MLX flushes subnormals to zero on the
way, where NumPy and torch round them.

Only ``mlx.core`` ships type information a checker accepts, so this module imports nothing
from ``mlx.nn`` and types the model structurally through ``Module``.
"""

import io
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol, Self, assert_never, runtime_checkable

import mlx.core as mx
from mlx.utils import tree_unflatten

from gradientmail.adapters import Adapter, Backend, Value
from gradientmail.serialization import (
    DType,
    ParameterSpec,
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
        return MlxValue(_as_array(tensor).astype(mx.float32))


class MlxAdapter(Adapter):
    """Adapter over an MLX module.

    ``schema`` rejects two paths that hold the same array object: MLX ties are
    identity-only and break at the first ``update``, so weights must be tied
    structurally (``embed.as_linear``), which lists them once.
    """

    def __init__(self, model: Module) -> None:
        self._model = model

    def schema(self) -> Schema:
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

    def to_bytes(self, dtype: DType) -> TensorBytes:
        return _tensor_bytes(self.data.astype(MLX_DTYPE[dtype]))


_DTYPE_CODE: Mapping[mx.Dtype, DType] = {dtype: code for code, dtype in MLX_DTYPE.items()}


def _as_array(t: TensorBytes) -> mx.array:
    # The only typed way from bytes to an array is MLX's own safetensors reader.
    loaded = mx.load(io.BytesIO(pack({"tensor": t})), format="safetensors")
    assert isinstance(loaded, dict)
    return loaded["tensor"]


def _tensor_bytes(array: mx.array) -> TensorBytes:
    # Flattening first makes a 0-d array viewable as uint8 and gives a transposed array its
    # logical row-major bytes; the buffer protocol evaluates the array.
    raw = memoryview(mx.reshape(array, (-1,)).view(mx.uint8)).tobytes()
    return TensorBytes(_DTYPE_CODE[array.dtype], array.shape, raw)
