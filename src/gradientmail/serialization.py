"""Tensor bytes in the safetensors format and the parameter schema they must match.

Payloads are produced by ``safetensors.serialize`` and read back by a small header
parser, so neither direction needs NumPy or an ML framework. Bytes are always
little-endian and C-contiguous: the format carries no strides, and adapters hand
over contiguous host copies.
"""

import ctypes
import hashlib
import itertools
import json
import math
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, TypedDict

import safetensors

if sys.byteorder != "little":
    raise ImportError("gradientmail requires a little-endian host")

type DType = Literal["F32", "F16", "BF16"]

ITEM_SIZE: Mapping[DType, int] = {"F32": 4, "F16": 2, "BF16": 2}
SPEC_NAME: Mapping[DType, str] = {"F32": "float32", "F16": "float16", "BF16": "bfloat16"}


@dataclass(frozen=True)
class TensorBytes:
    """One tensor as raw little-endian, C-contiguous bytes with its header dtype code."""

    dtype: DType
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        expected = ITEM_SIZE[self.dtype] * math.prod(self.shape)
        if len(self.data) != expected:
            raise ValueError(
                f"{self.dtype} tensor of shape {self.shape} needs {expected} bytes, "
                f"got {len(self.data)}"
            )


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    dtype: DType
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Schema:
    """Canonical trainable parameters, sorted by name with unique names."""

    parameters: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        if any(a.name >= b.name for a, b in itertools.pairwise(self.parameters)):
            raise ValueError("schema parameters must be sorted by name and unique")

    def digest(self) -> str:
        entries = [[p.name, p.dtype, list(p.shape)] for p in self.parameters]
        return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()

    def nbytes(self, name: str) -> int:
        spec = self._by_name[name]
        return ITEM_SIZE[spec.dtype] * math.prod(spec.shape)

    @cached_property
    def _by_name(self) -> Mapping[str, ParameterSpec]:
        return {p.name: p for p in self.parameters}


class SchemaMismatch(ValueError):
    """Tensors do not match the schema; ``names`` lists the offending parameters."""

    names: tuple[str, ...]

    def __init__(
        self, names: Iterable[str], message: str = "parameters do not match the schema"
    ) -> None:
        self.names = tuple(names)
        super().__init__(f"{message}: {', '.join(self.names)}")


def validate(schema: Schema, tensors: Mapping[str, TensorBytes], names: Iterable[str]) -> None:
    """Raise ``SchemaMismatch`` unless every name is present with the schema's dtype and shape."""

    def matches(name: str) -> bool:
        spec = specs.get(name)
        tensor = tensors.get(name)
        return (
            spec is not None
            and tensor is not None
            and (tensor.dtype, tensor.shape) == (spec.dtype, spec.shape)
        )

    specs = {p.name: p for p in schema.parameters}
    offending = tuple(name for name in names if not matches(name))
    if offending:
        raise SchemaMismatch(offending)


def pack(tensors: Mapping[str, TensorBytes]) -> bytes:
    """Serialize to safetensors bytes without metadata; deterministic for equal content."""

    def address(data: bytes) -> int:
        # typeshed's ctypes.cast rejects bytes and c_void_p.value is Optional; this is the
        # same address as ctypes.cast(data, c_void_p).value with an honest int type.
        return ctypes.addressof(
            ctypes.cast(ctypes.c_char_p(data), ctypes.POINTER(ctypes.c_char)).contents
        )

    # `tensors` keeps every buffer alive for the whole call, which the pointer API requires.
    specs = {
        name: safetensors.TensorSpec(
            dtype=SPEC_NAME[t.dtype],
            shape=list(t.shape),
            data_ptr=address(t.data),
            data_len=len(t.data),
        )
        for name, t in tensors.items()
    }
    return safetensors.serialize(specs, metadata=None)


def unpack(blob: bytes) -> Mapping[str, TensorBytes]:
    """Read safetensors bytes, ignoring ``__metadata__``.

    Raises ``ValueError`` when the header overruns the blob, an offset lies outside it,
    or a dtype code is not a ``DType``.
    """

    def dtype_of(code: str) -> DType:
        match code:
            case "F32" | "F16" | "BF16":
                return code
            case _:
                raise ValueError(f"unsupported dtype code {code!r}")

    def tensor(entry: _HeaderEntry) -> TensorBytes:
        start, stop = entry["data_offsets"]
        if not 0 <= start <= stop <= len(blob) - base:
            raise ValueError("tensor offsets outside the blob")
        return TensorBytes(
            dtype_of(entry["dtype"]),
            tuple(entry["shape"]),
            view[base + start : base + stop].tobytes(),
        )

    base = 8 + int.from_bytes(blob[:8], "little")
    if base > len(blob):
        raise ValueError("header overruns the blob")
    view = memoryview(blob)
    header: Mapping[str, _HeaderEntry] = {
        name: entry
        for name, entry in json.loads(view[8:base].tobytes()).items()
        if name != "__metadata__"
    }
    return {name: tensor(entry) for name, entry in header.items()}


class _HeaderEntry(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]
