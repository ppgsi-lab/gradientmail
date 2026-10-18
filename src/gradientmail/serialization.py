"""Tensor bytes in the safetensors format and the parameter schema they must match.

Payloads are produced by ``safetensors.serialize`` and read back by a small header
parser, so neither direction needs NumPy or an ML framework. Bytes are always
little-endian and C-contiguous: the format carries no strides, and adapters hand
over contiguous host copies.

Parameters are ``F32``, ``F16`` or ``BF16`` (``DType``, the formats torch and MLX share).
Payloads may also carry deltas quantized to ``F8_E4M3`` or ``F4`` (``Quantized``) and the
``F8_E8M0`` scales of the latter (``PayloadDType``). A quantized tensor travels as two
entries, its data under its name and its scales under ``scale_name(name)``, in the layout
``quantized_specs`` describes.
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
from typing import Literal, TypedDict, assert_never

import safetensors

if sys.byteorder != "little":
    raise ImportError("gradientmail requires a little-endian host")

type DType = Literal["F32", "F16", "BF16"]
type Quantized = Literal["F8_E4M3", "F4"]
type PayloadDType = DType | Quantized | Literal["F8_E8M0"]

ITEM_BITS: Mapping[PayloadDType, int] = {
    "F32": 32,
    "F16": 16,
    "BF16": 16,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "F4": 4,
}
SPEC_NAME: Mapping[PayloadDType, str] = {
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E8M0": "float8_e8m0fnu",
    "F4": "float4_e2m1fn_x2",
}
F4_BLOCK = 32
"""Elements per scale of an ``F4`` tensor, which is padded to a multiple of them."""


def nbytes(dtype: PayloadDType, shape: tuple[int, ...]) -> int:
    return ITEM_BITS[dtype] * math.prod(shape) // 8


@dataclass(frozen=True)
class TensorBytes:
    """One tensor as raw little-endian, C-contiguous bytes with its header dtype code.

    An ``F4`` tensor packs two elements per byte, the first in the low nibble, so its last
    axis must be even.
    """

    dtype: PayloadDType
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        if self.dtype == "F4" and (not self.shape or self.shape[-1] % 2):
            raise ValueError(f"F4 tensor of shape {self.shape} needs an even last axis")
        expected = nbytes(self.dtype, self.shape)
        if len(self.data) != expected:
            raise ValueError(
                f"{self.dtype} tensor of shape {self.shape} needs {expected} bytes, "
                f"got {len(self.data)}"
            )


@dataclass(frozen=True)
class ParameterSpec[D: PayloadDType]:
    name: str
    dtype: D
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Schema[D: PayloadDType]:
    """Named tensors with their dtypes and shapes, sorted by name with unique names: a model's
    canonical trainable parameters (``Schema[DType]``), or what a payload must hold."""

    parameters: tuple[ParameterSpec[D], ...]

    def __post_init__(self) -> None:
        if any(a.name >= b.name for a, b in itertools.pairwise(self.parameters)):
            raise ValueError("schema parameters must be sorted by name and unique")

    def digest(self) -> str:
        entries = [[p.name, p.dtype, list(p.shape)] for p in self.parameters]
        return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()

    def nbytes(self, name: str) -> int:
        spec = self._by_name[name]
        return nbytes(spec.dtype, spec.shape)

    @cached_property
    def _by_name(self) -> Mapping[str, ParameterSpec[D]]:
        return {p.name: p for p in self.parameters}


def scale_name(name: str) -> str:
    """The payload entry holding the scales of the quantized tensor under ``name``."""
    return f"{name}.scale"


def quantized_specs(
    name: str, dtype: Quantized, shape: tuple[int, ...]
) -> tuple[ParameterSpec[PayloadDType], ParameterSpec[PayloadDType]]:
    """The two entries a tensor of ``shape`` quantized to ``dtype`` travels as: its data under
    ``name`` and its scales, powers of two, under ``scale_name(name)``.

    ``F8_E4M3`` keeps the shape and has one F32 scale for the whole tensor. ``F4`` is
    flattened and zero-padded to a multiple of ``F4_BLOCK`` elements, with one F8_E8M0 scale
    per block, as the MX formats lay it out.
    """
    match dtype:
        case "F8_E4M3":
            return ParameterSpec(name, dtype, shape), ParameterSpec(scale_name(name), "F32", ())
        case "F4":
            blocks = -(-math.prod(shape) // F4_BLOCK)
            return (
                ParameterSpec(name, dtype, (blocks * F4_BLOCK,)),
                ParameterSpec(scale_name(name), "F8_E8M0", (blocks,)),
            )
        case _:
            assert_never(dtype)


class SchemaMismatch(ValueError):
    """Tensors do not match the schema; ``names`` lists the offending parameters."""

    names: tuple[str, ...]

    def __init__(
        self, names: Iterable[str], message: str = "parameters do not match the schema"
    ) -> None:
        self.names = tuple(names)
        super().__init__(f"{message}: {', '.join(self.names)}")


def validate(
    schema: Schema[PayloadDType], tensors: Mapping[str, TensorBytes], names: Iterable[str]
) -> None:
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

    def shape(t: TensorBytes) -> list[int]:
        # The F4 spec counts pairs along the last axis; the header records the elements.
        if t.dtype == "F4":
            return [*t.shape[:-1], t.shape[-1] // 2]
        return list(t.shape)

    # `tensors` keeps every buffer alive for the whole call, which the pointer API requires.
    specs = {
        name: safetensors.TensorSpec(
            dtype=SPEC_NAME[t.dtype],
            shape=shape(t),
            data_ptr=address(t.data),
            data_len=len(t.data),
        )
        for name, t in tensors.items()
    }
    return safetensors.serialize(specs, metadata=None)


def unpack(blob: bytes) -> Mapping[str, TensorBytes]:
    """Read safetensors bytes, ignoring ``__metadata__``.

    Raises ``ValueError`` when the header overruns the blob, an offset lies outside it,
    or a dtype code is not a ``PayloadDType``.
    """

    def dtype_of(code: str) -> PayloadDType:
        match code:
            case "F32" | "F16" | "BF16" | "F8_E4M3" | "F8_E8M0" | "F4":
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
