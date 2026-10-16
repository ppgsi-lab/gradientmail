"""Payload codecs: how a fragment's tensors become the bytes stored under a payload key.

``off`` stores the safetensors blob as is. ``zstd`` byte-shuffles every tensor first (the i-th
byte of every element side by side, so the exponent bytes form one run and the mantissa bytes
another) and compresses the blob with zstandard at level 1: measured on GPT-2's weights, 16%
smaller in fp32 and 31% in bf16 at over 1 GB/s, which is the entropy of the exponent bytes and
all a lossless codec gets from dense floats. The key's extension names the codec, so a reader
follows the writer without agreement; it needs the ``zstd`` extra only to read a compressed
payload.
"""

from collections.abc import Mapping
from types import ModuleType
from typing import Literal, assert_never

from gradientmail.adapters import MissingExtra
from gradientmail.serialization import ITEM_SIZE, TensorBytes, pack, unpack

__all__ = ["EXTENSION", "Compression", "compression_of", "decode", "encode"]

type Compression = Literal["off", "zstd"]

EXTENSION: Mapping[Compression, str] = {"off": ".safetensors", "zstd": ".safetensors.zst"}


def compression_of(key: str) -> Compression | None:
    """The codec a payload key's extension names; None when it names none."""
    for compression, extension in EXTENSION.items():
        if key.endswith(extension):
            return compression
    return None


def encode(tensors: Mapping[str, TensorBytes], compression: Compression) -> bytes:
    match compression:
        case "off":
            return pack(tensors)
        case "zstd":
            shuffled = {name: _shuffled(t) for name, t in tensors.items()}
            return _zstandard().ZstdCompressor(level=1).compress(pack(shuffled))
        case _:
            assert_never(compression)


def decode(blob: bytes, compression: Compression) -> Mapping[str, TensorBytes]:
    match compression:
        case "off":
            return unpack(blob)
        case "zstd":
            shuffled = unpack(_zstandard().ZstdDecompressor().decompress(blob))
            return {name: _unshuffled(t) for name, t in shuffled.items()}
        case _:
            assert_never(compression)


def _shuffled(t: TensorBytes) -> TensorBytes:
    size = ITEM_SIZE[t.dtype]
    return TensorBytes(t.dtype, t.shape, b"".join(t.data[i::size] for i in range(size)))


def _unshuffled(t: TensorBytes) -> TensorBytes:
    size = ITEM_SIZE[t.dtype]
    count = len(t.data) // size
    data = bytearray(len(t.data))
    for i in range(size):
        data[i::size] = t.data[i * count : (i + 1) * count]
    return TensorBytes(t.dtype, t.shape, bytes(data))


def _zstandard() -> ModuleType:
    try:
        import zstandard
    except ImportError as exc:
        raise MissingExtra("zstd") from exc
    return zstandard
