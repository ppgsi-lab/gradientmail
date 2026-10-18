"""Both codecs against the bytes they leave: exact round trips, smaller blobs, the key's name."""

import subprocess
import sys
from array import array
from collections.abc import Mapping

import pytest
from hypothesis import given
from hypothesis import strategies as st

from gradientmail.compression import Compression, compression_of, decode, encode
from gradientmail.serialization import PayloadDType, TensorBytes, nbytes, pack

CODECS: tuple[Compression, ...] = ("off", "zstd")


@st.composite
def tensors(draw: st.DrawFn) -> Mapping[str, TensorBytes]:
    def tensor() -> TensorBytes:
        dtype: PayloadDType = draw(
            st.sampled_from(["F32", "F16", "BF16", "F8_E4M3", "F8_E8M0", "F4"])
        )
        shape = tuple(draw(st.lists(st.integers(0, 5), max_size=3)))
        if dtype == "F4":
            shape = (*shape[:-1], 2 * shape[-1]) if shape else (2,)
        size = nbytes(dtype, shape)
        return TensorBytes(dtype, shape, draw(st.binary(min_size=size, max_size=size)))

    names = draw(
        st.lists(st.sampled_from(["w", "b", "scale"]), min_size=1, max_size=3, unique=True)
    )
    return {name: tensor() for name in names}


def describe_encode():
    @pytest.mark.parametrize("compression", CODECS)
    @given(tensors())
    def it_round_trips_every_dtype_shape_and_byte_pattern(
        compression: Compression, payload: Mapping[str, TensorBytes]
    ):
        assert dict(decode(encode(payload, compression), compression)) == payload

    def when_off():
        def it_stores_the_safetensors_blob_as_is():
            payload = {"w": TensorBytes("F32", (3,), bytes(range(12)))}

            assert encode(payload, "off") == pack(payload)

    def when_zstd():
        def it_shrinks_dense_floats():
            payload = {
                "w": TensorBytes(
                    "F32", (1024,), array("f", [1.0 + i / 4096 for i in range(1024)]).tobytes()
                )
            }

            assert len(encode(payload, "zstd")) < len(pack(payload))

        def it_needs_the_zstd_extra():
            script = "\n".join(
                [
                    "import sys",
                    "sys.modules['zstandard'] = None",
                    "from gradientmail.adapters import MissingExtra",
                    "from gradientmail.compression import encode",
                    "from gradientmail.serialization import TensorBytes",
                    "try:",
                    "    encode({'w': TensorBytes('F32', (1,), bytes(4))}, 'zstd')",
                    "except MissingExtra as exc:",
                    "    assert exc.extra == 'zstd', exc",
                    "else:",
                    "    raise SystemExit('encode() succeeded without zstandard')",
                ]
            )

            result = subprocess.run(
                [sys.executable, "-I", "-c", script], capture_output=True, text=True, check=False
            )

            assert result.returncode == 0, result.stderr


def describe_compression_of():
    def it_reads_the_codec_off_the_extension():
        assert compression_of("blobs/rounds/0000000003/d.safetensors") == "off"
        assert compression_of("blobs/rounds/0000000003/d.safetensors.zst") == "zstd"
        assert compression_of("blobs/rounds/0000000003/d.bin") is None
