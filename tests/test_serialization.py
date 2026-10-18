"""Our reader against the real safetensors format; the writer is safetensors itself."""

import ctypes
from collections.abc import Mapping

import pytest
import safetensors

from gradientmail.serialization import (
    ParameterSpec,
    Schema,
    SchemaMismatch,
    TensorBytes,
    pack,
    quantized_specs,
    unpack,
    validate,
)

TENSORS = {
    "fc1.weight": TensorBytes("F32", (2, 3), bytes(range(24))),
    "fc1.bias": TensorBytes("F16", (4,), bytes(range(100, 108))),
    "norm.weight": TensorBytes("BF16", (2, 2), bytes(range(200, 208))),
    "scale": TensorBytes("F32", (), bytes(range(50, 54))),
    "empty": TensorBytes("F32", (0, 3), b""),
    "q8": TensorBytes("F8_E4M3", (3,), bytes(range(60, 63))),
    "q8.scale": TensorBytes("F8_E8M0", (2,), bytes(range(120, 122))),
    "q4": TensorBytes("F4", (2, 4), bytes(range(4))),
}

SCHEMA = Schema(
    (
        ParameterSpec("fc1.bias", "F32", (2,)),
        ParameterSpec("fc1.weight", "F32", (2, 3)),
    )
)


def schema_of(tensors: Mapping[str, TensorBytes]) -> Schema:
    return Schema(
        tuple(ParameterSpec(name, t.dtype, t.shape) for name, t in sorted(tensors.items()))
    )


def describe_pack():
    def it_round_trips_f32_f16_bf16_scalars_and_empty_tensors():
        assert dict(unpack(pack(TENSORS))) == TENSORS

    def it_is_deterministic_across_insertion_order():
        reordered = dict(reversed(list(TENSORS.items())))

        assert pack(reordered) == pack(TENSORS)

    def it_is_readable_by_safetensors_deserialize():
        decoded = {
            name: (info["dtype"], tuple(info["shape"]), bytes(info["data"]))
            for name, info in safetensors.deserialize(pack(TENSORS))
        }

        assert decoded == {name: (t.dtype, t.shape, t.data) for name, t in TENSORS.items()}

    def it_rejects_a_length_mismatch_before_serializing():
        with pytest.raises(ValueError):
            TensorBytes("F32", (2,), bytes(7))
        with pytest.raises(ValueError):
            TensorBytes("F4", (2, 4), bytes(5))

    def it_rejects_an_f4_tensor_whose_last_axis_is_odd():
        with pytest.raises(ValueError):
            TensorBytes("F4", (4, 3), bytes(6))
        with pytest.raises(ValueError):
            TensorBytes("F4", (), b"")


def describe_unpack():
    def when_blob_is_truncated():
        @pytest.mark.parametrize("cut", [4, 13, -1], ids=["length", "header", "data"])
        def it_raises(cut: int):
            blob = pack(TENSORS)

            with pytest.raises(ValueError):
                unpack(blob[:cut])

    def when_dtype_code_is_unknown():
        def it_raises():
            data = bytes(8)
            spec = safetensors.TensorSpec(
                dtype="int64",
                shape=[1],
                data_ptr=ctypes.cast(data, ctypes.c_void_p).value,
                data_len=len(data),
            )
            blob = safetensors.serialize({"q": spec}, metadata=None)

            with pytest.raises(ValueError):
                unpack(blob)


def describe_validate():
    def it_accepts_a_fragment_whose_tensors_match():
        tensors = {"fc1.weight": TensorBytes("F32", (2, 3), bytes(24))}

        validate(SCHEMA, tensors, ["fc1.weight"])

    def when_a_tensor_has_the_wrong_shape():
        def it_names_the_parameter():
            tensors = {
                "fc1.weight": TensorBytes("F32", (3, 2), bytes(24)),
                "fc1.bias": TensorBytes("F32", (2,), bytes(8)),
            }

            with pytest.raises(SchemaMismatch) as excinfo:
                validate(SCHEMA, tensors, ["fc1.weight", "fc1.bias"])

            assert excinfo.value.names == ("fc1.weight",)

    def when_a_tensor_has_the_wrong_dtype():
        def it_names_the_parameter():
            tensors = {
                "fc1.weight": TensorBytes("F32", (2, 3), bytes(24)),
                "fc1.bias": TensorBytes("BF16", (2,), bytes(4)),
            }

            with pytest.raises(SchemaMismatch) as excinfo:
                validate(SCHEMA, tensors, ["fc1.weight", "fc1.bias"])

            assert excinfo.value.names == ("fc1.bias",)

    def when_a_fragment_is_incomplete():
        def it_rejects():
            tensors = {"fc1.weight": TensorBytes("F32", (2, 3), bytes(24))}

            with pytest.raises(SchemaMismatch) as excinfo:
                validate(SCHEMA, tensors, ["fc1.weight", "fc1.bias"])

            assert excinfo.value.names == ("fc1.bias",)


def describe_quantized_specs():
    def it_keeps_the_shape_with_one_scale_for_e4m3():
        assert quantized_specs("w", "F8_E4M3", (3, 5)) == (
            ParameterSpec("w", "F8_E4M3", (3, 5)),
            ParameterSpec("w.scale", "F32", ()),
        )

    def it_flattens_to_whole_blocks_with_a_scale_each_for_e2m1():
        assert quantized_specs("w", "F4", (3, 11)) == (
            ParameterSpec("w", "F4", (64,)),
            ParameterSpec("w.scale", "F8_E8M0", (2,)),
        )
        assert quantized_specs("w", "F4", (0, 3)) == (
            ParameterSpec("w", "F4", (0,)),
            ParameterSpec("w.scale", "F8_E8M0", (0,)),
        )


def describe_schema():
    def when_parameters_are_unsorted_or_duplicated():
        def it_rejects():
            bias = ParameterSpec("fc1.bias", "F32", (2,))
            weight = ParameterSpec("fc1.weight", "F32", (2, 3))

            with pytest.raises(ValueError):
                Schema((weight, bias))
            with pytest.raises(ValueError):
                Schema((bias, bias))

    def it_reports_nbytes_including_scalars_and_empty_tensors():
        schema = Schema(
            (
                ParameterSpec("empty", "F32", (0, 3)),
                ParameterSpec("scale", "BF16", ()),
                ParameterSpec("weight", "F16", (2, 3)),
            )
        )

        assert (schema.nbytes("empty"), schema.nbytes("scale"), schema.nbytes("weight")) == (
            0,
            2,
            12,
        )


def describe_schema_digest():
    def it_ignores_values_and_changes_with_dtype_or_shape():
        other_values = {
            name: TensorBytes(t.dtype, t.shape, bytes(len(t.data))) for name, t in TENSORS.items()
        }
        other_dtype = {**TENSORS, "fc1.weight": TensorBytes("BF16", (2, 3), bytes(12))}
        other_shape = {**TENSORS, "fc1.weight": TensorBytes("F32", (3, 2), bytes(24))}

        digest = schema_of(TENSORS).digest()

        assert digest == schema_of(other_values).digest()
        assert digest != schema_of(other_dtype).digest()
        assert digest != schema_of(other_shape).digest()
