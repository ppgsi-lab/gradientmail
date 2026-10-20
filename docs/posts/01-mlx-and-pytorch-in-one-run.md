# MLX and PyTorch learners in one run

*October 2026.*

A GradientMail run can include learners that train with PyTorch and learners that train with MLX. A Mac with Apple silicon and a machine with an NVIDIA GPU can contribute to the same model, each running its own framework's forward pass, autodiff and inner optimizer.

The two frameworks never exchange objects. Each learner writes its parameters to the mailbox as bytes and reads committed bytes back into its own model. Both models must therefore describe their parameters the same way, and the bytes must mean the same thing on both sides. This post covers what the library converts, what it checks, and what is left to the people who write the models.

## What crosses the mailbox

Tensor payloads are Safetensors blobs: an 8-byte length, a JSON header with each tensor's name, dtype code, shape and byte offsets, then the raw data, little-endian and C-contiguous. The dtype codes are `F32`, `F16` and `BF16`; both adapters reject parameters of any other dtype.

Records are JSON. The manifest, written once when the run is created, holds the parameter schema (each trainable parameter's name, dtype and shape, sorted by name), the algorithm configuration and references to the initial parameters, Θ⁰. Each snapshot record carries the schema's SHA-256 digest.

The model in the examples, two linear layers with a layer norm between them, has six parameters and 2,960 bytes in fp32. Its snapshot packs to 3,384 bytes, 416 of them header, shown here one entry per line:

```json
{"fc1.bias":{"dtype":"F32","shape":[32],"data_offsets":[0,128]},
 "fc1.weight":{"dtype":"F32","shape":[32,16],"data_offsets":[128,2176]},
 "fc2.bias":{"dtype":"F32","shape":[4],"data_offsets":[2176,2192]},
 "fc2.weight":{"dtype":"F32","shape":[4,32],"data_offsets":[2192,2704]},
 "norm.bias":{"dtype":"F32","shape":[32],"data_offsets":[2704,2832]},
 "norm.weight":{"dtype":"F32","shape":[32],"data_offsets":[2832,2960]}}
```

A PyTorch model and an MLX model holding the same values pack to identical bytes. Payload keys contain the SHA-256 of those bytes, and no record has a field for the framework.

## From a model to named tensors

Each framework has an adapter with three methods, `schema`, `snapshot` and `apply`. `Session` picks it from the modules of the model's base classes, without importing a framework to decide.

The [PyTorch adapter](../../src/gradientmail/adapters/torch.py) reads `model.named_parameters()`, which lists a tied parameter once. `snapshot` copies each tensor to host memory. `apply` writes with `Tensor.copy_` under `torch.no_grad()`, which moves the data to the parameter's device and writes through to tied aliases. The `nn.Parameter` is never replaced, so optimizer state keyed by it survives.

The [MLX adapter](../../src/gradientmail/adapters/mlx.py) flattens `model.trainable_parameters()`, a tree of dicts and lists, into dotted paths such as `fc1.weight` or `layers.0.bias`, the names `mlx.utils.tree_flatten` uses. `snapshot` reads each array through the buffer protocol, which evaluates it and yields its logical row-major bytes. `apply` decodes the payload with MLX's own Safetensors reader and sets the arrays with `Module.update`. MLX keys optimizer state by path, so it survives too.

MLX ties are identity only: two attributes hold the same array object, and the first `update` breaks the tie. The adapter rejects such a model with `SchemaMismatch` and asks for a structural tie, such as `embed.as_linear`. A PyTorch model whose head shares the embedding's `Parameter` and an MLX model using `embed.as_linear` both list one `embed.weight`.

## What has to match

Every learner's schema must equal the manifest's. [`torch_learner.py`](../../examples/torch_learner.py) and [`mlx_learner.py`](../../examples/mlx_learner.py) define the same model in each framework:

```python
class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.norm = torch.nn.LayerNorm(32)
        self.fc2 = torch.nn.Linear(32, 4)
```

```python
class MLP(nn.Module):  # import mlx.nn as nn
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 4)
```

Both produce the schema in the header above. Names come from the attribute structure, which is why the layers are attributes and not a `Sequential`: PyTorch's `nn.Sequential` names its first layer's weight `0.weight`, MLX's `layers.0.weight`. Some modules differ in structure too. At width 8, PyTorch's `nn.MultiheadAttention` packs the input projections into one `in_proj_weight` of shape `(24, 8)`; MLX's `nn.MultiHeadAttention` has separate `query_proj`, `key_proj` and `value_proj`, without biases by default.

Shapes follow each framework's layout. Both store a linear layer's weight as `(out, in)`, but `Conv2d(3, 8, 5)` has a weight of shape `(8, 3, 5, 5)` in PyTorch and `(8, 5, 5, 3)` in MLX.

Dtypes are part of the schema. A PyTorch learner training under bf16 autocast keeps fp32 parameters and has an `F32` schema; an MLX model converted with `set_dtype(mx.bfloat16)` has a `BF16` one. The two cannot join the same run.

Equal schemas are necessary but not sufficient. `Conv1d(3, 8, 3)` has a weight of shape `(8, 3, 3)` in both frameworks: `(out, in, kernel)` in PyTorch, `(out, kernel, in)` in MLX. The run accepts both. Loaded with the same bytes, their outputs on the same input differed by up to 1.69; with the weight's last two axes swapped, by 1.2e-7. The library checks names, dtypes and shapes, not what the forward pass does with them.

## When the models differ

A learner compares its model's schema with the manifest's when it joins, and `Session` raises `SchemaMismatch` naming every parameter whose name, dtype or shape differs. An MLX `Sequential` joining a run created by a PyTorch `Sequential` fails with:

```
SchemaMismatch: model does not match the run's schema: 0.bias, 0.weight, 1.bias, 1.weight, layers.0.bias, layers.0.weight, layers.1.bias, layers.1.weight
```

The exception comes from `session.join()`, or from `session.run` when it joins, before the communication thread starts, so the learner writes no snapshot or lease. An MLX copy of the demo model in bf16 gets all six of its parameters listed. A different `gm.DiLoCo` configuration raises `RunConfigMismatch`.

Payloads are validated again before every `apply`. Without that check, PyTorch's `copy_` would cast a wrong dtype, or broadcast a `(1, 16)` payload into a `(32, 16)` parameter; the adapter tests confirm that both adapters raise on a wrong shape before changing the model. The node closing a round also rejects a snapshot whose schema digest, delta dtype or shapes do not match, recording it in the round with reason `schema` and weight 0.

## Arithmetic in two frameworks

Each node computes in its own framework. A learner keeps a host copy of the committed bytes it last aligned with for each fragment, its anchor. When a round is due, it loads the anchor and its parameters as fp32 and publishes `anchor - θ` in the run's `delta_dtype`. The default keeps each parameter's own dtype; `"BF16"` halves an fp32 model's uploads, rounding each element to 8 significant bits; `"F8_E4M3"` and `"F4"` quantize them with power-of-two scales, which both backends pack to the same bytes.

Both frameworks convert to bf16 natively, rounding to nearest even. [`test_backends.py`](../../tests/test_backends.py) compares MLX with PyTorch on all 65,536 upper halves of an fp32 value, each combined with six lower halves. The results are equal bit for bit, except that MLX flushes subnormals to zero where PyTorch rounds them.

The node elected to close a round runs the merge and the Nesterov outer step in its framework, from code written once over a few fp32 operations. Each commit stores the fp32 outer momentum, plus an fp32 master copy for fp16 and bf16 parameters, so a closer in the other framework continues from the same state. The two backends' results are not bit-identical. The test that closes one round on both allows a relative difference of 1e-6, and in one check on two 4,096-element tensors, one matched exactly and the other differed in 149 elements.

This does not split the run. A round commits one payload, and every learner, in either framework, loads those bytes.

## Running the two examples

The docstring of `torch_learner.py` sets up a local MinIO bucket. One learner per framework then runs against the same bucket and prefix:

```sh
uv run python examples/torch_learner.py --rounds 6 --node-id torch
uv run python examples/mlx_learner.py --rounds 6 --node-id mlx
```

Both use `gm.DiLoCo(sync_every=20, fragments=2, quorum_size=2)`, so each round waits for both learners. The plan puts `fc1.weight` (2,048 bytes) in one fragment and the other five parameters (912 bytes) in the other. On one MacBook, with PyTorch on the CPU and MLX on the GPU, both scripts exited after six rounds. The MLX learner created the run, so Θ⁰ came from MLX's initialization, and the PyTorch learner loaded it before its first step. All six round records list both learners as participants and none as rejected.

## What the tests cover

[`test_interop.py`](../../tests/test_interop.py) runs when both frameworks are installed. Three tests check that both adapters produce the same literal schema for the demo model, and that bytes from either framework applied to the other read back unchanged. Those two also compare `fc1.weight` element by element; its `(32, 16)` shape is not square, so a disagreement about which axis is which would show.

A fourth test runs one PyTorch `Session` and one MLX `Session` through a moto S3 bucket. Both learners take part in each of the first three rounds, and no snapshot is rejected. After both rejoin, their parameters are byte-identical to each other and to the newest commit of every fragment, and differ from Θ⁰.

These tests compare parameters, not activations. On the same bytes and 256 random inputs, the demo model's outputs differed by up to 1.6e-3 between MLX on the GPU and PyTorch on the CPU, and by 4.8e-7 with MLX on the CPU.

## Limits

The library maps no names or layouts between frameworks. The models must produce the same schema with the same meaning. Only trainable parameters are exchanged, so batch normalization's running statistics stay local to each learner. Inner optimizer state stays local too, and the run's configuration does not describe the inner optimizer, so nothing checks that learners use the same one. Learners must report `samples` in the same unit, since contributions are weighted by it.

In this repository, the mixed-framework path is tested with the demo model only. The GPT-2 and Pythia examples are PyTorch only.
