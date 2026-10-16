# GradientMail

Distributed training over an S3-compatible bucket. Learners train locally in PyTorch or MLX and exchange parameter fragments through the bucket. One learner merges each round using [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428) and publishes the result. Another takes over if it stops.

Learners need no direct connection or separate coordination service. The core depends on the standard library and safetensors.

## Install

Requires Python 3.12+. From a checkout:

```sh
uv sync --extra s3 --extra torch
```

For MLX on Apple Silicon, replace `--extra torch` with `--extra mlx`. Each node needs `s3` (boto3) and its framework extra. JAX, Keras and TensorFlow extras are declared, but their backends are not implemented.

## Usage

Wrap your training loop in a `Session` and call `step` after each optimizer step:

```python
import gradientmail as gm

with gm.Session(
    model,
    algorithm=gm.DecoupledDiLoCo(sync_every=100, fragments=8),
    mailbox=gm.S3Mailbox(bucket="my-training", prefix="experiments/run-1"),
) as session:
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
        session.step(samples=len(batch))
```

Run this on each node with the same bucket, prefix and algorithm configuration. `model` can be a `torch.nn.Module`, an `mlx.nn.Module` or a `gradientmail.Adapter`. Parameter names, shapes and dtypes must match across nodes; mismatches raise `SchemaMismatch` or `RunConfigMismatch` when joining.

The first node initializes the run from its model. Later nodes load the latest committed parameters before the constructor returns. Each prefix holds one run; use a new prefix to start another.

`step(samples=...)` reports the number of tokens or samples processed in that optimizer step. Contributions are weighted by `samples² / steps`, accumulated per fragment. By default, `step` copies parameter bytes and processes queues without doing I/O or waiting for a commit. Training continues during synchronization; a committed fragment overwrites local updates made while it was in flight. Set `synchronous=True` on the session to wait once the configured number of rounds is outstanding.

Each node renews a lease every `lease_every` seconds (default 60) and is presumed gone after `lease_timeout` seconds (default 120) without a renewal, judged on each observer's own clock. A round closes at `quorum_size` accepted snapshots or once every live node has reported, so nodes can join and leave during the run; a crashed node delays the rounds it would have joined by one timeout, and a node that closes releases its lease at once.

Leaving the context calls `close()`, which publishes pending updates and waits for their commits. It stops waiting if the run stalls beyond the lease timeout plus the takeover timeout. For evaluation or inference, `gm.install(model, mailbox=...)` loads the latest committed parameters without joining or publishing.

Credentials come from boto3's default chain. Pass `endpoint_url=` to `S3Mailbox` for an S3-compatible service. Hugging Face Storage Buckets also require `request_checksums="when_required"`. The merge backend defaults to the model's framework. To merge on CUDA, import `TorchBackend` from `gradientmail.adapters.torch` and pass `backend=TorchBackend("cuda")` to the session.

## Algorithm

`DecoupledDiLoCo(sync_every=H, fragments=P)` groups parameters into up to `P` fragments balanced by size. Each fragment synchronizes once every `H` local steps, with one fragment per round. With `H=100, P=8`, rounds are scheduled 12 or 13 steps apart.

| Option | Default | Meaning |
|---|---|---|
| `in_flight` | `2` | Rounds a learner may publish ahead of its last installed round |
| `quorum_size` | `1` | Accepted snapshots required to close a round |
| `outer_lr` | `0.7` | Outer learning rate |
| `outer_momentum` | `0.9` | Nesterov momentum |
| `merge` | `Rda()` | Weighted RDA; weighted mean for names containing `embed` |

The configuration requires `1 <= in_flight <= fragments <= sync_every`. The model must also have enough parameter tensors to form at least `in_flight` fragments.

For standard DiLoCo, use `fragments=1`, `in_flight=1`, `synchronous=True` on the session and a quorum equal to the learner count. For Local SGD, also set `outer_lr=1`, `outer_momentum=0` and `merge=gm.WeightedMean()`. RDA changes the scale of the merged delta; tune `outer_lr` or use `WeightedMean()`.

## Examples

Run two synthetic learners in one process, without an object store:

```sh
uv run python examples/synthetic.py
```

This uses `MemoryMailbox` and prints each commit alongside its expected value. It needs either PyTorch or MLX installed for merging.

### PyTorch and MLX on two machines

[torch_learner.py](examples/torch_learner.py) and [mlx_learner.py](examples/mlx_learner.py) train the same small MLP with `sync_every=20, fragments=2, quorum_size=2`. Both must run for rounds to commit. Install the matching extras on each machine.

Start MinIO on one machine and create the bucket:

```sh
docker run -d -p 19000:9000 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
    bitnamilegacy/minio
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export GRADIENTMAIL_S3_ENDPOINT=http://127.0.0.1:19000
export GRADIENTMAIL_BUCKET=gradientmail GRADIENTMAIL_PREFIX=demo/run-1
uv run python -c "import boto3, os; boto3.client('s3', \
    endpoint_url=os.environ['GRADIENTMAIL_S3_ENDPOINT']).create_bucket(Bucket='gradientmail')"
```

On the second machine, export the same variables with the MinIO host's address in place of `127.0.0.1`. Run one learner on each machine:

```sh
uv run python examples/torch_learner.py --rounds 6 --device cuda
uv run python examples/mlx_learner.py --rounds 6
```

PyTorch also accepts `--device cpu` or `--device mps`. Both learners print installed rounds and stop after round 5. Bucket, prefix and endpoint can also be passed as `--bucket`, `--prefix` and `--endpoint`.

For AWS S3, create the bucket and leave `GRADIENTMAIL_S3_ENDPOINT` unset. For R2 or another S3-compatible store, set it to the service endpoint.

### GPT-2 on rented GPUs

[gpt2.py](examples/gpt2.py) trains GPT-2 small (124M) on FineWeb-Edu using four Salad RTX 3090s through [Skyward](https://github.com/gabfssilva/skyward) and a Hugging Face Storage Bucket. It includes distributed training, validation of the committed model and a single-node baseline. Setup and commands are in the module docstring.

## Protocol

Each session runs two background threads: one exchanges snapshots and committed fragments; the other participates in round coordination. The learner holding a round's claim collects snapshots, waits for quorum and an adaptive grace window, merges deltas against the committed anchor, then applies Nesterov to an fp32 master.

The mailbox uses `put_if_absent`, `get`, `list` and `delete` under the run's prefix. Safetensors payloads have SHA-256 keys and are uploaded before their JSON records. `manifest.json` defines the run; `rounds/` holds immutable commit records; `snapshots/` holds learner contributions; `claims/` coordinates who closes each round; `leases/` holds each learner's lease, one create-once object per renewal.

The first round record written wins if two learners race to commit. After a crash, another learner reloads state from the records and takes over after a timeout based on recent round durations, with a minimum of 30 poll intervals. Round records are kept; old snapshots and payloads are garbage-collected.

The store must provide strong read-after-write and list-after-write consistency, atomic `PutObject` with `If-None-Match: *`, and `ListObjectsV2` in UTF-8 byte order. S3 Express directory buckets, R2 behind a caching custom domain and MinIO on NFS are unsupported. `MemoryMailbox` provides the same contract within one process.

See [protocol.py](src/gradientmail/protocol.py) for records and object keys, and [syncer.py](src/gradientmail/syncer.py) for election, recovery and garbage collection.

## Limitations

- A quorum above the number of live learners closes with the live ones, so a round can have fewer participants than `quorum_size`; the record names them. There is no run termination signal.
- Late snapshots are excluded; stale, invalid, missing or corrupt snapshots are recorded as rejected. Local updates can be overwritten when a commit is installed.
- Learner IDs must be unique. Two processes using the same ID can silently lose one contribution. The default is `{hostname}-{pid}`.
- Restarts restore committed parameters and outer state. Local step counters reset; restoring the inner optimizer is the caller's responsibility.
- A learner can miss a payload that was already collected and wait for a newer version. If this happens while joining, `ProtocolError` requires restarting the session. Init payloads orphaned before the manifest write are not collected.
- Transient store errors retry indefinitely with backoff from 0.5 to 30 seconds. Permanent errors surface during construction or the next `step()`. Corrupt committed payloads raise `PayloadCorrupt`.
- S3 payloads above 5 GiB are rejected. Increase the number of fragments.
- PyTorch and MLX merges can differ in the last bit; MLX flushes bf16 subnormals to zero.
- Each node polls the store. Increase `poll_interval` to reduce requests and storage costs.

## Development

```sh
uv sync --extra s3 --extra torch --extra mlx
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyrefly check
```

Omit framework extras unavailable on your platform. Backend and interoperability tests skip when their frameworks are absent; syncer tests require at least one framework. The default store tests use memory and moto.

With the MinIO container and environment above:

```sh
GRADIENTMAIL_S3_ENDPOINT=http://127.0.0.1:19000 uv run pytest
```

For Hugging Face Storage Buckets:

```sh
AWS_PROFILE=hf GRADIENTMAIL_S3_ENDPOINT=https://s3.hf.co/<namespace> \
    GRADIENTMAIL_S3_CHECKSUMS=when_required GRADIENTMAIL_TEST_TIMEOUT=120 \
    uv run pytest -k endpoint
```

Endpoint tests create a `gradientmail-test` bucket using boto3 credentials. `GRADIENTMAIL_TEST_TIMEOUT` sets the training test timeout in seconds (default 30).
