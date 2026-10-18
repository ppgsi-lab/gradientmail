# GradientMail

Distributed training with [Decoupled DiLoCo](https://arxiv.org/abs/2604.21428) over a shared mailbox. Nodes exchange parameter fragments through S3, R2, MinIO or Hugging Face Storage Buckets, without P2P connections or a separate coordinator.

The cluster is heterogeneous and elastic: nodes with different GPUs and frameworks (PyTorch or MLX) can join an ongoing run, contribute to training for a period and leave. One of the training nodes merges each round and publishes the result. If it fails, another takes over.

## How it works

Each learner keeps a local copy of the model and runs its own optimizer. Parameters are split into fragments, synchronized on a staggered schedule. For each fragment, a learner publishes its update: the difference between the committed parameters it last aligned with and the ones it trained.

![Three nodes and a mailbox. Node A (MLX on Apple silicon), node B (PyTorch on an H100) and node C (PyTorch on an RTX 4090) each run a learner and a syncer. Each learner trains fragment k, puts its delta, the anchor minus its parameters, in the mailbox, polls for commits and aligns with them between its steps. Node B's syncer is elected for the round: it lists and gets the round's deltas, merges them with RDA, applies the Nesterov outer step and commits fragment k with the outer state. The other syncers follow it and take over if node B stalls. The mailbox, an S3, R2, MinIO or Hugging Face bucket, holds the deltas and the commits; every arrow is a request to it.](docs/assets/training-flow.svg)

An elected learner reads the updates for a round, merges them and applies the outer optimizer. It commits the updated fragment through the mailbox; learners poll for new commits and align their local models with them. All training and merge arithmetic runs on the learners.

## Install

Python 3.12+. From a checkout:

```sh
uv sync --extra s3 --extra torch
```

Use `--extra mlx` instead of `--extra torch` for MLX on Apple Silicon.

## Usage

Add a `Session` to your existing training loop:

```python
import gradientmail as gm

with gm.Session(
    model,
    algorithm=gm.DiLoCo(sync_every=100, fragments=8),
    mailbox=gm.S3Mailbox(bucket="my-training", prefix="experiments/run-1"),
) as session:
    for batch in dataloader:
        loss = train_step(model, optimizer, batch)
        session.step(samples=len(batch))
```

Run this on each node with the same bucket, prefix, algorithm configuration and parameter schema (names, shapes and dtypes). `model` can be a PyTorch or MLX module. The first node initializes the run; later nodes load its committed parameters before training. Use a new prefix for each run.

`sync_every=100, fragments=8` splits the model into up to eight parameter groups, each synchronized every 100 local steps. `step(samples=...)` reports the tokens or samples processed by that optimizer step. Leaving the context waits for the node's rounds in flight to commit, stopping the wait if the run stalls, and sets the model to the newest committed parameters. Steps trained since a fragment's last round are dropped, as in the DiLoCo papers; `flush_on_close=True` publishes them instead, and a round made only of those final contributions commits them without the outer momentum.

Training continues during synchronization. Aligning a fragment with its commit replaces local updates made while it was in flight. `overlap_steps=N` on `gm.DiLoCo` bounds that: a node waits once a round it published N steps ago is still not aligned with, and `0` is standard DiLoCo's barrier. `mixing=0.5` keeps that share of the local parameters when aligning, blending them with the commit instead of replacing them, which Streaming DiLoCo found to recover most of what overlapped steps cost.

The defaults follow Decoupled DiLoCo. `fragments=1, in_flight=1, overlap_steps=0, merge=gm.WeightedMean()` with `quorum_size` equal to the number of nodes is standard DiLoCo. `partition=gm.Strided()` deals the transformer blocks round the fragments, Streaming DiLoCo's pattern, instead of balancing the fragments by bytes.

`quorum_size` sets the target number of contributions per round (default 1). Rounds can close with fewer contributions once every live node has reported. Crashed nodes stop being counted after their lease expires (default 120 seconds).

`global_batch_size` fixes the batch the live nodes share. `session.batch_size` is this node's share, the global batch divided by the live nodes and rounded up, for the training loop to size its batches by; it changes as nodes join and leave. Without it, each node's batch is its own and the global batch grows with the node count.

`delta_dtype="BF16"` sends each update in bf16, halving an fp32 model's uploads at a rounding of 8 significant bits per element; `"F8_E4M3"` quarters them and `"F4"` takes an eighth, each element scaled by a power of two per tensor or per block of 32, as Streaming DiLoCo sends its outer gradients; by default it keeps each parameter's own dtype. `commit_dtype` publishes each commit's change in the same formats, so a node holding the previous version downloads that instead of the full parameters. `compression="zstd"` on `Session` stores this node's payloads byte-shuffled and zstandard-compressed (the `zstd` extra), about 16% smaller for fp32 and 31% for bf16; every node reads either format, so nodes of one run may differ.

`S3Mailbox` uses boto3 credentials. For other S3-compatible services, pass `endpoint_url=`; Hugging Face Storage Buckets also need `request_checksums="when_required"`.

For evaluation, `gm.align(model, mailbox=...)` loads the committed model without joining the run. GradientMail does not save your inner optimizer state; checkpoint it separately.

## Examples

Run two synthetic learners locally, with PyTorch or MLX installed:

```sh
uv run python examples/synthetic.py
```

For a PyTorch/MLX run across two machines, use [torch_learner.py](examples/torch_learner.py) and [mlx_learner.py](examples/mlx_learner.py). The PyTorch example includes the MinIO setup and commands.

## Further reading

- [Guide](docs/guide/README.md)

  A collection of explanations for understanding distributed training.

- [Shared Queue](docs/patterns/00-shared-queue.md)

  Distributing work across nodes with different processing speeds.

- [Barrier](docs/patterns/01-barrier.md)

  Holding nodes at a named point until a number of them have reached it.

- [Posts](docs/posts/README.md)

  A Pythia-14M run on four rented GPUs, MLX and PyTorch learners in one run, merging deltas with RDA, and implementing a mailbox backend.

## Development

```sh
uv run pytest
uv run ruff check
uv run ruff format --check
uv run pyrefly check
```

Tests use memory and moto stores by default. Set `GRADIENTMAIL_S3_ENDPOINT` to test against an S3-compatible store. Framework tests skip when their dependencies are absent.
