"""Phase 3 MLX learner: the demo MLP trained natively with SGD, one ``Session`` running the loop.

The same model, data and step as torch_learner.py, in MLX: the two learners train one
model through the mailbox with no remapping, because the attribute structure and so the
parameter names match. Run it against the same bucket and prefix as torch_learner.py, which
documents the MinIO setup; there is no syncer to run:

    uv run python examples/mlx_learner.py --rounds 6

The learner prints the rounds it aligns with, stops drawing batches once it holds round
``rounds - 1`` and returns once its rounds in flight have committed.
"""

import argparse
import os
from collections.abc import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import gradientmail as gm

ALGORITHM = gm.DiLoCo(sync_every=20, fragments=2, quorum_size=2)
BATCH = 32


class MLP(nn.Module):
    """Layers as attributes, not ``Sequential``: parameter names must match torch_learner.py's."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 4)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.norm(nn.relu(self.fc1(x))))


def loss(model: MLP, x: mx.array) -> mx.array:
    return mx.mean((model(x) - x[:, :4]) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX learner of the two-framework demo")
    parser.add_argument(
        "--bucket",
        default=os.environ.get("GRADIENTMAIL_BUCKET"),
        required="GRADIENTMAIL_BUCKET" not in os.environ,
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("GRADIENTMAIL_PREFIX"),
        required="GRADIENTMAIL_PREFIX" not in os.environ,
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("GRADIENTMAIL_S3_ENDPOINT"),
        help="S3-compatible endpoint URL (MinIO, R2); omit for AWS",
    )
    parser.add_argument("--rounds", type=int, default=6, help="stop after this many rounds")
    parser.add_argument("--node-id", default=None, help="defaults to hostname-pid")
    args = parser.parse_args()
    rounds: int = args.rounds

    model = MLP()
    optimizer = optim.SGD(learning_rate=0.01)
    loss_and_grad = nn.value_and_grad(model, loss)
    mailbox = gm.S3Mailbox(args.bucket, args.prefix, endpoint_url=args.endpoint)
    session = gm.Session(model, algorithm=ALGORITHM, mailbox=mailbox, node_id=args.node_id)
    session.join()
    print(f"joined: {session.rounds_seen + 1} round(s) already committed")
    step, seen = 0, session.rounds_seen

    def batches() -> Iterator[mx.array]:
        while session.rounds_seen < rounds - 1:
            yield mx.random.normal((BATCH, 16))

    def train_step(x: mx.array) -> int:
        nonlocal step, seen
        value, grads = loss_and_grad(model, x)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        step += 1
        if session.rounds_seen != seen:
            seen = session.rounds_seen
            print(f"step {step}  round {seen} aligned  loss {value.item():.4f}")
        return BATCH

    session.run(train_step, batches())
    print(f"done: holds round {session.rounds_seen}")


if __name__ == "__main__":
    main()
