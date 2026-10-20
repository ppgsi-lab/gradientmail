"""Phase 3 torch learner: the demo MLP trained natively with SGD, one ``Session`` running the loop.

Training is a synthetic regression (predict the first four inputs) on random batches. The step is
plain PyTorch that returns its batch size, and ``session.run`` calls it for every batch. Joining
creates the run from this model's parameters when it is first, or aligns with the run's global
parameters when it is not, so the two learners always start from one set of weights. There is no
syncer to run: each session closes rounds as elected through the mailbox. Both scripts share one
algorithm configuration (``ALGORITHM``) and must address the same bucket and prefix; with quorum
2 every round waits for both learners. Bucket, prefix and endpoint come from the arguments or from
``GRADIENTMAIL_BUCKET``, ``GRADIENTMAIL_PREFIX`` and ``GRADIENTMAIL_S3_ENDPOINT``; credentials
come from boto3's default chain. MinIO locally:

    docker run -d -p 19000:9000 -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin bitnamilegacy/minio
    export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
    export GRADIENTMAIL_S3_ENDPOINT=http://127.0.0.1:19000
    export GRADIENTMAIL_BUCKET=gradientmail GRADIENTMAIL_PREFIX=demo/run-1
    uv run python -c "import boto3, os; boto3.client('s3', \
        endpoint_url=os.environ['GRADIENTMAIL_S3_ENDPOINT']).create_bucket(Bucket='gradientmail')"

Then, in two terminals, in any order:

    uv run python examples/torch_learner.py --rounds 6 [--device cuda]
    uv run python examples/mlx_learner.py --rounds 6

Each learner prints the rounds it aligns with, stops drawing batches once it holds round
``rounds - 1`` and returns once its rounds in flight have committed. A prefix holds one run;
start a new run under a new prefix. Against AWS, omit the endpoint.
"""

import argparse
import os
from collections.abc import Iterator

import torch

import gradientmail as gm

ALGORITHM = gm.DiLoCo(sync_every=20, fragments=2, quorum_size=2)
BATCH = 32


class MLP(torch.nn.Module):
    """Layers as attributes, not ``Sequential``: parameter names must match mlx_learner.py's."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.norm = torch.nn.LayerNorm(32)
        self.fc2 = torch.nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(torch.relu(self.fc1(x))))


def main() -> None:
    parser = argparse.ArgumentParser(description="torch learner of the two-framework demo")
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
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps (default: cpu)")
    args = parser.parse_args()
    rounds: int = args.rounds
    device = torch.device(args.device)

    model = MLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    mailbox = gm.S3Mailbox(args.bucket, args.prefix, endpoint_url=args.endpoint)
    session = gm.Session(model, algorithm=ALGORITHM, mailbox=mailbox, node_id=args.node_id)
    session.join()
    print(f"joined: {session.rounds_seen + 1} round(s) already committed")
    step, seen = 0, session.rounds_seen

    def batches() -> Iterator[torch.Tensor]:
        while session.rounds_seen < rounds - 1:
            yield torch.randn(BATCH, 16, device=device)

    def train_step(x: torch.Tensor) -> int:
        nonlocal step, seen
        loss = ((model(x) - x[:, :4]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1
        if session.rounds_seen != seen:
            seen = session.rounds_seen
            print(f"step {step}  round {seen} aligned  loss {loss.item():.4f}")
        return BATCH

    session.run(train_step, batches())
    print(f"done: holds round {session.rounds_seen}")


if __name__ == "__main__":
    main()
