"""Phase 3 interoperability: torch and MLX exchange the demo MLP's bytes with no remapping.

Skipped unless both frameworks are installed. Cross-framework checks compare parameter
bytes and values, never activations: MLX GPU matmul differs from CPU at ~1e-3 relative.
"""

import asyncio
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence

import boto3.session
import pytest
from moto import mock_aws

pytest.importorskip("torch")
pytest.importorskip("mlx.core")

import mlx.core as mx
import mlx.nn
import mlx.optimizers
import torch
import torch.nn

from gradientmail.adapters import Adapter
from gradientmail.adapters.mlx import MlxAdapter
from gradientmail.adapters.torch import TorchAdapter
from gradientmail.algorithms.diloco import DecoupledDiLoCo
from gradientmail.clock import MonotonicClock
from gradientmail.learner import Learner
from gradientmail.mailbox import Mailbox
from gradientmail.mailbox.s3 import S3Mailbox
from gradientmail.protocol import Exchange, RoundRecord
from gradientmail.serialization import ParameterSpec, Schema, TensorBytes
from gradientmail.session import Session

# The interoperability claim: both adapters read this literal off their framework's demo MLP.
EXPECTED_MLP_SCHEMA = Schema(
    (
        ParameterSpec("fc1.bias", "F32", (32,)),
        ParameterSpec("fc1.weight", "F32", (32, 16)),
        ParameterSpec("fc2.bias", "F32", (4,)),
        ParameterSpec("fc2.weight", "F32", (4, 32)),
        ParameterSpec("norm.bias", "F32", (32,)),
        ParameterSpec("norm.weight", "F32", (32,)),
    )
)

NAMES = tuple(spec.name for spec in EXPECTED_MLP_SCHEMA.parameters)

POLL = 0.01
TIMEOUT = 30.0


class TorchMLP(torch.nn.Module):
    """The demo model: layers as attributes, not ``Sequential``, so names match MLX."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 32)
        self.norm = torch.nn.LayerNorm(32)
        self.fc2 = torch.nn.Linear(32, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(torch.relu(self.fc1(x))))


class MlxMLP(mlx.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = mlx.nn.Linear(16, 32)
        self.norm = mlx.nn.LayerNorm(32)
        self.fc2 = mlx.nn.Linear(32, 4)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.norm(mlx.nn.relu(self.fc1(x))))


def describe_cross_framework():
    def it_produces_the_expected_mlp_schema_from_both_frameworks():
        assert TorchAdapter(TorchMLP()).schema() == EXPECTED_MLP_SCHEMA
        assert MlxAdapter(MlxMLP()).schema() == EXPECTED_MLP_SCHEMA

    def it_applies_torch_bytes_to_mlx_and_back_unchanged():
        torch_model, mlx_model = TorchMLP(), MlxMLP()
        published = TorchAdapter(torch_model).snapshot(NAMES)

        MlxAdapter(mlx_model).apply(published)

        assert MlxAdapter(mlx_model).snapshot(NAMES) == published
        # (32, 16) is not square, so a layout disagreement the byte check tolerates shows here.
        assert mlx_model.fc1.weight.tolist() == torch_model.fc1.weight.tolist()

    def it_applies_mlx_bytes_to_torch_and_back_unchanged():
        torch_model, mlx_model = TorchMLP(), MlxMLP()
        published = MlxAdapter(mlx_model).snapshot(NAMES)

        TorchAdapter(torch_model).apply(published)

        assert TorchAdapter(torch_model).snapshot(NAMES) == published
        assert torch_model.fc1.weight.tolist() == mlx_model.fc1.weight.tolist()


def describe_two_frameworks_one_syncer():
    @pytest.fixture
    def connect() -> Iterator[Callable[[], Mailbox]]:
        """A fresh ``S3Mailbox`` per party over one moto bucket, as separate processes have."""
        # moto intercepts only clients created while the mock is active.
        with mock_aws():
            boto3.session.Session().client("s3", region_name="us-east-1").create_bucket(
                Bucket="gradientmail"
            )
            yield lambda: S3Mailbox("gradientmail", "run", region="us-east-1")

    def it_leaves_both_models_with_identical_bytes_after_three_rounds_through_moto(
        connect: Callable[[], Mailbox],
    ):
        algorithm = DecoupledDiLoCo(sync_every=8, fragments=2, quorum_size=2)
        torch_model, mlx_model = TorchMLP(), MlxMLP()
        initial = TorchAdapter(torch_model).snapshot(NAMES)
        MlxAdapter(mlx_model).apply(initial)
        stop = threading.Event()
        exchange = Exchange(connect(), clock=MonotonicClock())

        def session(model: object, learner_id: str) -> Session:
            return Session(
                model,
                algorithm=algorithm,
                mailbox=connect(),
                learner_id=learner_id,
                poll_interval=POLL,
            )

        def torch_learner(session: Session) -> None:
            optimizer = torch.optim.SGD(torch_model.parameters(), lr=0.01)
            with session:
                while not stop.is_set():
                    x, y = torch.randn(32, 16), torch.randn(32, 4)
                    loss = ((torch_model(x) - y) ** 2).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    session.step(samples=len(x))

        def mlx_learner(session: Session) -> None:
            def loss(model: MlxMLP, x: mx.array, y: mx.array) -> mx.array:
                return mx.mean((model(x) - y) ** 2)

            optimizer = mlx.optimizers.SGD(learning_rate=0.01)
            loss_and_grad = mlx.nn.value_and_grad(mlx_model, loss)
            with session:
                while not stop.is_set():
                    x, y = mx.random.normal((32, 16)), mx.random.normal((32, 4))
                    _, grads = loss_and_grad(mlx_model, x, y)
                    optimizer.update(mlx_model, grads)
                    mx.eval(mlx_model.parameters(), optimizer.state)
                    session.step(samples=x.shape[0])

        async def three_rounds() -> None:
            # The sessions close the rounds themselves, one on torch and one on MLX; one group,
            # so a failure on either learner ends the wait instead of running it out. Both hold
            # a lease before either trains, so every round waits for both.
            async with asyncio.timeout(TIMEOUT):
                while not {"torch", "mlx"} <= set(await exchange.leases()):
                    await asyncio.sleep(POLL)
            async with asyncio.TaskGroup() as group:
                group.create_task(asyncio.to_thread(torch_learner, torch_session))
                group.create_task(asyncio.to_thread(mlx_learner, mlx_session))
                try:
                    async with asyncio.timeout(TIMEOUT):
                        while len(await exchange.rounds_after(-1)) < 3:
                            await asyncio.sleep(POLL)
                finally:
                    stop.set()

        def rejoin(adapter: Adapter, learner_id: str) -> None:
            # A bare learner, so nothing closes rounds between the two rejoins.
            learner = Learner(
                adapter=adapter, algorithm=algorithm, mailbox=connect(), learner_id=learner_id
            )
            learner.start()
            learner.close()

        def newest(records: Sequence[RoundRecord]) -> Mapping[str, TensorBytes]:
            """The global parameters: the newest committed version of every fragment."""

            def fetched(record: RoundRecord) -> Mapping[str, TensorBytes]:
                tensors = asyncio.run(exchange.fetch(record.theta))
                assert tensors is not None, (
                    f"{record.theta.key} is the newest version but was swept"
                )
                return tensors

            by_fragment = {record.fragment: record for record in records}
            return {
                name: tensor
                for record in by_fragment.values()
                for name, tensor in fetched(record).items()
            }

        torch_session, mlx_session = session(torch_model, "torch"), session(mlx_model, "mlx")
        asyncio.run(three_rounds())
        # Between syncs each model drifts by its own training, by design; joining again installs
        # the newest committed version of every fragment, which is what the two must share.
        rejoin(TorchAdapter(torch_model), "torch")
        rejoin(MlxAdapter(mlx_model), "mlx")

        records = asyncio.run(exchange.rounds_after(-1))
        committed = newest(records)
        assert [record.round for record in records] == list(range(len(records)))
        assert len(records) >= 3
        # The three rounds were closed with both live; the first to close leaves the other
        # closing its last rounds alone.
        assert [tuple(p.learner for p in record.participants) for record in records[:3]] == [
            ("mlx", "torch")
        ] * 3
        assert [record.rejected for record in records] == [()] * len(records)
        assert TorchAdapter(torch_model).snapshot(NAMES) == MlxAdapter(mlx_model).snapshot(NAMES)
        assert dict(TorchAdapter(torch_model).snapshot(NAMES)) == dict(committed)
        assert all(committed[name] != initial[name] for name in NAMES)
