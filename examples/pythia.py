"""Pythia-14M re-trained from its own initialisation by four Salad RTX 3090s, through a Hugging
Face Storage Bucket, and scored against EleutherAI's checkpoints. One file: the compute, the
commands, the model and the data.

Pythia (arXiv 2304.01373) trained every model for 143,000 steps of 1024 sequences of 2048
tokens on the Pile, in one published order, and published the weights at step 0, at steps
1, 2, 4, ..., 512 and every 1000 steps. That makes the comparison exact up to the
synchronisation: every node starts from ``pythia-14m`` at step 0 and, at its local step ``s``,
trains on its share of the sequences the official run trained on at step ``s``, with the
official optimizer and schedule. After ``--steps`` steps the nodes together have seen what the
official run had seen by that step; the official run averaged gradients every step, the nodes
merge parameters through ``gm.Session`` every ``--sync-every`` steps. ``compare`` scores the
merged model, the official checkpoint at that step and the PolyPythias seeds
(``pythia-14m-seed1`` to ``seed9``, the same recipe from other initialisations and data
orders), whose spread is the noise a difference has to clear.

The model is GPT-NeoX as Pythia-14M configures it: 6 layers, 128 wide, 4 heads, rotary
embeddings on a quarter of each head, attention and MLP side by side on the block's input,
separate input and output embeddings over 50,304 tokens; 14M parameters, 12.9M of them the two
embeddings. Parameter names are the checkpoints' own, so a ``pytorch_model.bin`` from the Hub
loads as is once the attention buffers older transformers saved alongside (in the seeds'
files) are dropped. The MLP's GELU is the tanh approximation, which GPT-NeoX's fused
bias-GELU computed in training; transformers scores these checkpoints with the exact GELU,
and on the held-out batch the two give the same loss to the fifth decimal. The optimizer is
the official one: Adam with
decoupled weight decay 0.1 on matrices, betas 0.9 and 0.95, gradients clipped at norm 1, and a
learning rate rising linearly from 0 over the first 1430 steps to 1e-3, then following a cosine
that would reach 1e-4 at step 143,000. The official run trained in fp16 with loss scaling;
this one uses bf16 autocast.

Data: ``EleutherAI/pile-standard-pythia-preshuffled`` holds the run's 146,432,000 sequences of
2049 tokens (2048 inputs, each followed by its target) in training order, as uint16 across
21 files of up to 30 GB; step ``s`` is sequences ``1024·s`` to ``1024·s + 1023``. Node ``r`` of
``n`` reads its ``1024 / n`` sequences of each step with one HTTP range request, a step ahead on
a background thread, so a node of four downloads 1 MB per step and never a whole file. The
batch of step 140,000 is held out for validation. The official run trains on it long after
any step compared here; the seeds take the same sequences in other orders, so about 3% of it
falls in their first 4000 steps.

Each node trains exactly ``--steps`` steps, so the nodes together see the official run's
sequences up to that step and none after it; rounds commit behind the steps, and a node that
finishes first waits in ``close`` until the rounds of its last steps are committed. The
learning rate follows the step, as the official schedule does. With the defaults the two
embeddings, 26 MB each in fp32, are a fragment each and the blocks share the other two, and a
node publishes a fragment every 12 or 13 steps, about 1 MB/s to the bucket to be in every
round. ``--quorum`` is three of four: a node that cannot keep up is left out of the rounds
rather than stalling them, and its share of each batch does not reach the model; in two smoke
runs one Salad node uploaded at about 60 KB/s and never got a snapshot in. So before joining,
each node times a PUT the size of that fragment and refuses to train under ``--min-upload``
MB/s (default 2, twice the round's need): such a node fails fast, and the run is restarted on
another compute instead of comparing three nodes' data against the official four. The library
logs every upload, install and commit at INFO on each node's console.

Once: S3 credentials for the Hub (Settings → Access Tokens → the token's menu → Generate S3
credentials) as the ``hf`` profile in ``~/.aws/credentials``, ``GRADIENTMAIL_HF_NAMESPACE`` set
to your username or organization, a bucket (``hf buckets create gradientmail``), and a Skyward
daemon (``sky server start``) with ``SALAD_API_KEY``, ``SALAD_ORGANIZATION`` and
``SALAD_PROJECT`` in the environment. Then:

    sky run examples/pythia.py train --node all --steps 4000
    sky run examples/pythia.py compare --node 0 --step 4000
    sky run examples/pythia.py train --solo --rank 0 --nodes 1 --steps 128 --node 0
    sky run examples/pythia.py train --solo --steps 4000 --node 0

The third replays the official run on one node, every sequence of every step, to check this
file's model, optimizer and data against ``compare --only reference --step 128`` before
attributing a gap to the synchronisation. The fourth is the floor: one node's quarter of each
batch, without the others. The compute stays up between runs (``delete_on_exit=False``);
``sky compute delete`` ends it. A prefix holds one run; start another under a new prefix.
``evaluate`` and ``compare`` only read the bucket and the Hub, so they also run here:

    cd examples && AWS_PROFILE=hf uv run --frozen --with huggingface_hub \\
        --with ../../skyward python -c "import pythia; print(pythia.compare())"

``--rank`` and ``--nodes`` stand in for the node's own position when running off Skyward.
"""

from __future__ import annotations

import asyncio
import configparser
import contextlib
import itertools
import logging
import math
import os
import queue
import statistics
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import skyward as sky

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

    from gradientmail.mailbox import Mailbox

NAMESPACE = os.environ.get("GRADIENTMAIL_HF_NAMESPACE", "")
ENDPOINT = f"https://s3.hf.co/{NAMESPACE}"
MODEL = "EleutherAI/pythia-14m"
DATASET = "datasets/EleutherAI/pile-standard-pythia-preshuffled"
SHARD_BYTES = 30_000_000_000
BATCH = 1024
SEQUENCE = 2049
CONTEXT = SEQUENCE - 1
HELD_OUT = 140_000
TOTAL_STEPS = 143_000
WARMUP = 1430
LR = 1e-3
MIN_LR = 1e-4
LAYERS = 6
DIM = 128
HEADS = 4
HEAD = DIM // HEADS
ROTARY = HEAD // 4
VOCAB = 50304
STALE_BUFFERS = ("attention.bias", "attention.masked_bias", "attention.rotary_emb.inv_freq")


def s3_credentials() -> dict[str, str]:
    """The Hub's S3 keys: the ``hf`` AWS profile on this machine, the environment on a node."""
    profiles = configparser.ConfigParser()
    profiles.read(Path.home() / ".aws" / "credentials")
    if profiles.has_section("hf"):
        return {
            "AWS_ACCESS_KEY_ID": profiles["hf"]["aws_access_key_id"],
            "AWS_SECRET_ACCESS_KEY": profiles["hf"]["aws_secret_access_key"],
        }
    names = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    return {name: os.environ.get(name, "") for name in names}


def upload_rate(box: Mailbox, rank: int) -> float:
    """MB/s of one PUT the size of the embedding fragment to the bucket, deleted afterwards."""
    key = f"preflight/node-{rank}.bin"
    data = os.urandom(VOCAB * DIM * 4)

    async def measured() -> float:
        await box.delete([key])
        started = time.monotonic()
        await box.put_if_absent(key, data)
        seconds = time.monotonic() - started
        await box.delete([key])
        return len(data) / seconds / 1e6

    return asyncio.run(measured())


gpu = sky.app(
    provider=sky.Salad(priority="high"),
    accelerator=sky.accelerators.RTX_3090(),
    nodes=4,
    image=sky.Image(
        python="3.12",
        pip=["torch", "huggingface_hub", "safetensors~=0.8", "boto3~=1.36"],
        includes=["../src/gradientmail"],
        env={**s3_credentials(), "GRADIENTMAIL_HF_NAMESPACE": NAMESPACE},
        bootstrap_timeout=1800,
    ),
    options=sky.Options(ready_timeout=1800),
    delete_on_exit=False,
)


@dataclass(frozen=True)
class Place:
    """This process's position among the run's nodes."""

    rank: int
    nodes: int


@dataclass(frozen=True)
class Pythia:
    """The torch half, as ``pythia()`` builds it where torch is installed."""

    model: Callable[[], nn.Module]
    checkpoint: Callable[[str, int], dict[str, Tensor]]
    batches: Callable[[int, int, torch.device], Iterator[Tensor]]
    held_out: Callable[[], Tensor]
    pick_device: Callable[[], torch.device]
    autocast: Callable[[torch.device], contextlib.AbstractContextManager[object]]
    adamw: Callable[[nn.Module, torch.device], torch.optim.Optimizer]
    validation: Callable[[nn.Module, Tensor, int, torch.device], float]


@gpu
def train(
    bucket: str = "gradientmail",
    prefix: str = "pythia-14m",
    steps: int = 4000,
    sync_every: int = 50,
    fragments: int = 4,
    quorum: int = 3,
    micro_batch: int = 8,
    min_upload: float = 2.0,
    solo: bool = False,
    rank: int = -1,
    nodes: int = 0,
) -> dict[str, float]:
    """Re-train Pythia-14M from step 0 to STEPS across the run's nodes; --solo trains one
    node's share alone."""
    import torch

    import gradientmail as gm
    from gradientmail.adapters.torch import TorchBackend

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    nano = pythia()
    place = where(rank, nodes)
    share, rest = divmod(BATCH, place.nodes)
    if rest or share % micro_batch:
        raise ValueError(
            f"{BATCH} sequences do not split among {place.nodes} nodes "
            f"in micro-batches of {micro_batch}"
        )
    accumulation = share // micro_batch
    device = nano.pick_device()
    torch.set_float32_matmul_precision("high")
    model = nano.model()
    model.load_state_dict(nano.checkpoint(MODEL, 0))
    model.to(device)
    optimizer = nano.adamw(model, device)
    per_step = share * CONTEXT
    batches = nano.batches(place.rank, place.nodes, device)
    box = None if solo else mailbox(bucket, prefix)
    if box is not None:
        rate = upload_rate(box, place.rank)
        print(
            f"rank {place.rank}: uploads {VOCAB * DIM * 4 / 1e6:.0f} MB at {rate:.2f} MB/s",
            flush=True,
        )
        if rate < min_upload:
            raise RuntimeError(
                f"rank {place.rank} uploads at {rate:.2f} MB/s, under --min-upload {min_upload}: "
                "it would be in no round"
            )
    session = (
        None
        if box is None
        else gm.Session(
            model,
            algorithm=gm.DiLoCo(
                sync_every=sync_every,
                fragments=fragments,
                quorum_size=quorum,
                merge=gm.WeightedMean(),
            ),
            mailbox=box,
            learner_id=f"node-{place.rank}",
            poll_interval=5.0,
            backend=TorchBackend(str(device)),
        )
    )

    print(
        f"rank {place.rank} of {place.nodes}: {steps} steps of {per_step:,} tokens on {device}",
        flush=True,
    )
    started = time.monotonic()
    ema = math.nan
    seen = -1
    step = 0
    with contextlib.ExitStack() as stack:
        if session is not None:
            stack.enter_context(session)
        while step < steps:
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = 0.0
            for chunk in next(batches).split(micro_batch):
                with nano.autocast(device):
                    micro = model(chunk[:, :-1], chunk[:, 1:]) / accumulation
                micro.backward()
                loss += micro.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if session is not None:
                session.step(samples=per_step)
            step += 1
            ema = loss if math.isnan(ema) else 0.95 * ema + 0.05 * loss
            rounds_seen = -1 if session is None else session.rounds_seen
            if step == 1 or step % 20 == 0 or rounds_seen != seen:
                seen = rounds_seen
                rate = step * per_step / (time.monotonic() - started)
                print(
                    f"step {step}  {step / steps:.0%}  loss {ema:.3f}  lr {lr:.2e}  "
                    f"{rate:,.0f} tok/s  round {rounds_seen}",
                    flush=True,
                )
    return {
        "steps": step,
        "tokens": step * per_step,
        "rounds": -1 if session is None else session.rounds_seen,
        "loss": ema,
        "val_loss": nano.validation(model, nano.held_out(), micro_batch, device),
        "seconds": time.monotonic() - started,
    }


@gpu
def evaluate(
    bucket: str = "gradientmail",
    prefix: str = "pythia-14m",
    micro_batch: int = 8,
) -> dict[str, float]:
    """Validation loss of the run's newest committed parameters."""
    import gradientmail as gm

    nano = pythia()
    device = nano.pick_device()
    model = nano.model().to(device)
    newest = gm.install(model, mailbox=mailbox(bucket, prefix))
    return {
        "round": newest,
        "val_loss": nano.validation(model, nano.held_out(), micro_batch, device),
    }


@gpu
def compare(
    bucket: str = "gradientmail",
    prefix: str = "pythia-14m",
    step: int = 4000,
    seeds: int = 9,
    micro_batch: int = 8,
    only: str = "",
) -> dict[str, float]:
    """Validation loss of the run, of the official checkpoint at STEP and of SEEDS PolyPythias
    seeds at STEP; --only reference|ours for one side."""
    import torch

    import gradientmail as gm

    if only not in ("", "reference", "ours"):
        raise ValueError(f"--only takes reference or ours, not {only!r}")

    def scored(name: str) -> float:
        loss = nano.validation(model, tokens, micro_batch, device)
        print(f"{name}: val_loss {loss:.4f}", flush=True)
        return loss

    def official(repo: str) -> float:
        model.load_state_dict(nano.checkpoint(repo, step))
        return scored(f"{repo.removeprefix('EleutherAI/')} step {step}")

    def reference() -> dict[str, float]:
        losses = {MODEL: official(MODEL)}
        spread = [official(f"{MODEL}-seed{seed}") for seed in range(1, seeds + 1)]
        if len(spread) > 1:
            print(
                f"seeds: mean {statistics.mean(spread):.4f}  "
                f"stdev {statistics.stdev(spread):.4f}  "
                f"range {min(spread):.4f}-{max(spread):.4f}",
                flush=True,
            )
        return losses | {f"{MODEL}-seed{n}": loss for n, loss in enumerate(spread, start=1)}

    def run() -> dict[str, float]:
        newest = gm.install(model, mailbox=mailbox(bucket, prefix))
        return {"round": newest, prefix: scored(f"{prefix} round {newest}")}

    nano = pythia()
    device = nano.pick_device()
    torch.set_float32_matmul_precision("high")
    model = nano.model().to(device)
    tokens = nano.held_out()
    print(f"{len(tokens)} held-out sequences of step {HELD_OUT}, on {device}", flush=True)
    return {
        **({} if only == "ours" else reference()),
        **({} if only == "reference" else run()),
    }


def learning_rate(step: int) -> float:
    """The official rate for update ``step``, counted from 0: GPT-NeoX's linear warmup from 0
    over the first 1% of the run, then a cosine from ``LR`` towards ``MIN_LR``."""
    if step <= WARMUP:
        return LR * step / WARMUP
    progress = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return MIN_LR + (LR - MIN_LR) * 0.5 * (1 + math.cos(math.pi * progress))


def where(rank: int, nodes: int) -> Place:
    """The position given on the command line, or the node's own."""
    if rank >= 0 and nodes > 0:
        return Place(rank, nodes)
    info = sky.instance_info()
    return Place(info.rank, info.nodes)


def mailbox(bucket: str, prefix: str) -> Mailbox:
    import gradientmail as gm

    return gm.S3Mailbox(
        bucket,
        prefix,
        endpoint_url=ENDPOINT,
        region="us-east-1",
        request_checksums="when_required",
    )


def pythia() -> Pythia:
    """The model, the checkpoints, the data and the scorer, built where they run.

    ``sky run`` imports this file on this machine to find the commands, and torch is only
    on the nodes, so everything that needs it is defined here, when a command calls.
    """
    import torch
    from huggingface_hub import HfFileSystem, hf_hub_download
    from torch import Tensor, nn
    from torch.nn import functional as F

    hub = HfFileSystem()

    class GPTNeoX(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gpt_neox = Trunk()
            self.embed_out = nn.Linear(DIM, VOCAB, bias=False)

        def forward(self, inputs: Tensor, targets: Tensor | None = None) -> Tensor:
            """The mean loss against ``targets``, or the logits when there are none."""
            logits = self.embed_out(self.gpt_neox(inputs))
            if targets is None:
                return logits
            return F.cross_entropy(logits.view(-1, VOCAB), targets.reshape(-1))

    class Trunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_in = nn.Embedding(VOCAB, DIM)
            self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
            self.final_layer_norm = nn.LayerNorm(DIM)
            frequencies = 1 / 10000 ** (torch.arange(0, ROTARY, 2) / ROTARY)
            angles = torch.outer(torch.arange(CONTEXT, dtype=torch.float32), frequencies)
            self.register_buffer("cos", angles.repeat(1, 2).cos(), persistent=False)
            self.register_buffer("sin", angles.repeat(1, 2).sin(), persistent=False)

        def forward(self, inputs: Tensor) -> Tensor:
            length = inputs.shape[1]
            x = self.embed_in(inputs)
            for layer in self.layers:
                x = layer(x, self.cos[:length], self.sin[:length])
            return self.final_layer_norm(x)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layernorm = nn.LayerNorm(DIM)
            self.post_attention_layernorm = nn.LayerNorm(DIM)
            self.attention = Attention()
            self.mlp = MLP()

        def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            attended = self.attention(self.input_layernorm(x), cos, sin)
            return x + attended + self.mlp(self.post_attention_layernorm(x))

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query_key_value = nn.Linear(DIM, 3 * DIM)
            self.dense = nn.Linear(DIM, DIM)

        def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            def rotated(t: Tensor) -> Tensor:
                """Rotary embedding on the first ``ROTARY`` dimensions of each head."""
                head, tail = t[..., :ROTARY], t[..., ROTARY:]
                first, second = head.chunk(2, dim=-1)
                turned = torch.cat((-second, first), dim=-1)
                spun = head * cos.to(t.dtype) + turned * sin.to(t.dtype)
                return torch.cat((spun, tail), dim=-1)

            batch, length, _ = x.shape
            # Each head's query, key and value lie side by side, not the three in blocks.
            qkv = self.query_key_value(x).view(batch, length, HEADS, 3 * HEAD).transpose(1, 2)
            q, k, v = qkv.chunk(3, dim=-1)
            y = F.scaled_dot_product_attention(rotated(q), rotated(k), v, is_causal=True)
            return self.dense(y.transpose(1, 2).reshape(batch, length, DIM))

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dense_h_to_4h = nn.Linear(DIM, 4 * DIM)
            self.dense_4h_to_h = nn.Linear(4 * DIM, DIM)

        def forward(self, x: Tensor) -> Tensor:
            return self.dense_4h_to_h(F.gelu(self.dense_h_to_4h(x), approximate="tanh"))

    def checkpoint(repo: str, step: int) -> dict[str, Tensor]:
        """``repo``'s published parameters at ``step``; ``pytorch_model.bin`` is the one file
        every repo has at every step."""
        path = hf_hub_download(repo, "pytorch_model.bin", revision=f"step{step}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        return {name: t for name, t in state.items() if not name.endswith(STALE_BUFFERS)}

    def sequences(start: int, count: int) -> Tensor:
        """Sequences ``start`` to ``start + count - 1`` of the preshuffled Pile, one range
        request per file they lie in."""
        begin, end = start * SEQUENCE * 2, (start + count) * SEQUENCE * 2
        data = bytearray()
        while begin < end:
            shard, offset = divmod(begin, SHARD_BYTES)
            size = min(end - begin, SHARD_BYTES - offset)
            path = f"{DATASET}/document-{shard:05d}-of-00020.bin"
            with hub.open(path, "rb", cache_type="none") as file:
                file.seek(offset)
                data += file.read(size)
            begin += size
        return torch.frombuffer(data, dtype=torch.uint16).to(torch.long).view(count, SEQUENCE)

    class Batches(Iterator[Tensor]):
        """This node's share of every official step's sequences, from step 0 on, read ahead on
        a background thread."""

        def __init__(self, rank: int, nodes: int, device: torch.device, depth: int = 8) -> None:
            self._queue: queue.Queue[Tensor | BaseException] = queue.Queue(maxsize=depth)
            self._device = device
            share = BATCH // nodes
            threading.Thread(target=self._produce, args=(rank * share, share), daemon=True).start()

        def __next__(self) -> Tensor:
            match self._queue.get():
                case BaseException() as failure:
                    raise failure
                case chunk:
                    return chunk.to(self._device)

        def _produce(self, first: int, share: int) -> None:
            try:
                for step in itertools.count():
                    self._queue.put(sequences(step * BATCH + first, share))
            except BaseException as failure:
                self._queue.put(failure)

    def held_out() -> Tensor:
        return sequences(HELD_OUT * BATCH, BATCH)

    def pick_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def autocast(device: torch.device) -> torch.autocast:
        """bf16 on CUDA; elsewhere full precision, which is what MPS and CPU do well."""
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def adamw(model: nn.Module, device: torch.device) -> torch.optim.AdamW:
        """GPT-NeoX's optimizer: weight decay on matrices only, none on biases and norms."""
        matrices = [p for p in model.parameters() if p.dim() >= 2]
        vectors = [p for p in model.parameters() if p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": matrices, "weight_decay": 0.1}, {"params": vectors, "weight_decay": 0.0}],
            lr=LR,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=device.type == "cuda",
        )

    def validation(
        model: nn.Module, tokens: Tensor, micro_batch: int, device: torch.device
    ) -> float:
        """Mean loss over every token of ``tokens``' sequences."""
        with torch.no_grad(), autocast(device):
            total = sum(
                model(chunk[:, :-1], chunk[:, 1:]).item() * len(chunk)
                for chunk in tokens.to(device).split(micro_batch)
            )
        return total / len(tokens)

    return Pythia(
        model=GPTNeoX,
        checkpoint=checkpoint,
        batches=Batches,
        held_out=held_out,
        pick_device=pick_device,
        autocast=autocast,
        adamw=adamw,
        validation=validation,
    )
