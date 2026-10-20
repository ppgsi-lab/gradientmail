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
21 files of up to 30 GB; step ``s`` is sequences ``1024·s`` to ``1024·s + 1023``. A node reads
a ``1024 / n`` slice of a step's sequences with one HTTP range request, a step ahead on a
background thread, so a node of four downloads 1 MB per step and never a whole file. The batch
of step 140,000 is held out for validation. The official run trains on it long after any step
compared here; the seeds take the same sequences in other orders, so about 3% of it falls in
their first 4000 steps.

The work is a pool of units, ``--unit`` consecutive steps of one of the ``n`` slices (4000 steps
in four slices are 800 units of 20), claimed as create-once objects (``units/``) through the
mailbox the session uses, earliest steps first. A node claims its next unit when its data
thread gets there, so a fast node takes more units than a slow one and the nodes finish within
one unit of each other whatever their speeds; with a fixed split, the slowest node of a smoke
run would have trained its last 18% alone, closing rounds by itself. Every sequence up to
``--steps`` is trained once, at the official learning rate of its step, in slices of the
official batch; only how many a node takes varies. Within a block the nodes try their own
slice first, so they seldom claim one unit at once; when they do (the gateway has accepted two
create-once PUTs made in the same instant), the object is read back and the node it does not
name moves on. A unit claimed by a node that died is not redone. ``--solo`` keeps the fixed
split: node ``r`` trains slice ``r`` of every step, alone.

Rounds commit behind the steps, and a node that finishes first waits in ``close`` until the
rounds of its last steps are committed. With the defaults the two embeddings, 26 MB each in
fp32, are a fragment each and the blocks share the other two, and a node publishes a fragment
every 12 or 13 steps, about 1 MB/s to the bucket to be in every round. ``--quorum`` is three
of four: a node that cannot keep up is left out of the rounds
rather than stalling them, and its share of each batch does not reach the model; in two smoke
runs one Salad node uploaded at about 60 KB/s and never got a snapshot in. ``--follow`` is
how many others' snapshots in a round make a node publish ahead of its schedule (one short
of the quorum by default; 1 lets the first node to reach a round set the pace; 0 never
publishes early), ``--grace``
fixes the seconds a round keeps collecting after quorum, and ``--alignment`` is what the
commit does to the steps a node trained since its snapshot: ``overwrite`` drops them, the
papers' rule and what the four-node run did, which cost it 22% of its steps; ``carry`` keeps
them all; ``blend:0.5`` keeps that share; ``adaptive`` carries while the round's deltas agree
(mean cosine 0.6 or more, as every commit records) and overwrites once they do not (under
0.4), fragment by fragment, with ``adaptive:0.5`` switching at one value and
``adaptive:0.4:0.6`` setting the two. ``--delta-dtype`` and ``--commit-dtype`` send the deltas
and the commits' updates in ``BF16``, ``F8_E4M3`` or ``F4`` instead of fp32, a half, a quarter
or an eighth of the bytes, and ``--compression zstd`` shuffles and compresses each node's
payloads on top of that.

Salad nodes are home computers on whatever uplink their owner has: of the nodes rented for
these runs, uplinks ran from 0.7 to 27 MB/s and one node's downlink starved its GPU at a
quarter of the others' pace. So the compute carries a health check, ``healthy``: before a node
is admitted, it downloads 26 MB of the dataset from the Hub and uploads 26 MB, an embedding
fragment, to the bucket. A node under ``PYTHIA_MIN_MBPS`` (default 20 megabits per second) in
either direction twice in a row is lost to Skyward, which buys another and runs ``train`` on
it; a learner that arrives mid-run joins at the newest commit and claims units like the rest.
Skyward keeps probing every two minutes, but a node that has passed answers at once from then
on, so the run's traffic is its own. The function travels to the nodes as bytecode, so
``sky run`` must run on their Python, 3.12. Skyward also replaces a node that takes over four
minutes to be allocated or to install its environment (``PYTHIA_BOOTSTRAP_TIMEOUT``, 240
seconds; a node on a good link takes 40) or over two to start its worker: a slow install is
a slow link. The image, ``PYTHIA_BOOTSTRAP_TIMEOUT`` and ``PYTHIA_NODES`` with it, names the
compute, so ``compare`` reaches a run's compute only with the values the run was launched
with. Once every node holds the run's parameters, a barrier releases them at one instant of
the bucket's clock, so training starts together instead of the first node up training alone
while the rest boot. The library logs every upload, alignment and commit at INFO on each
node's console.

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
``PYTHIA_NODES`` sets how many nodes train (default 4). Each trains ``1024 / n`` sequences
a step, so with eight a step takes about half as long, a fragment is due twice as often, and
a node needs twice the upload rate to be in every round:

    PYTHIA_NODES=8 PYTHIA_MIN_MBPS=40 sky run examples/pythia.py train --node all \\
        --steps 4000 --prefix pythia-14m-8n-carry --quorum 7 --alignment carry

``evaluate`` and ``compare`` only read the bucket and the Hub, so they also run here:

    cd examples && AWS_PROFILE=hf uv run --frozen --with huggingface_hub \\
        --with ../../skyward python -c "import pythia; print(pythia.compare())"

``--rank`` and ``--nodes`` together stand in for the node's own position when running off
Skyward.
"""

from __future__ import annotations

import configparser
import contextlib
import logging
import math
import os
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import skyward as sky

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

    from gradientmail.algorithms import Alignment
    from gradientmail.compression import Compression
    from gradientmail.mailbox import Mailbox
    from gradientmail.serialization import DType, Quantized

NAMESPACE = os.environ.get("GRADIENTMAIL_HF_NAMESPACE", "")
ENDPOINT = f"https://s3.hf.co/{NAMESPACE}"
NODES = int(os.environ.get("PYTHIA_NODES", "4"))
MIN_MBPS = float(os.environ.get("PYTHIA_MIN_MBPS", "20"))
HEALTH_EVERY = 120.0
BOOTSTRAP_TIMEOUT = int(os.environ.get("PYTHIA_BOOTSTRAP_TIMEOUT", "240"))
BUCKET = "gradientmail"
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

log = logging.getLogger("pythia")


@dataclass(frozen=True)
class Unit:
    """``count`` official steps from ``first``, slice ``lane`` of ``lanes`` of each one's batch."""

    first: int
    count: int
    lane: int
    lanes: int


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


def healthy(info: sky.Info) -> bool | str:
    """Whether this node's network carries the run: ``MIN_MBPS`` or more both downloading the
    dataset from the Hub and uploading an embedding fragment to the bucket. Skyward runs it on
    the node, with the image's credentials, before admitting it and every ``HEALTH_EVERY``
    seconds after; a node that has passed answers at once from then on. The verdict is the
    reason a failing node is replaced for."""
    if os.environ.get("PYTHIA_HEALTHY"):
        return True
    import urllib.request

    import boto3
    from botocore.config import Config

    size = VOCAB * DIM * 4
    request = urllib.request.Request(
        f"https://huggingface.co/{DATASET}/resolve/main/document-00000-of-00020.bin",
        headers={"Range": f"bytes=0-{size - 1}"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=60) as response:
        received = len(response.read())
    down = received * 8 / (time.monotonic() - started) / 1e6
    s3 = boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        config=Config(request_checksum_calculation="when_required"),
    )
    key = f"health/{info.node}.bin"
    started = time.monotonic()
    s3.put_object(Bucket=BUCKET, Key=key, Body=os.urandom(size))
    up = size * 8 / (time.monotonic() - started) / 1e6
    s3.delete_object(Bucket=BUCKET, Key=key)
    verdict = f"{down:.0f} Mbit/s down, {up:.0f} Mbit/s up, {MIN_MBPS:.0f} required"
    print(f"health: {verdict}", flush=True)
    if min(down, up) < MIN_MBPS:
        return verdict
    os.environ["PYTHIA_HEALTHY"] = "1"
    return True


gpu = sky.app(
    provider=sky.Salad(priority="high", min_download_mbps=300, min_upload_mbps=200),
    accelerator=sky.accelerators.RTX_3090(),
    nodes=NODES,
    image=sky.Image(
        python="3.12",
        pip=["torch", "huggingface_hub", "safetensors~=0.8", "boto3~=1.36", "zstandard>=0.23"],
        includes=["../src/gradientmail"],
        env={
            **s3_credentials(),
            "GRADIENTMAIL_HF_NAMESPACE": NAMESPACE,
            "PYTHIA_NODES": str(NODES),
        },
        bootstrap_timeout=BOOTSTRAP_TIMEOUT,
    ),
    options=sky.Options(
        provision_timeout=240.0,
        worker_timeout=120.0,
        ready_timeout=1800,
        health_checker=sky.HealthChecker(
            healthy, interval=HEALTH_EVERY, timeout=120.0, consecutive_failures=2
        ),
    ),
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
    load_unit: Callable[[Unit], Iterator[tuple[int, Tensor]]]
    held_out: Callable[[], Tensor]
    pick_device: Callable[[], torch.device]
    autocast: Callable[[torch.device], contextlib.AbstractContextManager[object]]
    adamw: Callable[[nn.Module, torch.device], torch.optim.Optimizer]
    validation: Callable[[nn.Module, Tensor, int, torch.device], float]


@gpu
def train(
    bucket: str = BUCKET,
    prefix: str = "pythia-14m",
    steps: int = 4000,
    sync_every: int = 50,
    fragments: int = 4,
    quorum: int = 3,
    follow: int | None = None,
    grace: float | None = None,
    alignment: str = "overwrite",
    delta_dtype: str = "",
    commit_dtype: str = "",
    compression: str = "off",
    micro_batch: int = 8,
    unit: int = 20,
    solo: bool = False,
    rank: int = -1,
    nodes: int = 0,
) -> dict[str, float]:
    """Re-train Pythia-14M from step 0 to STEPS across the run's nodes, in units of UNIT steps
    claimed from the bucket; --solo trains one node's fixed slice alone. --alignment is
    overwrite, carry, blend:<mixing> or adaptive; --delta-dtype and --commit-dtype send deltas
    and commits in F32, F16, BF16, F8_E4M3 or F4 (the parameters' own dtype by default);
    --compression zstd compresses this node's payloads."""
    import torch

    import gradientmail as gm
    from gradientmail.adapters.torch import TorchBackend
    from gradientmail.patterns.barrier import Barrier
    from gradientmail.patterns.shared_queue import shared_queue

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
    box = None if solo else mailbox(bucket, prefix)
    lanes = (
        (place.rank,) if solo else tuple((place.rank + i) % place.nodes for i in range(place.nodes))
    )
    units = {
        f"{first // unit:06d}-{lane}.json": Unit(first, min(unit, steps - first), lane, place.nodes)
        for first in range(0, steps, unit)
        for lane in lanes
    }
    session = (
        None
        if box is None
        else gm.Session(
            model,
            algorithm=gm.DiLoCo(
                sync_every=sync_every,
                fragments=fragments,
                quorum_size=quorum,
                follow=follow,
                grace_seconds=grace,
                alignment=aligned_by(alignment),
                merge=gm.WeightedMean(),
                delta_dtype=payload_dtype(delta_dtype),
                commit_dtype=payload_dtype(commit_dtype),
            ),
            mailbox=box,
            learner_id=f"node-{place.rank}",
            poll_interval=5.0,
            backend=TorchBackend(str(device)),
            compression=compressed_by(compression),
        )
    )

    print(
        f"rank {place.rank} of {place.nodes}: {'slice ' + str(place.rank) if solo else 'units'} "
        f"of {steps} steps, {per_step:,} tokens per step, on {device}",
        flush=True,
    )
    ema = math.nan
    seen = -1
    done = 0
    with contextlib.ExitStack() as stack:
        if session is not None:
            stack.enter_context(session)
        batches = stack.enter_context(
            shared_queue(
                units,
                mailbox=box if box is not None else gm.MemoryMailbox(),
                learner_id=f"node-{place.rank}",
                load=nano.load_unit,
            )
        )
        if box is not None:
            Barrier(mailbox=box, learner_id=f"node-{place.rank}").wait(
                "start", parties=place.nodes, timeout=1800
            )
            print(f"rank {place.rank}: training", flush=True)
        started = time.monotonic()
        for step, batch in batches:
            batch = batch.to(device)
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = 0.0
            for chunk in batch.split(micro_batch):
                with nano.autocast(device):
                    micro = model(chunk[:, :-1], chunk[:, 1:]) / accumulation
                micro.backward()
                loss += micro.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if session is not None:
                session.step(samples=per_step)
            done += 1
            ema = loss if math.isnan(ema) else 0.95 * ema + 0.05 * loss
            rounds_seen = -1 if session is None else session.rounds_seen
            if done == 1 or done % 20 == 0 or rounds_seen != seen:
                seen = rounds_seen
                rate = done * per_step / (time.monotonic() - started)
                print(
                    f"step {step}  {(step + 1) / steps:.0%}  done {done}  loss {ema:.3f}  "
                    f"lr {lr:.2e}  {rate:,.0f} tok/s  round {rounds_seen}",
                    flush=True,
                )
    return {
        "steps": done,
        "tokens": done * per_step,
        "rounds": -1 if session is None else session.rounds_seen,
        "loss": ema,
        "val_loss": nano.validation(model, nano.held_out(), micro_batch, device),
        "seconds": time.monotonic() - started,
    }


@gpu
def evaluate(
    bucket: str = BUCKET,
    prefix: str = "pythia-14m",
    micro_batch: int = 8,
) -> dict[str, float]:
    """Validation loss of the run's newest committed parameters."""
    import gradientmail as gm

    nano = pythia()
    device = nano.pick_device()
    model = nano.model().to(device)
    newest = gm.align(model, mailbox=mailbox(bucket, prefix))
    return {
        "round": newest,
        "val_loss": nano.validation(model, nano.held_out(), micro_batch, device),
    }


@gpu
def compare(
    bucket: str = BUCKET,
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
        newest = gm.align(model, mailbox=mailbox(bucket, prefix))
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
    """The position given on the command line, or this node's rank among ``NODES``, the count
    the run was launched with and the image carries; ``instance_info`` counts only the nodes
    up when the task reached this one."""
    if rank >= 0 and nodes > 0:
        return Place(rank, nodes)
    return Place(sky.instance_info().rank, nodes if nodes > 0 else NODES)


def aligned_by(name: str) -> Alignment:
    """``overwrite``, ``carry``, ``blend:<mixing>``, ``adaptive``, ``adaptive:<threshold>`` or
    ``adaptive:<overwrite_below>:<carry_above>``."""
    import gradientmail as gm

    match name.split(":"):
        case ["overwrite"]:
            return gm.Overwrite()
        case ["carry"]:
            return gm.Carry()
        case ["blend", mixing]:
            return gm.Blend(float(mixing))
        case ["adaptive"]:
            return gm.Adaptive()
        case ["adaptive", threshold]:
            return gm.Adaptive(float(threshold))
        case ["adaptive", below, above]:
            return gm.Adaptive(carry_above=float(above), overwrite_below=float(below))
        case _:
            raise ValueError(
                "--alignment takes overwrite, carry, blend:<mixing>, adaptive, "
                f"adaptive:<threshold> or adaptive:<overwrite_below>:<carry_above>, not {name!r}"
            )


def payload_dtype(name: str) -> DType | Quantized | None:
    """``F32``, ``F16``, ``BF16``, ``F8_E4M3``, ``F4``, or the parameters' own dtype when empty."""
    match name:
        case "":
            return None
        case "F32" | "F16" | "BF16" | "F8_E4M3" | "F4" as dtype:
            return dtype
        case _:
            raise ValueError(f"a payload dtype is F32, F16, BF16, F8_E4M3 or F4, not {name!r}")


def compressed_by(name: str) -> Compression:
    match name:
        case "off" | "zstd" as compression:
            return compression
        case _:
            raise ValueError(f"--compression takes off or zstd, not {name!r}")


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
    from torch import nn
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

    def load_unit(unit: Unit) -> Iterator[tuple[int, Tensor]]:
        share = BATCH // unit.lanes
        last = unit.first + unit.count - 1
        log.info("unit: steps %d-%d, slice %d", unit.first, last, unit.lane)
        for step in range(unit.first, last + 1):
            yield step, sequences(step * BATCH + unit.lane * share, share)

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
        load_unit=load_unit,
        held_out=held_out,
        pick_device=pick_device,
        autocast=autocast,
        adamw=adamw,
        validation=validation,
    )
