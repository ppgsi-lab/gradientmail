"""GPT-2 small pre-trained from scratch on FineWeb-Edu by four Salad RTX 3090s, through a
Hugging Face Storage Bucket.

Every node runs ``train``: a nanoGPT-style decoder (12 layers, 768 wide, 124M parameters;
``nanogpt.py``) with AdamW, bf16 autocast and a cosine schedule, one ``gm.Session`` around the
loop. The nodes never talk to each other. The run lives in a Storage Bucket reached through the
Hub's S3 gateway, and the sessions settle among themselves who closes each round. ``evaluate``
loads the newest committed parameters with ``gm.install`` and measures validation loss, so the
merged model is judged, not a node's local copy. ``train --solo`` is the baseline: the same
loop without a session, for one node's share of the tokens.

The run ends when ``tokens · fragments / (nodes · tokens per step · sync_every)`` rounds are
committed, not after a number of local steps, so nodes of different speeds stop together: a
slower node contributes fewer steps per round, publishing when the round is about to close
without it, and the learning-rate schedule follows the rounds rather than the local step.

Data: node ``r`` of ``n`` reads files ``r, r + n, ...`` of the first thirteen ``sample/10BT``
parquet files; file 013 is held out for validation.

Once: S3 credentials for the Hub (Settings → Access Tokens → the token's menu → Generate S3
credentials) as the ``hf`` profile in ``~/.aws/credentials``, ``GRADIENTMAIL_HF_NAMESPACE`` set
to your username or organization, a bucket (``hf buckets create gradientmail``), and a Skyward
daemon (``sky server start``) with ``SALAD_API_KEY``, ``SALAD_ORGANIZATION`` and
``SALAD_PROJECT`` in the environment. The credentials travel to the nodes in the image's
environment. Then:

    sky run examples/gpt2.py train --node all --tokens 1e9
    sky run examples/gpt2.py evaluate --node 0
    sky run examples/gpt2.py train --solo --prefix gpt2-solo --node 0

The compute stays up between runs (``delete_on_exit=False``); ``sky compute delete`` ends it.
A prefix holds one run; start another under a new prefix. Since ``evaluate`` only reads the
bucket, it also runs here:

    cd examples && AWS_PROFILE=hf uv run --with tiktoken --with pyarrow --with huggingface_hub \\
        --with ../../skyward python -c "import gpt2; print(gpt2.evaluate())"

With ``--sync-every 200`` and ``--fragments 4`` a node uploads a quarter of the model, 124 MB
in fp32, about every two minutes; raise ``--sync-every`` on a slow uplink. ``--rank`` and
``--nodes`` stand in for the node's own position when running off Skyward.
"""

import configparser
import contextlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import skyward as sky

if TYPE_CHECKING:
    from gradientmail.mailbox import Mailbox

NAMESPACE = os.environ.get("GRADIENTMAIL_HF_NAMESPACE", "")
ENDPOINT = f"https://s3.hf.co/{NAMESPACE}"
BLOCK = 1024


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


gpu = sky.app(
    provider=sky.Salad(priority="high"),
    accelerator=sky.accelerators.RTX_3090(),
    nodes=4,
    image=sky.Image(
        python="3.12",
        pip=[
            "torch",
            "tiktoken",
            "pyarrow",
            "huggingface_hub",
            "safetensors~=0.8",
            "boto3~=1.36",
        ],
        includes=["nanogpt.py", "../src/gradientmail"],
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


@gpu
def train(
    bucket: str = "gradientmail",
    prefix: str = "gpt2",
    tokens: float = 1e9,
    sync_every: int = 200,
    fragments: int = 4,
    quorum: int = 3,
    micro_batch: int = 16,
    accumulation: int = 4,
    layers: int = 12,
    dim: int = 768,
    heads: int = 12,
    solo: bool = False,
    rank: int = -1,
    nodes: int = 0,
) -> dict[str, float]:
    """Pre-train for TOKENS tokens across the run's nodes; --solo trains one node's share alone."""
    import torch
    from nanogpt import (
        GPT,
        TRAIN_FILES,
        Batches,
        Config,
        adamw,
        autocast,
        learning_rate,
        pick_device,
        validation,
    )

    import gradientmail as gm
    from gradientmail.adapters.torch import TorchBackend

    place = where(rank, nodes)
    device = pick_device()
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    model = GPT(Config(layers, dim, heads)).to(device)
    optimizer = adamw(model, device)
    per_step = micro_batch * BLOCK * accumulation
    share = tokens / (place.nodes * per_step)
    # One node's share, in steps alone or in rounds of sync_every / fragments steps each.
    steps = max(1, round(share))
    rounds = max(1, round(share * fragments / sync_every))
    batches = Batches(TRAIN_FILES[place.rank :: place.nodes], micro_batch, device)
    session = (
        None
        if solo
        else gm.Session(
            model,
            algorithm=gm.DecoupledDiLoCo(
                sync_every=sync_every,
                fragments=fragments,
                quorum_size=quorum,
                merge=gm.WeightedMean(),
            ),
            mailbox=mailbox(bucket, prefix),
            learner_id=f"node-{place.rank}",
            poll_interval=5.0,
            backend=TorchBackend(str(device)),
        )
    )

    def progress(step: int) -> float:
        """How far the run is, in [0, 1]: by rounds installed, or by this node's steps alone."""
        return step / steps if session is None else (session.rounds_seen + 1) / rounds

    target = f"{steps} steps" if session is None else f"{rounds} rounds"
    print(
        f"rank {place.rank} of {place.nodes}: {target} of {per_step:,}-token steps on {device}",
        flush=True,
    )
    started = time.monotonic()
    ema = math.nan
    seen = -1
    step = 0
    with contextlib.ExitStack() as stack:
        if session is not None:
            stack.enter_context(session)
        while (at := progress(step)) < 1:
            lr = learning_rate(step, at)
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = 0.0
            for _ in range(accumulation):
                inputs, targets = next(batches)
                with autocast(device):
                    micro = model(inputs, targets) / accumulation
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
                    f"step {step}  {at:.0%}  loss {ema:.3f}  lr {lr:.2e}  "
                    f"{rate:,.0f} tok/s  round {rounds_seen}",
                    flush=True,
                )
    return {
        "steps": step,
        "tokens": step * per_step,
        "rounds": -1 if session is None else session.rounds_seen,
        "loss": ema,
        "val_loss": validation(model, micro_batch, 32, device),
        "seconds": time.monotonic() - started,
    }


@gpu
def evaluate(
    bucket: str = "gradientmail",
    prefix: str = "gpt2",
    layers: int = 12,
    dim: int = 768,
    heads: int = 12,
    micro_batch: int = 16,
    batches: int = 64,
) -> dict[str, float]:
    """Validation loss of the run's newest committed parameters."""
    from nanogpt import GPT, Config, pick_device, validation

    import gradientmail as gm

    device = pick_device()
    model = GPT(Config(layers, dim, heads)).to(device)
    newest = gm.install(model, mailbox=mailbox(bucket, prefix))
    return {"round": newest, "val_loss": validation(model, micro_batch, batches, device)}


def where(rank: int, nodes: int) -> Place:
    """The position given on the command line, or the node's own."""
    if rank >= 0 and nodes > 0:
        return Place(rank, nodes)
    info = sky.instance_info()
    return Place(info.rank, info.nodes)


def mailbox(bucket: str, prefix: str) -> "Mailbox":
    import gradientmail as gm

    return gm.S3Mailbox(
        bucket,
        prefix,
        endpoint_url=ENDPOINT,
        region="us-east-1",
        request_checksums="when_required",
    )
