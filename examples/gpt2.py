"""GPT-2 small pre-trained from scratch on FineWeb-Edu by four Salad RTX 3090s, through a
Hugging Face Storage Bucket, and scored against OpenAI's checkpoint. One file: the compute,
the commands, the model and the data.

Every node runs ``train``: a nanoGPT-style decoder (12 layers, 768 wide, 124M parameters;
pre-norm blocks, GELU MLPs, tied embeddings, GPT-2's initialisation) with AdamW, bf16
autocast and a cosine schedule, one ``gm.Session`` around the loop. The nodes never talk to
each other. The run lives in a Storage Bucket reached through the Hub's S3 gateway, and the
sessions settle among themselves who closes each round. ``evaluate`` loads the newest
committed parameters with ``gm.install`` and measures validation loss, so the merged model is
judged, not a node's local copy. ``train --solo`` is the baseline: the same loop without a
session, for one node's share of the tokens.

``compare`` scores the run and OpenAI's 124M checkpoint through one scorer, on the two
measures llm.c uses to call that model reproduced: the validation loss, in-distribution for
ours and not for OpenAI's, and HellaSwag zero-shot, where the four endings of each validation
example are scored by the log-likelihood of their tokens after the context; ``acc`` picks the
ending with the highest total, ``acc_norm`` the highest per token. Chance is 25%; OpenAI's
checkpoint gets 29.6% ``acc_norm`` and GPT-3's 125M 33.7%. OpenAI's weights come from the Hub
(``openai-community/gpt2``) without transformers, the Conv1D matrices transposed and the
vocabulary padded as ``VOCAB`` pads it. Validation and HellaSwag are scored over GPT-2's
50,257 tokens, so the padding never counts. ``--only openai`` skips the bucket and checks the
scorer against the published number.

The run ends when ``tokens · fragments / (nodes · tokens per step · sync_every)`` rounds are
committed, not after a number of local steps, so nodes of different speeds stop together: a
slower node contributes fewer steps per round, publishing when the round is about to close
without it, and the learning-rate schedule follows the rounds rather than the local step.

Data: node ``r`` of ``n`` reads files ``r, r + n, ...`` of the first thirteen ``sample/10BT``
parquet files of HuggingFaceFW/fineweb-edu, downloaded from the Hub when reached and tokenized
with tiktoken's GPT-2 encoding on a background thread; file 013 is held out for validation.

Once: S3 credentials for the Hub (Settings → Access Tokens → the token's menu → Generate S3
credentials) as the ``hf`` profile in ``~/.aws/credentials``, ``GRADIENTMAIL_HF_NAMESPACE`` set
to your username or organization, a bucket (``hf buckets create gradientmail``), and a Skyward
daemon (``sky server start``) with ``SALAD_API_KEY``, ``SALAD_ORGANIZATION`` and
``SALAD_PROJECT`` in the environment. The credentials travel to the nodes in the image's
environment. Then:

    sky run examples/gpt2.py train --node all --tokens 1e9
    sky run examples/gpt2.py evaluate --node 0
    sky run examples/gpt2.py compare --node 0
    sky run examples/gpt2.py train --solo --prefix gpt2-solo --node 0

The compute stays up between runs (``delete_on_exit=False``); ``sky compute delete`` ends it.
A prefix holds one run; start another under a new prefix. ``evaluate`` and ``compare`` only
read the bucket and the Hub, so they also run here:

    cd examples && AWS_PROFILE=hf uv run --frozen --with tiktoken --with pyarrow \\
        --with huggingface_hub --with ../../skyward python -c "import gpt2; print(gpt2.compare())"

With ``--sync-every 200`` and ``--fragments 4`` a node uploads a quarter of the model, 124 MB
in fp32, about every two minutes; raise ``--sync-every`` on a slow uplink. ``--rank`` and
``--nodes`` stand in for the node's own position when running off Skyward.
"""

from __future__ import annotations

import configparser
import contextlib
import itertools
import math
import os
import queue
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import skyward as sky

if TYPE_CHECKING:
    import torch
    from torch import Tensor, nn

    from gradientmail.mailbox import Mailbox

NAMESPACE = os.environ.get("GRADIENTMAIL_HF_NAMESPACE", "")
ENDPOINT = f"https://s3.hf.co/{NAMESPACE}"
DATASET = "HuggingFaceFW/fineweb-edu"
TRAIN_FILES = tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in range(13))
VAL_FILE = "sample/10BT/013_00000.parquet"
HELLASWAG = ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet")
OPENAI = "openai-community/gpt2"
BLOCK = 1024
VOCAB = 50304  # GPT-2's 50257, padded to a multiple of 64
GPT2_VOCAB = 50257
EOT = 50256
LR = 6e-4
WARMUP = 300
# This decoder's names for a block's parameters against OpenAI's checkpoint's.
LINEARS = {
    "attn.qkv": "attn.c_attn",
    "attn.proj": "attn.c_proj",
    "mlp.0": "mlp.c_fc",
    "mlp.2": "mlp.c_proj",
}
NORMS = {"ln1": "ln_1", "ln2": "ln_2"}


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
class Config:
    layers: int = 12
    dim: int = 768
    heads: int = 12


@dataclass(frozen=True)
class Choice:
    """One HellaSwag example: the context's tokens, the four endings' tokens, the right one."""

    context: Sequence[int]
    endings: Sequence[Sequence[int]]
    label: int


@dataclass(frozen=True)
class Scores:
    val_loss: float
    acc: float
    acc_norm: float


@dataclass(frozen=True)
class NanoGPT:
    """The torch half, as ``nanogpt()`` builds it where torch is installed."""

    model: Callable[[Config], nn.Module]
    batches: Callable[[Sequence[str], int, torch.device], Iterator[tuple[Tensor, Tensor]]]
    pick_device: Callable[[], torch.device]
    autocast: Callable[[torch.device], contextlib.AbstractContextManager[object]]
    adamw: Callable[[nn.Module, torch.device], torch.optim.Optimizer]
    validation: Callable[[nn.Module, int, int, torch.device], float]
    openai_gpt2: Callable[[torch.device], nn.Module]
    hellaswag: Callable[[int], Sequence[Choice]]
    accuracy: Callable[[nn.Module, Sequence[Choice], torch.device], tuple[float, float]]


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

    import gradientmail as gm
    from gradientmail.adapters.torch import TorchBackend

    nano = nanogpt()
    place = where(rank, nodes)
    device = nano.pick_device()
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    model = nano.model(Config(layers, dim, heads)).to(device)
    optimizer = nano.adamw(model, device)
    per_step = micro_batch * BLOCK * accumulation
    share = tokens / (place.nodes * per_step)
    # One node's share, in steps alone or in rounds of sync_every / fragments steps each.
    steps = max(1, round(share))
    rounds = max(1, round(share * fragments / sync_every))
    batches = nano.batches(TRAIN_FILES[place.rank :: place.nodes], micro_batch, device)
    session = (
        None
        if solo
        else gm.Session(
            model,
            algorithm=gm.DiLoCo(
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
                with nano.autocast(device):
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
        "val_loss": nano.validation(model, micro_batch, 32, device),
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
    import gradientmail as gm

    nano = nanogpt()
    device = nano.pick_device()
    model = nano.model(Config(layers, dim, heads)).to(device)
    newest = gm.install(model, mailbox=mailbox(bucket, prefix))
    return {"round": newest, "val_loss": nano.validation(model, micro_batch, batches, device)}


@gpu
def compare(
    bucket: str = "gradientmail",
    prefix: str = "gpt2",
    examples: int = 10042,
    batches: int = 64,
    micro_batch: int = 16,
    only: str = "",
) -> dict[str, dict[str, float]]:
    """Validation loss and HellaSwag of the run and of OpenAI's 124M; --only openai|ours for one."""
    import torch

    import gradientmail as gm

    if only not in ("", "openai", "ours"):
        raise ValueError(f"--only takes openai or ours, not {only!r}")

    def scored(name: str, model: nn.Module) -> dict[str, float]:
        acc, acc_norm = nano.accuracy(model, choices, device)
        scores = Scores(nano.validation(model, micro_batch, batches, device), acc, acc_norm)
        print(
            f"{name}: val_loss {scores.val_loss:.3f}  "
            f"hellaswag acc {scores.acc:.1%}  acc_norm {scores.acc_norm:.1%}",
            flush=True,
        )
        return asdict(scores)

    def run() -> dict[str, float]:
        model = nano.model(Config()).to(device)
        newest = gm.install(model, mailbox=mailbox(bucket, prefix))
        return {"round": newest, **scored(prefix, model)}

    nano = nanogpt()
    device = nano.pick_device()
    torch.set_float32_matmul_precision("high")
    choices = nano.hellaswag(examples)
    print(f"{len(choices)} HellaSwag examples, {batches} validation batches, on {device}")
    return {
        **({} if only == "ours" else {"openai": scored("openai", nano.openai_gpt2(device))}),
        **({} if only == "openai" else {prefix: run()}),
    }


def learning_rate(step: int, progress: float) -> float:
    """Linear warmup over this node's first steps, then cosine on the run's ``progress`` in
    [0, 1] down to a tenth of the peak."""
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR / 10 + (LR - LR / 10) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


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


def nanogpt() -> NanoGPT:
    """The decoder, the data stream and the scorer, built where they run.

    ``sky run`` imports this file on this machine to find the commands, and torch is only
    on the nodes, so everything that needs it is defined here, when a command calls.
    """
    import pyarrow.parquet as pq
    import tiktoken
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from torch import Tensor, nn
    from torch.nn import functional as F

    class GPT(nn.Module):
        def __init__(self, config: Config) -> None:
            super().__init__()
            self.wte = nn.Embedding(VOCAB, config.dim)
            self.wpe = nn.Embedding(BLOCK, config.dim)
            self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
            self.ln_f = nn.LayerNorm(config.dim)
            self.lm_head = nn.Linear(config.dim, VOCAB, bias=False)
            self.lm_head.weight = self.wte.weight
            self.apply(init_weights)
            # Residual projections start smaller, so the residual stream's variance stays put.
            for name, parameter in self.named_parameters():
                if name.endswith(("attn.proj.weight", "mlp.2.weight")):
                    nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.layers))

        def forward(self, inputs: Tensor, targets: Tensor | None = None) -> Tensor:
            """The mean loss against ``targets``, or the logits when there are none."""
            positions = torch.arange(inputs.shape[1], device=inputs.device)
            x = self.wte(inputs) + self.wpe(positions)
            for block in self.blocks:
                x = block(x)
            logits = self.lm_head(self.ln_f(x))
            if targets is None:
                return logits
            return F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1))

    class Block(nn.Module):
        def __init__(self, config: Config) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(config.dim)
            self.attn = Attention(config)
            self.ln2 = nn.LayerNorm(config.dim)
            self.mlp = nn.Sequential(
                nn.Linear(config.dim, 4 * config.dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(4 * config.dim, config.dim),
            )

        def forward(self, x: Tensor) -> Tensor:
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class Attention(nn.Module):
        def __init__(self, config: Config) -> None:
            super().__init__()
            self.heads = config.heads
            self.qkv = nn.Linear(config.dim, 3 * config.dim)
            self.proj = nn.Linear(config.dim, config.dim)

        def forward(self, x: Tensor) -> Tensor:
            def heads(t: Tensor) -> Tensor:
                return t.view(batch, length, self.heads, dim // self.heads).transpose(1, 2)

            batch, length, dim = x.shape
            q, k, v = self.qkv(x).split(dim, dim=2)
            y = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), is_causal=True)
            return self.proj(y.transpose(1, 2).reshape(batch, length, dim))

    def init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    class Batches(Iterator[tuple[Tensor, Tensor]]):
        """``(inputs, targets)`` micro-batches of ``BLOCK`` tokens, tokenized on a background
        thread.

        Documents are concatenated with GPT-2's end-of-text token between them and cut into
        contiguous blocks, so every token is a target exactly once. Files are read in order,
        each downloaded from the Hub when reached.
        """

        def __init__(
            self, files: Sequence[str], micro_batch: int, device: torch.device, depth: int = 8
        ) -> None:
            self._queue: queue.Queue[Tensor | None] = queue.Queue(maxsize=depth)
            self._device = device
            self._size = micro_batch * BLOCK + 1
            threading.Thread(target=self._produce, args=(files,), daemon=True).start()

        def __next__(self) -> tuple[Tensor, Tensor]:
            chunk = self._queue.get()
            if chunk is None:
                raise StopIteration
            chunk = chunk.to(self._device)
            return chunk[:-1].view(-1, BLOCK), chunk[1:].view(-1, BLOCK)

        def _produce(self, files: Sequence[str]) -> None:
            encoding = tiktoken.get_encoding("gpt2")
            buffer: list[int] = []
            for file in files:
                path = hf_hub_download(DATASET, file, repo_type="dataset")
                for rows in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
                    for ids in encoding.encode_ordinary_batch(rows.column("text").to_pylist()):
                        buffer.extend(ids)
                        buffer.append(encoding.eot_token)
                    while len(buffer) >= self._size:
                        self._queue.put(torch.tensor(buffer[: self._size], dtype=torch.long))
                        # The chunk's last token is the next chunk's first input.
                        del buffer[: self._size - 1]
            self._queue.put(None)

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
        """GPT-2's optimizer: weight decay on matrices only, none on biases and norms."""
        matrices = [p for p in model.parameters() if p.dim() >= 2]
        vectors = [p for p in model.parameters() if p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": matrices, "weight_decay": 0.1}, {"params": vectors, "weight_decay": 0.0}],
            lr=LR,
            betas=(0.9, 0.95),
            fused=device.type == "cuda",
        )

    def validation(model: nn.Module, micro_batch: int, batches: int, device: torch.device) -> float:
        """Mean loss over the first ``batches`` micro-batches of the held-out file, over GPT-2's
        50,257 tokens."""

        def loss(inputs: Tensor, targets: Tensor) -> float:
            logits = model(inputs)[..., :GPT2_VOCAB].reshape(-1, GPT2_VOCAB)
            return F.cross_entropy(logits, targets.reshape(-1)).item()

        stream = Batches((VAL_FILE,), micro_batch, device)
        with torch.no_grad(), autocast(device):
            losses = [loss(x, y) for x, y in itertools.islice(stream, batches)]
        return sum(losses) / len(losses)

    def openai_gpt2(device: torch.device) -> nn.Module:
        """OpenAI's 124M checkpoint on this file's decoder."""

        def block(i: int) -> dict[str, Tensor]:
            def at(name: str) -> Tensor:
                return hub[f"h.{i}.{name}"]

            named = LINEARS | NORMS
            # Conv1D keeps (in, out); nn.Linear keeps (out, in).
            weights = {
                f"{ours}.weight": at(f"{theirs}.weight").t() for ours, theirs in LINEARS.items()
            }
            norms = {f"{ours}.weight": at(f"{theirs}.weight") for ours, theirs in NORMS.items()}
            biases = {f"{ours}.bias": at(f"{theirs}.bias") for ours, theirs in named.items()}
            return {f"blocks.{i}.{name}": t for name, t in (weights | norms | biases).items()}

        hub = load_file(hf_hub_download(OPENAI, "model.safetensors"))
        config = Config()
        wte = torch.cat([hub["wte.weight"], torch.zeros(VOCAB - GPT2_VOCAB, config.dim)])
        state = {
            "wte.weight": wte,
            "lm_head.weight": wte,
            "wpe.weight": hub["wpe.weight"],
            "ln_f.weight": hub["ln_f.weight"],
            "ln_f.bias": hub["ln_f.bias"],
            **{name: t for i in range(config.layers) for name, t in block(i).items()},
        }
        model = GPT(config)
        model.load_state_dict(state)
        return model.to(device)

    def hellaswag(examples: int) -> Sequence[Choice]:
        """The first ``examples`` of HellaSwag's validation split, tokenized."""
        encoding = tiktoken.get_encoding("gpt2")
        rows = pq.read_table(hf_hub_download(*HELLASWAG, repo_type="dataset")).to_pylist()
        return tuple(
            Choice(
                encoding.encode_ordinary(row["ctx"]),
                tuple(encoding.encode_ordinary(" " + ending) for ending in row["endings"]),
                int(row["label"]),
            )
            for row in itertools.islice(rows, examples)
        )

    def accuracy(
        model: nn.Module, choices: Sequence[Choice], device: torch.device, group: int = 8
    ) -> tuple[float, float]:
        """Zero-shot HellaSwag: the share of ``choices`` whose right ending has the highest
        log-likelihood (``acc``) and the highest log-likelihood per token (``acc_norm``)."""

        def likelihoods(batch: Sequence[Choice]) -> tuple[Tensor, Tensor]:
            """Per choice, the total and the per-token log-likelihood of its four endings."""
            rows = [(*choice.context, *ending) for choice in batch for ending in choice.endings]
            starts = [len(choice.context) for choice in batch for _ in choice.endings]
            length = max(len(row) for row in rows)
            tokens = torch.full((len(rows), length), EOT, dtype=torch.long)
            scored = torch.zeros(len(rows), length - 1)
            for r, (row, start) in enumerate(zip(rows, starts, strict=True)):
                tokens[r, : len(row)] = torch.tensor(row)
                # Position t scores token t + 1; the ending is row[start:].
                scored[r, start - 1 : len(row) - 1] = 1
            tokens, scored = tokens.to(device), scored.to(device)
            with torch.no_grad(), autocast(device):
                logits = model(tokens)[:, :-1, :GPT2_VOCAB].float()
            logprobs = logits.gather(2, tokens[:, 1:, None]).squeeze(2) - logits.logsumexp(2)
            total = (logprobs * scored).sum(1).view(len(batch), 4)
            return total, total / scored.sum(1).view(len(batch), 4)

        def batches() -> Iterator[tuple[Tensor, Tensor]]:
            for n, start in enumerate(range(0, len(choices), group), start=1):
                yield likelihoods(choices[start : start + group])
                if n % 125 == 0:
                    print(
                        f"  hellaswag {min(start + group, len(choices))}/{len(choices)}", flush=True
                    )

        totals, means = (torch.cat(parts).cpu() for parts in zip(*batches(), strict=True))
        labels = torch.tensor([choice.label for choice in choices])
        acc = (totals.argmax(1) == labels).float().mean().item()
        acc_norm = (means.argmax(1) == labels).float().mean().item()
        return acc, acc_norm

    return NanoGPT(
        model=GPT,
        batches=Batches,
        pick_device=pick_device,
        autocast=autocast,
        adamw=adamw,
        validation=validation,
        openai_gpt2=openai_gpt2,
        hellaswag=hellaswag,
        accuracy=accuracy,
    )
