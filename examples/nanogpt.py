"""The plain-PyTorch half of examples/gpt2.py: a GPT-2-shaped decoder and a FineWeb-Edu stream.

Nothing here knows about GradientMail or Skyward. The model follows nanoGPT (pre-norm blocks,
GELU MLPs, tied input and output embeddings, GPT-2's initialisation); the data is the
``sample/10BT`` parquet files of HuggingFaceFW/fineweb-edu, downloaded from the Hub when
reached and tokenized with tiktoken's GPT-2 encoding on a background thread.
"""

import itertools
import math
import queue
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pyarrow.parquet as pq
import tiktoken
import torch
from huggingface_hub import hf_hub_download
from torch import Tensor, nn
from torch.nn import functional as F

DATASET = "HuggingFaceFW/fineweb-edu"
TRAIN_FILES = tuple(f"sample/10BT/{i:03d}_00000.parquet" for i in range(13))
VAL_FILE = "sample/10BT/013_00000.parquet"
BLOCK = 1024
VOCAB = 50304  # GPT-2's 50257, padded to a multiple of 64
LR = 6e-4
WARMUP = 300


@dataclass(frozen=True)
class Config:
    layers: int = 12
    dim: int = 768
    heads: int = 12


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

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        positions = torch.arange(inputs.shape[1], device=inputs.device)
        x = self.wte(inputs) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
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
    """``(inputs, targets)`` micro-batches of ``BLOCK`` tokens, tokenized on a background thread.

    Documents are concatenated with GPT-2's end-of-text token between them and cut into
    contiguous blocks, so every token is a target exactly once. Files are read in order, each
    downloaded from the Hub when reached.
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


def learning_rate(step: int, progress: float) -> float:
    """Linear warmup over this node's first steps, then cosine on the run's ``progress`` in
    [0, 1] down to a tenth of the peak."""
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    return LR / 10 + (LR - LR / 10) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def validation(model: nn.Module, micro_batch: int, batches: int, device: torch.device) -> float:
    """Mean loss over the first ``batches`` micro-batches of the held-out file."""
    stream = Batches((VAL_FILE,), micro_batch, device)
    model.eval()
    with torch.no_grad(), autocast(device):
        losses = [model(x, y).item() for x, y in itertools.islice(stream, batches)]
    model.train()
    return sum(losses) / len(losses)
