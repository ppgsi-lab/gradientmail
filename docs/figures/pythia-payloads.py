"""Re-encode the retained output-embedding commit of the eight-replica H=25 run.

Run with a local sample directory containing round.json (456), previous.json (452), and
anchor, theta, outer and update payloads as <name>.safetensors.zst. Their references and
SHA-256 hashes are in the round records under itsmechico/gradientmail,
pythia-14m-8n-carry-f8-h25. No training is repeated: the master, anchor and momentum are fixed.

    python docs/figures/pythia-payloads.py /path/to/sample
    python docs/figures/build.py pythia-payloads
"""

import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
import zstandard

from gradientmail.adapters.torch import TorchBackend
from gradientmail.algorithms import Precision, decoded, encoded
from gradientmail.compression import decode, encode
from gradientmail.protocol import decode_round_record


@dataclass(frozen=True)
class Measurement:
    precision: str
    update_raw: int
    update_zstd: int
    commit_raw: int
    commit_zstd: int


def main(sample: Path) -> None:
    def chart(rows: Sequence[Measurement]) -> str:
        def text(x: float, y: float, cls: str, content: str) -> str:
            return f'<text x="{x}" y="{y}" class="{cls}">{content}</text>'

        out = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="648" '
            'viewBox="0 0 960 648" role="img" aria-label="Measured payload sizes for the same '
            "Pythia-14M output-embedding update in FP32, BF16, FP8 and FP4, with and without "
            "zstd. Global commit update and complete-commit sizes share one decimal megabyte axis. "
            "The default FP32 commit stores the fragment and momentum; quantized commits "
            "also store an fp32 master and a separate update. Re-encoding round 456 does "
            'not measure the training quality of alternative formats.">',
            '<rect width="960" height="648" rx="6" class="ground"/>',
            text(24, 36, "title", "Payload size by precision and compression"),
            text(
                24,
                58,
                "small muted",
                "The same output-embedding global commit update, re-encoded from round 456.",
            ),
            '<rect x="24" y="82" width="24" height="12" rx="2" class="param"/>',
            text(56, 92, "small", "Without zstd"),
            '<rect x="208" y="82" width="24" height="12" rx="2" class="commit"/>',
            text(240, 92, "small", "With zstd"),
            text(936, 118, "small end muted", "Raw / zstd, MB"),
        ]
        left, right = 152, 744
        for tick in range(0, 101, 20):
            x = left + (right - left) * tick / 100
            out.append(f'<path d="M {x} 144 V 520" class="guide"/>')
            out.append(text(x, 562, "small mid muted", str(tick)))
        for group, first_y in (
            ("Global commit update (learner download)", 160),
            ("Complete commit (synchronizer upload)", 380),
        ):
            out.append(text(24, first_y - 28, "b", group))
            for i, row in enumerate(rows):
                raw, compressed = (
                    (row.update_raw, row.update_zstd)
                    if first_y == 160
                    else (row.commit_raw, row.commit_zstd)
                )
                y = first_y + i * 42
                label = (
                    "FP32 (default)"
                    if row.precision == "FP32" and first_y == 380
                    else row.precision
                )
                out.append(text(24, y + 8, "small", label))
                for size, offset, cls in ((raw, -10, "param"), (compressed, 5, "commit")):
                    width = (right - left) * size / 100_000_000
                    out.append(
                        f'<rect x="{left}" y="{y + offset}" width="{width:.2f}" '
                        f'height="10" rx="2" class="{cls}"/>'
                    )
                out.append(
                    text(936, y + 8, "small end", f"{raw / 1e6:.2f} / {compressed / 1e6:.2f}")
                )
        out.extend(
            [
                '<path d="M 24 326 H 936" class="rule"/>',
                f'<path d="M {left} 540 H {right}" class="axis"/>',
                text((left + right) / 2, 588, "small mid muted", "Payload size (MB, decimal)"),
                text(
                    24,
                    618,
                    "small muted",
                    "Quantized commits retain fp32 fragment, momentum and master, plus the update.",
                ),
                "</svg>",
            ]
        )
        return "\n".join(out) + "\n"

    current = decode_round_record((sample / "round.json").read_bytes())
    previous = decode_round_record((sample / "previous.json").read_bytes())
    assert current.update is not None
    assert current.fragment == previous.fragment and current.version == previous.version + 1
    references = {
        "anchor": previous.theta,
        "theta": current.theta,
        "outer": current.outer,
        "update": current.update.payload,
    }
    payloads = {}
    for name, ref in references.items():
        blob = (sample / f"{name}.safetensors.zst").read_bytes()
        assert len(blob) == ref.size and sha256(blob).hexdigest() == ref.sha256
        payloads[name] = decode(blob, "zstd")

    backend = TorchBackend()
    name = "embed_out.weight"
    anchor = backend.load(payloads["anchor"][name])
    master = backend.load(payloads["outer"][f"master.{name}"])
    momentum = payloads["outer"][f"momentum.{name}"]
    shape = payloads["theta"][name].shape
    displacement = master - anchor
    formats: tuple[tuple[str, Precision], ...] = (
        ("FP32", None),
        ("BF16", "BF16"),
        ("FP8", "F8_E4M3"),
        ("FP4", "F4"),
    )
    rows: list[Measurement] = []
    observed_size = current.theta.size + current.outer.size + current.update.payload.size
    for label, precision in formats:
        update = encoded(name, displacement, "F32", precision)
        theta = {name: master.to_bytes("F32")}
        outer = {f"momentum.{name}": momentum}
        if precision is not None:
            moved = decoded(backend, update, name, shape, precision)
            theta = {name: (anchor + moved).to_bytes("F32")}
            outer = outer | {f"master.{name}": master.to_bytes("F32")}
        commit = (theta, outer) if precision is None else (theta, outer, update)
        row = Measurement(
            label,
            len(encode(update, "off")),
            len(encode(update, "zstd")),
            sum(len(encode(part, "off")) for part in commit),
            sum(len(encode(part, "zstd")) for part in commit),
        )
        if label == "FP8":
            assert update == payloads["update"] and theta == payloads["theta"]
            assert outer == payloads["outer"] and row.commit_zstd == observed_size
            for part, ref in (
                (theta, current.theta),
                (outer, current.outer),
                (update, current.update.payload),
            ):
                assert sha256(encode(part, "zstd")).hexdigest() == ref.sha256
        rows.append(row)
        print(
            f"{label}: update {row.update_zstd / 1e6:.2f} MB, "
            f"commit {row.commit_zstd / 1e6:.2f} MB",
            flush=True,
        )

    result = {
        "source": {
            "bucket": "https://huggingface.co/buckets/itsmechico/gradientmail",
            "prefix": "pythia-14m-8n-carry-f8-h25",
            "round": current.round,
            "previous_round": previous.round,
            "fragment": current.fragment,
            "tensor": name,
            "shape": shape,
            "payloads": {key: asdict(ref) for key, ref in references.items()},
            "fp8_reproduces_stored_payloads": True,
            "torch": str(torch.__version__),
            "zstandard": zstandard.__version__,
        },
        "observed": {"commit_zstd": observed_size},
        "measurements": [asdict(row) for row in rows],
    }
    destination = Path(__file__).parent
    (destination / "pythia-payloads.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "pythia-payloads.svg").write_text(chart(rows))
    print("Wrote payload measurements and figure source", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
