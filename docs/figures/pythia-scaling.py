"""Write ``docs/figures/pythia-loss.svg``, ``docs/figures/pythia-scaling.svg``,
``docs/figures/pythia-ablation.svg``, ``docs/figures/pythia-speedup.svg`` and
``docs/figures/pythia-cosine.svg``, the five charts of the Pythia-14M post.

The first plots the nodes' mean training loss of the recipe's runs and of the single replica
against the dataset step, with a legend and final-bin labels. The second plots the
held-out loss at dataset step 4,000 against the number of replicas for the recipe, Decoupled
DiLoCo with ``Carry`` at H = 25 and F8 deltas, and for the same rule at H = 50 in fp32, over the
official checkpoint and the nine PolyPythia seeds. The third places every eight-replica run on one
loss axis. The fourth plots the training time against the replica count on logarithmic axes. The
fifth plots the mean cosine between the deltas of a round, averaged over each tenth of a run, for
the H = 50 runs. The numbers are the runs' own, from ``compare``, the round records and the nodes'
consoles; they are typed here rather than read, so the figures build without them. Losses are at
each run's last round.

The sources draw with the classes of ``figure.css`` only; ``build.py`` turns them into assets::

    uv run --frozen python docs/figures/pythia-scaling.py
    uv run --frozen --with fonttools --with brotli python docs/figures/build.py \\
        pythia-loss pythia-scaling pythia-ablation pythia-speedup pythia-cosine
"""

import math
from collections.abc import Iterable
from pathlib import Path

FIGURES = Path(__file__).parent
WIDTH = 960
HEIGHT = 440

OFFICIAL = 3.781
SEEDS = (3.732, 3.762)

# Held-out loss at the last round and training time in minutes, first unit to last commit, by
# replica count: the recipe, and Carry at H = 50 in fp32, whose single replica is the baseline.
RECIPE = {2: 3.5592, 4: 3.6052, 8: 3.6009, 16: 3.7494}
RECIPE_MINUTES = {2: 248, 4: 129, 8: 72, 16: 37}
H50 = {1: 3.585, 2: 3.584, 4: 3.689, 8: 3.678, 16: 3.847}
H50_MINUTES = {1: 472, 2: 275, 4: 135, 8: 69, 16: 36}

# The nodes' mean training loss and the minutes since the run's first step, per 100-step bin, from
# the step lines of each node's console: the recipe at 2 to 16 replicas and the single replica.
CURVES: dict[int, list[tuple[int, float, float]]] = {
    1: [
        (50, 6.1, 10.639),
        (150, 17.8, 9.239),
        (250, 29.5, 7.592),
        (350, 41.4, 6.696),
        (450, 53.1, 6.140),
        (550, 65.0, 5.731),
        (650, 76.8, 5.413),
        (750, 88.6, 5.176),
        (850, 100.4, 4.989),
        (950, 112.2, 4.809),
        (1050, 123.9, 4.618),
        (1150, 135.8, 4.400),
        (1250, 147.6, 4.240),
        (1350, 159.2, 4.136),
        (1450, 171.2, 4.053),
        (1550, 182.8, 3.997),
        (1650, 194.7, 3.945),
        (1750, 206.5, 3.907),
        (1850, 218.3, 3.870),
        (1950, 230.1, 3.845),
        (2050, 241.8, 3.816),
        (2150, 253.6, 3.792),
        (2250, 265.4, 3.773),
        (2350, 277.3, 3.755),
        (2450, 289.1, 3.738),
        (2550, 300.8, 3.725),
        (2650, 312.7, 3.708),
        (2750, 324.4, 3.689),
        (2850, 336.2, 3.685),
        (2950, 348.0, 3.670),
        (3050, 359.9, 3.663),
        (3150, 371.6, 3.653),
        (3250, 383.5, 3.642),
        (3350, 395.3, 3.639),
        (3450, 407.0, 3.628),
        (3550, 418.8, 3.625),
        (3650, 430.7, 3.618),
        (3750, 442.4, 3.610),
        (3850, 454.3, 3.605),
        (3950, 466.0, 3.602),
    ],
    2: [
        (50, 3.5, 10.593),
        (150, 9.5, 9.028),
        (250, 15.5, 7.416),
        (350, 21.8, 6.606),
        (450, 28.0, 6.070),
        (550, 34.4, 5.670),
        (650, 40.5, 5.368),
        (750, 46.7, 5.135),
        (850, 52.8, 4.950),
        (950, 59.2, 4.767),
        (1050, 65.4, 4.583),
        (1150, 71.7, 4.358),
        (1250, 78.0, 4.189),
        (1350, 84.0, 4.081),
        (1450, 90.1, 3.996),
        (1550, 96.3, 3.939),
        (1650, 102.4, 3.888),
        (1750, 108.7, 3.849),
        (1850, 114.8, 3.813),
        (1950, 121.1, 3.790),
        (2050, 127.3, 3.762),
        (2150, 133.7, 3.739),
        (2250, 139.6, 3.722),
        (2350, 145.9, 3.705),
        (2450, 152.3, 3.690),
        (2550, 158.5, 3.679),
        (2650, 164.6, 3.664),
        (2750, 171.0, 3.648),
        (2850, 177.2, 3.645),
        (2950, 183.3, 3.633),
        (3050, 189.4, 3.629),
        (3150, 195.6, 3.620),
        (3250, 201.9, 3.610),
        (3350, 208.2, 3.610),
        (3450, 214.5, 3.601),
        (3550, 220.8, 3.601),
        (3650, 226.9, 3.596),
        (3750, 232.9, 3.588),
        (3850, 239.6, 3.591),
        (3950, 245.6, 3.590),
    ],
    4: [
        (50, 1.8, 10.596),
        (150, 4.9, 9.297),
        (250, 8.1, 7.721),
        (350, 11.4, 6.874),
        (450, 14.5, 6.361),
        (550, 17.7, 5.991),
        (650, 20.9, 5.696),
        (750, 24.4, 5.436),
        (850, 27.3, 5.236),
        (950, 30.6, 5.042),
        (1050, 33.7, 4.876),
        (1150, 36.8, 4.678),
        (1250, 40.0, 4.470),
        (1350, 43.4, 4.300),
        (1450, 46.7, 4.174),
        (1550, 49.7, 4.093),
        (1650, 53.0, 4.020),
        (1750, 56.2, 3.967),
        (1850, 59.4, 3.923),
        (1950, 62.7, 3.888),
        (2050, 65.9, 3.852),
        (2150, 69.2, 3.825),
        (2250, 72.3, 3.805),
        (2350, 75.4, 3.784),
        (2450, 78.7, 3.761),
        (2550, 81.9, 3.745),
        (2650, 85.2, 3.728),
        (2750, 88.4, 3.712),
        (2850, 91.6, 3.705),
        (2950, 94.8, 3.689),
        (3050, 98.1, 3.683),
        (3150, 101.3, 3.672),
        (3250, 104.5, 3.662),
        (3350, 107.7, 3.660),
        (3450, 110.8, 3.653),
        (3550, 114.2, 3.649),
        (3650, 117.5, 3.639),
        (3750, 120.5, 3.630),
        (3850, 123.7, 3.628),
        (3950, 127.1, 3.625),
    ],
    8: [
        (50, 1.8, 10.603),
        (150, 3.4, 9.214),
        (250, 5.2, 7.639),
        (350, 6.8, 6.869),
        (450, 8.6, 6.371),
        (550, 10.3, 6.010),
        (650, 12.0, 5.716),
        (750, 13.8, 5.470),
        (850, 15.4, 5.269),
        (950, 17.2, 5.094),
        (1050, 18.9, 4.949),
        (1150, 20.7, 4.787),
        (1250, 22.4, 4.597),
        (1350, 24.1, 4.418),
        (1450, 25.9, 4.274),
        (1550, 27.6, 4.178),
        (1650, 29.3, 4.095),
        (1750, 31.0, 4.037),
        (1850, 32.7, 3.988),
        (1950, 34.5, 3.950),
        (2050, 36.2, 3.911),
        (2150, 37.9, 3.879),
        (2250, 39.6, 3.853),
        (2350, 41.4, 3.831),
        (2450, 43.1, 3.808),
        (2550, 44.9, 3.793),
        (2650, 46.7, 3.773),
        (2750, 48.4, 3.753),
        (2850, 50.1, 3.747),
        (2950, 51.9, 3.730),
        (3050, 53.6, 3.724),
        (3150, 55.2, 3.715),
        (3250, 57.0, 3.707),
        (3350, 58.8, 3.710),
        (3450, 60.6, 3.701),
        (3550, 62.3, 3.697),
        (3650, 64.1, 3.689),
        (3750, 65.8, 3.679),
        (3850, 67.5, 3.675),
        (3950, 69.3, 3.673),
    ],
    16: [
        (50, 0.5, 10.625),
        (150, 1.4, 9.422),
        (250, 2.2, 8.002),
        (350, 3.1, 7.160),
        (450, 4.0, 6.674),
        (550, 4.8, 6.333),
        (650, 5.7, 6.053),
        (750, 6.6, 5.816),
        (850, 7.4, 5.618),
        (950, 8.3, 5.452),
        (1050, 9.1, 5.320),
        (1150, 10.1, 5.183),
        (1250, 10.9, 5.054),
        (1350, 11.8, 4.929),
        (1450, 12.7, 4.784),
        (1550, 13.6, 4.631),
        (1650, 14.5, 4.494),
        (1750, 15.5, 4.392),
        (1850, 16.3, 4.307),
        (1950, 17.1, 4.247),
        (2050, 18.1, 4.194),
        (2150, 19.1, 4.146),
        (2250, 20.0, 4.109),
        (2350, 20.9, 4.078),
        (2450, 21.8, 4.045),
        (2550, 22.8, 4.023),
        (2650, 23.6, 3.997),
        (2750, 24.6, 3.968),
        (2850, 25.5, 3.957),
        (2950, 26.4, 3.933),
        (3050, 27.4, 3.923),
        (3150, 28.4, 3.908),
        (3250, 29.3, 3.894),
        (3350, 30.2, 3.887),
        (3450, 31.3, 3.870),
        (3550, 32.2, 3.863),
        (3650, 33.2, 3.851),
        (3750, 34.2, 3.839),
        (3850, 35.2, 3.832),
        (3950, 36.2, 3.823),
    ],
}
CURVE_CLASS = {1: "dotted", 2: "line-a", 4: "line-b", 8: "line-c", 16: "line-d"}

# Every eight-replica run, the recipe first and the rest from best to worst.
ABLATION = [
    ("Carry, H = 25, F8 + zstd: the recipe", 3.6009),
    ("Carry, H = 50, fp32", 3.678),
    ("Adaptive, H = 50, fp32", 3.705),
    ("Overwrite, H = 50, fp32", 3.723),
    ("Carry, H = 50, F8 + zstd", 3.733),
    ("Blend(0.5), H = 50, fp32", 3.770),
    ("Carry, H = 250, bf16", 3.836),
]

# Mean cosine between a round's deltas, by tenth of the run, the class its line draws with and
# how far its label sits from the line's end.
COSINES = {
    "2 nodes, Carry": (
        [0.96, 0.86, 0.72, 0.65, 0.51, 0.42, 0.37, 0.35, 0.33, 0.31],
        "line-ink thin",
        0,
    ),
    "4 nodes, Carry": (
        [0.91, 0.80, 0.66, 0.62, 0.51, 0.41, 0.50, 0.47, 0.42, 0.38],
        "line-ink thin",
        0,
    ),
    "16 nodes, Carry": (
        [0.88, 0.80, 0.66, 0.54, 0.53, 0.44, 0.36, 0.32, 0.29, 0.25],
        "line-ink thin",
        0,
    ),
    "8 nodes, Carry": (
        [0.84, 0.71, 0.57, 0.51, 0.40, 0.30, 0.24, 0.20, 0.17, 0.17],
        "line-ink",
        -3,
    ),
    "8 nodes, Overwrite": (
        [0.88, 0.77, 0.53, 0.41, 0.37, 0.27, 0.19, 0.16, 0.12, 0.11],
        "line-sig",
        -6,
    ),
    "8 nodes, Adaptive": (
        [0.90, 0.75, 0.53, 0.46, 0.36, 0.29, 0.19, 0.14, 0.12, 0.11],
        "line-acc",
        8,
    ),
    "8 nodes, Blend(0.5)": (
        [0.82, 0.63, 0.35, 0.28, 0.29, 0.13, 0.07, 0.06, 0.07, 0.06],
        "line-muted",
        5,
    ),
}


def fmt(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def text(x: float, y: float, cls: str, content: str) -> str:
    return f'<text x="{fmt(x)}" y="{fmt(y)}" class="{cls}">{content}</text>'


def path(cls: str, parts: Iterable[str]) -> str:
    return f'<path d="{" ".join(parts)}" class="{cls}"/>'


def line(cls: str, points: Iterable[tuple[float, float]]) -> str:
    return path(
        cls, [f"{'M' if i == 0 else 'L'} {fmt(x)} {fmt(y)}" for i, (x, y) in enumerate(points)]
    )


def dot(x: float, y: float, cls: str) -> str:
    return f'<circle cx="{fmt(x)}" cy="{fmt(y)}" r="5" class="{cls}"/>'


def duration(minutes: int) -> str:
    hours, rest = divmod(minutes, 60)
    return f"{hours} h {rest:02d} min" if hours else f"{rest} min"


def open_svg(aria: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{aria}">',
        f'<rect width="{WIDTH}" height="{HEIGHT}" rx="6" class="ground"/>',
    ]


def training_loss() -> str:
    left, right, top, bottom = 88, 768, 154, 414
    height = 520
    lo, hi = 3.55, 5.0

    def py(loss: float) -> float:
        return bottom - (bottom - top) * (loss - lo) / (hi - lo)

    def px(step: float) -> float:
        return left + (right - left) * (step - 750) / (4000 - 750)

    ends = "; ".join(
        f"{n} {'replicas' if n > 1 else 'replica'} {curve[-1][2]:.3f}"
        for n, curve in CURVES.items()
    )
    aria = (
        "Mean local training cross-entropy in nats per token against dataset step for 1, 2, 4, 8 "
        "and 16 replicas. A legend identifies every curve. One replica uses H = 50 and fp32; "
        "the distributed runs use H = 25 and F8 with zstd. The view shows loss below 5 after "
        "the initial drop. Dots mark the last logged 100-step bin, centered at step 3,950. "
        f"At that bin: {ends}. These are training losses, not held-out evaluation losses."
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" aria-label="{aria}">',
        f'<rect width="{WIDTH}" height="{height}" rx="6" class="ground"/>',
        text(24, 36, "title", "Training loss over 4,000 dataset steps"),
        text(24, 58, "small muted", "Mean local cross-entropy in nats per token; lower is better."),
        '<g aria-label="Legend">',
    ]
    for i, n in enumerate(CURVES):
        x = left + i * 160
        out.append(line(f"{CURVE_CLASS[n]} strong", [(x, 86), (x + 30, 86)]))
        out.append(text(x + 40, 90, "small", f"{n} {'replicas' if n > 1 else 'replica'}"))
    out.extend(
        [
            "</g>",
            text(24, 118, "small muted", "1 replica: H = 50, fp32. Other runs: H = 25, F8 + zstd."),
        ]
    )
    ticks = [3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0]
    out.append(path("rule", [f"M {left} {fmt(py(t))} H {right}" for t in ticks]))
    out.append(path("axis", [f"M {left} {top} V {bottom} H {right}"]))
    out.extend(text(left - 12, py(t) + 4, "small end muted", f"{t:.1f}") for t in ticks)
    out.append(
        f'<text x="0" y="0" class="small mid muted" transform="translate(28 {(top + bottom) // 2}) '
        'rotate(-90)">Training loss</text>'
    )
    step_ticks = range(1000, 4001, 500)
    out.append(path("tick", [f"M {fmt(px(s))} {bottom} V {bottom + 4}" for s in step_ticks]))
    out.extend(text(px(s), bottom + 20, "small mid muted", f"{s:,}") for s in step_ticks)
    out.append(text((left + right) // 2, 464, "small mid muted", "Dataset step"))
    out.append(text(808, 328, "small muted", "Last logged bin"))
    label_y = {16: 358, 8: 385, 4: 412, 1: 439, 2: 466}
    dot_class = {1: "dot-muted", 2: "sA", 4: "sB", 8: "sC", 16: "sD"}
    out.append('<g role="list" aria-label="Training loss curves">')
    for n, curve in CURVES.items():
        shown = [(s, loss) for s, _, loss in curve if loss <= hi]
        cls = CURVE_CLASS[n]
        step, loss = shown[-1]
        out.append(f'<g role="listitem" aria-label="{n} {"replicas" if n > 1 else "replica"}">')
        out.append(line(f"{cls} strong", [(px(s), py(value)) for s, value in shown]))
        out.append(
            line(f"{cls} thin", [(px(step), py(loss)), (788, py(loss)), (800, label_y[n] - 4)])
        )
        out.append(dot(px(step), py(loss), dot_class[n]))
        label = f"{n} {'replicas' if n > 1 else 'replica'} · {loss:.3f}"
        out.append(text(808, label_y[n], "small", label))
        out.append("</g>")
    out.extend(
        [
            "</g>",
            text(
                24,
                496,
                "small muted",
                "100-step bins. Dots mark the last logged bin at step 3,950. "
                "Initial loss above 5 is outside the view.",
            ),
            "</svg>",
        ]
    )
    return "\n".join(out) + "\n"


def scaling() -> str:
    left, right, top, bottom = 96, 900, 40, 372
    nodes = [1, 2, 4, 8, 16]
    lo, hi = 3.50, 3.90

    def px(n: int) -> float:
        return left + (right - left) * nodes.index(n) / (len(nodes) - 1)

    def py(loss: float) -> float:
        return bottom - (bottom - top) * (loss - lo) / (hi - lo)

    out = open_svg(
        "Held-out loss at dataset step 4,000 against the number of replicas, at a global batch of "
        f"1,024 sequences. The recipe, Carry at H = 25 with F8 deltas, scores {RECIPE[2]:.3f} with "
        f"2 replicas, {RECIPE[4]:.3f} with 4, {RECIPE[8]:.3f} with 8 and {RECIPE[16]:.3f} with 16. "
        f"Carry at H = 50 in fp32 scores {H50[1]:.3f} with one replica, {H50[2]:.3f} with 2, "
        f"{H50[4]:.3f} with 4, {H50[8]:.3f} with 8 and {H50[16]:.3f} with 16. The official "
        "checkpoint scores 3.781 and the nine PolyPythia seeds 3.732 to 3.762."
    )
    band = py(SEEDS[0]) - py(SEEDS[1])
    out.append(
        f'<rect x="{left}" y="{fmt(py(SEEDS[1]))}" width="{right - left}" height="{fmt(band)}" '
        'class="store"/>'
    )
    out.append(path("guide", [f"M {left} {fmt(py(OFFICIAL))} H {right}"]))
    out.append(text(left + 10, py(OFFICIAL) - 6, "small muted", "official pythia-14m, 3.781"))
    out.append(
        text(left + 10, py(SEEDS[0]) + 14, "small muted", "nine PolyPythia seeds, 3.732 to 3.762")
    )
    ticks = [3.50, 3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90]
    out.append(path("rule", [f"M {left} {fmt(py(t))} H {right}" for t in ticks[1:]]))
    out.extend(text(left - 10, py(t) + 4, "small end muted", f"{t:.2f}") for t in ticks)
    out.append(
        f'<text x="0" y="0" class="small mid muted" transform="translate(36 {(top + bottom) // 2}) '
        'rotate(-90)">held-out loss at step 4,000 (nats per token)</text>'
    )
    out.append(path("axis", [f"M {left} {top - 8} V {bottom} H {right}"]))
    out.append(path("tick", [f"M {fmt(px(n))} {bottom} V {bottom + 4}" for n in nodes]))
    out.extend(text(px(n), bottom + 20, "small mid muted", str(n)) for n in nodes)
    out.append(
        text(
            (left + right) // 2,
            bottom + 44,
            "small mid muted",
            "replicas, each training 1,024 / n sequences per step",
        )
    )
    out.append(line("dotted", [(px(n), py(loss)) for n, loss in H50.items()]))
    out.append(line("line-ink", [(px(n), py(loss)) for n, loss in RECIPE.items()]))
    for n, loss in H50.items():
        out.append(dot(px(n), py(loss), "dot-muted"))
        label = f"H = 50, fp32, {loss:.3f}" if n in (1, 16) else f"{loss:.3f}"
        if n == 16:
            out.append(text(px(n) - 12, py(loss) + 4, "small end muted", label))
        else:
            out.append(text(px(n) + 12, py(loss) - 8, "small muted", label))
    for n, loss in RECIPE.items():
        out.append(dot(px(n), py(loss), "dot-ink"))
        label = f"H = 25, F8, {loss:.3f}" if n == 16 else f"{loss:.3f}"
        if n == 16:
            out.append(text(px(n) - 12, py(loss) + 4, "small end", label))
        else:
            out.append(text(px(n) + 12, py(loss) + 16, "small", label))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def ablation() -> str:
    left, right, top, bottom = 360, 900, 40, 372
    lo, hi = 3.55, 3.90

    def px(loss: float) -> float:
        return left + (right - left) * (loss - lo) / (hi - lo)

    def py(row: int) -> float:
        return top + 16 + (bottom - top - 32) * row / (len(ABLATION) - 1)

    out = open_svg(
        "Held-out loss at dataset step 4,000 of every eight-replica run. "
        + "; ".join(f"{label}, {loss:.3f}" for label, loss in ABLATION)
        + ". One replica scores 3.585, the official checkpoint 3.781 and the nine PolyPythia seeds "
        "3.732 to 3.762."
    )
    out.append(
        f'<rect x="{fmt(px(SEEDS[0]))}" y="{top - 8}" width="{fmt(px(SEEDS[1]) - px(SEEDS[0]))}" '
        f'height="{bottom - top + 8}" class="store"/>'
    )
    out.append(path("guide", [f"M {fmt(px(OFFICIAL))} {top - 8} V {bottom}"]))
    out.append(path("guide", [f"M {fmt(px(H50[1]))} {top - 8} V {bottom}"]))
    out.append(text(px(OFFICIAL) + 6, top - 14, "small muted", "official, 3.781"))
    out.append(text(px(SEEDS[0]) - 6, top - 14, "small end muted", "seeds"))
    out.append(text(px(H50[1]) + 6, top - 14, "small muted", "one replica, 3.585"))
    ticks = [3.55, 3.60, 3.65, 3.70, 3.75, 3.80, 3.85, 3.90]
    out.append(path("axis", [f"M {left} {bottom} H {right}"]))
    out.append(path("tick", [f"M {fmt(px(t))} {bottom} V {bottom + 4}" for t in ticks]))
    out.extend(text(px(t), bottom + 20, "small mid muted", f"{t:.2f}") for t in ticks)
    out.append(
        text(
            (left + right) // 2,
            bottom + 44,
            "small mid muted",
            "held-out loss at step 4,000 (nats per token), eight replicas",
        )
    )
    for row, (label, loss) in enumerate(ABLATION):
        y = py(row)
        recipe = row == 0
        out.append(path("rule", [f"M {left} {fmt(y)} H {fmt(px(loss))}"]))
        out.append(dot(px(loss), y, "dot-ink" if recipe else "dot-muted"))
        out.append(text(left - 12, y + 4, "small end" if recipe else "small end muted", label))
        out.append(text(px(loss) + 12, y + 4, "small" if recipe else "small muted", f"{loss:.3f}"))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def cosine() -> str:
    left, right, top, bottom = 96, 700, 40, 372

    def px(tenth: float) -> float:
        return left + (right - left) * tenth / 10

    def py(c: float) -> float:
        return bottom - (bottom - top) * c

    out = open_svg(
        "Mean cosine between the deltas of a round, by tenth of the run, at H = 50. Every run "
        "starts near 0.9 and falls as the replicas diverge. With Carry it ends at 0.31 for 2 "
        "replicas, 0.38 for 4, 0.17 for 8 and 0.25 for 16. At 8 replicas Overwrite and Adaptive "
        "follow one curve to 0.11 and Blend(0.5) falls to 0.13 in the sixth tenth and to about "
        "0.06 after."
    )
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    out.append(path("rule", [f"M {left} {fmt(py(t))} H {right}" for t in ticks[1:]]))
    out.extend(text(left - 10, py(t) + 4, "small end muted", f"{t:.1f}") for t in ticks)
    out.append(
        f'<text x="0" y="0" class="small mid muted" transform="translate(36 {(top + bottom) // 2}) '
        'rotate(-90)">mean cosine between the deltas of a round</text>'
    )
    out.append(path("axis", [f"M {left} {top - 8} V {bottom} H {right}"]))
    out.append(path("tick", [f"M {fmt(px(i))} {bottom} V {bottom + 4}" for i in range(11)]))
    out.extend(text(px(i), bottom + 20, "small mid muted", f"{i * 10}%") for i in range(0, 11, 2))
    out.append(
        text((left + right) // 2, bottom + 44, "small mid muted", "share of the run, by rounds")
    )
    for name, (series, cls, dy) in COSINES.items():
        out.append(line(cls, [(px(i + 0.5), py(c)) for i, c in enumerate(series)]))
        label = f"{name}, {series[-1]:.2f}"
        out.append(text(px(9.5) + 10, py(series[-1]) + 4 + dy, "small", label))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def speedup() -> str:
    left, right, top, bottom = 96, 900, 40, 372
    nodes = [1, 2, 4, 8, 16]
    lo, hi = math.log2(24), math.log2(600)
    single = H50_MINUTES[1]

    def px(n: int) -> float:
        return left + (right - left) * nodes.index(n) / (len(nodes) - 1)

    def py(minutes: float) -> float:
        return bottom - (bottom - top) * (math.log2(minutes) - lo) / (hi - lo)

    out = open_svg(
        "Training time against the number of replicas, both on logarithmic scales. One replica "
        f"takes {duration(single)}. The recipe takes "
        + ", ".join(f"{duration(m)} with {n}" for n, m in RECIPE_MINUTES.items())
        + "; Carry at H = 50 takes "
        + ", ".join(f"{duration(m)} with {n}" for n, m in H50_MINUTES.items() if n > 1)
        + ". The dashed line is the ideal, the single-replica time divided by the replica count."
    )
    ticks = [(30, "30 min"), (60, "1 h"), (120, "2 h"), (240, "4 h"), (480, "8 h")]
    out.append(path("rule", [f"M {left} {fmt(py(m))} H {right}" for m, _ in ticks]))
    out.extend(text(left - 10, py(m) + 4, "small end muted", label) for m, label in ticks)
    out.append(
        f'<text x="0" y="0" class="small mid muted" transform="translate(36 {(top + bottom) // 2}) '
        'rotate(-90)">training time, first unit to last commit</text>'
    )
    out.append(path("axis", [f"M {left} {top - 8} V {bottom} H {right}"]))
    out.append(path("tick", [f"M {fmt(px(n))} {bottom} V {bottom + 4}" for n in nodes]))
    out.extend(text(px(n), bottom + 20, "small mid muted", str(n)) for n in nodes)
    out.append(text((left + right) // 2, bottom + 44, "small mid muted", "replicas"))
    out.append(line("guide", [(px(n), py(single / n)) for n in nodes]))
    out.append(
        text(px(16) - 12, py(single / 16) + 18, "small end muted", f"ideal, {duration(single)} / n")
    )
    out.append(line("dotted", [(px(n), py(m)) for n, m in H50_MINUTES.items()]))
    out.extend(dot(px(n), py(m), "dot-muted") for n, m in H50_MINUTES.items())
    recipe = {1: single} | RECIPE_MINUTES
    out.append(line("line-ink", [(px(n), py(m)) for n, m in recipe.items()]))
    for n, m in recipe.items():
        x, y = px(n), py(m)
        out.append(dot(x, y, "dot-ink"))
        label = duration(m) if n == 1 else f"{duration(m)}, {single / m / n:.0%}"
        if n == 16:
            out.append(text(x - 12, y + 22, "small end", label))
        else:
            out.append(text(x + 12, y - 8, "small", label))
    out.append("</svg>")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    (FIGURES / "pythia-loss.svg").write_text(training_loss())
    (FIGURES / "pythia-scaling.svg").write_text(scaling())
    (FIGURES / "pythia-ablation.svg").write_text(ablation())
    (FIGURES / "pythia-speedup.svg").write_text(speedup())
    (FIGURES / "pythia-cosine.svg").write_text(cosine())
