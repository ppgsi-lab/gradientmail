"""Write ``docs/figures/hero.svg``, the home page's animation.

Above, four learners walk down a loss valley round by round; below, their timelines. The schedule
follows the learner and syncer rules with one fragment: quorum 2, a publish every 6 own steps or
as soon as the round is one short of quorum, a grace window of 1.2, a merge of 0.4 and an align at
the next own step boundary that keeps the steps since the snapshot, as ``Carry()`` does, with
times in steps of A. The first merge plays slowed down. Each learner pulls toward its own minimum,
offset from the shared one by less after every round. The outer step has a learning rate of 1.0 and
a momentum of 0.4, not the defaults' 0.7 and 0.9: in a valley crossed in three rounds, momentum 0.9
would carry Θ several rounds' walk past the learners.

The source draws with the classes of ``figure.css`` only; ``build.py`` turns it into the asset::

    uv run --frozen python docs/figures/hero.py
    uv run --frozen --with fonttools --with brotli python docs/figures/build.py hero
"""

import itertools
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

type Point = tuple[float, float]


@dataclass(frozen=True)
class Snapshot:
    learner: int
    round: int
    sent: float
    steps: int


@dataclass(frozen=True)
class Round:
    quorum: float
    close: float
    commit: float


@dataclass(frozen=True)
class Merge:
    anchor: Point
    mean: Point
    rda: Point
    length: float
    next: Point
    weights: Mapping[Snapshot, float]


SOURCE = Path(__file__).parent / "hero.svg"
NAMES = ("A", "B", "C", "D")
FILLS = ("sA", "sB", "sC", "sD")
LINES = ("line-a", "line-b", "line-c", "line-d")
STEP = (1.0, 1.25, 1.6, 1.4)
SPAN = 23.5
SNAPSHOTS = (
    Snapshot(0, 0, 6, 6),
    Snapshot(3, 0, 7, 5),
    Snapshot(1, 0, 7.5, 6),
    Snapshot(2, 0, 8, 5),
    Snapshot(0, 1, 12, 6),
    Snapshot(2, 1, 12.8, 3),
    Snapshot(1, 1, 13.75, 5),
    Snapshot(3, 1, 14, 5),
    Snapshot(0, 2, 18, 6),
    Snapshot(2, 2, 19.2, 4),
    Snapshot(3, 2, 19.6, 4),
    Snapshot(1, 2, 20, 5),
)
ROUNDS = (Round(7.5, 8.5, 8.9), Round(13.3, 14.5, 14.9), Round(19.7, 20.5, 20.9))
ALIGNS = ((9, 15, 21), (10, 16.25, 22.5), (9.6, 16, 22.4), (9.8, 15.4, 22.4))

SECONDS = 0.85
SLOW_FROM, SLOW_TO, SLOW_SECONDS = 8.5, 8.9, 5.6
HOLD, FADE = 0.8, 1.0
LOOP = SPAN * SECONDS + SLOW_SECONDS - (SLOW_TO - SLOW_FROM) * SECONDS + HOLD + FADE
BEAT_WEIGHTS, BEAT_MEAN, BEAT_RDA, BEAT_OUTER = 0.3, 1.4, 2.6, 3.8

CENTER, ROTATION, THETA0 = (390, 50), -0.3, (64, 152)
CURVATURE = (0.055, 0.28)
OFFSETS = ((-55, 115), (80, 66), (30, -118), (-66, -66))
SPREAD = (1, 0.28, 0.1)
ETA, BETA = 1.0, 0.4

ALT = (
    "Above, the shared parameters Θ over loss contours; below, the four learners' timelines. Each"
    " learner leaves Θ at its own pace toward its own data's minimum and publishes a snapshot when"
    " it is due, while it keeps training. At quorum the syncer waits a grace window, weights the"
    " snapshots by tokens squared over steps and merges them with RDA, which keeps the mean length"
    " the weighted mean loses; the Nesterov step carries momentum past the learners. Each learner"
    " aligns with the new Θ between its own steps and keeps what it trained since its snapshot,"
    " which its next snapshot carries. Over three rounds the learners agree more and Θ settles in"
    " the valley."
)


def main() -> None:
    SOURCE.write_text(hero())
    print(f"{SOURCE.name}: {SOURCE.stat().st_size / 1024:.0f} KB, loop {LOOP:.2f} s")


def hero() -> str:
    """The figure's source: every element and its animation over one loop of ``LOOP`` seconds."""
    thetas, merges, positions, starts = _model()
    return (
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="448"'
                f' viewBox="0 0 480 448" role="img" aria-label="{ALT}">',
                "<!-- Written by hero.py: change that and run it, not this file. -->",
                _defs(),
                '<rect width="480" height="448" class="ground"/>',
                *_valley(thetas, merges, positions, starts),
                *_timelines(merges),
                *_legend(),
                "</svg>",
            ]
        )
        + "\n"
    )


_ANIM = f'dur="{LOOP:.3f}s" repeatCount="indefinite"'
_LANES, _SYNCER = (278, 300, 322, 344), 380


def _model() -> tuple[
    Sequence[Point], Sequence[Merge], Mapping[Snapshot, Point], Sequence[Sequence[Point]]
]:
    """Θ after each round, the rounds' merges, where each snapshot froze and where each learner
    starts training from each version: Θ plus what it trained since its snapshot."""
    thetas: list[Point] = [THETA0]
    merges: list[Merge] = []
    positions: dict[Snapshot, Point] = {}
    starts: list[list[Point]] = [[THETA0] for _ in NAMES]
    momentum = (0.0, 0.0)
    for g in range(len(ROUNDS)):
        anchor = thetas[g]
        members = [s for s in SNAPSHOTS if s.round == g]
        positions |= {s: _learner(s.learner, s.sent, starts) for s in members}
        total = sum(s.steps for s in members)
        weights = {s: s.steps / total for s in members}
        deltas = [(positions[s][0] - anchor[0], positions[s][1] - anchor[1]) for s in members]
        norms = [math.hypot(*d) for d in deltas]
        pairs = list(zip(members, deltas, norms, strict=True))
        mean = (
            sum(weights[s] * d[0] for s, d, _ in pairs),
            sum(weights[s] * d[1] for s, d, _ in pairs),
        )
        direction = (
            sum(weights[s] * d[0] / n for s, d, n in pairs),
            sum(weights[s] * d[1] / n for s, d, n in pairs),
        )
        length = sum(weights[s] * n for s, _, n in pairs)
        size = math.hypot(*direction)
        rda = (direction[0] / size * length, direction[1] / size * length)
        momentum = (BETA * momentum[0] - rda[0], BETA * momentum[1] - rda[1])
        after = (
            anchor[0] + ETA * (rda[0] - BETA * momentum[0]),
            anchor[1] + ETA * (rda[1] - BETA * momentum[1]),
        )
        thetas.append(after)
        merges.append(Merge(anchor, mean, rda, length, after, weights))
        for s in members:
            before, at = _before(s.learner, g, starts), positions[s]
            starts[s.learner].append(_add(after, (before[0] - at[0], before[1] - at[1])))
    return thetas, merges, positions, starts


def _defs() -> str:
    reveal = (
        (0, 4),
        (_real(SLOW_FROM), _n1(56 + SLOW_FROM * 32 - 52)),
        (_real(SLOW_TO), _n1(56 + SLOW_TO * 32 - 52)),
        (_real(12), 388),
        (LOOP, 388),
    )
    return (
        '<defs><clipPath id="hero-plot"><rect x="16" y="16" width="448" height="228" rx="6"/>'
        '</clipPath><clipPath id="hero-reveal"><rect x="52" y="260" width="4" height="140">'
        f"{_track('width', reveal)}</rect></clipPath></defs>"
    )


def _valley(
    thetas: Sequence[Point],
    merges: Sequence[Merge],
    positions: Mapping[Snapshot, Point],
    starts: Sequence[Sequence[Point]],
) -> list[str]:
    """The parameter space: contours, Θ's path, the learners, their snapshots and the merges."""
    commits = [_real(r.commit) for r in ROUNDS]
    samples = list(_steps(0, LOOP + 1e-9, 0.1))

    def trail(i: int, v: int) -> str:
        def carried() -> list[Point]:
            """The steps between the last snapshot and aligning with ``v``, moved to start from
            Θ, as aligning moves them."""
            if not v:
                return []
            s = next(s for s in SNAPSHOTS if s.learner == i and s.round == v - 1)
            before = ALIGNS[i][v - 2] if v > 1 else 0
            shift = (thetas[v][0] - positions[s][0], thetas[v][1] - positions[s][1])
            ks = _steps(
                (s.sent - before) / STEP[i], (ALIGNS[i][v - 1] - before) / STEP[i] - 1e-9, 0.2
            )
            return [_add(_drift(i, starts[i][v - 1], k, v - 1), shift) for k in ks]

        start = ALIGNS[i][v - 1] if v else 0
        end = ALIGNS[i][v] if v < len(ROUNDS) else SPAN
        steps = (end - start) / STEP[i]
        prefix = carried()
        points = [*prefix, *(_drift(i, starts[i][v], k, v) for k in _steps(0, steps + 1e-9, 0.2))]
        points.append(_drift(i, starts[i][v], steps, v))
        lengths = list(
            itertools.accumulate(
                (math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in itertools.pairwise(points)),
                initial=0.0,
            )
        )
        shown, hidden = (_real(start + 0.4) if v else 0), _real(end)

        def drawn(t: float) -> str:
            k = max(0, (_sim(t)[0] - start) / STEP[i])
            j = min(len(points) - 1, len(prefix) + math.floor(k / 0.2 + 0.5))
            return f"{1 - lengths[j] / lengths[-1]:.3f}"

        offsets = [(t, drawn(t)) for t in samples if shown <= t <= hidden]
        d = "".join(f"{'L' if j else 'M'}{_n1(x)} {_n1(y)}" for j, (x, y) in enumerate(points))
        return (
            f'<path d="{d}" class="{LINES[i]} strong" pathLength="1" stroke-dasharray="1 1"'
            ' stroke-dashoffset="1" opacity="0">'
            + _track("stroke-dashoffset", [(shown, "1"), *offsets, (hidden, "0")])
            + f"{_visible(shown, hidden)}</path>"
        )

    def explained() -> list[str]:
        first = merges[0]
        mean_tip, rda_tip = _add(first.anchor, first.mean), _add(first.anchor, first.rda)
        heading = math.atan2(first.rda[1], first.rda[0])
        done = max(a[0] for a in ALIGNS)

        def arc(turn: float) -> str:
            x = first.anchor[0] + first.length * math.cos(heading + turn)
            y = first.anchor[1] + first.length * math.sin(heading + turn)
            return f"{_n1(x)} {_n1(y)}"

        radius = _n1(first.length)
        shortened = math.hypot(*first.mean) / first.length
        return [
            '<g opacity="0">'
            + _track(
                "opacity",
                ((_beat(0), 0), (_beat(0), 1), (_real(done + 0.5), 1), (_real(done + 1.5), 0)),
            ),
            f'<path d="M{arc(-0.9)}A{radius} {radius} 0 0 1 {arc(0.9)}" class="guide"'
            f' opacity="0">{_visible(_beat(BEAT_RDA), LOOP, 0.6)}</path>',
            _growing(first.anchor, mean_tip, "arrow-ink", _beat(BEAT_MEAN), _beat(BEAT_MEAN + 0.6)),
            f'<text x="{_n1(mean_tip[0] + 4)}" y="{_n1(mean_tip[1] + 16)}" class="small b halo"'
            f' opacity="0">mean · {shortened:.2f}'
            f"{_visible(_beat(BEAT_MEAN + 0.5), LOOP, 0.3)}</text>",
            f'<text x="{_n1(rda_tip[0] - 2)}" y="{_n1(rda_tip[1] - 9)}" class="small b acc halo"'
            f' text-anchor="end" opacity="0">RDA · 1.00'
            f"{_visible(_beat(BEAT_RDA + 0.5), LOOP, 0.3)}</text>",
            "</g>",
        ]

    def stepped(g: int) -> str:
        r, m = ROUNDS[g], merges[g]
        tip = _add(m.anchor, m.rda)
        done = max(a[g] for a in ALIGNS)
        if g == 0:
            grow = (_beat(BEAT_RDA), _beat(BEAT_RDA + 0.6))
            reach = (_beat(BEAT_OUTER), _beat(BEAT_OUTER + 0.6))
        else:
            grow = (_real(r.close), _real(r.close + 0.3))
            reach = (_real(r.commit), _real(r.commit + 0.3))
        left = f"{1 - (SPAN - done - 0.5) / 1.0:.2f}" if done + 1.5 > SPAN else "0"
        return (
            "<g>"
            + _track("opacity", ((_real(done + 0.5), 1), (_real(min(SPAN, done + 1.5)), left)))
            + _growing(m.anchor, tip, "arrow-acc strong", *grow)
            + _growing(tip, m.next, "life-s", *reach)
            + "</g>"
        )

    def frozen(s: Snapshot) -> str:
        gone = _real(ROUNDS[s.round].commit + (1.2 if s.round == 0 else 0.3))
        return (
            f'{_circle(positions[s], 3.5, "snap")} opacity="0">'
            f"{_visible(_real(s.sent), gone)}</circle>"
        )

    def version(g: int) -> Iterator[str]:
        at, end = thetas[g + 1], commits[g + 1] if g + 1 < len(ROUNDS) else LOOP
        yield (
            f'<path d="{_diamond(at, 5.5)}" class="commit ring" opacity="0">'
            f"{_visible(commits[g])}</path>"
        )
        yield (
            f'<text x="{_n1(at[0] - 8)}" y="{_n1(at[1] + 18)}" class="small b acc halo"'
            f' text-anchor="end" opacity="0">v{4 + g}{_visible(commits[g], end)}</text>'
        )

    def learner(i: int) -> str:
        at = [(t, _learner(i, _sim(t)[0], starts)) for t in samples]
        return (
            f"{_circle(at[0][1], 4, f'{FILLS[i]} ring')}>"
            + _track("cx", [(t, _n1(p[0])) for t, p in at])
            + _track("cy", [(t, _n1(p[1])) for t, p in at])
            + "</circle>"
        )

    def weight(s: Snapshot) -> str:
        at, anchor = positions[s], merges[0].anchor
        d = (at[0] - anchor[0], at[1] - anchor[1])
        n = math.hypot(*d)
        side = (-d[1] / n, d[0] / n)
        flip = -1 if side[1] < 0 else 1
        x, y = at[0] + side[0] * 14 * flip, at[1] + side[1] * 14 * flip + 4
        align = "end" if side[0] * flip < 0 else "start"
        shown = (
            (_beat(BEAT_WEIGHTS), 0),
            (_beat(BEAT_WEIGHTS + 0.4), 1),
            (_real(SLOW_TO + 0.3), 1),
            (_real(SLOW_TO + 0.9), 0),
        )
        return (
            f'<text x="{_n1(x)}" y="{_n1(y)}" class="small halo" text-anchor="{align}"'
            f' opacity="0"><tspan class="muted">w </tspan>{merges[0].weights[s]:.2f}'
            f"{_track('opacity', shown)}</text>"
        )

    contours = "".join(
        f'<ellipse cx="{CENTER[0]}" cy="{CENTER[1]}" rx="{rx}" ry="{ry}"/>'
        for rx, ry in ((44, 17), (100, 38), (165, 62), (240, 90), (325, 122), (420, 158))
    )
    rotation = f"{math.degrees(ROTATION):.3f} {CENTER[0]} {CENTER[1]}"
    return [
        '<g clip-path="url(#hero-plot)">',
        f'<g class="rule" transform="rotate({rotation})">{contours}</g>',
        "<g>",
        *(
            f'<line x1="{_n1(a[0])}" y1="{_n1(a[1])}" x2="{_n1(b[0])}" y2="{_n1(b[1])}"'
            f' class="life-s" opacity="0">{_visible(commits[g])}</line>'
            for g, (a, b) in enumerate(itertools.pairwise(thetas))
        ),
        *(trail(i, v) for i in range(len(NAMES)) for v in range(len(ROUNDS) + 1)),
        *explained(),
        *(stepped(g) for g in range(len(ROUNDS))),
        *(frozen(s) for s in SNAPSHOTS),
        f"{_circle(THETA0, 4.5, 'dot-ink')}/>",
        *(line for g in range(len(ROUNDS)) for line in version(g)),
        *(learner(i) for i in range(len(NAMES))),
        *(weight(s) for s in SNAPSHOTS if s.round == 0),
        f"{_fade_out()}</g>",
        f'<text x="{THETA0[0] - 20}" y="{THETA0[1] + 22}" class="m i" text-anchor="end">Θ</text>'
        f'<text x="{THETA0[0] - 16}" y="{THETA0[1] + 22}" class="small muted">v3</text>',
        "</g>",
        '<rect x="16" y="16" width="448" height="228" rx="6" class="rule"/>',
        '<text x="452" y="234" class="small muted" text-anchor="end">'
        "loss contours, schematic</text>",
    ]


def _timelines(merges: Sequence[Merge]) -> list[str]:
    """The four lanes and the syncer's: revealed up to now, scrolled once 12 steps of A passed."""

    def x(u: float) -> float:
        return 56 + u * 32

    def pills(i: int) -> Iterator[str]:
        d = STEP[i]
        starts = itertools.takewhile(lambda s: s + d <= SPAN + d, _accumulated(0, d))
        return (
            f'<rect x="{_n1(x(s) + 2)}" y="{_LANES[i] - 8}" width="{_n1(d * 32 - 4)}"'
            f' height="16" rx="8" class="{FILLS[i]}"/>'
            for s in starts
        )

    def closed(r: Round, m: Merge) -> Iterator[str]:
        members = sorted(s.learner for s in m.weights)
        left, n = x(r.close) + 3, len(members)
        yield (
            f'<rect x="{_n1(x(r.quorum))}" y="{_SYNCER - 8}"'
            f' width="{_n1((r.close - r.quorum) * 32)}" height="16" rx="8" class="wait"/>'
        )
        yield (
            f'<rect x="{_n1(left)}" y="{_SYNCER - 7}" width="{_n1(12 + 5.5 * n + 3.5 * (n - 1))}"'
            ' height="14" rx="7" class="round"/>'
        )
        yield from (
            f'<circle cx="{_n1(left + 8.75 + 9 * k)}" cy="{_SYNCER}" r="2.75" class="{FILLS[i]}"/>'
            for k, i in enumerate(members)
        )

    shift = ((0, "0 0"), (_real(12), "0 0"), (_real(SPAN), f"{_js(-(SPAN - 12) * 32)} 0"))
    values = ";".join(["0 0", *(v for _, v in shift), shift[-1][1]])
    times = ";".join(["0", *(_kt(t) for t, _ in shift), "1"])
    now = [
        (t, _n1(v))
        for t, v in (
            (0, 56),
            (_real(SLOW_FROM), x(SLOW_FROM)),
            (_real(SLOW_TO), x(SLOW_TO)),
            (_real(12), 440),
            (LOOP, 440),
        )
    ]
    return [
        *(
            f'<text x="36" y="{_LANES[i] + 4}" class="b" text-anchor="middle">{name}</text>'
            for i, name in enumerate(NAMES)
        ),
        f'<text x="48" y="{_SYNCER + 4}" class="small muted" text-anchor="end">syncer</text>',
        '<g clip-path="url(#hero-reveal)"><g><animateTransform attributeName="transform"'
        f' type="translate" values="{values}" keyTimes="{times}" calcMode="linear" {_ANIM}/>',
        *(line for i in range(len(NAMES)) for line in pills(i)),
        *(line for r, m in zip(ROUNDS, merges, strict=True) for line in closed(r, m)),
        *(f"{_circle((x(s.sent), _LANES[s.learner]), 4, 'snap')}/>" for s in SNAPSHOTS),
        *(
            f'<path d="{_diamond((x(a), _LANES[i]), 5.5)}" class="commit ring"/>'
            for i, aligns in enumerate(ALIGNS)
            for a in aligns
        ),
        f"</g>{_fade_out()}</g>",
        f'<line x1="56" y1="266" x2="56" y2="392" class="axis">{_track("x1", now)}'
        f"{_track('x2', now)}</line>",
        '<text x="440" y="404" class="small muted" text-anchor="end">time →</text>',
    ]


def _legend() -> list[str]:
    return [
        '<circle cx="28" cy="424" r="4" class="snap"/>'
        '<text x="38" y="428" class="small muted">snapshot</text>',
        '<rect x="90" y="418" width="22" height="12" rx="6" class="wait"/>'
        '<text x="118" y="428" class="small muted">grace</text>',
        '<rect x="158" y="417" width="27" height="14" rx="7" class="round"/>'
        '<circle cx="167" cy="424" r="2.75" class="sA"/>'
        '<circle cx="176" cy="424" r="2.75" class="sB"/>'
        '<text x="191" y="428" class="small muted">round, with who is in it</text>',
        f'<path d="{_diamond((318, 424), 5)}" class="commit"/>'
        '<text x="328" y="428" class="small muted">commit, aligned</text>',
    ]


def _drift(i: int, anchor: Point, k: float, v: int) -> Point:
    """Learner ``i`` after ``k`` steps from ``anchor`` at version ``v``: gradient descent on a
    quadratic valley whose minimum, for this learner, sits ``OFFSETS[i] * SPREAD[v]`` off the
    shared one, in the valley's own axes."""
    cos, sin = math.cos(ROTATION), math.sin(ROTATION)
    x, y = anchor[0] - CENTER[0], anchor[1] - CENTER[1]
    spread = SPREAD[min(v, len(SPREAD) - 1)]
    u = (cos * x + sin * y, -sin * x + cos * y)
    m = (OFFSETS[i][0] * spread, OFFSETS[i][1] * spread)
    w = [m[a] + math.pow(1 - CURVATURE[a], k) * (u[a] - m[a]) for a in (0, 1)]
    return CENTER[0] + cos * w[0] - sin * w[1], CENTER[1] + sin * w[0] + cos * w[1]


def _learner(i: int, u: float, starts: Sequence[Sequence[Point]]) -> Point:
    """Learner ``i`` at step time ``u``: training from where it started at the version it aligned
    with last, or on its way there during the 0.4 steps after aligning."""
    v = _aligned(i, u)
    start = ALIGNS[i][v - 1] if v else 0
    if not v or u - start >= 0.4:
        return _drift(i, starts[i][v], (u - start) / STEP[i], v)
    before, after = _before(i, v - 1, starts), starts[i][v]
    f = max(0, min(1, (u - start) / 0.4))
    f = f * f * (3 - 2 * f)
    return before[0] + f * (after[0] - before[0]), before[1] + f * (after[1] - before[1])


def _before(i: int, v: int, starts: Sequence[Sequence[Point]]) -> Point:
    """Learner ``i`` as it aligns with the version after ``v``, trained from where it started."""
    start = ALIGNS[i][v - 1] if v else 0
    return _drift(i, starts[i][v], (ALIGNS[i][v] - start) / STEP[i], v)


def _aligned(i: int, u: float) -> int:
    return sum(1 for a in ALIGNS[i] if a <= u)


def _sim(real: float) -> tuple[float, float]:
    """The step time at ``real`` seconds into the loop, and the seconds into the slowed merge."""
    slow = SLOW_FROM * SECONDS
    if real < slow:
        return real / SECONDS, -1
    if real < slow + SLOW_SECONDS:
        return SLOW_FROM + (SLOW_TO - SLOW_FROM) * (real - slow) / SLOW_SECONDS, real - slow
    return min(SPAN, SLOW_TO + (real - slow - SLOW_SECONDS) / SECONDS), SLOW_SECONDS


def _real(u: float) -> float:
    """The seconds into the loop at step time ``u``."""
    if u <= SLOW_FROM:
        return u * SECONDS
    if u <= SLOW_TO:
        return SLOW_FROM * SECONDS + ((u - SLOW_FROM) / (SLOW_TO - SLOW_FROM)) * SLOW_SECONDS
    return SLOW_FROM * SECONDS + SLOW_SECONDS + (u - SLOW_TO) * SECONDS


def _beat(seconds: float) -> float:
    """The seconds into the loop that are ``seconds`` into the slowed merge."""
    return SLOW_FROM * SECONDS + seconds


def _steps(start: float, stop: float, step: float) -> Iterator[float]:
    return itertools.takewhile(lambda t: t <= stop, _accumulated(start, step))


def _accumulated(start: float, step: float) -> Iterator[float]:
    """``start``, ``start + step``, ``start + step + step``, ...: summed, as the frames are."""
    return itertools.accumulate(itertools.repeat(step), initial=start)


def _track(attribute: str, pairs: Sequence[tuple[float, str | int]], mode: str = "linear") -> str:
    """An animation of ``attribute`` through (seconds, value) pairs: the first value holds before
    the first time and the last after the last."""
    points = sorted(((max(0, min(LOOP, t)), v) for t, v in pairs), key=lambda p: p[0])
    points = (
        ([(0, points[0][1])] if points[0][0] > 0 else [])
        + points
        + ([(LOOP, points[-1][1])] if points[-1][0] < LOOP else [])
    )
    values = ";".join(str(v) for _, v in points)
    times = ";".join(_kt(t) for t, _ in points)
    return (
        f'<animate attributeName="{attribute}" values="{values}" keyTimes="{times}"'
        f' calcMode="{mode}" {_ANIM}/>'
    )


def _fade_out() -> str:
    return _track("opacity", ((LOOP - FADE, 1), (LOOP, 0)))


def _visible(start: float, end: float = LOOP, fade: float = 0) -> str:
    return _track("opacity", ((start, 0), (start + fade, 1), (end, 1), (end, 0)))


def _growing(a: Point, b: Point, classes: str, start: float, end: float) -> str:
    """A line drawn from ``a`` to ``b`` between ``start`` and ``end``, hidden before."""
    return (
        f'<line x1="{_n1(a[0])}" y1="{_n1(a[1])}" x2="{_n1(a[0])}" y2="{_n1(a[1])}"'
        f' class="{classes}" opacity="0">'
        + _track("x2", ((start, _n1(a[0])), (end, _n1(b[0]))))
        + _track("y2", ((start, _n1(a[1])), (end, _n1(b[1]))))
        + _track("opacity", ((start, 0), (start, 1)), "discrete")
        + "</line>"
    )


def _circle(at: Point, r: float, classes: str) -> str:
    return f'<circle cx="{_n1(at[0])}" cy="{_n1(at[1])}" r="{_js(r)}" class="{classes}"'


def _diamond(at: Point, h: float) -> str:
    return (
        f"M{_n1(at[0])} {_n1(at[1] - h)}l{_js(h)} {_js(h)}l{_js(-h)} {_js(h)}l{_js(-h)} {_js(-h)}z"
    )


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _kt(t: float) -> str:
    return f"{max(0, min(1, t / LOOP)):.4f}"


def _n1(v: float) -> str:
    """``v`` to one decimal, halves rounded up, without a trailing ``.0``."""
    return _js(math.floor(v * 10 + 0.5) / 10)


def _js(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else repr(float(v))


if __name__ == "__main__":
    main()
