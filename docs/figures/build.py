"""Build the documentation figures.

Each ``docs/figures/<name>.svg`` becomes ``docs/assets/<name>.svg``. A source draws only with the
classes of ``figure.css``: it sets no color and no font. The build adds the design system's colors
and font stacks (``tokens.json``) as CSS variables, light by default and dark under
``prefers-color-scheme: dark``, the shared patterns and arrowheads, and a subset of every face the
figure sets text in, as a woff2 data URI: GitHub shows an SVG through ``<img>``, which loads no web
fonts. The faces come from google/fonts at a pinned commit, cached under
``~/.cache/gradientmail/fonts``.

    uv run --frozen --with fonttools --with brotli python docs/figures/build.py [name ...]
"""

import base64
import functools
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

SOURCES = Path(__file__).parent
ASSETS = SOURCES.parent / "assets"
CACHE = Path.home() / ".cache" / "gradientmail" / "fonts"
FONTS = (
    "https://raw.githubusercontent.com/google/fonts/9710da1eacb3be272583c3224dcb70f9da6eadbb/ofl/"
)
SVG = "{http://www.w3.org/2000/svg}"
THEMES = ("light", "dark")


@dataclass(frozen=True)
class Face:
    family: str
    style: str
    weight: int
    file: str
    sha256: str
    axes: tuple[tuple[str, float], ...] = ()


SANS = Face(
    "Source Sans 3",
    "normal",
    400,
    "sourcesans3/SourceSans3[wght].ttf",
    "042fe2cc0b933e328410d7acbd0aa6a1873dca5aef81875f4bc214b08825c7b9",
    (("wght", 400),),
)
SANS_BOLD = Face(SANS.family, "normal", 600, SANS.file, SANS.sha256, (("wght", 600),))
MATH = Face(
    "Source Serif 4",
    "normal",
    400,
    "sourceserif4/SourceSerif4[opsz,wght].ttf",
    "97b2d4da6e3cb494b5a1e66ae176914d852ccabef49e0c02c0df25f3e39aca0b",
    (("wght", 400), ("opsz", 16)),
)
MATH_ITALIC = Face(
    MATH.family,
    "italic",
    400,
    "sourceserif4/SourceSerif4-Italic[opsz,wght].ttf",
    "15fbc7e4679489a501998c3669272637a6646388ef7e4bd77eebb5bf967a1f42",
    MATH.axes,
)
MONO = Face(
    "Source Code Pro",
    "normal",
    400,
    "sourcecodepro/SourceCodePro[wght].ttf",
    "b400fc584e10aff25d0e775ce181b4fc1c5ea1b5dc37b81aeb2084375b945790",
    (("wght", 400),),
)
SYMBOLS = Face(
    "Noto Sans Math",
    "normal",
    400,
    "notosansmath/NotoSansMath-Regular.ttf",
    "3f495fe933c06786e4d5f6d86b8ee70b6753a68ee3b9d87528726de0f6e2c47d",
)
BOLD = frozenset({"b", "title", "digit"})
HEADS = ("ink", "muted", "accent", "signal", "a", "b", "c")
DEFS = (
    "<defs>"
    '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
    'patternTransform="rotate(45)"><rect width="6" height="6" class="hatch-bg"/>'
    '<path d="M 0 0 V 6" class="hatch-line"/></pattern>'
    '<pattern id="lost" width="6" height="6" patternUnits="userSpaceOnUse" '
    'patternTransform="rotate(45)"><rect width="6" height="6" class="lost-bg"/>'
    '<path d="M 0 0 V 6" class="lost-line"/></pattern>'
    + "".join(
        f'<marker id="head-{head}" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="8" '
        'markerHeight="8" markerUnits="userSpaceOnUse" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 Z" class="head-{head}"/></marker>'
        for head in HEADS
    )
    + "</defs>"
)


def face_of(classes: frozenset[str]) -> Face:
    if "var" in classes:
        return MATH_ITALIC
    if "m" in classes:
        return MATH_ITALIC if "i" in classes else MATH
    if "code" in classes:
        return MONO
    return SANS_BOLD if classes & BOLD else SANS


@functools.cache
def static(face: Face) -> bytes:
    """The face as a static TrueType font: downloaded once, checked, pinned at its axes."""
    path = CACHE / face.file
    if not path.exists():
        print(f"downloading {face.file}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FONTS + urllib.parse.quote(face.file)) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != face.sha256:
        raise SystemExit(f"{path}: checksum mismatch; delete it to download again")
    font = TTFont(io.BytesIO(data), recalcTimestamp=False)
    if face.axes:
        font = instancer.instantiateVariableFont(font, dict(face.axes))
    out = io.BytesIO()
    font.save(out)
    return out.getvalue()


@functools.cache
def cmap(face: Face) -> frozenset[int]:
    table = TTFont(io.BytesIO(static(face))).getBestCmap()
    if table is None:
        raise SystemExit(f"{face.file}: no Unicode cmap")
    return frozenset(table)


def characters(root: ElementTree.Element) -> dict[Face, set[str]]:
    """The characters each face sets, following the classes text inherits."""
    used: dict[Face, set[str]] = {}

    def add(classes: frozenset[str], text: str | None) -> None:
        if text:
            used.setdefault(face_of(classes), set()).update(" " if c.isspace() else c for c in text)

    def walk(element: ElementTree.Element, inherited: frozenset[str], inside: bool) -> None:
        classes = inherited | frozenset(element.get("class", "").split())
        inside = inside or element.tag == f"{SVG}text"
        if inside:
            add(classes, element.text)
        for child in element:
            walk(child, classes, inside)
            if inside:
                add(classes, child.tail)

    walk(root, frozenset(), False)
    return used


def faces(name: str, root: ElementTree.Element) -> dict[Face, str]:
    """Each face's subset text; what the math face lacks goes to the symbols face its stack
    falls back on."""
    assigned: dict[Face, set[str]] = {}
    for face, chars in characters(root).items():
        missing = {c for c in chars if ord(c) not in cmap(face)}
        assigned.setdefault(face, set()).update(chars - missing)
        if not missing:
            continue
        unset = {c for c in missing if ord(c) not in cmap(SYMBOLS)}
        if face.family != MATH.family or unset:
            raise SystemExit(f"{name}: {face.family} has no glyph for {''.join(sorted(missing))!r}")
        assigned.setdefault(SYMBOLS, set()).update(missing)
    return {face: "".join(sorted(chars)) for face, chars in assigned.items() if chars}


def woff2(face: Face, text: str) -> str:
    font = TTFont(io.BytesIO(static(face)), recalcTimestamp=False)
    options = subset.Options()
    options.flavor = "woff2"
    subsetter = subset.Subsetter(options)
    subsetter.populate(text=text)
    subsetter.subset(font)
    font.flavor = "woff2"
    out = io.BytesIO()
    font.save(out)
    return base64.b64encode(out.getvalue()).decode()


def variables(path: Path) -> str:
    """The tokens' font families and light colors on ``svg``, and their dark colors under
    ``prefers-color-scheme: dark``."""
    match json.loads(path.read_text()):
        case {"color": {"tokens": list(colors)}, "type": {"families": dict(families)}}:
            pass
        case _:
            raise SystemExit(f"{path}: expected color tokens and font families")
    stacks = [f"--{key}: {stack};" for key, stack in families.items() if isinstance(stack, str)]
    themed: dict[str, list[str]] = {theme: [] for theme in THEMES}
    for token in colors:
        match token:
            case {"name": str(name), "value": Mapping() as values}:
                for theme in THEMES:
                    match values.get(theme):
                        case str(value):
                            themed[theme].append(f"--{name}: {value};")
                        case _:
                            raise SystemExit(f"{path}: {name} has no {theme} value")
            case _:
                raise SystemExit(f"{path}: malformed color token {token!r}")
    light = " ".join(themed["light"] + stacks)
    dark = " ".join(themed["dark"])
    return f"svg {{ {light} }}\n@media (prefers-color-scheme: dark) {{ svg {{ {dark} }} }}"


def build(source: Path, css: str, known: frozenset[str], palette: str) -> None:
    text = source.read_text()
    root = ElementTree.fromstring(text)
    used = {c for element in root.iter() for c in element.get("class", "").split()}
    if unknown := used - known:
        raise SystemExit(f"{source.name}: classes not in figure.css: {', '.join(sorted(unknown))}")
    head = re.match(r"\s*<svg\b[^>]*>", text)
    if head is None or "<style" in text:
        raise SystemExit(f"{source.name}: expected one <svg> element and no <style>")
    fonts = "\n".join(
        f'@font-face {{ font-family: "{face.family}"; font-style: {face.style}; '
        f"font-weight: {face.weight}; "
        f'src: url(data:font/woff2;base64,{woff2(face, chars)}) format("woff2"); }}'
        for face, chars in faces(source.name, root).items()
    )
    style = f"<style>\n{palette}\n{fonts}\n{css}</style>\n{DEFS}"
    target = ASSETS / source.name
    target.write_text(f"{text[: head.end()]}\n{style}{text[head.end() :]}")
    print(f"{target.relative_to(ASSETS.parent.parent)}: {target.stat().st_size / 1024:.0f} KB")


def main(names: list[str]) -> None:
    css = (SOURCES / "figure.css").read_text()
    known = frozenset(re.findall(r"\.([A-Za-z][\w-]*)", css))
    palette = variables(SOURCES / "tokens.json")
    sources = [SOURCES / f"{name}.svg" for name in names] or sorted(SOURCES.glob("*.svg"))
    for source in sources:
        build(source, css, known, palette)


if __name__ == "__main__":
    main(sys.argv[1:])
