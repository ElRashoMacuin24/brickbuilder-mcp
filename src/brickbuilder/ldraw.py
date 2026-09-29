"""Access to the LDraw parts library: part geometry bounds, titles and colours.

LDraw conventions: 1 stud = 20 LDU, 1 plate = 8 LDU, 1 brick = 24 LDU, -Y is up.
Standard parts have their origin at the centre of the top surface (studs stick
up into negative Y).
"""

from __future__ import annotations

import functools
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LDRAW_DIR = Path(os.environ.get("LDRAW_DIR", PROJECT_ROOT / "data" / "ldraw"))


class LDrawError(Exception):
    pass


_PART_ID = re.compile(r"(s/)?[a-z0-9][a-z0-9_.-]{0,63}")


def normalize_part_id(part: str) -> str:
    """Canonical part id. Rejects anything that is not a plain LDraw part name
    (e.g. path separators or '..'), so user input can never reach files outside
    the parts library."""
    part = str(part).strip().lower().replace("\\", "/")
    if part.endswith(".dat"):
        part = part[:-4]
    if not _PART_ID.fullmatch(part) or ".." in part:
        raise LDrawError(f"Invalid part id '{part}'")
    return part


def _find(name: str) -> Path | None:
    name = name.strip().lower().replace("\\", "/")
    root = LDRAW_DIR.resolve()
    for sub in ("parts", "p"):
        path = (LDRAW_DIR / sub / name).resolve()
        if not path.is_relative_to(root):
            return None  # never follow references outside the library
        if path.exists():
            return path
    return None


def library_available() -> bool:
    return (LDRAW_DIR / "parts").is_dir() and (LDRAW_DIR / "LDConfig.ldr").exists()


@functools.lru_cache(maxsize=8192)
def _points(name: str) -> np.ndarray:
    """All vertices of a part/subpart/primitive, fully resolved, in its local frame."""
    path = _find(name)
    if path is None:
        return np.zeros((0, 3))
    chunks = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "1" and len(tok) >= 15:
                vals = list(map(float, tok[2:14]))
                pos = np.array(vals[:3])
                mat = np.array(vals[3:]).reshape(3, 3)
                sub = _points(" ".join(tok[14:]))
                if len(sub):
                    chunks.append(sub @ mat.T + pos)
            elif tok[0] in ("3", "4"):
                n = int(tok[0])
                if len(tok) >= 2 + 3 * n:
                    chunks.append(np.array(list(map(float, tok[2 : 2 + 3 * n]))).reshape(n, 3))
    return np.vstack(chunks) if chunks else np.zeros((0, 3))


@functools.lru_cache(maxsize=8192)
def part_title(part: str) -> str | None:
    path = _find(normalize_part_id(part) + ".dat")
    if path is None:
        return None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        first = fh.readline().strip()
    return first[2:].strip() if first.startswith("0 ") else first


def resolve_moved(part: str) -> str:
    """Follow '~Moved to xxx' aliases to the canonical part id."""
    part = normalize_part_id(part)
    for _ in range(5):
        title = part_title(part) or ""
        m = re.match(r"~Moved to (\S+)", title)
        if not m:
            break
        part = normalize_part_id(m.group(1))
    return part


@dataclass(frozen=True)
class PartBounds:
    """Body bounding box in LDU (studs excluded), local part frame, Y down."""

    x0: float
    x1: float
    y_top: float
    y_bottom: float
    z0: float
    z1: float

    @property
    def height_plates(self) -> int:
        return round((self.y_bottom - self.y_top) / 8)


@functools.lru_cache(maxsize=4096)
def part_bounds(part: str) -> PartBounds:
    part = normalize_part_id(part)
    pts = _points(part + ".dat")
    if len(pts) == 0:
        raise LDrawError(f"Part '{part}' not found in the LDraw library")
    lo, hi = pts.min(0), pts.max(0)
    # Snap footprint to the stud grid and height to whole plates. Stud tops
    # (4 LDU) are dropped by the floor: 28 -> 24 for bricks, 12 -> 8 for plates.
    # Snap by width, not by edges: odd-width parts have edges at +-10/30 LDU.
    x0 = round(lo[0] / 10) * 10
    x1 = x0 + max(1, round((hi[0] - lo[0]) / 20)) * 20
    z0 = round(lo[2] / 10) * 10
    z1 = z0 + max(1, round((hi[2] - lo[2]) / 20)) * 20
    height = max(8, int((hi[1] - lo[1] + 2) // 8) * 8)
    y_bottom = round(hi[1])
    return PartBounds(x0, x1, y_bottom - height, y_bottom, z0, z1)


# ---------------------------------------------------------------- colours


@dataclass(frozen=True)
class Colour:
    code: int
    name: str
    rgb: tuple[int, int, int]
    alpha: int = 255
    material: str = "solid"  # solid | transparent | chrome | pearlescent | metal | rubber | other

    @property
    def hex(self) -> str:
        return "#%02X%02X%02X" % self.rgb


def _norm_colour_name(name: str) -> str:
    return re.sub(r"[\s_\-]", "", name.lower()).replace("grey", "gray")


@functools.lru_cache(maxsize=1)
def colours() -> dict[int, Colour]:
    out: dict[int, Colour] = {}
    path = LDRAW_DIR / "LDConfig.ldr"
    if not path.exists():
        raise LDrawError("LDConfig.ldr missing - run scripts/fetch_ldraw.sh")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"0\s+!COLOUR\s+(\S+)\s+CODE\s+(\d+)\s+VALUE\s+#([0-9A-Fa-f]{6})", line.strip())
        if not m:
            continue
        name, code, hexval = m.group(1), int(m.group(2)), m.group(3)
        alpha_m = re.search(r"ALPHA\s+(\d+)", line)
        alpha = int(alpha_m.group(1)) if alpha_m else 255
        material = "solid"
        for kw, mat in (("CHROME", "chrome"), ("PEARLESCENT", "pearlescent"), ("METAL", "metal"),
                        ("RUBBER", "rubber"), ("MATERIAL", "other")):
            if kw in line:
                material = mat
                break
        if alpha < 255:
            material = "transparent"
        rgb = tuple(int(hexval[i : i + 2], 16) for i in (0, 2, 4))
        out[code] = Colour(code, name.replace("_", " "), rgb, alpha, material)
    return out


def resolve_colour(value: int | str) -> Colour:
    table = colours()
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        code = int(value)
        if code not in table:
            raise LDrawError(f"Unknown LDraw colour code {code}")
        return table[code]
    key = _norm_colour_name(value)
    for c in table.values():
        if _norm_colour_name(c.name) == key:
            return c
    matches = [c for c in table.values() if key in _norm_colour_name(c.name)]
    if len(matches) == 1:
        return matches[0]
    hint = ", ".join(f"{c.name} ({c.code})" for c in matches[:8])
    raise LDrawError(f"Unknown colour '{value}'" + (f" - did you mean: {hint}" if hint else ""))


# ---------------------------------------------------------------- search


@functools.lru_cache(maxsize=1)
def _title_index() -> list[tuple[str, str]]:
    index = []
    for path in (LDRAW_DIR / "parts").glob("*.dat"):
        with open(path, encoding="utf-8", errors="ignore") as fh:
            title = fh.readline().strip()[2:].strip()
        if title.startswith("~") or title.startswith("="):
            continue  # moved aliases, sub-assemblies and duplicates
        index.append((path.stem.lower(), title))
    index.sort()
    return index


def search_library(query: str, limit: int = 40) -> list[tuple[str, str]]:
    squash = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
    phrase, words = squash(query), query.lower().split()
    hits = [(pid, title) for pid, title in _title_index()
            if all(w in title.lower() or w == pid for w in words)]
    # Exact phrase first, then short, plain titles (the "basic" version of a part).
    hits.sort(key=lambda h: (phrase not in squash(h[1]), h[0].startswith("u"), len(h[1]), h[0]))
    return hits[:limit]
