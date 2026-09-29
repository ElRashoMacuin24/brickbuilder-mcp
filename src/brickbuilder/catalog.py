"""Curated part catalog with shape information.

Any part in the LDraw library can be placed; parts outside this catalog are
treated as plain studded boxes of their bounding size for collision checks,
connectivity and previews.

Shapes are prisms: a profile in the local (z, y) plane extruded along local X.
All slopes in LDraw descend toward local -Z, which is the model's front at
rotation 0.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import ldraw


@dataclass(frozen=True)
class Shape:
    kind: str = "box"  # box | slope | slope_inv | ridge
    flat_rows: int = 0  # slopes: studded rows at the high (back) end
    studs: bool = True  # box: studs on top


@dataclass(frozen=True)
class CatalogPart:
    id: str
    category: str
    shape: Shape


_BOX = Shape()
_TILE = Shape(studs=False)
_S1 = Shape("slope", flat_rows=1)
_CHEESE = Shape("slope", flat_rows=0)
_INV = Shape("slope_inv", flat_rows=1)
_RIDGE = Shape("ridge", studs=False)

_ENTRIES = [
    # bricks
    *[(p, "brick", _BOX) for p in
      ("3005", "3004", "3622", "3010", "3009", "3008", "3003", "3002", "3001", "2456", "3007", "3006",
       "3245c", "2454")],
    # plates
    *[(p, "plate", _BOX) for p in
      ("3024", "3023b", "3623", "3710", "3666", "3460", "3022", "3021", "3020", "3795", "3034",
       "3031", "3032", "3035", "3958", "3030", "3036")],
    # tiles
    *[(p, "tile", _TILE) for p in ("3070b", "3069b", "63864", "2431", "3068b", "87079")],
    # slopes
    ("3040b", "slope", _S1), ("3039", "slope", _S1), ("3038", "slope", _S1), ("3037", "slope", _S1),
    ("4286", "slope", _S1), ("3298", "slope", _S1), ("60481a", "slope", _S1), ("4460b", "slope", _S1),
    ("54200", "slope", _CHEESE), ("85984", "slope", _CHEESE),
    ("3665a", "slope", _INV), ("3660a", "slope", _INV), ("3747b", "slope", _INV),
    ("3044b", "slope", _RIDGE),
    # round / misc (previewed as boxes)
    ("3062b", "round", _BOX), ("6141", "round", _BOX), ("4032a", "round", _BOX),
    ("3941", "round", _BOX), ("98138", "round", _TILE), ("4589", "round", _BOX),
    ("3659", "misc", _BOX), ("30136", "misc", _BOX), ("2877", "misc", _BOX), ("3700", "misc", _BOX),
    ("3794b", "misc", _BOX),
]

CATALOG: dict[str, CatalogPart] = {pid: CatalogPart(pid, cat, shape) for pid, cat, shape in _ENTRIES}


def lookup(part: str) -> tuple[str, Shape, str]:
    """Return (canonical id, shape, category) for any library part."""
    pid = ldraw.normalize_part_id(part)
    if pid not in CATALOG:
        pid = ldraw.resolve_moved(pid)
    if pid in CATALOG:
        entry = CATALOG[pid]
        return pid, entry.shape, entry.category
    ldraw.part_bounds(pid)  # raises if missing
    return pid, _BOX, "other"


def footprint(part: str) -> tuple[int, int, int]:
    """(studs along local X, studs along local Z, height in plates)."""
    b = ldraw.part_bounds(part)
    return round((b.x1 - b.x0) / 20), round((b.z1 - b.z0) / 20), b.height_plates


def packing_sizes(kind: str) -> list[tuple[str, int, int]]:
    """Rectangular parts usable by the automatic layer filler: (id, sx, sz)."""
    cat = {"brick": "brick", "plate": "plate", "tile": "tile"}[kind]
    out = []
    for pid, entry in CATALOG.items():
        if entry.category != cat or entry.shape.kind != "box":
            continue
        sx, sz, h = footprint(pid)
        if cat == "brick" and h != 3:
            continue
        out.append((pid, sx, sz))
    return out
