"""Turn voxel layer maps into real bricks, and pictures into LEGO-colour grids."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from . import catalog, ldraw
from .model import BuildError, Model

EMPTY = {".", " ", "_"}

# Common, widely available solid colours used when quantizing pictures.
STANDARD_PALETTE = [0, 1, 2, 4, 14, 15, 19, 25, 27, 28, 70, 71, 72, 73, 85, 191, 212, 226, 272, 288,
                    308, 320, 321, 322, 326, 5, 13, 29, 30, 31, 78, 84, 92, 378, 379, 484, 26, 10, 69, 6]


def parse_legend(legend: dict[str, int | str]) -> dict[str, int]:
    out = {}
    for ch, colour in legend.items():
        if len(ch) != 1:
            raise BuildError(f"Legend keys must be single characters, got '{ch}'")
        out[ch] = ldraw.resolve_colour(colour).code
    return out


def fill_layers(model: Model, layers: list[list[str]], legend: dict[str, int | str], x: int = 0, y: int = 0,
                z: int = 0, kind: str = "brick", upright: bool = False) -> dict:
    """Pack each layer map into the largest bricks that fit, staggering seams.

    Each layer is a list of rows seen from above: the first row is the back,
    the last row is the front (lowest z). Layers go bottom to top. With
    upright=True each layer is instead a front-facing wall: the first row is
    the top and every row becomes its own course of bricks at depth z.
    """
    codes = parse_legend(legend)
    sizes = catalog.packing_sizes(kind)
    height = {"brick": 3, "plate": 1, "tile": 1}[kind]
    placed, skipped = [], []

    courses: list[tuple[int, dict[tuple[int, int], int]]] = []
    if upright:
        # layers[0] is a picture of a wall; flatten into one course per row.
        for li, rows in enumerate(layers):
            for ri, row in enumerate(reversed(rows)):
                cells = {}
                for ci, ch in enumerate(row):
                    if ch not in EMPTY:
                        cells[(x + ci, z + li)] = _code(codes, ch)
                courses.append((y + ri * height, cells))
    else:
        for li, rows in enumerate(layers):
            cells = {}
            for ri, row in enumerate(rows):
                zz = z + len(rows) - 1 - ri
                for ci, ch in enumerate(row):
                    if ch not in EMPTY:
                        cells[(x + ci, zz)] = _code(codes, ch)
            courses.append((y + li * height, cells))

    for index, (yy, cells) in enumerate(courses):
        free = {c: col for c, col in cells.items() if not any((c[0], yy + k, c[1]) in model.occ for k in range(height))}
        skipped += [(cx, yy, cz) for (cx, cz) in cells if (cx, cz) not in free]
        below = {(cx, cz): model.occ.get((cx, yy - 1, cz)) for (cx, cz) in free}
        for pid, sx, sz, rot, cx, cz, col in _pack(free, below, sizes, prefer_x=index % 2 == 0):
            p = model.add(pid, col, cx, yy, cz, rot)
            placed.append(p.id)
    return {"placed": placed, "skipped_occupied_cells": skipped}


def _code(codes: dict[str, int], ch: str) -> int:
    if ch not in codes:
        raise BuildError(f"Character '{ch}' is not in the legend (use '.' for empty)")
    return codes[ch]


def _pack(cells: dict[tuple[int, int], int], below: dict, sizes, prefer_x: bool):
    """Greedy rectangle packing in scan order. At each first-unfilled cell pick
    the candidate covering the most studs, then bridging the most parts below."""
    todo = dict(cells)
    order = sorted(todo, key=lambda c: (c[1], c[0]) if prefer_x else (c[0], c[1]))
    out = []
    for cell in order:
        if cell not in todo:
            continue
        col = todo[cell]
        best = None
        for pid, sx, sz in sizes:
            for rot, (wx, wz) in ((0, (sx, sz)), (90, (sz, sx))):
                if rot == 90 and sx == sz:
                    continue
                # Every cell before this one in scan order is filled, so any
                # rectangle covering it must have it as its min corner.
                cover = [(cell[0] + i, cell[1] + j) for i in range(wx) for j in range(wz)]
                if not all(todo.get(c) == col for c in cover):
                    continue
                parts_below = {below.get(c) for c in cover} - {None}
                along_pref = (wx >= wz) if prefer_x else (wz >= wx)
                aligned = _aligned_seams(cell, wx, wz, cells, below)
                score = (-aligned, len(cover), len(parts_below), along_pref)
                if best is None or score > best[0]:
                    best = (score, pid, sx, sz, rot, cover)
        if best is None:
            raise BuildError(f"No part fits cell {cell}")  # 1x1 always fits, so unreachable
        _, pid, sx, sz, rot, cover = best
        for c in cover:
            del todo[c]
        out.append((pid, sx, sz, rot, cell[0], cell[1], col))
    return out


def _aligned_seams(cell, wx, wz, cells, below) -> int:
    """Count places where this rectangle's far edges sit exactly over a seam
    between two parts in the course below (weak, stacked joints). Edges at the
    boundary of the filled region are unavoidable and not counted."""
    count = 0
    x, z = cell
    for j in range(wz):
        inside, beyond = (x + wx - 1, z + j), (x + wx, z + j)
        if beyond in cells and below.get(inside) is not None and below.get(inside) != below.get(beyond):
            count += 1
    for i in range(wx):
        inside, beyond = (x + i, z + wz - 1), (x + i, z + wz)
        if beyond in cells and below.get(inside) is not None and below.get(inside) != below.get(beyond):
            count += 1
    return count


# ---------------------------------------------------------------- pictures


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def image_to_grid(path: str, width: int, height: int | None = None, max_colors: int = 12,
                  palette: list[int | str] | None = None, crop: list[float] | None = None,
                  aspect: float = 1.0, alpha_threshold: int = 128) -> dict:
    """Downsample a picture to width x height cells in LEGO colours.

    aspect is the cell's height/width ratio (1.0 for studs seen from above,
    1.2 for bricks in an upright wall, 0.4 for plates in an upright wall).
    """
    img = Image.open(Path(path).expanduser()).convert("RGBA")
    if crop:
        W, H = img.size
        l, t, r, b = crop
        img = img.crop((int(l * W), int(t * H), int(r * W), int(b * H)))
    if height is None:
        height = max(1, round(width * img.height / img.width / aspect))
    small = img.resize((width, height), Image.Resampling.BOX)
    arr = np.asarray(small).astype(float)
    rgb, alpha = arr[..., :3], arr[..., 3]

    codes = [ldraw.resolve_colour(c).code for c in palette] if palette else STANDARD_PALETTE
    table = ldraw.colours()
    codes = [c for c in codes if c in table]
    pal_lab = _srgb_to_lab(np.array([table[c].rgb for c in codes], dtype=float))
    lab = _srgb_to_lab(rgb)

    def assign(allowed: list[int]) -> np.ndarray:
        d = ((lab[:, :, None, :] - pal_lab[None, None, allowed, :]) ** 2).sum(-1)
        return np.array(allowed)[d.argmin(-1)]

    idx = assign(list(range(len(codes))))
    visible = alpha >= alpha_threshold
    if max_colors and len(set(idx[visible].ravel())) > max_colors:
        counts = np.bincount(idx[visible].ravel(), minlength=len(codes))
        keep = list(np.argsort(counts)[::-1][:max_colors])
        idx = assign(keep)

    used = [codes[i] for i in sorted(set(idx[visible].ravel()), key=lambda i: -int((idx[visible] == i).sum()))]
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    legend = {chars[n]: code for n, code in enumerate(used)}
    lookup = {code: ch for ch, code in legend.items()}
    rows = ["".join(lookup[codes[idx[r, c]]] if visible[r, c] else "." for c in range(width)) for r in range(height)]
    return {"rows": rows, "legend": legend, "width": width, "height": height}
