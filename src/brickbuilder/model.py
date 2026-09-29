"""Brick model on a stud grid.

Grid coordinates (what tools use):
  x: studs, increasing to the right (as seen from the front)
  z: studs, increasing toward the back (z=0 is the front row)
  y: plates, increasing upward (y=0 is the ground; a brick is 3 plates tall)
A part's (x, y, z) is the minimum corner of its body after rotation.

LDraw coordinates: X = 20*x, Z = 20*z, Y = -8*y (LDraw -Y is up).
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import catalog, ldraw

FACINGS = {"front": 0, "right": 90, "back": 180, "left": 270}


class BuildError(Exception):
    pass


def rot_matrix(rot: int) -> np.ndarray:
    """Rotation about the vertical axis. rot=90 turns a slope's downhill face
    (local -Z, the front) to face +X (right)."""
    t = math.radians(rot)
    c, s = round(math.cos(t)), round(math.sin(t))
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=float)


def parse_rot(rot: int | str | None) -> int:
    if rot is None:
        return 0
    if isinstance(rot, str):
        key = rot.strip().lower()
        if key in FACINGS:
            return FACINGS[key]
        if not re.fullmatch(r"-?\d+", key):
            raise BuildError(f"rot must be 0/90/180/270 or front/right/back/left, got '{rot}'")
        rot = int(key)
    rot = int(rot) % 360
    if rot % 90:
        raise BuildError(f"rot must be a multiple of 90, got {rot}")
    return rot


@dataclass
class Placement:
    id: int
    part: str
    color: int
    x: int
    y: int
    z: int
    rot: int = 0


@dataclass
class Geometry:
    matrix: np.ndarray
    pos: np.ndarray  # LDraw position of the part origin
    cells: list[tuple[int, int, int]]  # (x, y, z) occupied
    top_studs: set[tuple[int, int]]  # (x, z) columns with a stud on the top level
    bottom_conn: set[tuple[int, int]]  # (x, z) columns that accept studs from below
    y_top: int  # highest occupied level
    shape: catalog.Shape = field(default_factory=catalog.Shape)


def compute_geometry(p: Placement) -> Geometry:
    pid, shape, _ = catalog.lookup(p.part)
    b = ldraw.part_bounds(pid)
    m = rot_matrix(p.rot)
    corners = np.array([[x, 0, z] for x in (b.x0, b.x1) for z in (b.z0, b.z1)], dtype=float) @ m.T
    mx, mz = corners[:, 0].min(), corners[:, 2].min()
    pos = np.array([20 * p.x - mx, -8 * p.y - b.y_bottom, 20 * p.z - mz])

    nx, nz = round((b.x1 - b.x0) / 20), round((b.z1 - b.z0) / 20)
    h = b.height_plates
    cells, top, bottom = [], set(), set()
    for ix in range(nx):
        for iz in range(nz):
            local = np.array([b.x0 + 20 * ix + 10, 0, b.z0 + 20 * iz + 10])
            w = m @ local + pos
            gx, gz = math.floor(w[0] / 20), math.floor(w[2] / 20)
            for gy in range(p.y, p.y + h):
                cells.append((gx, gy, gz))
            high_row = iz >= nz - shape.flat_rows
            if shape.kind == "box":
                has_stud = shape.studs
            elif shape.kind == "slope":
                has_stud = high_row
            elif shape.kind == "slope_inv":
                has_stud = True
            else:
                has_stud = False
            if has_stud:
                top.add((gx, gz))
            if shape.kind != "slope_inv" or high_row:
                bottom.add((gx, gz))
    return Geometry(m, pos, cells, top, bottom, p.y + h - 1, shape)


class Model:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.parts: dict[int, Placement] = {}
        self.next_id = 1
        self._geom: dict[int, Geometry] = {}
        self.occ: dict[tuple[int, int, int], int] = {}
        self._history: list[tuple[dict[int, Placement], int]] = []

    # ------------------------------------------------------------ editing

    def checkpoint(self) -> None:
        self._history.append((copy.deepcopy(self.parts), self.next_id))
        del self._history[:-50]

    def undo(self) -> bool:
        if not self._history:
            return False
        parts, next_id = self._history.pop()
        self._rebuild(parts, next_id)
        return True

    def _rebuild(self, parts: dict[int, Placement], next_id: int) -> None:
        self.parts, self.next_id = {}, next_id
        self._geom.clear()
        self.occ.clear()
        for p in parts.values():
            self._insert(p)

    def _insert(self, p: Placement) -> None:
        g = compute_geometry(p)
        self.parts[p.id] = p
        self._geom[p.id] = g
        for c in g.cells:
            self.occ[c] = p.id

    def geometry(self, pid: int) -> Geometry:
        return self._geom[pid]

    def add(self, part: str, color: int | str, x: int, y: int, z: int, rot: int | str | None = 0) -> Placement:
        pid, _, _ = catalog.lookup(part)
        colour = ldraw.resolve_colour(color)
        p = Placement(self.next_id, pid, colour.code, int(x), int(y), int(z), parse_rot(rot))
        g = compute_geometry(p)
        clashes = sorted({self.occ[c] for c in g.cells if c in self.occ})
        if clashes:
            others = ", ".join(f"#{i} {self.parts[i].part}" for i in clashes[:5])
            raise BuildError(f"{pid} at x={x} y={y} z={z} rot={p.rot} collides with {others}")
        if min(c[1] for c in g.cells) < 0:
            raise BuildError(f"{pid} at y={y} would go below the ground (y<0)")
        self.next_id += 1
        self._insert(p)
        return p

    def remove(self, ids: list[int]) -> list[int]:
        removed = []
        for i in ids:
            if i in self.parts:
                for c in self._geom[i].cells:
                    self.occ.pop(c, None)
                del self.parts[i], self._geom[i]
                removed.append(i)
        return removed

    def parts_in_region(self, x0, x1, y0, y1, z0, z1) -> list[int]:
        return sorted({pid for (x, y, z), pid in self.occ.items()
                       if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1})

    def clear(self) -> None:
        self._rebuild({}, 1)

    # ------------------------------------------------------------ queries

    def bounds(self) -> tuple[int, int, int, int, int, int] | None:
        if not self.occ:
            return None
        xs, ys, zs = zip(*self.occ.keys())
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def connections(self) -> dict[int, set[int]]:
        """Stud connections between parts (undirected adjacency)."""
        adj: dict[int, set[int]] = {i: set() for i in self.parts}
        for i, g in self._geom.items():
            for (x, z) in g.top_studs:
                above = self.occ.get((x, g.y_top + 1, z))
                if above is not None and above != i and (x, z) in self._geom[above].bottom_conn:
                    adj[i].add(above)
                    adj[above].add(i)
        return adj

    def check(self) -> dict:
        adj = self.connections()
        seen, groups = set(), []
        for start in self.parts:
            if start in seen:
                continue
            stack, group = [start], []
            seen.add(start)
            while stack:
                n = stack.pop()
                group.append(n)
                for m in adj[n]:
                    if m not in seen:
                        seen.add(m)
                        stack.append(m)
            groups.append(sorted(group))
        groups.sort(key=len, reverse=True)
        grounded = lambda grp: any(self.parts[i].y == 0 for i in grp)
        floating = [grp for grp in groups if not grounded(grp)]
        # Parts that sit only on tiles / slopes rest without being attached.
        return {
            "parts": len(self.parts),
            "connected_groups": len(groups),
            "group_sizes": [len(g) for g in groups[:10]],
            "floating_groups": floating[:10],
            "loose_grounded_groups": [g for g in groups[1:] if grounded(g)][:10],
        }

    def layer_map(self, y: int, legend: dict[int, str]) -> list[str]:
        b = self.bounds()
        if b is None:
            return []
        x0, x1, _, _, z0, z1 = b
        rows = []
        for z in range(z1, z0 - 1, -1):  # back row first, front row last
            row = []
            for x in range(x0, x1 + 1):
                pid = self.occ.get((x, y, z))
                row.append("." if pid is None else legend.setdefault(self.parts[pid].color, _next_char(legend)))
            rows.append("".join(row))
        return rows

    # ------------------------------------------------------------ persistence

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "next_id": self.next_id,
            "parts": [asdict(p) for p in self.parts.values()],
        }

    @classmethod
    def from_json(cls, data: dict) -> "Model":
        m = cls(data["name"], data.get("description", ""))
        m._rebuild({p["id"]: Placement(**p) for p in data["parts"]}, data.get("next_id", 1))
        return m

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{safe_name(self.name)}.json"
        path.write_text(json.dumps(self.to_json(), indent=1))
        return path

    def to_ldr(self, legacy_ids: bool = False) -> str:
        lines = [
            f"0 {self.name}",
            f"0 Name: {safe_name(self.name)}.ldr",
            "0 Author: brickbuilder-mcp",
            f"0 // {self.description}" if self.description else None,
            f"0 // Exported {datetime.now():%Y-%m-%d %H:%M}",
        ]
        lines = [line for line in lines if line]
        # One build step per plate level, bottom to top.
        ordered = sorted(self.parts.values(), key=lambda p: (p.y, p.z, p.x))
        last_y = None
        for p in ordered:
            if last_y is not None and p.y != last_y:
                lines.append("0 STEP")
            last_y = p.y
            g = self._geom[p.id]
            m = g.matrix
            nums = [*g.pos, *m.flatten()]
            part = legacy_part_id(p.part) if legacy_ids else p.part
            lines.append(f"1 {p.color} " + " ".join(_fmt(v) for v in nums) + f" {part}.dat")
        lines.append("0 STEP")
        return "\n".join(lines) + "\n"


def legacy_part_id(pid: str) -> str:
    """Map e.g. 3023b -> 3023 when the plain number is an alias of this part."""
    m = re.fullmatch(r"(\d+)([a-z])", pid)
    if m and ldraw.resolve_moved(m.group(1)) == pid:
        return m.group(1)
    return pid


def _fmt(v: float) -> str:
    v = round(float(v), 3)
    return str(int(v)) if v == int(v) else f"{v:g}"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "model"


_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@$%&*+=?"


def _next_char(legend: dict) -> str:
    used = set(legend.values())
    for ch in _CHARS:
        if ch not in used:
            return ch
    return "#"
