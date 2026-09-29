"""Software preview renderer: simplified brick shapes, z-buffered, outlined.

Render space is LDraw with Y flipped so +Y is up: (X, Y, Z) = (x_ldu, -y_ldu, z_ldu).
The front of the model faces -Z.
"""

from __future__ import annotations

import io
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import ldraw
from .model import Model

VIEWS = {
    "front": (0, 0), "back": (180, 0), "left": (-90, 0), "right": (90, 0), "top": (0, 90),
    "iso_front_left": (-45, 30), "iso_front_right": (45, 30),
    "iso_back_left": (-135, 30), "iso_back_right": (135, 30),
}
DEFAULT_VIEWS = ["iso_front_left", "iso_back_right", "front", "top"]
BACKGROUND = np.array([226, 229, 233], dtype=float)
LIGHT = np.array([-0.45, 1.0, -0.65]) / np.linalg.norm([-0.45, 1.0, -0.65])


def _profile(shape, b) -> list[tuple[float, float]]:
    """Convex cross-section in the local (z, y) plane, y down."""
    z0, z1, yt, yb = b.z0, b.z1, b.y_top, b.y_bottom
    zf = z1 - 20 * shape.flat_rows
    lip = min(4, (yb - yt) / 4)
    if shape.kind == "slope":
        if shape.flat_rows == 0:
            return [(z1, yt), (z1, yb), (z0, yb)]
        return [(zf, yt), (z1, yt), (z1, yb), (z0, yb), (z0, yb - lip)]
    if shape.kind == "slope_inv":
        return [(z0, yt), (z1, yt), (z1, yb), (zf, yb), (z0, yt + lip)]
    if shape.kind == "ridge":
        zm = (z0 + z1) / 2
        return [(zm, yt), (z1, yb - lip), (z1, yb), (z0, yb), (z0, yb - lip)]
    return [(z0, yt), (z1, yt), (z1, yb), (z0, yb)]


class Mesh:
    def __init__(self):
        self.tris: list[np.ndarray] = []
        self.colors: list[np.ndarray] = []
        self.fids: list[int] = []
        self._next_fid = 1

    def polygon(self, pts: np.ndarray, color: np.ndarray, inside: np.ndarray, fid: int | None = None) -> None:
        """Add a convex planar polygon, oriented so its normal points away from `inside`."""
        if fid is None:
            fid = self.new_fid()
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.dot(n, pts.mean(0) - inside) < 0:
            pts = pts[::-1]
        for i in range(1, len(pts) - 1):
            self.tris.append(np.array([pts[0], pts[i], pts[i + 1]]))
            self.colors.append(color)
            self.fids.append(fid)

    def new_fid(self) -> int:
        self._next_fid += 1
        return self._next_fid


def build_mesh(model: Model) -> Mesh:
    mesh = Mesh()
    flip = np.array([1.0, -1.0, 1.0])
    table = ldraw.colours()
    for pid, p in model.parts.items():
        g = model.geometry(pid)
        b = ldraw.part_bounds(p.part)
        col = table[p.color]
        rgb = np.array(col.rgb, dtype=float)
        if col.alpha < 255:
            rgb = rgb * 0.6 + 255 * 0.4

        def world(local: np.ndarray) -> np.ndarray:
            return (local @ g.matrix.T + g.pos) * flip

        prof = _profile(g.shape, b)
        left = np.array([[b.x0, y, z] for z, y in prof], dtype=float)
        right = np.array([[b.x1, y, z] for z, y in prof], dtype=float)
        wl, wr = world(left), world(right)
        centre = np.vstack([wl, wr]).mean(0)
        mesh.polygon(wl, rgb, centre)
        mesh.polygon(wr, rgb, centre)
        n = len(prof)
        for i in range(n):
            j = (i + 1) % n
            mesh.polygon(np.array([wl[i], wl[j], wr[j], wr[i]]), rgb, centre)

        top_y = 8 * (g.y_top + 1)
        for (x, z) in g.top_studs:
            if (x, g.y_top + 1, z) in model.occ:
                continue
            cx, cz = 20 * x + 10, 20 * z + 10
            ang = np.linspace(0, 2 * math.pi, 9)[:-1]
            ring = np.stack([cx + 6 * np.cos(ang), np.zeros(8), cz + 6 * np.sin(ang)], 1)
            lo, hi = ring + [0, top_y, 0], ring + [0, top_y + 4, 0]
            inside = np.array([cx, top_y + 2, cz])
            mesh.polygon(hi, rgb * 1.04, inside)
            side = mesh.new_fid()
            for i in range(8):
                j = (i + 1) % 8
                mesh.polygon(np.array([lo[i], lo[j], hi[j], hi[i]]), rgb, inside, fid=side)
    return mesh


def _camera(az: float, el: float):
    a, e = math.radians(az), math.radians(el)
    v = np.array([math.sin(a) * math.cos(e), math.sin(e), -math.cos(a) * math.cos(e)])
    r = np.array([math.cos(a), 0.0, math.sin(a)])
    u = np.array([-math.sin(e) * math.sin(a), math.cos(e), math.sin(e) * math.cos(a)])
    return r, u, v


def render_view(mesh: Mesh, az: float, el: float, width: int, height: int, ss: int = 2) -> Image.Image:
    W, H = width * ss, height * ss
    img = np.tile(BACKGROUND, (H, W, 1))
    if not mesh.tris:
        return Image.fromarray(img.astype(np.uint8)).resize((width, height))
    tris = np.array(mesh.tris)
    colors = np.array(mesh.colors)
    fids = np.array(mesh.fids)
    r, u, v = _camera(az, el)

    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    ok = lengths > 1e-9
    tris, colors, fids, normals = tris[ok], colors[ok], fids[ok], normals[ok] / lengths[ok, None]
    visible = normals @ v > 1e-6
    tris, colors, fids, normals = tris[visible], colors[visible], fids[visible], normals[visible]

    shade = 0.42 + 0.48 * np.clip(normals @ LIGHT, 0, 1) + 0.18 * np.clip(normals @ v, 0, 1)
    shaded = np.clip(colors * shade[:, None], 0, 255)

    sx, sy, depth = tris @ r, tris @ u, tris @ v
    allx, ally = np.array(mesh.tris) @ r, np.array(mesh.tris) @ u
    margin = 0.06
    span_x = max(allx.max() - allx.min(), 1e-6)
    span_y = max(ally.max() - ally.min(), 1e-6)
    scale = min(W * (1 - 2 * margin) / span_x, H * (1 - 2 * margin) / span_y)
    cx, cy = (allx.max() + allx.min()) / 2, (ally.max() + ally.min()) / 2
    px = (sx - cx) * scale + W / 2
    py = H / 2 - (sy - cy) * scale

    zbuf = np.full((H, W), -np.inf)
    fbuf = np.zeros((H, W), dtype=np.int64)
    for i in range(len(px)):
        x0, x1, x2 = px[i]
        y0, y1, y2 = py[i]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        minx, maxx = max(int(math.floor(min(x0, x1, x2))), 0), min(int(math.ceil(max(x0, x1, x2))), W - 1)
        miny, maxy = max(int(math.floor(min(y0, y1, y2))), 0), min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if minx > maxx or miny > maxy:
            continue
        xs = np.arange(minx, maxx + 1) + 0.5
        ys = np.arange(miny, maxy + 1)[:, None] + 0.5
        w0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / area
        w1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / area
        w2 = 1 - w0 - w1
        eps = -1e-4
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not inside.any():
            continue
        d0, d1, d2 = depth[i]
        z = w0 * d0 + w1 * d1 + w2 * d2
        region = zbuf[miny:maxy + 1, minx:maxx + 1]
        m = inside & (z > region + 1e-3)
        region[m] = z[m]
        fbuf[miny:maxy + 1, minx:maxx + 1][m] = fids[i]
        img[miny:maxy + 1, minx:maxx + 1][m] = shaded[i]

    edge = np.zeros((H, W), dtype=bool)
    edge[:, :-1] |= fbuf[:, :-1] != fbuf[:, 1:]
    edge[:-1, :] |= fbuf[:-1, :] != fbuf[1:, :]
    edge[:, 1:] |= edge[:, :-1] & (fbuf[:, 1:] == 0)  # thicken silhouette onto background side
    img[edge] = img[edge] * 0.35 + np.array([20, 20, 25]) * 0.65
    out = Image.fromarray(img.astype(np.uint8))
    return out.resize((width, height), Image.Resampling.LANCZOS) if ss > 1 else out


def parse_view(name: str) -> tuple[str, float, float]:
    key = name.strip().lower()
    if key in VIEWS:
        return key, *VIEWS[key]
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", key)
    if m:
        return key, float(m.group(1)), float(m.group(2))
    raise ValueError(f"Unknown view '{name}'. Use {', '.join(VIEWS)} or 'azimuth,elevation' in degrees")


def render(model: Model, views: list[str] | None = None, size: int = 520,
           reference: str | None = None) -> bytes:
    views = views or DEFAULT_VIEWS
    parsed = [parse_view(v) for v in views]
    mesh = build_mesh(model)
    panels = []
    if reference:
        ref = Image.open(Path(reference).expanduser()).convert("RGB")
        ref.thumbnail((size, size))
        canvas = Image.new("RGB", (size, size), tuple(int(c) for c in BACKGROUND))
        canvas.paste(ref, ((size - ref.width) // 2, (size - ref.height) // 2))
        panels.append(("reference", canvas))
    for name, az, el in parsed:
        panels.append((name, render_view(mesh, az, el, size, size)))

    cols = 1 if len(panels) == 1 else 2 if len(panels) <= 4 else 3
    rows = math.ceil(len(panels) / cols)
    sheet = Image.new("RGB", (cols * size, rows * size), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, (name, panel) in enumerate(panels):
        ox, oy = (i % cols) * size, (i // cols) * size
        sheet.paste(panel, (ox, oy))
        draw.rectangle([ox, oy, ox + size - 1, oy + size - 1], outline=(255, 255, 255), width=2)
        draw.text((ox + 8, oy + 6), name, fill=(40, 40, 50))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
