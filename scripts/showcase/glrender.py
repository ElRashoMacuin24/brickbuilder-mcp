"""GPU renderer for showcase images and videos (needs an OpenGL 3.3+ GPU).

Same simplified brick shapes as the built-in preview renderer, but drawn with
OpenGL so a 40k-part model renders in well under a second, with perspective,
soft lighting and LEGO-instruction-style outlines. It can also animate a build:
parts appear course by course, dropping into place.

    uv run --with moderngl python scripts/showcase/glrender.py helms_deep out.png
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder import ldraw  # noqa: E402
from brickbuilder.model import Model  # noqa: E402
from brickbuilder.render import _profile  # noqa: E402

CACHE = REPO / "scripts" / "showcase" / "cache"


def load_model(name: str) -> Model:
    return Model.from_json(json.loads((REPO / "models" / f"{name}.json").read_text()))


def build_mesh(model: Model) -> dict[str, np.ndarray]:
    """Triangles with per-triangle colour, part level (y) and face id."""
    tris, cols, ys, fids = [], [], [], []
    flip = np.array([1.0, -1.0, 1.0])
    table = ldraw.colours()
    fid = 0
    ang = np.linspace(0, 2 * math.pi, 13)[:-1]

    def poly(pts, rgb, inside, y, f):
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if np.dot(n, pts.mean(0) - inside) < 0:
            pts = pts[::-1]
        for i in range(1, len(pts) - 1):
            tris.append((pts[0], pts[i], pts[i + 1]))
            cols.append(rgb)
            ys.append(y)
            fids.append(f)

    for pid, p in model.parts.items():
        g = model.geometry(pid)
        b = ldraw.part_bounds(p.part)
        col = table[p.color]
        rgb = np.array(col.rgb, dtype=float) / 255
        if col.alpha < 255:
            rgb = rgb * 0.6 + 0.4
        world = lambda local: (local @ g.matrix.T + g.pos) * flip  # noqa: E731
        prof = _profile(g.shape, b)
        wl = world(np.array([[b.x0, y, z] for z, y in prof], dtype=float))
        wr = world(np.array([[b.x1, y, z] for z, y in prof], dtype=float))
        centre = np.vstack([wl, wr]).mean(0)
        fid += 1
        poly(wl, rgb, centre, p.y, fid)
        fid += 1
        poly(wr, rgb, centre, p.y, fid)
        n = len(prof)
        for i in range(n):
            j = (i + 1) % n
            fid += 1
            poly(np.array([wl[i], wl[j], wr[j], wr[i]]), rgb, centre, p.y, fid)
        top_y = 8 * (g.y_top + 1)
        for (x, z) in g.top_studs:
            if (x, g.y_top + 1, z) in model.occ:
                continue
            cx, cz = 20 * x + 10, 20 * z + 10
            ring = np.stack([cx + 6 * np.cos(ang), np.zeros(12), cz + 6 * np.sin(ang)], 1)
            lo, hi = ring + [0, top_y, 0], ring + [0, top_y + 4, 0]
            inside = np.array([cx, top_y + 2, cz])
            fid += 1
            poly(hi, np.minimum(rgb * 1.06, 1), inside, p.y, fid)
            fid += 1
            for i in range(12):
                j = (i + 1) % 12
                poly(np.array([lo[i], lo[j], hi[j], hi[i]]), rgb, inside, p.y, fid)
    t = np.array(tris, dtype=np.float32)
    return {"tris": t, "cols": np.array(cols, np.float32), "ys": np.array(ys, np.float32),
            "fids": np.array(fids, np.int32)}


def mesh_for(name: str) -> dict[str, np.ndarray]:
    CACHE.mkdir(parents=True, exist_ok=True)
    src = REPO / "models" / f"{name}.json"
    f = CACHE / f"{name}.npz"
    if f.exists() and f.stat().st_mtime > src.stat().st_mtime:
        return dict(np.load(f))
    mesh = build_mesh(load_model(name))
    np.savez(f, **mesh)
    return mesh


VERT = """
#version 330
uniform mat4 mvp;
uniform float level;      // parts with y <= level are placed; the next course drops in
uniform float drop;
in vec3 in_pos; in vec3 in_norm; in vec3 in_col; in float in_y; in float in_fid;
out vec3 v_norm; out vec3 v_col; out vec3 v_pos; flat out int v_fid;
void main() {
    vec3 p = in_pos;
    float t = in_y - level;
    if (t > 1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    if (t > 0.0) p.y += t * t * drop;
    v_norm = in_norm; v_col = in_col; v_pos = p; v_fid = int(in_fid);
    gl_Position = mvp * vec4(p, 1.0);
}
"""
FRAG = """
#version 330
uniform vec3 eye; uniform vec3 light;
in vec3 v_norm; in vec3 v_col; in vec3 v_pos; flat in int v_fid;
layout(location=0) out vec4 f_col;
layout(location=1) out int f_id;
void main() {
    vec3 n = normalize(v_norm);
    vec3 v = normalize(eye - v_pos);
    float diff = max(dot(n, light), 0.0);
    float hemi = 0.5 + 0.5 * n.y;
    float spec = pow(max(dot(reflect(-light, n), v), 0.0), 24.0) * 0.12;
    vec3 c = v_col * (0.30 + 0.22 * hemi + 0.58 * diff) + spec;
    f_col = vec4(pow(c, vec3(0.95)), 1.0);
    f_id = v_fid;
}
"""


def _look_at(eye, target, up=(0, 1, 0)):
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def _perspective(fovy, aspect, near, far):
    f = 1 / math.tan(math.radians(fovy) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = f / aspect, f
    m[2, 2], m[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


_CTX = None


def _context() -> moderngl.Context:
    """One shared OpenGL context: several renderers can then be used side by side."""
    global _CTX
    if _CTX is None:
        _CTX = moderngl.create_standalone_context()
    return _CTX


class Renderer:
    def __init__(self, mesh: dict[str, np.ndarray], width: int, height: int, ss: int = 2,
                 background=((236, 239, 243), (206, 212, 221))):
        self.w, self.h, self.ss = width, height, ss
        self.ctx = _context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        # LDraw is left-handed (x right, y down, z back); flip z for right-handed OpenGL
        # so the model isn't mirrored. The reflection reverses winding, hence the minus.
        t = mesh["tris"] * np.array([1, 1, -1], np.float32)
        n = -np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        k = len(t)
        data = np.concatenate([
            t.reshape(-1, 3),
            np.repeat(n, 3, 0),
            np.repeat(mesh["cols"], 3, 0),
            np.repeat(mesh["ys"], 3)[:, None],
            np.repeat(mesh["fids"].astype(np.float32), 3)[:, None],
        ], 1).astype("f4")
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        vbo = self.ctx.buffer(data.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [(vbo, "3f 3f 3f 1f 1f", "in_pos", "in_norm", "in_col",
                                                      "in_y", "in_fid")])
        W, H = width * ss, height * ss
        self.col_tex = self.ctx.texture((W, H), 4)
        self.id_tex = self.ctx.texture((W, H), 1, dtype="i4")
        self.depth = self.ctx.depth_renderbuffer((W, H))
        self.fbo = self.ctx.framebuffer([self.col_tex, self.id_tex], self.depth)
        pts = t.reshape(-1, 3)
        self.lo, self.hi = pts.min(0), pts.max(0)
        self.centre = (self.lo + self.hi) / 2
        self.radius = float(np.linalg.norm(self.hi - self.lo) / 2)
        self.ymax = float(mesh["ys"].max())
        top, bot = np.array(background[0], float), np.array(background[1], float)
        grad = np.linspace(0, 1, H)[:, None, None]
        self.bg = (top * (1 - grad) + bot * grad).repeat(W, 1)
        self.triangles = k

    def frame(self, az: float, el: float, zoom: float = 1.0, level: float | None = None,
              target=None, fov: float = 30.0) -> Image.Image:
        W, H = self.w * self.ss, self.h * self.ss
        target = self.centre if target is None else np.asarray(target, float)
        a, e = math.radians(az), math.radians(el)
        d = self.radius / math.sin(math.radians(fov) / 2) * 0.92 / zoom
        if W > H:
            d *= 0.85
        eye = target + d * np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
        mvp = _perspective(fov, W / H, d * 0.05, d * 4) @ _look_at(eye, target)
        self.prog["mvp"].write(mvp.T.astype("f4").tobytes())
        self.prog["eye"].value = tuple(eye)
        lv = np.array([-0.45, 1.0, 0.65])
        self.prog["light"].value = tuple(lv / np.linalg.norm(lv))
        self.prog["level"].value = self.ymax + 2 if level is None else level
        self.prog["drop"].value = 60.0
        self.fbo.use()
        self.ctx.clear(0, 0, 0, 0, depth=1.0)
        self.id_tex.write(np.zeros((H, W), np.int32).tobytes())
        self.vao.render()
        rgba = np.frombuffer(self.col_tex.read(), np.uint8).reshape(H, W, 4)[::-1].astype(float)
        ids = np.frombuffer(self.id_tex.read(), np.int32).reshape(H, W)[::-1]
        img = np.where((ids > 0)[..., None], rgba[..., :3], self.bg)
        edge = np.zeros((H, W), bool)
        edge[:, :-1] |= ids[:, :-1] != ids[:, 1:]
        edge[:-1, :] |= ids[:-1, :] != ids[1:, :]
        img[edge] = img[edge] * 0.35 + np.array([22, 22, 28]) * 0.65
        out = Image.fromarray(img.clip(0, 255).astype(np.uint8))
        return out.resize((self.w, self.h), Image.Resampling.LANCZOS) if self.ss > 1 else out


if __name__ == "__main__":
    name, out = sys.argv[1], sys.argv[2]
    az = float(sys.argv[3]) if len(sys.argv) > 3 else -35
    el = float(sys.argv[4]) if len(sys.argv) > 4 else 30
    r = Renderer(mesh_for(name), 1600, 1000)
    r.frame(az, el).save(out)
    print(f"{r.triangles} triangles -> {out}")
