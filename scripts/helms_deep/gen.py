"""Generate a minifig-scale Helm's Deep as a brickbuilder model.

Grid: x right, z back (z=0 front / the plain), y plates up.
Design is done as a colour voxel grid G[x, y, z], hollowed to a shell,
then packed into bricks (3-plate courses) and plates.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder import catalog, ldraw  # noqa: E402
from brickbuilder.model import Model, compute_geometry, Placement  # noqa: E402

rng = np.random.default_rng(7)
W, HMAX, D = 192, 180, 160
G = np.zeros((W, HMAX, D), np.uint8)

# colour indices -> LDraw codes
AIR, WALL, ROCK, DIRT, GRAVEL, BLACK, WOOD, MOSS, WATER, CRYST, CRYSTP, DWOOD, ROCK2, WALL2, PAVE = range(15)
LDRAW = {WALL: 71, ROCK: 72, DIRT: 28, GRAVEL: 19, BLACK: 0, WOOD: 70, MOSS: 378, WATER: 43,
         CRYST: 47, CRYSTP: 52, DWOOD: 308, ROCK2: 71, WALL2: 72, PAVE: 71}

X, Z = np.meshgrid(np.arange(W), np.arange(D), indexing="ij")
XC, ZC = X + 0.5, Z + 0.5
CX, CZ = 128.0, 84.0
DX, DZ = XC - CX, ZC - CZ
R = np.hypot(DX, DZ)
TH = np.degrees(np.arctan2(DZ, DX)) % 360


def noise(cells_x, cells_z, seed):
    r = np.random.default_rng(seed).random((cells_z, cells_x)).astype(np.float32)
    img = Image.fromarray(r, mode="F").resize((W, D), Image.BICUBIC)
    return np.asarray(img).T  # (W, D)


def fnoise(seed):
    a = noise(6, 5, seed) * 0.6 + noise(14, 12, seed + 1) * 0.3 + noise(40, 34, seed + 2) * 0.1
    return (a - a.min()) / (a.max() - a.min())


def fill(mask, y0, top, col):
    """Fill columns in mask from y0 (inclusive) to top (exclusive); top int or 2-D array."""
    top = np.broadcast_to(np.asarray(top), mask.shape)
    for y in range(y0, int(top[mask].max()) if mask.any() else y0):
        sel = mask & (top > y)
        G[:, y, :][sel] = col


def carve(mask, y0, y1=HMAX):
    top = np.broadcast_to(np.asarray(y1), mask.shape)
    for y in range(y0, int(top[mask].max()) if mask.any() else y0):
        sel = mask & (top > y)
        G[:, y, :][sel] = AIR


def paint(mask, y0, y1, col, only_solid=True):
    for y in range(y0, y1):
        sel = mask & (G[:, y, :] != AIR) if only_solid else mask
        G[:, y, :][sel] = col


def q3(h):
    """Quantise heights to course tops (2 + 3k)."""
    return (np.round((h - 2) / 3) * 3 + 2).astype(int)


# ------------------------------------------------------------------ ground
n_ground = fnoise(11)
G[:, 0, :] = DIRT
ground = np.where(n_ground > 0.62, GRAVEL, DIRT)
ground = np.where((n_ground < 0.18), ROCK, ground)
G[:, 1, :] = ground

# ------------------------------------------------------------------ rock massifs
n1, n2 = fnoise(21), fnoise(31)
XW = X + (fnoise(51) - 0.5) * 16
ZW = Z + (fnoise(61) - 0.5) * 16
H = np.zeros((W, D))
# back wall of the Deep
H = np.maximum(H, np.where(ZW >= 134, np.minimum(60 + (ZW - 134) * 5, 138), 0))
# left cliff where the Deeping Wall ends
H = np.maximum(H, np.where((XW <= 20) & (Z >= 66), np.minimum(45 + (20 - XW) * 6 + (Z - 66) * 0.6, 130), 0))
H = np.maximum(H, np.where((XW <= 12) & (Z < 66), np.maximum(0, 12 + (12 - XW) * 5 - (66 - Z) * 0.8), 0))
# right massif against the Hornburg
right_edge = np.where(Z >= 60, 160, 168) + (fnoise(71) - 0.5) * 8
H = np.maximum(H, np.where(X >= right_edge, np.minimum(45 + (X - right_edge) * 6, 140) * np.clip((Z - 6) / 36, 0.25, 1), 0))
# massif behind the keep (tower and great hall are cut into it)
back_edge = 104 + np.maximum(0, 100 - X) * 0.8 + np.where(X > 100, (fnoise(81) - 0.5) * 4, (fnoise(81) - 0.5) * 12)
back_edge = np.where((X >= 118) & (X <= 154), np.maximum(back_edge, 104), back_edge)
H = np.maximum(H, np.where((X >= 78) & (Z >= back_edge), 112 + (Z - back_edge) * 1.5, 0))
H = np.minimum(H, 140)
rock_top = np.where(H > 2, q3(H + (n1 - 0.5) * 30), 0)
# rocky skirt under the Hornburg's outer wall ("Helm's Gate")
ring_ang = (TH >= 150) | (TH <= 10)
skirt = (R >= 33) & (R < 40 + n2 * 4) & ((TH >= 195) | (TH <= 10))
skirt_top = q3(2 + (40 - R).clip(0) * 2.2 + n2 * 8)
rock_top = np.where(skirt, np.maximum(rock_top, skirt_top), rock_top)
# a few boulders on the plain and in the coomb
for bx, bz, br, bh in [(30, 20, 3.5, 8), (58, 36, 2.5, 5), (150, 18, 3, 8), (22, 48, 4, 11), (70, 110, 3, 8),
                        (40, 90, 2.5, 5), (86, 128, 3, 8), (176, 30, 4, 11), (9, 30, 3, 8)]:
    m = np.hypot(XC - bx, ZC - bz) < br
    rock_top = np.where(m, np.maximum(rock_top, bh), rock_top)
rock_top = np.minimum(rock_top, 146)
patch = fnoise(41)
fill(rock_top > 2, 2, rock_top, ROCK)
# lighter rock bands for texture
for y in range(2, HMAX):
    sl = G[:, y, :]
    band = (sl == ROCK) & (((patch + ((y - 2) // 3) * 0.039) % 0.25) < 0.035)
    sl[band] = ROCK2

# ------------------------------------------------------------------ carve fortress zones
ring_zone = ring_ang & (R < 35.5)
carve(ring_zone, 2)
terrace_zone = (R < 34) & ~ring_ang & (Z < 104)
carve(terrace_zone | (R < 22), 2)

# ------------------------------------------------------------------ Deeping Wall geometry
P0 = np.array([94.5, 81.0])
P1 = np.array([2.0, 125.0])
nsmp = 600
t = np.linspace(0, 1, nsmp)
dirv = (P1 - P0) / np.linalg.norm(P1 - P0)
nf = np.array([dirv[1], -dirv[0]])  # front normal (towards the plain, -z)
if nf[1] > 0:
    nf = -nf
bow = 5.0
pts = P0 + np.outer(t, P1 - P0) + np.outer(np.sin(np.pi * t) * bow, nf)
seg = np.diff(pts, axis=0)
seglen = np.hypot(seg[:, 0], seg[:, 1])
U = np.concatenate([[0], np.cumsum(seglen)])
tang = np.vstack([seg, seg[-1:]])
tang /= np.hypot(tang[:, 0], tang[:, 1])[:, None]
norm = np.stack([tang[:, 1], -tang[:, 0]], 1)
norm[norm[:, 1] > 0] *= -1
flatP = np.stack([XC.ravel(), ZC.ravel()], 1)
best_d2 = np.full(len(flatP), np.inf)
best_i = np.zeros(len(flatP), int)
for i0 in range(0, nsmp, 50):
    d2 = ((flatP[:, None, :] - pts[None, i0:i0 + 50, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    v = d2[np.arange(len(flatP)), j]
    better = v < best_d2
    best_d2[better] = v[better]
    best_i[better] = i0 + j[better]
rel = flatP - pts[best_i]
DW_D = (rel * norm[best_i]).sum(1).reshape(W, D)  # + = front
DW_U = (U[best_i] + (rel * tang[best_i]).sum(1)).reshape(W, D)
DW_L = U[-1]
outside_ring = ~(ring_ang & (R < 34.5))
dw_core = (np.abs(DW_D) <= 4.0) & (DW_U >= -1) & (DW_U <= DW_L + 2) & outside_ring
u_exit = DW_U[dw_core & (R >= 34.5)].min()
dw_top = (35 + np.clip(np.floor(u_exit + 5.5 - DW_U) * 2, 0, 9)).astype(int)
dw_top = np.minimum(dw_top, 44)

# ------------------------------------------------------------------ Hornburg: fills
# courtyard ring (between outer and inner walls), floor top y=24
court = ring_ang & (R >= 21.5) & (R < 29.5)
fill(court, 2, 23, WALL)
# upper terrace (inside inner wall + back terrace up to the great hall), top y=45
terr = (R < 17.5) | terrace_zone
fill(terr, 2, 44, WALL)

# outer wall
outer = ring_ang & (R >= 29) & (R < 34.2)
fill(outer, 2, 44, WALL)
fill(ring_ang & (R >= 34.2) & (R < 35.6), 2, 11, WALL)   # battered base
fill(ring_ang & (R >= 35.6) & (R < 36.8), 2, 5, WALL)
THU = np.where(TH >= 150, TH, TH + 360)
arc_s = np.floor(33.5 * np.radians(THU)).astype(int)
dw_join = (np.abs(DW_D) <= 4.5) & (DW_U < u_exit + 3)
o_par = ring_ang & (R >= 32.7) & (R < 34.2) & ~dw_join
fill(o_par, 44, 50, WALL)
fill(o_par & (arc_s % 3 != 2), 50, 53, WALL)
# arrow slits on the outer face
slit = ring_ang & (R >= 33.2) & (R < 34.2) & (arc_s % 9 == 4)
paint(slit, 29, 35, BLACK)
paint(slit, 14, 20, BLACK)

# inner wall
in_ang = (TH >= 160) | (TH <= 20)
inner = in_ang & (R >= 17) & (R < 22.2)
fill(inner, 2, 65, WALL)
arc_i = np.floor(21.5 * np.radians(np.where(TH >= 160, TH, TH + 360))).astype(int)
i_par = in_ang & (R >= 20.7) & (R < 22.2)
fill(i_par, 65, 71, WALL)
fill(i_par & (arc_i % 3 != 2), 71, 74, WALL)
slit_i = in_ang & (R >= 21.2) & (R < 22.2) & (arc_i % 8 == 3)
paint(slit_i, 50, 56, BLACK)
paint(slit_i, 29, 35, BLACK)

# ------------------------------------------------------------------ Deeping Wall
dw = dw_core
fill(dw, 2, dw_top, WALL)
fill((DW_D > 4) & (DW_D <= 5.5) & (DW_U >= 0) & (DW_U <= DW_L + 2) & outside_ring, 2, 11, WALL)
fill((DW_D > 5.5) & (DW_D <= 6.7) & (DW_U >= 0) & (DW_U <= DW_L + 2) & outside_ring, 2, 5, WALL)
dpar = (DW_D >= 2.7) & (DW_D <= 4.2) & (DW_U >= u_exit + 6) & (DW_U <= DW_L + 2) & outside_ring
fill(dpar, 35, 41, WALL)
fill(dpar & (np.floor(DW_U).astype(int) % 3 != 2), 41, 44, WALL)
# back stairs down into the Deep
for ua in (0.30 * DW_L, 0.72 * DW_L):
    st = (DW_D < -4) & (DW_D >= -7.6) & (DW_U >= ua) & (DW_U < ua + 17)
    fill(st, 2, (35 - np.floor(DW_U - ua).astype(int) * 2).clip(2), WALL)
# culvert
uc = 0.5 * DW_L
cul = (np.abs(DW_U - uc) < 2.6) & (np.abs(DW_D) <= 7)
carve(cul, 2, 11)
carve((np.abs(DW_U - uc) < 1.6) & (np.abs(DW_D) <= 7), 11, 14)
bars = cul & (DW_D >= 3.2) & (DW_D <= 4.4) & (np.floor(DW_U - uc).astype(int) % 2 == 0)
fill(bars, 2, 11, BLACK)
# weathering / moss low on the front
mossy = (DW_D > 2) & (rng.random((W, D)) < 0.25)
paint(mossy & dw, 2, 8, MOSS)
paint(mossy & ((DW_D > 4) & (DW_D <= 6.7)), 2, 8, MOSS)

# ------------------------------------------------------------------ gatehouse, landing, causeway
gh = (X >= 100) & (X <= 123) & (Z >= 42) & (Z <= 70) & (R >= 29)
carve(gh, 44)
fill(gh, 2, 44, WALL)
t1 = (X >= 100) & (X <= 108) & (Z >= 42) & (Z <= 56)
t2 = (X >= 115) & (X <= 123) & (Z >= 42) & (Z <= 54)
for tw, (xa, xb, za, zb) in ((t1, (100, 108, 42, 56)), (t2, (115, 123, 42, 54))):
    fill(tw, 44, 56, WALL)
    rim = tw & ((X == xa) | (X == xb) | (Z == za))
    fill(rim, 56, 62, WALL)
    per = np.where(Z == za, X, Z)
    fill(rim & (per % 3 != 1), 62, 65, WALL)
    paint(tw & ((Z == za) | (X == xa) | (X == xb)) & (((X + Z) % 4) == 0), 47, 53, BLACK)
gmid = (X >= 109) & (X <= 114) & (Z >= 42) & (Z <= 43)
fill(gmid, 44, 50, WALL)
fill(gmid & (X % 2 == 0), 50, 53, WALL)
# gate passage and gate
passage = (X >= 109) & (X <= 114) & (Z >= 42) & (Z <= 63)
carve(passage, 23, 41)
fill((X >= 109) & (X <= 114) & (Z == 46), 23, 38, WOOD)
fill((X >= 109) & (X <= 114) & (Z == 46) & (X % 2 == 0), 26, 29, DWOOD)
fill((X >= 109) & (X <= 114) & (Z == 46) & (X % 2 == 0), 32, 35, DWOOD)
# landing in front of the gate
land = ((X >= 103) & (X <= 124) & (Z >= 35) & (Z <= 41)) | ((X >= 115) & (X <= 124) & (Z >= 29) & (Z <= 41))
fill(land, 2, 23, WALL)
lpar = land & (((Z == 35) & (X < 115)) | (X == 103) | ((X == 115) & (Z < 35)) | ((Z == 29) & (X < 119)))
fill(lpar, 23, 29, WALL)
fill(lpar & (((X + Z) % 3) != 2), 29, 32, WALL)
# causeway ramp along the Hornburg's front
ramp = (R >= 46.5) & (R < 53.5) & (TH >= 259) & (TH <= 320)
tr = np.clip((TH - 262) / 56, 0, 1)
ramp_top = np.round(23 - 21 * tr).astype(int)
carve(ramp, 2)
fill(ramp & (ramp_top > 2), 2, ramp_top, WALL)
rpar = ramp & (R >= 52.2)
fill(rpar & (ramp_top > 2), 2, ramp_top + 5, WALL)
paint(ramp & (rng.random((W, D)) < 0.2), 2, 8, MOSS)

# ------------------------------------------------------------------ inner doorway + stair up to the terrace
door_x = (X >= 126) & (X <= 129)
carve(door_x & (Z >= 62) & (Z <= 67), 23, 35)
trench = door_x & (Z >= 68) & (Z <= 79)
carve(trench, 23, 44)
fill(trench, 23, np.minimum(23 + (Z - 67) * 2, 44), WALL)

# courtyard stairs up to the outer walkway (against the outer wall's inner face)
for ths in (292.0, 165.0):
    a = 27.8 * np.radians((THU - ths) % 360)
    st = ring_ang & (R >= 26.3) & (R < 29) & (a >= 0) & (a < 11)
    fill(st, 23, np.minimum(23 + np.floor(a).astype(int) * 2 + 2, 44), WALL)
# terrace stair up to the inner walkway
a = 15.6 * np.radians((TH - 205) % 360)
st = (R >= 14.2) & (R < 17.2) & (a >= 0) & (a < 11) & in_ang
fill(st, 44, np.minimum(44 + np.floor(a).astype(int) * 2 + 2, 65), WALL)
a = 15.6 * np.radians((TH - 318) % 360)
st = (R >= 14.2) & (R < 17.2) & (a >= 0) & (a < 11) & in_ang
fill(st, 44, np.minimum(44 + np.floor(a).astype(int) * 2 + 2, 65), WALL)

# ------------------------------------------------------------------ great hall (cut into the cliff)
hall_x0, hall_x1 = 120, 151
# grand stair from the terrace (45) to the hall floor (51)
gst = (X >= 124) & (X <= 147) & (Z >= 98) & (Z <= 103)
fill(gst, 44, 44 + (Z - 97), WALL)
fac = (X >= hall_x0) & (X <= hall_x1) & (Z >= 104) & (Z <= 105)
carve(fac, 44)
fill(fac, 44, 83, WALL)
cornice = (X >= hall_x0 - 1) & (X <= hall_x1 + 1) & (Z >= 103) & (Z <= 105)
fill(cornice & (Z == 103), 80, 83, WALL)
fill(cornice, 83, 84, WALL)
room = (X >= 122) & (X <= 149) & (Z >= 106) & (Z <= 123)
carve(room, 50, 77)
fill(room, 44, 50, WALL)
arch_parts = []
for i in range(5):
    ox = 122 + 6 * i
    carve((X >= ox) & (X <= ox + 3) & (Z >= 104) & (Z <= 105), 50, 56)
    for zz in (104, 105):
        arch_parts.append(("3307", WALL, ox - 1, 56, zz, 0))
for k in range(10):
    wx = 122 + 3 * k
    carve((X == wx) & (Z >= 104) & (Z <= 105), 65, 74)
for px in (127, 133, 139, 145):
    for pz in (111, 118):
        fill((X >= px) & (X <= px + 1) & (Z >= pz) & (Z <= pz + 1), 50, 77, WALL)
# dark interior back wall so the arcade reads
paint((X >= 122) & (X <= 149) & (Z == 124), 50, 77, BLACK)

# ------------------------------------------------------------------ Hornburg tower
tx0, tx1, tz0, tz1 = 102, 113, 97, 108
tw = (X >= tx0) & (X <= tx1) & (Z >= tz0) & (Z <= tz1)
ex = np.minimum(X - tx0, tx1 - X)
ez = np.minimum(Z - tz0, tz1 - Z)
tw &= (ex + ez) >= 2
carve(tw, 2)
fill(tw, 2, 140, WALL)
rim = tw & ((ex == 0) | (ez == 0) | ((ex + ez) == 2))
fill(rim, 140, 146, WALL)
fill(rim & (((X + Z) % 3) != 0), 146, 149, WALL)
for (px0, px1) in ((102, 105), (110, 113)):
    prong = (X >= px0) & (X <= px1) & (Z >= tz0) & (Z <= tz0 + 3) & tw
    fill(prong, 140, 161, WALL)
    inner_p = prong & (X > px0) & (X < px1) & (Z > tz0) & (Z < tz0 + 3)
    fill(inner_p, 161, 167, WALL)
# windows and slits on the exposed faces
face = tw & ((ez == 0) | (ex == 0))
for y0 in (53, 83, 113):
    paint(face & (((ez == 0) & (np.abs(X - 107.5) < 1)) | ((ex == 0) & (np.abs(Z - 102.5) < 1))), y0, y0 + 12, BLACK)
for y0 in (68, 98, 125):
    paint(face & (((ez == 0) & ((X == 104) | (X == 111))) | ((ex == 0) & (Z == 100))), y0, y0 + 6, BLACK)
carve(tw & (Z == tz0) & (X >= 106) & (X <= 109), 44, 56)
paint(tw & (Z == tz0 + 1) & (X >= 106) & (X <= 109), 44, 56, BLACK)
# walkway from terrace edge to the tower door
fill((X >= 105) & (X <= 110) & (Z >= 92) & (Z < tz0) & ~ring_ang, 2, 44, WALL)

# ------------------------------------------------------------------ citadel on the upper terrace
cx0, cx1, cz0, cz1 = 114, 126, 85, 96
cit = (X >= cx0) & (X <= cx1) & (Z >= cz0) & (Z <= cz1)
carve(cit, 44)
fill(cit, 44, 71, WALL)
crim = cit & ((X == cx0) | (X == cx1) | (Z == cz0) | (Z == cz1))
fill(crim, 71, 74, WALL)
fill(crim & (((X + Z) % 2) == 0), 74, 77, WALL)
carve(cit & (Z == cz0) & (X >= 119) & (X <= 121), 44, 56)
paint(cit & (Z == cz0 + 1) & (X >= 119) & (X <= 121), 44, 56, BLACK)
for y0 in (59,):
    paint(cit & (Z == cz0) & ((X == 116) | (X == 124)), y0, y0 + 9, BLACK)
    paint(cit & ((X == cx0) | (X == cx1)) & ((Z == 88) | (Z == 93)), y0, y0 + 9, BLACK)
# steps up to the citadel door
fill((X >= 118) & (X <= 122) & (Z >= 82) & (Z <= 84), 44, 44 + (Z - 81) * 1, WALL)
# stable / armoury doors in the courtyard (base of the inner wall)
sdoor = in_ang & (R >= 21.2) & (R < 22.2) & ((arc_i % 14) >= 6) & ((arc_i % 14) <= 8) & (TH > 280)
paint(sdoor, 23, 35, BLACK)
sdoor2 = in_ang & (R >= 21.2) & (R < 22.2) & ((arc_i % 14) >= 6) & ((arc_i % 14) <= 8) & (TH < 20)
paint(sdoor2, 23, 35, BLACK)

# ------------------------------------------------------------------ Glittering Caves and the Deeping Stream
cave = (X >= 50) & (X <= 60) & (Z >= 128) & (Z <= 158)
cave_top = 17 - (np.abs(XC - 55.5) > 3.5) * 3 - (np.abs(XC - 55.5) > 4.5) * 3
carve(cave, 2, cave_top)
for _ in range(40):
    cx, cz = rng.integers(51, 60), rng.integers(136, 158)
    h = int(rng.integers(3, 12))
    col = CRYST if rng.random() < 0.6 else CRYSTP
    if G[cx, 2, cz] == AIR:
        G[cx, 2:2 + h, cz] = col
# stream: cave mouth -> culvert -> out onto the plain
cpt = pts[np.argmin(np.abs(U - uc))]
path = np.array([[55.5, 132], [56, 118], [cpt[0] + 6, cpt[1] + 14], cpt, cpt - nf * 14, [cpt[0] - 20, 20], [cpt[0] - 26, 0]])
sp = np.concatenate([np.linspace(path[i], path[i + 1], 80) for i in range(len(path) - 1)])
dmin = np.min(np.hypot(XC[..., None] - sp[:, 0], ZC[..., None] - sp[:, 1]), axis=-1)
stream = dmin < 1.6
G[:, 1, :][stream & (G[:, 1, :] != AIR)] = WATER
G[:, 1, :][stream] = WATER
carve(stream & (G[:, 2, :] == ROCK), 2, 3)

# ------------------------------------------------------------------ weathering on fortress walls
for y0 in range(2, 59, 3):
    r = rng.random((W, D)) < (0.10 if y0 < 10 else 0.03)
    for y in range(y0, y0 + 3):
        sl = G[:, y, :]
        sl[(sl == WALL) & r] = MOSS if y0 < 10 else WALL2

print("voxels solid:", int((G != AIR).sum()))

# ------------------------------------------------------------------ hollow the solid mass into a shell
np.save(Path(__file__).with_name("design.npy"), G)
solid = G != AIR
air = ~solid
up = air.copy()
for k in range(1, 5):
    up[:, :-k, :] |= air[:, k:, :]
    up[:, HMAX - k:, :] = True
up[:, 3:, :] |= air[:, 2:-1, :]  # underside of overhangs
def dilate(a, r):
    pad = np.pad(a, ((r, r), (0, 0), (r, r)), constant_values=True)
    out = np.zeros_like(a)
    for ox in range(-r, r + 1):
        for oz in range(-r, r + 1):
            out |= pad[r + ox:r + ox + W, :, r + oz:r + oz + D]
    return out
rockish = (G == ROCK) | (G == ROCK2)
keep = (rockish & dilate(up, 1)) | (~rockish & dilate(up, 2))
pill = ((X % 12) < 2) & ((Z % 12) < 2)
keep |= pill[:, None, :]
keep[:, :2, :] = True
G[solid & ~keep] = AIR
print("voxels after hollowing:", int((G != AIR).sum()))

# ------------------------------------------------------------------ 45-degree slopes on rock ledges
slopes = []
used = np.zeros((W, D), bool)
isrock = lambda v: (v == ROCK) | (v == ROCK2)
solid = G != AIR
topy = np.where(solid.any(1), HMAX - 1 - np.argmax(solid[:, ::-1, :], axis=1), -1)
dirs = [(0, -1, 0), (0, 1, 180), (1, 0, 90), (-1, 0, 270)]
cand = np.argwhere((topy >= 4) & ((topy - 4) % 3 == 0))
rng2 = np.random.default_rng(5)
rng2.shuffle(cand)
for x, z in cand:
    if used[x, z]:
        continue
    y = topy[x, z]
    col = G[x, y, z]
    if not isrock(col) or not all(isrock(G[x, yy, z]) for yy in (y - 1, y - 2)):
        continue
    for ddx, ddz, rot in dirs:
        ox, oz, ix, iz = x + ddx, z + ddz, x - ddx, z - ddz
        if not (0 <= ox < W and 0 <= oz < D and 0 <= ix < W and 0 <= iz < D):
            continue
        if used[ix, iz] or topy[ox, oz] > y - 3 or topy[ix, iz] < y:
            continue
        if not all(isrock(G[ix, yy, iz]) for yy in (y - 2, y - 1, y)):
            continue
        # the slope's high row must not be the top of a thin spike; keep it simple
        mx, mz = min(x, ix), min(z, iz)
        slopes.append(("3040b", int(G[ix, y, iz]), int(mx), int(y - 2), int(mz), rot, (x, z), (ix, iz)))
        used[x, z] = used[ix, iz] = True
        break
print("slopes:", len(slopes))
import pickle as _pk
_pk.dump(slopes, open(Path(__file__).with_name("slopes_raw.pkl"), "wb"))

np.save(Path(__file__).with_name("grid.npy"), G)
import pickle  # noqa: E402
pickle.dump(arch_parts, open(Path(__file__).with_name("specials.pkl"), "wb"))
