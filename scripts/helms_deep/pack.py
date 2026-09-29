"""Pack the voxel grid into bricks/plates and save as a brickbuilder model."""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder import catalog  # noqa: E402
from brickbuilder.model import Model, Placement, compute_geometry  # noqa: E402

HERE = Path(__file__).parent
G = np.load(HERE / "grid.npy")
W, HMAX, D = G.shape
LDRAW = {1: 71, 2: 72, 3: 28, 4: 19, 5: 0, 6: 70, 7: 378, 8: 43, 9: 47, 10: 52, 11: 308, 12: 71, 13: 72, 14: 71}
specials = pickle.load(open(HERE / "specials.pkl", "rb"))
specials += [("30528", 1, 108, 23, 42, 0), ("3455", 1, 125, 35, 62, 0)]
specials += [s for s in pickle.load(open(HERE / "extra_specials.pkl", "rb"))] if (HERE / "extra_specials.pkl").exists() else []

DESIGN = np.load(HERE / "design.npy")
G0 = G.copy()


def build(G):
    global m, occ_id
    m = Model(sys.argv[1] if len(sys.argv) > 1 else "helms_deep",
              "Minifig-scale Helm's Deep: Deeping Wall, Hornburg rings, gatehouse, causeway, tower, great hall, Glittering Caves")

    # --- specials first; clear their cells from the voxel grid
    for pid, col, x, y, z, rot in specials:
        code = LDRAW.get(col, col) if isinstance(col, int) and col < 20 else col
        g = compute_geometry(Placement(0, catalog.lookup(pid)[0], 0, x, y, z, rot))
        for (cx, cy, cz) in g.cells:
            if 0 <= cx < W and 0 <= cz < D and 0 <= cy < HMAX:
                G[cx, cy, cz] = 0
        m.add(pid, code, x, y, z, rot)

    BRICKS = [("3005", 1, 1), ("3004", 2, 1), ("3622", 3, 1), ("3010", 4, 1), ("3009", 6, 1), ("3008", 8, 1),
              ("3003", 2, 2), ("3002", 3, 2), ("3001", 4, 2), ("2456", 6, 2), ("3007", 8, 2), ("3006", 10, 2)]
    PLATES = [("3024", 1, 1), ("3023b", 2, 1), ("3623", 3, 1), ("3710", 4, 1), ("3666", 6, 1), ("3460", 8, 1),
              ("3022", 2, 2), ("3021", 3, 2), ("3020", 4, 2), ("3795", 6, 2), ("3034", 8, 2),
              ("3031", 4, 4), ("3032", 6, 4), ("3035", 8, 4), ("3958", 6, 6), ("3030", 10, 4), ("3036", 8, 6)]

    global occ_id
    occ_id = np.full((W, HMAX, D), -1, np.int32)  # part id per cell, for seam checks
    for pid, p in m.parts.items():
        for (cx, cy, cz) in m.geometry(pid).cells:
            if 0 <= cx < W and 0 <= cz < D and 0 <= cy < HMAX:
                occ_id[cx, cy, cz] = pid


    def cands(sizes):
        out = []
        for pid, sx, sz in sizes:
            out.append((pid, sx, sz, 0))
            if sx != sz:
                out.append((pid, sz, sx, 90))
        out.sort(key=lambda c: -c[1] * c[2])
        return out


    BC, PC = cands(BRICKS), cands(PLATES)


    def pack_level(cells, y, h, cl, prefer_x):
        """cells: (W, D) int array of LDraw colour codes (0 = empty). Places parts of height h at y."""
        todo = cells.copy()
        below = occ_id[:, y - 1, :] if y > 0 else np.full((W, D), -1, np.int32)
        xs, zs = np.nonzero(todo)
        order = np.lexsort((xs, zs)) if prefer_x else np.lexsort((zs, xs))
        placed = 0
        for k in order:
            x, z = int(xs[k]), int(zs[k])
            col = todo[x, z]
            if col == 0:
                continue
            best = None
            for pid, wx, wz, rot in cl:
                if x + wx > W or z + wz > D:
                    continue
                blk = todo[x:x + wx, z:z + wz]
                if not (blk == col).all():
                    continue
                area = wx * wz
                if best is not None and area < best[0][1]:
                    break
                # aligned seams at the far edges (weak stacked joints)
                al = 0
                if x + wx < W:
                    a, b = below[x + wx - 1, z:z + wz], below[x + wx, z:z + wz]
                    al += int(((a >= 0) & (b >= 0) & (a != b) & (todo[x + wx, z:z + wz] > 0)).sum())
                if z + wz < D:
                    a, b = below[x:x + wx, z + wz - 1], below[x:x + wx, z + wz]
                    al += int(((a >= 0) & (b >= 0) & (a != b) & (todo[x:x + wx, z + wz] > 0)).sum())
                along = (wx >= wz) if prefer_x else (wz >= wx)
                score = (-al, area, along)
                if best is None or score > best[0]:
                    best = ((-al, area, along), (pid, wx, wz, rot))
            pid, wx, wz, rot = best[1]
            todo[x:x + wx, z:z + wz] = 0
            p = m.add(pid, 0 if col == 10000 else int(col), x, y, z, rot)
            occ_id[x:x + wx, y:y + h, z:z + wz] = p.id
            placed += 1
        return placed


    lut = np.zeros(256, np.int32)
    for k, v in LDRAW.items():
        lut[k] = v
    C = lut[G]  # LDraw codes, 0 = empty (Black is code 0!) -> remap black to -1 sentinel
    BLACK_CODE = 0
    C = np.where(G == 5, 10000, C)  # temporary code for black
    t0 = time.time()
    total = 0
    total += pack_level(C[:, 0, :], 0, 1, PC, True)
    total += pack_level(C[:, 1, :], 1, 1, PC, False)
    course = 0
    for y0 in range(2, HMAX, 3):
        if y0 + 2 >= HMAX:
            for y in range(y0, HMAX):
                total += pack_level(C[:, y, :], y, 1, PC, (y % 2) == 0)
            break
        c0, c1, c2 = C[:, y0, :], C[:, y0 + 1, :], C[:, y0 + 2, :]
        brick = (c0 > 0) & (c0 == c1) & (c1 == c2)
        total += pack_level(np.where(brick, c0, 0), y0, 3, BC, course % 2 == 0)
        total += pack_level(np.where(brick, 0, c0), y0, 1, PC, course % 2 == 1)
        total += pack_level(np.where(brick, 0, c1), y0 + 1, 1, PC, course % 2 == 0)
        total += pack_level(np.where(brick, 0, c2), y0 + 2, 1, PC, course % 2 == 1)
        course += 1
        if course % 10 == 0:
            print(f"y={y0} parts={len(m.parts)} {time.time() - t0:.0f}s", flush=True)

    return m


for it in range(6):
    m = build(G.copy())
    chk = m.check()
    fl = [i for grp in chk["floating_groups"] for i in grp]
    adj = m.connections()
    # recompute all non-main groups, not just the first 10
    seen, groups = set(), []
    for s0 in m.parts:
        if s0 in seen:
            continue
        st, grp = [s0], []
        seen.add(s0)
        while st:
            n = st.pop(); grp.append(n)
            for k in adj[n]:
                if k not in seen:
                    seen.add(k); st.append(k)
        groups.append(grp)
    groups.sort(key=len, reverse=True)
    bad = [i for grp in groups[1:] for i in grp]
    print(f"iter {it}: parts={len(m.parts)} groups={len(groups)} stray parts={len(bad)}", flush=True)
    if not bad:
        break
    added = 0
    for i in bad:
        for (cx, cy, cz) in m.geometry(i).cells:
            if not (0 <= cx < W and 0 <= cz < D):
                continue
            y = cy - 1
            while y >= 0 and G[cx, y, cz] == 0 and DESIGN[cx, y, cz] != 0:
                G[cx, y, cz] = DESIGN[cx, y, cz]; added += 1; y -= 1
    print("   restored voxels:", added)
    if added == 0:
        break
np.save(HERE / "grid_final.npy", G)
if bad:
    print("pruning stray parts:", len(bad))
    m.remove(bad)
print("parts:", len(m.parts))
out = m.save(REPO / "models")
print("saved", out)
chk = m.check()
print({k: (v if not isinstance(v, list) else [len(g) if isinstance(g, list) else g for g in v]) for k, v in chk.items()})
pickle.dump(chk, open(HERE / "check.pkl", "wb"))
