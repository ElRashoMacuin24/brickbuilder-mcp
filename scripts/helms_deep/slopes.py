import pickle, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder.model import compute_geometry, Placement
HERE = Path(__file__).parent
raw = pickle.load(open(HERE / "slopes_raw.pkl", "rb"))
ok, bad = [], 0
for pid, col, mx, y, mz, rot, low, high in raw:
    g = compute_geometry(Placement(0, pid, 0, mx, y, mz, rot))
    cols = {(c[0], c[2]) for c in g.cells}
    if cols == {tuple(low), tuple(high)} and g.top_studs == {tuple(high)}:
        ok.append((pid, col, mx, y, mz, rot))
    else:
        bad += 1
        if bad < 4: print("mismatch", rot, low, high, cols, g.top_studs)
print("ok", len(ok), "bad", bad)
pickle.dump(ok, open(HERE / "extra_specials.pkl", "wb"))
