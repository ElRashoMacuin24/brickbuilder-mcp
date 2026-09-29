import sys, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder.model import Model
from brickbuilder import render
name = sys.argv[1]; out = sys.argv[2]; views = sys.argv[3].split(';'); size = int(sys.argv[4]) if len(sys.argv) > 4 else 700
ref = sys.argv[5] if len(sys.argv) > 5 else None
t=time.time()
m = Model.from_json(json.load(open(REPO / "models"/(name+".json"))))
png = render.render(m, views, size, ref)
Path(out).write_bytes(png); print("rendered", time.time()-t)
