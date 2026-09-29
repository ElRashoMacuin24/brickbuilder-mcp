import sys, json, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder.model import Model
from brickbuilder import render
name, out, views = sys.argv[1], sys.argv[2], sys.argv[3].split(';')
x0,x1,z0,z1,ymax = map(int, sys.argv[4].split(','))
size = int(sys.argv[5]) if len(sys.argv) > 5 else 800
d = json.load(open(REPO / "models"/(name+".json")))
d["parts"] = [p for p in d["parts"] if x0 <= p["x"] <= x1 and z0 <= p["z"] <= z1 and p["y"] <= ymax]
m = Model.from_json(d)
t=time.time(); Path(out).write_bytes(render.render(m, views, size, None)); print(len(d["parts"]), "parts", time.time()-t)
