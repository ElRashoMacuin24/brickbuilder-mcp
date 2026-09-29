import sys, json, collections
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from brickbuilder.model import Model
m = Model.from_json(json.load(open(REPO / "models"/((sys.argv[1] if len(sys.argv)>1 else "helms_deep")+".json"))))
adj = m.connections()
seen=set(); groups=[]
for s in m.parts:
    if s in seen: continue
    st=[s]; seen.add(s); g=[]
    while st:
        n=st.pop(); g.append(n)
        for k in adj[n]:
            if k not in seen: seen.add(k); st.append(k)
    groups.append(g)
groups.sort(key=len, reverse=True)
fl=[g for g in groups[1:]]
print("groups", len(groups), "non-main parts", sum(map(len,fl)))
cnt=collections.Counter()
for g in fl:
    ps=[m.parts[i] for i in g]
    xs=[p.x for p in ps]; ys=[p.y for p in ps]; zs=[p.z for p in ps]
    cols=collections.Counter(p.color for p in ps).most_common(2)
    key=(min(xs)//16*16, min(zs)//16*16)
    cnt[key]+=len(g)
for g in fl[:25]:
    ps=[m.parts[i] for i in g]
    print(len(g), "x",min(p.x for p in ps),max(p.x for p in ps),"y",min(p.y for p in ps),max(p.y for p in ps),"z",min(p.z for p in ps),max(p.z for p in ps), collections.Counter(p.color for p in ps).most_common(2), collections.Counter(p.part for p in ps).most_common(2))
print(sorted(cnt.items(), key=lambda kv:-kv[1])[:20])
