import collections, json, sys
import wandb
api = wandb.Api()
runs = list(api.runs("bbakay22-ko-university/spatial_memory"))
by = collections.defaultdict(list)
for r in runs:
    by[r.name].append(r)
print(f"{len(runs)} runs, {len(by)} distinct names\n")
for name, rs in sorted(by.items()):
    if len(rs) == 1 and rs[0].state != "running":
        continue
    states = ",".join(r.state for r in rs)
    ids = ",".join(r.id[:12] for r in rs)
    from pathlib import Path
    hist = Path(f"/group/compact-3dmem/campaigns/spatial_memory/{name}/history.json")
    print(f"{name:60s} n={len(rs)} states=[{states}] history={'YES' if hist.exists() else 'NO'}")
