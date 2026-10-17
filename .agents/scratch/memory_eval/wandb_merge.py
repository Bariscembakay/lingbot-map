"""Rebuild one canonical wandb run per fragmented config from its on-disk
history.json, then delete the fragments. Skips any config with a RUNNING run
except for deleting its already-dead fragments."""
import json, re, sys
from pathlib import Path
import wandb

ENTITY_PROJECT = "bbakay22-ko-university/spatial_memory"
CAMP = Path("/group/compact-3dmem/campaigns/spatial_memory")

api = wandb.Api()
runs = list(api.runs(ENTITY_PROJECT))
import collections
by = collections.defaultdict(list)
for r in runs:
    by[r.name].append(r)

for name, rs in sorted(by.items()):
    if len(rs) == 1 and rs[0].state != "running":
        continue
    canon_id = re.sub(r"[^a-z0-9_-]", "-", name.lower())[:110] + "-m"
    dead = [r for r in rs if r.state != "running"]
    live = [r for r in rs if r.state == "running"]
    hist_p = CAMP / name / "history.json"
    if not dead:
        print(f"[skip] {name}: only a live run"); continue
    if not hist_p.exists():
        print(f"[skip] {name}: no history.json"); continue
    rebuild = not live  # if a live fragment still logs, only clean dead ones;
                        # canonical gets rebuilt after the config finishes.
    have_canon = any(r.id == canon_id for r in dead)
    dead = [r for r in dead if r.id != canon_id]
    if rebuild and not have_canon:
        hist = json.loads(hist_p.read_text())
        cfg = json.loads((CAMP / name / "config.json").read_text()) \
            if (CAMP / name / "config.json").exists() else {}
        run = wandb.init(project="spatial_memory", id=canon_id, name=name,
                         config=cfg, resume="never", reinit=True,
                         settings=wandb.Settings(quiet=True))
        for rec in hist:
            run.log(rec, step=int(rec.get("step", 0)))
        run.finish()
        print(f"[new ] {name}: canonical {canon_id} <- {len(hist)} records")
    for r in dead:
        r.delete()
        print(f"[del ] {name}: fragment {r.id} ({r.state})")
print("done")
