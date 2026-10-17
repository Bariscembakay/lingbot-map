import json, re
from pathlib import Path
import wandb
name = "scenes96_96f_b4_write4_read2_lingbothead"
canon = re.sub(r"[^a-z0-9_-]", "-", name.lower())[:110] + "-m"
hist = json.loads((Path("/group/compact-3dmem/campaigns/spatial_memory") / name / "history.json").read_text())
cfgp = Path("/group/compact-3dmem/campaigns/spatial_memory") / name / "config.json"
cfg = json.loads(cfgp.read_text()) if cfgp.exists() else {}
run = wandb.init(project="spatial_memory", id=canon, name=name + " (merged)",
                 config=cfg, resume="allow")
start = run.step or 0
n = 0
for rec in hist:
    s = int(rec.get("step", 0))
    if s >= start:
        run.log(rec, step=s); n += 1
run.finish()
print(f"[merged] {canon}: appended {n} records (from step {start}, total hist {len(hist)})")
