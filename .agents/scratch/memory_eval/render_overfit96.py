import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import from_viz

CAMP = Path("/group/compact-3dmem/campaigns/spatial_memory")
OUT = CAMP / "overfit96f_train285efbc7cf_step3000_96frames"
jobs = [
    (CAMP / "overfit1scene_96f_smallread/viz/probe_train_003000.npz", "write"),
    (CAMP / "overfit1scene_96f_smallread_nowrite/viz/probe_train_003000.npz", "nowrite"),
]
for npz, name in jobs:
    with tempfile.TemporaryDirectory() as td:
        os.symlink(npz, Path(td) / "probe_only.npz")
        from_viz(Path(td), OUT, conf_pct=0.0, frame="selfpose", name=name)
print("rendered ->", OUT)
