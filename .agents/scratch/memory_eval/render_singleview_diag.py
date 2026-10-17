"""Diagnostic: single-view pred vs GT for scene 1730c7d709 (worst fused scene).
If the 'holes' were a projection bug, single views would show missing regions.
If it's depth migration, each view is fully covered but locally displaced."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import emit

d = np.load("/scratch/baris_bakay/recall96/ours_1730c7d709.npz")
OUT = Path("/group/compact-3dmem/campaigns/spatial_memory/"
           "scenes32_96f_b4_write4_read2_lingbothead/recall96_singleview_diag_1730c7d709")
for q in (10, 48, 90):
    v = d["valid"][q]
    p, g = d["pred_pts_world"][q][v], d["gt_pts_world"][q][v]
    r = emit(OUT, f"view{q:02d}_alone", p, g, np.ones(len(p), bool),
             np.ones(len(p), np.float32), 0.0)
    err = np.linalg.norm(p - g, axis=-1)
    print(f"view {q}: {len(p)} pts (100% of valid px), L21 mean {err.mean():.3f} "
          f"med {np.median(err):.3f} p95 {np.percentile(err,95):.3f}")
print("done ->", OUT)
