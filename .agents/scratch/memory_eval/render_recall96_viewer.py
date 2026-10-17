#!/usr/bin/env python3
"""Render the recall96 benchmark predictions into the viser campaign layout:
one fused all-96-query cloud per scene per model, under each model's arm dir."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import emit  # noqa: E402

SRC = Path("/scratch/baris_bakay/recall96")
CAMP = Path("/group/compact-3dmem/campaigns/spatial_memory")
SCENES = ["210f741378", "260db9cf5a", "2ab7bea148", "47eb87b5bb",
          "0a7cc12c0e", "1730c7d709", "1841a0b525", "4291be3b44"]
SUB = 4   # pixel subsample per view for the fused cloud

DSTS = {
    "ours": CAMP / "scenes32_96f_b4_write4_read2_lingbothead" / "recall96_val8_allqueries_step5400",
    "fullbptt": CAMP / "scenes32_16f_b4_fullbptt" / "recall96_val8_allqueries_step4000",
    "cut3r": CAMP / "CUT3R_zeroshot_512dpt" / "recall96_val8_allqueries",
}

def fused(pred, gt, valid):
    ps, gs = [], []
    for i in range(len(pred)):
        v = valid[i]
        ps.append(pred[i][v][::SUB]); gs.append(gt[i][v][::SUB])
    return np.concatenate(ps), np.concatenate(gs)

for scene in SCENES:
    ours = np.load(SRC / f"ours_{scene}.npz")
    gt_w, valid = ours["gt_pts_world"], ours["valid"]
    # ours
    p, g = fused(ours["pred_pts_world"], gt_w, valid)
    emit(DSTS["ours"], f"val_{scene}", p, g, np.ones(len(p), bool),
         np.ones(len(p), np.float32), 0.0)
    # fullbptt
    fb = np.load(SRC / "fullbptt" / f"ours_{scene}.npz")
    p, g = fused(fb["pred_pts_world"], fb["gt_pts_world"], fb["valid"])
    emit(DSTS["fullbptt"], f"val_{scene}", p, g, np.ones(len(p), bool),
         np.ones(len(p), np.float32), 0.0)
    # cut3r: own scale + resolution; median-magnitude alignment, GT from ours
    ct = np.load(SRC / f"cut3r_{scene}.npz")
    cp = ct["pred_pts_world"].astype(np.float32)
    flat = cp.reshape(-1, 3)[:: SUB * 4]
    s = float(np.median(np.linalg.norm(gt_w[valid], axis=-1))
              / max(np.median(np.linalg.norm(flat, axis=-1)), 1e-9))
    cp = (cp * s).reshape(len(cp), -1, 3)[:, ::SUB]
    cpts = cp.reshape(-1, 3)
    _, gref = fused(ours["pred_pts_world"], gt_w, valid)
    # error color: nearest-GT distance is what the eval measured; approximate
    # with per-point distance to the (subsampled) GT cloud via a coarse KDTree.
    from scipy.spatial import cKDTree
    d, _ = cKDTree(gref[::4]).query(cpts, workers=-1)
    from scripts.memory.render_state import write_ply, heat
    out = DSTS["cut3r"]; tag = f"val_{scene}"
    (out / tag).mkdir(parents=True, exist_ok=True)
    lo, hi = np.percentile(d, 5), np.percentile(d, 95)
    write_ply(out / tag / "pred_cloud.ply", cpts, heat(d, float(lo), float(hi)))
    write_ply(out / f"{tag}-gt.ply", gref,
              np.full((len(gref), 3), 160, np.uint8))
    print(f"[done] {scene}")
print("all rendered")
