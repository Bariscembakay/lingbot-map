#!/usr/bin/env python3
"""Rewrite the NRGBD render dirs' GT PLYs with true RGB from the source images
(same unprojection order as the emit scripts: per query view, valid mask,
fixed pixel subsample)."""
import sys
from pathlib import Path
import numpy as np, cv2, json

sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import write_ply  # noqa: E402

ROOT = Path("/data/NRGBD/breakfast_room")
CAMP = Path("/group/compact-3dmem/campaigns/spatial_memory")
K = np.array([[554.2562584220408, 0, 320], [0, 554.2562584220408, 240],
              [0, 0, 1]], np.float64)

def load_poses():
    lines = [l for l in (ROOT / "poses.txt").read_text().splitlines() if l.strip()]
    P = np.array([[list(map(float, lines[4*i+j].split())) for j in range(4)]
                  for i in range(len(lines)//4)], np.float64)
    P[:, :, 1:3] *= -1.0
    return P

POSES = load_poses()

def gt_pts_cols(img_id, c2w_rel, sub):
    d = cv2.imread(str(ROOT / "depth" / f"depth{img_id}.png"), cv2.IMREAD_UNCHANGED)
    d = np.nan_to_num(d.astype(np.float32)) / 1000.0
    d[(d > 10) | (d < 1e-3)] = 0
    rgb = cv2.cvtColor(cv2.imread(str(ROOT / "images" / f"img{img_id}.png")),
                       cv2.COLOR_BGR2RGB)
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    x = (u - K[0, 2]) * d / K[0, 0]
    y = (v - K[1, 2]) * d / K[1, 1]
    pts = np.stack([x, y, d], -1)
    valid = d > 0
    R, t = c2w_rel[:3, :3], c2w_rel[:3, 3]
    return (pts[valid] @ R.T + t)[::sub], rgb[valid][::sub]

def rebuild(ids0, qs, sub, out_ply, scale=1.0):
    base = np.linalg.inv(POSES[ids0[0]])
    P, C = [], []
    for q in qs:
        c2w_rel = base @ POSES[ids0[q]]
        c2w_rel = c2w_rel.copy(); c2w_rel[:3, 3] /= scale
        p, c = gt_pts_cols(ids0[q], c2w_rel, sub)
        P.append(p / 1.0 if scale == 1.0 else p)
        C.append(c)
    P, C = np.concatenate(P), np.concatenate(C)
    if scale != 1.0:
        pass  # points already in canonical via pose scaling + depth below
    write_ply(out_ply, P.astype(np.float32), C.astype(np.uint8))
    print(f"[recolor] {out_ply}  {len(P)} pts")

# 1. stride-2 state snapshots (metric units)
ids = [i * 2 for i in range(500)]
d1 = CAMP / "CUT3R_zeroshot_512dpt/nrgbd_breakfast_room_recall_stride2"
rebuild(ids, list(range(0, 300, 8)), 3, d1 / "state300_recall-gt.ply")
rebuild(ids, list(range(0, 500, 8)), 3, d1 / "state500_recall-gt.ply")

# 2. 96f stride-5 pair
ids5 = [i * 5 for i in range(96)]
d2 = CAMP / "CUT3R_zeroshot_512dpt/nrgbd_breakfastroom_96f_stride5_recall"
if (d2 / "state96_recall-gt.ply").exists():
    rebuild(ids5, list(range(0, 96, 1)), 3, d2 / "state96_recall-gt.ply")
# ours side: canonical units -- divide GT (metres) by the cache's gt_scale
d3 = CAMP / ("scenes96_96f_b4_write4_read2_lingbothead/"
             "nrgbd_breakfastroom_96f_stride5_recall")
cachemeta = Path("/group/compact-3dmem/campaigns/spatial_memory/"
                 "eval_nrgbd_cache/breakfast_room_96f_s5/meta.json")
if (d3 / "breakfast_room-gt.ply").exists() and cachemeta.exists():
    s = float(json.loads(cachemeta.read_text())["gt_scale"])
    base = np.linalg.inv(POSES[0])
    P, C = [], []
    for q in range(96):
        c2w_rel = base @ POSES[q * 5]
        p, c = gt_pts_cols(q * 5, c2w_rel, 3)
        P.append(p / s); C.append(c)
    P, C = np.concatenate(P), np.concatenate(C)
    write_ply(d3 / "breakfast_room-gt.ply", P.astype(np.float32), C.astype(np.uint8))
    print(f"[recolor] {d3 / 'breakfast_room-gt.ply'}  {len(P)} pts (canonical /{s:.3f})")
print("done")
