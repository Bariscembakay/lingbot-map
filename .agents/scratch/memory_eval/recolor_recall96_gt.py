#!/usr/bin/env python3
"""RGB-color the recall96 val GT clouds (ScanNet++): rebuild each scene's
fused GT in the exact order of render_recall96_viewer.py (per view, valid
mask, ::4 pixel subsample) and pair colors from the decoded video frames."""
import sys, json
from pathlib import Path
import numpy as np, cv2

sys.path.insert(0, "/home/baris_bakay/lingbot-map")
sys.path.insert(0, "/home/baris_bakay/lingbot-map/.agents/scratch/memory_eval")
from scripts.memory.render_state import write_ply           # noqa: E402
from run_cut3r_recall import decode_frames, video_resolution  # noqa: E402

SRC = Path("/scratch/baris_bakay/recall96")
CAMP = Path("/group/compact-3dmem/campaigns/spatial_memory")
DIRS = [CAMP / "CUT3R_zeroshot_512dpt/recall96_val8_allqueries",
        CAMP / "scenes32_96f_b4_write4_read2_lingbothead/recall96_val8_allqueries_step5400"]
SCENES = ["210f741378", "260db9cf5a", "2ab7bea148", "47eb87b5bb",
          "0a7cc12c0e", "1730c7d709", "1841a0b525", "4291be3b44"]
SUB = 4

for scene in SCENES:
    npz = SRC / f"ours_{scene}.npz"
    d = np.load(npz)
    gt_w, valid = d["gt_pts_world"], d["valid"]
    n, h, w = valid.shape
    video = Path(f"/data/ScanNetpp/data/{scene}/iphone/rgb.mkv")
    # cache indices 0,2,..,190 at cache stride 20 -> video frames 0,40,..,3800
    vids = [i * 40 for i in range(n)]
    W_nat, H_nat = video_resolution(video)
    frames = decode_frames(video, vids, W_nat, H_nat)
    P, C = [], []
    for i in range(n):
        rgb = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_AREA)
        v = valid[i]
        P.append(gt_w[i][v][::SUB]); C.append(rgb[v][::SUB])
    P, C = np.concatenate(P), np.concatenate(C)
    for D in DIRS:
        out = D / f"val_{scene}-gt.ply"
        if out.parent.exists():
            write_ply(out, P.astype(np.float32), C.astype(np.uint8))
    print(f"[recolor] val_{scene}: {len(P)} pts -> {len(DIRS)} dirs", flush=True)
print("done")
