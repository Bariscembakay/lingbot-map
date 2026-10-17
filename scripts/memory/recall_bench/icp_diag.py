#!/usr/bin/env python3
"""Diagnostic: how much of each method's NRGBD recall error is a global Sim(3)?

Starts from the alignment the benchmark already applies (ours: none, posed;
baselines selfpose: Umeyama on camera centres), refines it with trimmed
point-to-point Sim(3) ICP against the GT cloud, and reports acc/comp before
and after on the same subsample. The fitted scale says how far off the
recalled cloud's scale was. Subsampled, so numbers are close to but not
identical with metrics.json.
Usage: icp_diag.py --scene kitchen --tier 100 --methods a,b,c --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_recall_score import load_poses, umeyama_sim3  # noqa: E402

CAMP = Path("/group/compact-3dmem/campaigns/spatial_memory")
MAN = Path(__file__).resolve().parent / "manifests"


def gt_cloud(scene, tier, rng, n_keep):
    man = json.loads((MAN / f"{scene}.json").read_text())
    root, ids = Path(man["root"]), man["ingest_frames"][:tier]
    focal = float(Path(root / man["focal"]).read_text().split()[0])
    poses = load_poses(root / man["poses"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    c2w = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]
    lo, hi = man["depth_valid_range_m"]
    pts = []
    for q, i in enumerate(ids):
        d = cv2.imread(str(root / f"depth/depth{i}.png"), cv2.IMREAD_UNCHANGED)
        d = np.nan_to_num(d.astype(np.float64)) / man["depth_scale_mm"]
        H, W = d.shape
        jj, ii = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        v = (d > lo) & (d < hi)
        pc = np.stack([(jj[v] - W / 2) / focal * d[v], (ii[v] - H / 2) / focal * d[v], d[v]], -1)
        pts.append(pc[::16] @ c2w[q, :3, :3].T + c2w[q, :3, 3])
    G = np.concatenate(pts)
    return G[rng.choice(len(G), min(n_keep, len(G)), replace=False)], c2w


def metrics(P, G, tG):
    da, _ = tG.query(P, workers=-1)
    dc, _ = cKDTree(P).query(G, workers=-1)
    return dict(acc_mean=float(da.mean()), acc_med=float(np.median(da)),
                comp_mean=float(dc.mean()), comp_med=float(np.median(dc)),
                chamfer=float((da.mean() + dc.mean()) / 2))


def sim3_icp(P, tG, G, rng, iters=40, n_fit=200_000, trim=0.25):
    """Trimmed ICP: each step fits Umeyama to the nearest-neighbour pairs closer
    than `trim` m. Returns the cumulative (s, R, t) mapping P into the GT frame."""
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    sub = P[rng.choice(len(P), min(n_fit, len(P)), replace=False)]
    for _ in range(iters):
        X = s * sub @ R.T + t
        d, j = tG.query(X, workers=-1)
        k = d < trim
        if k.sum() < 1000:
            break
        ds, dR, dt = umeyama_sim3(X[k], G[j[k]])
        s, R, t = ds * s, dR @ R, ds * (dR @ t) + dt
        if abs(ds - 1) < 1e-5 and np.linalg.norm(dt) < 1e-5:
            break
    return s, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    G, c2w_gt = gt_cloud(args.scene, args.tier, rng, 2_000_000)
    tG = cKDTree(G)
    res = {}
    for m in args.methods.split(","):
        d = np.load(CAMP / "recallbench_dumps" / m / f"{args.scene}_n{args.tier}.npz")
        posed = "pw_selfpose" not in d.files
        P = d["pw_gtpose" if posed else "pw_selfpose"].reshape(-1, 3).astype(np.float64)
        P = P[np.isfinite(P).all(1)]
        P = P[rng.choice(len(P), min(1_000_000, len(P)), replace=False)]
        if not posed:
            sA, RA, tA = umeyama_sim3(d["c2w_pred"][:args.tier, :3, 3], c2w_gt[:, :3, 3])
            P = sA * P @ RA.T + tA
        before = metrics(P, G, tG)
        s, R, t = sim3_icp(P, tG, G, rng)
        after = metrics(s * P @ R.T + t, G, tG)
        rot = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        res[m] = dict(before=before, after=after, icp_scale=s, icp_rot_deg=rot,
                      icp_trans=float(np.linalg.norm(t)))
        print(f"[{m}|{args.scene}|n{args.tier}] scale {s:.3f} rot {rot:.2f}deg "
              f"chamfer {before['chamfer']:.3f} -> {after['chamfer']:.3f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
