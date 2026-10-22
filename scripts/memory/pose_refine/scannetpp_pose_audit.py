#!/usr/bin/env python3
"""ScanNet++ iPhone GT audit: ARKit aligned_pose vs the COLMAP reconstruction of the iPhone frames (every
10th frame, registered to the laser scan). Per scene: COLMAP coverage, ARKit-COLMAP disagreement, and the
fraction of frames inside 'valid stretches' -- consecutive registered COLMAP frames whose ARKit relative
motion agrees with COLMAP's (the stretches a COLMAP-anchored ARKit GT can use).
Usage: scannetpp_pose_audit.py --out audit.json [--trans-tol 0.05] [--rot-tol 2]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path("/data/ScanNetpp/data")


def q2R(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def colmap_c2w(path):
    out = {}
    lines = [l for l in open(path) if not l.startswith("#")]
    for l in lines[0::2]:
        p = l.split()
        if len(p) < 10:
            continue
        R, t = q2R(*map(float, p[1:5])), np.array(list(map(float, p[5:8])))
        M = np.eye(4)
        M[:3, :3], M[:3, 3] = R.T, -R.T @ t
        out[int(p[9].split("_")[1].split(".")[0])] = M
    return out


def rel_err(Ta, Tb, Ca, Cb):
    ra, rc = np.linalg.inv(Ta) @ Tb, np.linalg.inv(Ca) @ Cb
    dt = np.linalg.norm(ra[:3, 3] - rc[:3, 3])
    m = ra[:3, :3].T @ rc[:3, :3]
    return dt, np.degrees(np.arccos(np.clip((np.trace(m) - 1) / 2, -1, 1)))


def audit(seq, tt=0.05, rt=2.0):
    ip = ROOT / seq / "iphone"
    try:
        a = json.loads((ip / "pose_intrinsic_imu.json").read_text())
        c = colmap_c2w(ip / "colmap/images.txt")
    except Exception as e:
        return seq, {"error": str(e)[:100]}
    n = len(a)
    A = {i: np.asarray(a[f"frame_{i:06d}"]["aligned_pose"], np.float64) for i in c if f"frame_{i:06d}" in a}
    idx = sorted(A)
    if len(idx) < 2:
        return seq, {"frames": n, "colmap": len(c), "valid_frac": 0.0}
    d = np.array([np.linalg.norm(A[i][:3, 3] - c[i][:3, 3]) for i in idx])
    step = int(np.median(np.diff(idx)))
    valid = 0
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if i1 - i0 == step:
            dt, dr = rel_err(A[i0], A[i1], c[i0], c[i1])
            if dt <= tt and dr <= rt:
                valid += step
    return seq, {"frames": n, "colmap": len(c), "colmap_step": step, "colmap_cover": len(c) * step / n,
                 "abs_dist_median": float(np.median(d)), "abs_dist_max": float(d.max()), "valid_frac": valid / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    seqs = sorted(p.name for p in ROOT.iterdir() if (p / "iphone/colmap/images.txt").exists())
    with ProcessPoolExecutor(args.workers) as ex:
        res = dict(ex.map(audit, seqs, chunksize=4))
    args.out.write_text(json.dumps(res, indent=1))
    v = np.array([r.get("valid_frac", 0) for r in res.values()])
    print(f"{len(res)} scenes; valid-frame fraction: median {np.median(v):.3f}, "
          f"scenes >=0.95 {int((v >= 0.95).sum())}, 0.5-0.95 {int(((v >= 0.5) & (v < 0.95)).sum())}, <0.5 {int((v < 0.5).sum())}; "
          f"frames kept {sum(r.get('valid_frac', 0) * r.get('frames', 0) for r in res.values()) / sum(r.get('frames', 0) for r in res.values()):.3f}")


if __name__ == "__main__":
    main()
