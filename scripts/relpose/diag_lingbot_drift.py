#!/usr/bin/env python3
"""Does released lingbot's local accuracy degrade along the stream? (the CUT3R frame-0 diagnostic)

From the cached full-length benchmark runs in lingbot-poserefine-v1 (KITTI 00-10, VBR): lingbot's
poses and GT. One Sim3 per sequence (Umeyama on all camera centres) puts lingbot in metres; then,
per 10-frame window and binned by stream index and by GT distance from frame 0:
  speed   predicted / GT distance travelled in the window (1 = right scale; CUT3R collapsed to ~0.2)
  rpe_t   translation error of the window's relative pose, metres per 10 frames
  rpe_r   rotation error of the window's relative pose, degrees per 10 frames
Rows pool all sequences of a benchmark. numpy only.
Usage: diag_lingbot_drift.py [--out FILE.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/lingbot-poserefine-v1")
W = 10
INDEX_BINS = [0, 250, 500, 1000, 2000, 3000, 5000, 20000]
DIST_BINS = [0, 50, 100, 200, 400, 800, 1600, 5000]


def quat_to_mat(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    s = 2.0 / (q * q).sum(-1)
    return np.stack([1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
                     s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
                     s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def enc_to_c2w(enc):
    M = np.tile(np.eye(4), (len(enc), 1, 1))
    M[:, :3, :3] = quat_to_mat(enc[:, 3:7].astype(np.float64))
    M[:, :3, 3] = enc[:, :3]
    return M


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    return (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src)), R


def windows(P, G):
    s, _ = umeyama(P[:, :3, 3], G[:, :3, 3])
    i = np.arange(0, len(P) - W, W)
    rp = np.linalg.inv(P[i]) @ P[i + W]
    rg = np.linalg.inv(G[i]) @ G[i + W]
    dg = np.linalg.norm(rg[:, :3, 3], axis=1)
    speed = s * np.linalg.norm(rp[:, :3, 3], axis=1) / np.maximum(dg, 1e-6)
    rpe_t = np.linalg.norm(s * rp[:, :3, 3] - rg[:, :3, 3], axis=1)
    m = np.einsum("nji,njk->nik", rp[:, :3, :3], rg[:, :3, :3])
    rpe_r = np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    dist = np.linalg.norm(G[i, :3, 3] - G[0, :3, 3], axis=1)
    return dict(index=i, dist=dist, moving=dg > 0.05 * W, speed=speed, rpe_t=rpe_t, rpe_r=rpe_r)


def binned(rows, key, bins):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = [(r[key] >= lo) & (r[key] < hi) for r in rows]
        n = int(sum(s.sum() for s in sel))
        if n == 0:
            continue
        cat = {m: np.concatenate([r[m][s] for r, s in zip(rows, sel)]) for m in ("speed", "rpe_t", "rpe_r", "moving")}
        mv = cat["moving"]
        out.append({"bin": f"{lo}-{hi}", "n": n, "speed_median": float(np.median(cat["speed"][mv])) if mv.any() else None,
                    "rpe_t_mean": float(cat["rpe_t"].mean()), "rpe_r_mean": float(cat["rpe_r"].mean())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {}
    for bench in ("kitti", "vbr"):
        rows = []
        for f in sorted((ROOT / bench / "specials").glob("*.npz")):
            z = np.load(f)
            rows.append(windows(enc_to_c2w(z["pose_enc"]), z["gt_c2w"].astype(np.float64)))
        report[bench] = {"by_index": binned(rows, "index", INDEX_BINS), "by_distance_m": binned(rows, "dist", DIST_BINS)}
        for k, tab in report[bench].items():
            print(f"\n{bench} {k}  (speed: median over moving windows; rpe per {W} frames)")
            print(f"{'bin':>12} {'n':>6} {'speed':>7} {'rpe_t m':>8} {'rpe_r deg':>9}")
            for r in tab:
                sp = f"{r['speed_median']:.3f}" if r["speed_median"] is not None else "-"
                print(f"{r['bin']:>12} {r['n']:>6} {sp:>7} {r['rpe_t_mean']:>8.3f} {r['rpe_r_mean']:>9.3f}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
