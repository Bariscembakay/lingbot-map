#!/usr/bin/env python3
"""Offline test of fusing a relative front end with an absolute one, no learning: chain the relative method's
steps (ABot-Recon) and pull every frame gently toward the absolute method's pose (lingbot): a causal
complementary filter, so it would run in streaming. Rotation moves a fraction alpha of the way to the
absolute rotation (on SO(3)), the centre a fraction beta; small gains keep the steps' high frequencies and
the absolute poses' low frequencies (anchoring, gravity, no accumulation). The relative steps are first put
in the absolute method's scale by the median ratio of 100-frame displacements (robust to its jitter).
Prints a (alpha, beta) sweep of ATE and gap errors, and writes the fused trajectories for one setting.
Usage: fuse_abs_rel.py --gt-root DIR --scenes kitti/00 ... --abs SPEC --rel SPEC [--out DIR --alpha A --beta B]
SPECs as in traj_diagnostics.py (bss:<root> | npz:<dir>[:key]).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
import traj_diagnostics as td  # noqa: E402


def step_scale(L, A, gap=100):
    if len(L) <= gap:
        gap = max(1, len(L) // 2)
    dl = np.linalg.norm(L[gap:, :3, 3] - L[:-gap, :3, 3], axis=-1)
    da = np.linalg.norm(A[gap:, :3, 3] - A[:-gap, :3, 3], axis=-1)
    ok = da > 1e-9
    return float(np.median(dl[ok] / da[ok])) if ok.any() else 1.0


def fuse(L, A, alpha, beta, alpha_tilt=0.0, up=None):
    """L: absolute poses [N,4,4]; A: relative method's poses (only its steps are used). alpha_tilt pulls only
    the camera's gravity direction (world vertical `up`, in L's world frame) toward the absolute pose's,
    leaving heading to the steps."""
    S = td.rel(A, 1)
    S[:, :3, 3] *= step_scale(L, A)
    X = np.empty_like(L)
    X[0] = L[0]
    Rl = Rot.from_matrix(L[:, :3, :3])
    for t in range(1, len(L)):
        P = X[t - 1] @ S[t - 1]
        Rp = Rot.from_matrix(P[:3, :3])
        R = Rp * Rot.from_rotvec(alpha * (Rp.inv() * Rl[t]).as_rotvec())
        if alpha_tilt:
            gp, gl = R.inv().apply(up), Rl[t].inv().apply(up)     # vertical in the two camera frames
            ax = np.cross(gp, gl)
            sn = np.linalg.norm(ax)
            if sn > 1e-12:
                ang = np.arctan2(sn, gp @ gl)
                R = R * Rot.from_rotvec(-alpha_tilt * ang * ax / sn)
        X[t, :3, :3] = R.as_matrix()
        X[t, :3, 3] = P[:3, 3] + beta * (L[t, :3, 3] - P[:3, 3])
        X[t, 3] = (0, 0, 0, 1)
    return X


def summary(R):
    g = lambda d, f: np.mean([r["gap"][d][f] for r in R if d in r["gap"]])
    return (f"ATE {np.mean([r['ate'] for r in R]):6.2f} | rot@1 {g(1, 'rot_deg'):.2f} @16 {g(16, 'rot_deg'):.2f} "
            f"@256 {g(256, 'rot_deg'):.2f} @1024 {g(1024, 'rot_deg'):.2f} | tr@1 {g(1, 'trans_m'):.2f} "
            f"@16 {g(16, 'trans_m'):.2f} @256 {g(256, 'trans_m'):.1f} | t_rel "
            f"{np.mean([r['kitti_odometry']['t_rel_pct'] for r in R if r['kitti_odometry']]):.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--abs", required=True)
    ap.add_argument("--rel", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=0.01)
    args = ap.parse_args()
    data = []
    for sc in args.scenes:
        bench, scene = sc.split("/")
        G = td.load_bss(args.gt_root / bench / scene / "gt/traj.txt")
        data.append((sc, G, td.load(args.abs, bench, scene), td.load(args.rel, bench, scene)))
    print(f"{'absolute only':>24}: {summary([td.diagnostics(L, G)[0] for _, G, L, _ in data])}")
    print(f"{'relative only':>24}: {summary([td.diagnostics(A, G)[0] for _, G, _, A in data])}")
    for a in args.alphas:
        for b in args.betas:
            if a == 0 and b == 0:
                continue
            R = [td.diagnostics(fuse(L, A, a, b), G)[0] for _, G, L, A in data]
            print(f"{f'fused a={a:g} b={b:g}':>24}: {summary(R)}", flush=True)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for sc, G, L, A in data:
            np.savez(args.out / f"{sc.replace('/', '_')}.npz", c2w=fuse(L, A, args.alpha, args.beta))


if __name__ == "__main__":
    main()
