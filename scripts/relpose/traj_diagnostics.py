#!/usr/bin/env python3
"""What ATE hides: per-scene trajectory diagnostics of several methods against GT, from saved trajectories
only (CPU). Every method is first Sim3-aligned to GT on camera centres (as ATE is), then:
  gap      relative rotation (deg) and translation (m, and % of the GT distance) error at frame gaps 1..4096:
           flat = jitter, ~sqrt(gap) = random-walk drift, ~gap = a constant per-step bias
  bias     mean signed per-step rotation error per camera axis (deg/frame): a constant bias shows here
  scale    predicted / GT path length over sliding 100-frame segments: scale drift along the stream
  stream   gap-10 error over the first vs the last 20% of the frames: degradation with stream length
  kitti    KITTI odometry t_rel (%) and r_rel (deg/100 m) over 100..800 m segments
  jitter   RMS second difference of centres and of step rotations, relative to GT's
  still    predicted motion while GT is stationary
  revisit  distance error between frames > 300 apart that GT puts within 3 m of each other
Sources (--method name=spec):  bss:<root>  -> <root>/<bench>/<scene>/lingbot_map/traj.txt
                               npz:<dir>[:key] -> <dir>/<bench>_<scene>.npz, array key (default c2w)
Writes <out>/diagnostics.json and per-scene figures <out>/<bench>_<scene>.png.
Usage: traj_diagnostics.py --gt-root DIR --scenes kitti/00 vbr/campus_train0 ... --method NAME=SPEC ... --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GAPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def load_bss(p):
    a = np.loadtxt(p, comments="#")
    c2w = np.tile(np.eye(4), (len(a), 1, 1))
    c2w[:, :3] = a[:, 1:].reshape(-1, 3, 4)
    return c2w


def load(spec, bench, scene):
    kind, rest = spec.split(":", 1)
    if kind == "bss":
        return load_bss(Path(rest) / bench / scene / "lingbot_map/traj.txt")
    d, _, key = rest.partition(":")
    f = Path(d) / f"{bench}_{scene}.npz"
    if not f.exists():
        f = Path(d) / "cache" / bench / f"{scene}.npz"        # eval_lingbot_rel's refiner-cache layout
    return np.load(f)[key or "c2w"].astype(np.float64) if f.exists() else None


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / xs.var(0).sum()
    return s, R, mu_d - s * R @ mu_s


def align(P, G):
    s, R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
    A = P.copy()
    A[:, :3, :3] = R @ P[:, :3, :3]
    A[:, :3, 3] = s * P[:, :3, 3] @ R.T + t
    return A, s


def log_rot(R):
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    k = np.where(th > 1e-7, th / (2 * np.sin(np.maximum(th, 1e-7))), 0.5)
    return v * k[..., None]


def angle_deg(R):
    """Rotation magnitude; log_rot's axis scaling is ill-conditioned near 180 deg, the angle is not."""
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def rel(M, d):
    return np.linalg.inv(M[:-d]) @ M[d:]


def gap_curve(A, G):
    out = {}
    for d in GAPS:
        if d >= len(G):
            break
        rp, rg = rel(A, d), rel(G, d)
        rot = angle_deg(np.swapaxes(rp[:, :3, :3], -1, -2) @ rg[:, :3, :3])
        et = np.linalg.norm(rp[:, :3, 3] - rg[:, :3, 3], axis=-1)
        dist = np.linalg.norm(rg[:, :3, 3], axis=-1)
        ok = dist > 0.5
        out[d] = {"rot_deg": float(rot.mean()), "trans_m": float(et.mean()),
                  "trans_pct": float(100 * np.median(et[ok] / dist[ok])) if ok.any() else None}
    return out


def kitti_odometry(A, G, lengths=(100, 200, 300, 400, 500, 600, 700, 800), step=10):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=-1))]
    te, re = [], []
    for i in range(0, len(G), step):
        for L in lengths:
            j = np.searchsorted(seg, seg[i] + L)
            if j >= len(G):
                continue
            E = np.linalg.inv(np.linalg.inv(A[i]) @ A[j]) @ (np.linalg.inv(G[i]) @ G[j])
            te.append(np.linalg.norm(E[:3, 3]) / L)
            re.append(angle_deg(E[:3, :3]) / L)
    if not te:
        return None
    return {"t_rel_pct": float(100 * np.mean(te)), "r_rel_deg_per_100m": float(100 * np.mean(re)), "segments": len(te)}


def diagnostics(P, G):
    A, s = align(P, G)
    r = {"ate": float(np.sqrt((np.linalg.norm(A[:, :3, 3] - G[:, :3, 3], axis=-1) ** 2).mean())), "sim3_scale": float(s)}
    r["gap"] = gap_curve(A, G)
    sp, sg = rel(A, 1), rel(G, 1)
    step_err = log_rot(np.swapaxes(sp[:, :3, :3], -1, -2) @ sg[:, :3, :3])
    r["bias_deg_per_frame_xyz"] = np.degrees(step_err.mean(0)).tolist()
    # scale drift: path-length ratio over sliding 100-frame segments with > 1 m of GT motion
    lp = np.linalg.norm(sp[:, :3, 3], axis=-1)
    lg = np.linalg.norm(sg[:, :3, 3], axis=-1)
    cp, cg = np.r_[0, np.cumsum(lp)], np.r_[0, np.cumsum(lg)]
    starts = np.arange(0, len(lg) - 100, 50)
    seg_p, seg_g = cp[starts + 100] - cp[starts], cg[starts + 100] - cg[starts]
    ok = seg_g > 1.0
    ratio = seg_p[ok] / seg_g[ok]
    r["scale"] = {"frames": starts[ok].tolist(), "ratio": ratio.tolist(),
                  "log2_p10_p50_p90": np.percentile(np.log2(ratio), [10, 50, 90]).tolist() if ok.any() else None}
    # stream position: gap-10 errors in the first vs the last 20%
    n = len(G)
    if n > 200:
        rp, rg = rel(A, 10), rel(G, 10)
        rot = angle_deg(np.swapaxes(rp[:, :3, :3], -1, -2) @ rg[:, :3, :3])
        et = np.linalg.norm(rp[:, :3, 3] - rg[:, :3, 3], axis=-1)
        q = len(rot) // 5
        r["stream"] = {"gap10_rot_deg_first_last": [float(rot[:q].mean()), float(rot[-q:].mean())],
                       "gap10_trans_m_first_last": [float(et[:q].mean()), float(et[-q:].mean())]}
    r["kitti_odometry"] = kitti_odometry(A, G)
    # jitter: RMS second differences relative to GT's
    acc = lambda M: np.diff(M[:, :3, 3], 2, axis=0)
    rv = lambda M: log_rot(rel(M, 1)[:, :3, :3])
    r["jitter"] = {"pos_rms_ratio": float(np.sqrt((acc(A) ** 2).sum(-1).mean() / max((acc(G) ** 2).sum(-1).mean(), 1e-12))),
                   "rot_rms_ratio": float(np.sqrt((np.diff(rv(A), axis=0) ** 2).sum(-1).mean()
                                                  / max((np.diff(rv(G), axis=0) ** 2).sum(-1).mean(), 1e-12)))}
    still = lg < 0.01
    r["still"] = {"frames": int(still.sum()),
                  "pred_m_per_frame": float(lp[still].mean()) if still.any() else None}
    # revisits: GT-close frames far apart in time
    from scipy.spatial import cKDTree
    pg = G[:, :3, 3]
    pairs = np.array([(i, j) for i, j in cKDTree(pg).query_pairs(3.0) if abs(i - j) > 300])
    if len(pairs):
        pairs = pairs[np.random.default_rng(0).choice(len(pairs), min(len(pairs), 4000), replace=False)]
        dp = np.linalg.norm(A[pairs[:, 0], :3, 3] - A[pairs[:, 1], :3, 3], axis=-1)
        dg = np.linalg.norm(pg[pairs[:, 0]] - pg[pairs[:, 1]], axis=-1)
        r["revisit"] = {"pairs": int(len(pairs)), "dist_err_m_mean": float(np.abs(dp - dg).mean()),
                        "dist_err_m_p90": float(np.percentile(np.abs(dp - dg), 90))}
    else:
        r["revisit"] = {"pairs": 0}
    return r, A


def plane_axes(G):
    """The two principal axes of the GT centres (a top-down view for driving and walking)."""
    X = G[:, :3, 3] - G[:, :3, 3].mean(0)
    return np.linalg.svd(X, full_matrices=False)[2][:2]


def figure(path, bench, scene, G, aligned, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.2))
    V = plane_axes(G)
    g2 = (G[:, :3, 3] - G[:, :3, 3].mean(0)) @ V.T
    ax[0].plot(g2[:, 0], g2[:, 1], "k-", lw=2.2, label="GT")
    for name, A in aligned.items():
        a2 = (A[:, :3, 3] - G[:, :3, 3].mean(0)) @ V.T
        ax[0].plot(a2[:, 0], a2[:, 1], lw=1.0, label=f"{name} ({results[name]['ate']:.1f} m)")
    ax[0].set_aspect("equal")
    ax[0].set_title(f"{bench}/{scene}: top-down, Sim3-aligned")
    ax[0].legend(fontsize=7)
    for name in aligned:
        g = results[name]["gap"]
        ds = sorted(int(d) for d in g)
        ax[1].loglog(ds, [g[d]["rot_deg"] for d in ds], "o-", ms=3, label=name)
        ax[2].loglog(ds, [g[d]["trans_m"] for d in ds], "o-", ms=3, label=name)
        sc = results[name]["scale"]
        if sc["frames"]:
            ax[3].plot(sc["frames"], np.log2(sc["ratio"]), lw=0.8, label=name)
    ax[1].set_title("relative rotation error vs gap (deg)")
    ax[2].set_title("relative translation error vs gap (m)")
    ax[3].set_title("log2 path-length ratio, 100-frame segments")
    ax[3].axhline(0, color="k", lw=0.6)
    for a in ax[1:3]:
        a.set_xlabel("frame gap")
    ax[3].set_xlabel("frame")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", type=Path, required=True, help="BSS root holding <bench>/<scene>/gt/traj.txt")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--method", action="append", required=True, help="NAME=SPEC, see the module docstring")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    methods = dict(m.split("=", 1) for m in args.method)
    args.out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for sc in args.scenes:
        bench, scene = sc.split("/")
        G = load_bss(args.gt_root / bench / scene / "gt/traj.txt")
        res, aligned = {}, {}
        for name, spec in methods.items():
            P = load(spec, bench, scene)
            if P is None:
                continue
            assert len(P) == len(G), (name, sc, len(P), len(G))
            res[name], aligned[name] = diagnostics(P, G)
        allres[sc] = res
        figure(args.out / f"{bench}_{scene}.png", bench, scene, G, aligned, res)
        print(f"[diag] {sc}: " + "  ".join(f"{n} ATE {r['ate']:.2f}" for n, r in res.items()), flush=True)
    (args.out / "diagnostics.json").write_text(json.dumps(allres, indent=1))


if __name__ == "__main__":
    main()
