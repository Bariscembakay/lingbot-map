#!/usr/bin/env python
"""Scores reset-and-chain CUT3R (`--order chunked`) against the continuous
stream (forward pass of the palindrome arm, same code and numerics).

Chaining: chunk j+1 starts on the last frame o of chunk j, which therefore has
a pose in both chunk frames; chunk j+1 -> chunk j is T_j(o) @ inv(T_j+1(o)).
CUT3R is metric, so the links are rigid -- no scale is fitted.
"""
import argparse
import glob
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = "/group/compact-3dmem/campaigns/frame_anchor"
CONT = "cut3r_pub_palindrome_s1"
COLORS = {"continuous": "#2a78d6", 25: "#eb6834", 50: "#1baf7a", 100: "#eda100", 200: "#e87ba4"}
GT_COLOR, INK2 = "#52514e", "#52514e"
KITTI_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def chain(r, m):
    """Per-source-frame poses in the first chunk's frame, plus the per-frame
    records of each frame's first prediction."""
    pos, fid, T = r["pos"].astype(int), r["fid"].astype(int), r["c2w_pred"]
    resets = np.array(m.get("resets") or [], int)
    chunk = np.searchsorted(resets, pos, side="left")
    N = m["n_frames"]
    G = np.full((N, 4, 4), np.nan)
    first_pos = np.full(N, -1)
    local = {}
    M = np.eye(4)
    for j in range(chunk.max() + 1):
        sel = np.where(chunk == j)[0]
        if j:
            q = sel[0]
            o = fid[q]
            M = M @ local[(j - 1, o)] @ np.linalg.inv(T[q])
        for p in sel:
            local[(j, fid[p])] = T[p]
            if first_pos[fid[p]] < 0:
                G[fid[p]] = M @ T[p]
                first_pos[fid[p]] = p
    return G, first_pos


def umeyama(src, dst, with_scale):
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U) * np.linalg.det(Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (xs ** 2).sum(1).mean() if with_scale else 1.0
    return s, R, md - s * R @ ms


def ate(P, G, with_scale):
    ok = np.isfinite(G).all(axis=(1, 2)) & np.isfinite(P).all(axis=(1, 2))
    a, b = P[ok, :3, 3], G[ok, :3, 3]
    s, R, t = umeyama(a, b, with_scale)
    return float(np.sqrt(((s * a @ R.T + t - b) ** 2).sum(1).mean()))


def kitti_odometry_error(P, G, step=10):
    """KITTI devkit metric: mean translation error (%) and rotation error
    (deg/100 m) over sub-trajectories of 100..800 m. No alignment."""
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1))]
    te, re = [], []
    for first in range(0, len(G), step):
        for L in KITTI_LENGTHS:
            last = np.searchsorted(dist, dist[first] + L)
            if last >= len(G):
                continue
            Eg = np.linalg.inv(G[first]) @ G[last]
            Ep = np.linalg.inv(P[first]) @ P[last]
            E = np.linalg.inv(Ep) @ Eg
            te.append(np.linalg.norm(E[:3, 3]) / L)
            re.append(rot_deg(E[:3, :3]) / L)
    return 100 * float(np.mean(te)), 100 * float(np.mean(re))


def rpe(P, G, lag=10):
    Ep = np.linalg.inv(P[:-lag]) @ P[lag:]
    Eg = np.linalg.inv(G[:-lag]) @ G[lag:]
    E = np.linalg.inv(Ep) @ Eg
    return float(np.nanmedian(rot_deg(E[:, :3, :3]))), float(np.nanmedian(np.linalg.norm(E[:, :3, 3], axis=1)))


def forward_speed(P, G, lag=10):
    dp = P[lag:, :3, 3] - P[:-lag, :3, 3]
    dg = G[lag:, :3, 3] - G[:-lag, :3, 3]
    fp, fg = (dp * P[:-lag, :3, 2]).sum(1), (dg * G[:-lag, :3, 2]).sum(1)
    return float(np.nanmedian(np.where(fg > 1.0, fp / np.maximum(fg, 1e-6), np.nan)))


def load_run(d):
    r = dict(np.load(f"{d}/frames.npz"))
    m = json.load(open(f"{d}/meta.json"))
    N = m["n_frames"]
    if m["order"] == "chunked":
        P, fp = chain(r, m)
        recs = {k: v[fp] for k, v in r.items() if v.shape[:1] == r["pos"].shape}
    else:
        P = r["c2w_pred"][:N]
        recs = {k: v[:N] for k, v in r.items() if v.shape[:1] == r["pos"].shape}
    gt = r["c2w_gt"][:N] if m["order"] != "chunked" else r["c2w_gt"][fp]
    has_gt = bool(np.isfinite(gt).all(axis=(1, 2)).any())
    G = None
    if has_gt:
        g0 = gt[np.isfinite(gt).all(axis=(1, 2))][0]
        G = np.linalg.inv(g0) @ gt
    return m, P, G, recs, r


def pose_scores(P, G, kitti):
    s = dict(ate_sim3=ate(P, G, True), ate_se3=ate(P, G, False))
    s["rpe10_rot"], s["rpe10_trans"] = rpe(P, G)
    s["fwd_speed"] = forward_speed(P, G)
    s["max_dist"] = float(np.nanmax(np.linalg.norm(P[:, :3, 3], axis=1)))
    s["final_dist_ratio"] = float(np.linalg.norm(P[-1, :3, 3]) / max(np.linalg.norm(G[-1, :3, 3]), 1e-6))
    if kitti:
        s["t_err"], s["r_err"] = kitti_odometry_error(P, G)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+", type=int, default=[25, 50, 100, 200])
    ap.add_argument("--out", default=f"{C}/_analysis_resetchain")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    md, summary = ["# Reset-and-chain vs continuous CUT3R\n"], {}
    for bench in ("kitti", "nrgbd", "kitti_depthval"):
        arms = {"continuous": CONT, **{K: f"cut3r_pub_chunk{K}_ov1_s1" for K in args.chunks}}
        scenes = sorted(osp.basename(d.rstrip("/")) for d in glob.glob(f"{C}/{CONT}/{bench}/*/"))
        avail = {a: arm for a, arm in arms.items()
                 if all(osp.exists(f"{C}/{arm}/{bench}/{s}/frames.npz") for s in scenes)}
        if len(avail) < 2:
            continue
        rows, traj, frame_rows = {}, {}, {}
        for a, arm in avail.items():
            per = {}
            for s in scenes:
                m, P, G, recs, raw = load_run(f"{C}/{arm}/{bench}/{s}")
                if a != "continuous":
                    # chunk 0 is causal and starts from the same state: must match bitwise
                    K = m["chunk"]
                    _, Pc, _, _, _ = load_run(f"{C}/{CONT}/{bench}/{s}")
                    assert np.array_equal(P[:K], Pc[:K]), f"chunk 0 differs from continuous: {arm}/{s}"
                if G is not None:
                    per[s] = pose_scores(P, G, bench == "kitti")
                    traj[(a, s)] = (P, G)
                fr = {k: float(np.nanmedian(recs[k])) for k in
                      ("world_shape_med", "self_shape_med", "self_absrel", "sw_res_med") if k in recs
                      and np.isfinite(recs[k]).any()}
                if "world_shape_med" in fr:
                    fr["ws_ratio"] = float(np.nanmedian(recs["world_shape_med"] / recs["self_shape_med"]))
                if fr:
                    frame_rows.setdefault(a, []).append(fr)
            if per:
                rows[a] = per
        summary[bench] = dict(pose=rows, frame=frame_rows)

        md.append(f"\n## {bench} — {len(scenes)} scenes (all), mean over scenes\n")
        if rows:
            keys = ["t_err", "r_err"] if bench == "kitti" else []
            keys += ["ate_sim3", "ate_se3", "rpe10_rot", "rpe10_trans", "fwd_speed", "max_dist"]
            keys = [k for k in keys if any(np.isfinite(v[k]) for per in rows.values() for v in per.values())]
            md.append("| run | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 1))
            for a, per in rows.items():
                lab = "continuous" if a == "continuous" else f"reset every {a}"
                md.append(f"| {lab} | " + " | ".join(f"{np.mean([v[k] for v in per.values()]):.3g}" for k in keys) + " |")
            if bench == "kitti":
                md.append("\nPer sequence, KITTI t_err (%):\n\n| seq | " + " | ".join(
                    "continuous" if a == "continuous" else f"K={a}" for a in rows) + " |\n|" + "---|" * (len(rows) + 1))
                for s in scenes:
                    md.append(f"| {s} | " + " | ".join(f"{rows[a][s]['t_err']:.1f}" for a in rows) + " |")
        if frame_rows:
            keys = [k for k in ("world_shape_med", "self_shape_med", "ws_ratio", "self_absrel", "sw_res_med")
                    if all(k in r for rs in frame_rows.values() for r in rs)]
            md.append("\nPer-frame geometry, median per scene then mean over scenes:\n\n| run | " + " | ".join(keys)
                      + " |\n|" + "---|" * (len(keys) + 1))
            for a, rs in frame_rows.items():
                lab = "continuous" if a == "continuous" else f"reset every {a}"
                md.append(f"| {lab} | " + " | ".join(f"{np.mean([r[k] for r in rs]):.4g}" for k in keys) + " |")

        for fname, show in (("kitti_trajectories_allK.png", list(avail)),
                            ("kitti_trajectories.png", [a for a in ("continuous", 25, 50) if a in avail])):
          if bench == "kitti" and traj:
            fig, axes = plt.subplots(3, 4, figsize=(17, 12.5))
            for ax, s in zip(axes.flat, scenes):
                G = traj[("continuous", s)][1]
                ax.plot(G[:, 0, 3], G[:, 2, 3], color=GT_COLOR, lw=1.6, ls="--", label="GT")
                for a in show:
                    if (a, s) in traj:
                        P = traj[(a, s)][0]
                        ax.plot(P[:, 0, 3], P[:, 2, 3], color=COLORS[a], lw=1.3,
                                label="continuous" if a == "continuous" else f"reset every {a}")
                t = np.linspace(0, 2 * np.pi, 200)
                ax.plot(297.94 * np.sin(t), 297.94 * np.cos(t), color="#e34948", lw=0.7, ls=":")
                ax.set_aspect("equal", adjustable="datalim")
                ax.set_title(f"KITTI {s}", fontsize=9)
                ax.grid(alpha=0.25, lw=0.5)
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
            a = axes.flat[-1]
            a.axis("off")
            h, l = axes.flat[0].get_legend_handles_labels()
            a.legend(h, l, frameon=False, fontsize=9, loc="center left")
            a.text(0, 0.1, "top-down (x, z), metres, frame-0 coordinates\ndotted red: 298 m from frame 0",
                   fontsize=8, color=INK2, transform=a.transAxes)
            fig.tight_layout()
            fig.savefig(f"{args.out}/{fname}", dpi=95)
            plt.close(fig)

    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    open(f"{args.out}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
