#!/usr/bin/env python
"""Scores the repeated straight-segment runs (`--order repeat`): can the
predicted camera leave the ~298 m shell around frame 0, and does a state reset
start a new coordinate frame?"""
import argparse
import glob
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK2 = "#52514e"


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def score(d):
    r = dict(np.load(f"{d}/frames.npz"))
    m = json.load(open(f"{d}/meta.json"))
    N, K = m["n_frames"], m["repeat"]
    c = r["c2w_pred"]
    g = np.linalg.inv(r["c2w_gt"][0]) @ r["c2w_gt"]
    dist = np.linalg.norm(c[:, :3, 3], axis=1)
    passes = []
    for p in range(K):
        s = slice(p * N, (p + 1) * N)
        cp, gp = c[s], g[s]
        step_p = np.linalg.norm(np.diff(cp[:, :3, 3], axis=0), axis=1)
        step_g = np.linalg.norm(np.diff(gp[:, :3, 3], axis=0), axis=1)
        mv = step_g > 0.3
        row = dict(
            max_dist=float(dist[s].max()), raw_norm_max=float(np.log1p(dist[s].max())),
            end_dist=float(dist[s][-1]),
            within_pass_disp=float(np.linalg.norm(cp[-1, :3, 3] - cp[0, :3, 3])),
            gt_disp=float(np.linalg.norm(gp[-1, :3, 3] - gp[0, :3, 3])),
            step_ratio=float(np.median(step_p[mv] / step_g[mv])),
            first_t=float(np.linalg.norm(cp[0, :3, 3])), first_rot=rot_deg(cp[0, :3, :3]))
        if p:
            d0 = np.linalg.norm(cp[:, :3, 3] - c[:N, :3, 3], axis=1)
            row.update(vs_pass1_med=float(np.median(d0)), vs_pass1_max=float(d0.max()))
        # source frame at which pass-1 prediction stops tracking GT distance
        if p == 0:
            gd = np.linalg.norm(gp[:, :3, 3], axis=1)
            bad = np.where((gd > 20) & (dist[s] / np.maximum(gd, 1e-6) < 0.9))[0]
            row["gt_dist_at_first_10pct_shortfall"] = float(gd[bad[0]]) if len(bad) else None
        passes.append(row)
    return m, c, g, dist, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", default="/group/compact-3dmem/campaigns/frame_anchor")
    ap.add_argument("--arms", nargs="+", default=["cut3r_pub_repeat4_s1", "cut3r_pub_repeat4_reset_s1"])
    ap.add_argument("--out", default="/group/compact-3dmem/campaigns/frame_anchor/_analysis_repeat4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {}
    for arm in args.arms:
        for d in sorted(glob.glob(f"{args.campaign}/{arm}/kitti/*/")):
            runs[(arm, osp.basename(d.rstrip("/")))] = score(d.rstrip("/"))

    lines = []
    for (arm, scene), (m, c, g, dist, passes) in runs.items():
        lines.append(f"\n### {arm} / kitti {scene} — {m['n_frames']} frames × {m['repeat']}, resets after {m['resets']}\n")
        lines.append("| pass | max pred dist (m) | raw norm | end-of-pass dist | displacement in pass (pred / GT, m) | step ratio | first frame |t| (m) / rot (deg) | vs pass 1 median / max (m) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, p in enumerate(passes):
            vs = f"{p['vs_pass1_med']:.2f} / {p['vs_pass1_max']:.2f}" if "vs_pass1_med" in p else "—"
            lines.append(f"| {i + 1} | {p['max_dist']:.1f} | {p['raw_norm_max']:.3f} | {p['end_dist']:.1f} | "
                         f"{p['within_pass_disp']:.1f} / {p['gt_disp']:.1f} | {p['step_ratio']:.2f} | "
                         f"{p['first_t']:.2f} / {p['first_rot']:.2f} | {vs} |")
        lines.append(f"\nPass 1 stops tracking GT distance (pred < 0.9 GT) at GT {passes[0]['gt_dist_at_first_10pct_shortfall']} m.")

    scenes = sorted({s for _, s in runs})
    fig, axes = plt.subplots(len(args.arms) * len(scenes), 2, figsize=(12, 3.0 * len(args.arms) * len(scenes)),
                             gridspec_kw=dict(width_ratios=[2.2, 1]))
    axes = np.atleast_2d(axes)
    row = 0
    for scene in scenes:
        for arm in args.arms:
            if (arm, scene) not in runs:
                continue
            m, c, g, dist, passes = runs[(arm, scene)]
            N, K = m["n_frames"], m["repeat"]
            ax, ax2 = axes[row]
            x = np.arange(len(dist))
            gd = np.linalg.norm(g[:, :3, 3], axis=1)
            ax.plot(x, gd, color=INK2, lw=1.2, ls="--", label="GT, per pass")
            for p in range(K):
                s = slice(p * N, (p + 1) * N)
                ax.plot(x[s], dist[s], color=PASS_COLORS[p], lw=1.8, label=f"pass {p + 1}")
                ax2.plot(c[s, 0, 3], c[s, 2, 3], color=PASS_COLORS[p], lw=1.5)
            ax2.plot(g[:N, 0, 3], g[:N, 2, 3], color=INK2, lw=1.2, ls="--")
            ax.axhline(297.94, color="#e34948", lw=0.9, ls=":", label="298 m (max seen so far)")
            for p in range(1, K):
                ax.axvline(p * N - 0.5, color=INK2, lw=0.6, alpha=0.5)
            ax.set_ylabel("dist. from\nstream origin (m)")
            ax.set_title(f"kitti {scene} — {'reset between passes' if m['resets'] else 'no reset'}", fontsize=9)
            ax2.set_title("top-down (x, z); dashed = GT", fontsize=9)
            ax2.set_aspect("equal", adjustable="datalim")
            for a in (ax, ax2):
                a.grid(alpha=0.25, lw=0.5)
                for sp in ("top", "right"):
                    a.spines[sp].set_visible(False)
            if row == 0:
                ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
            row += 1
    axes[-1, 0].set_xlabel("stream position")
    fig.tight_layout()
    fig.savefig(f"{args.out}/repeat4.png", dpi=110)
    open(f"{args.out}/summary.md", "w").write("\n".join(lines) + "\n")
    json.dump({f"{a}/{s}": v[4] for (a, s), v in runs.items()}, open(f"{args.out}/summary.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
