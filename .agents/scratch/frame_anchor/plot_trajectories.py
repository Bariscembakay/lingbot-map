#!/usr/bin/env python
"""Ceiling or drift? Splits CUT3R's KITTI trajectory error into heading error
(the camera's predicted optical axis vs GT, in frame-0 coordinates) and forward
speed (predicted motion along the camera's own optical axis over GT forward
motion, 10-frame windows).

  drift   -> heading error grows, forward speed stays ~1, the path curls
  ceiling -> heading stays aligned, forward speed falls to ~0 near the plateau
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = "/group/compact-3dmem/campaigns/frame_anchor"
OUT = f"{C}/_analysis_trajectories"
BLUE, ORANGE, INK2, RED = "#2a78d6", "#eb6834", "#52514e", "#e34948"
LAG = 10


def load(d, n=None):
    r = np.load(f"{d}/frames.npz")
    m = json.load(open(f"{d}/meta.json"))
    n = n or m["n_frames"]
    c = r["c2w_pred"][:n]
    g = np.linalg.inv(r["c2w_gt"][0]) @ r["c2w_gt"][:n]
    return m, c, g


def decompose(c, g):
    yaw = lambda T: np.degrees(np.arctan2(T[:, 0, 2], T[:, 2, 2]))  # noqa: E731  optical axis, x-z plane
    herr = (yaw(c) - yaw(g) + 180) % 360 - 180
    dp = c[LAG:, :3, 3] - c[:-LAG, :3, 3]
    dg = g[LAG:, :3, 3] - g[:-LAG, :3, 3]
    fwd_p = (dp * c[:-LAG, :3, 2]).sum(1)
    fwd_g = (dg * g[:-LAG, :3, 2]).sum(1)
    speed = np.where(fwd_g > 1.0, fwd_p / np.maximum(fwd_g, 1e-6), np.nan)
    dist_p = np.linalg.norm(c[:, :3, 3], axis=1)
    arc_g = np.r_[0, np.cumsum(np.linalg.norm(np.diff(g[:, :3, 3], axis=0), axis=1))]
    return herr, speed, dist_p, arc_g


def style(ax):
    ax.grid(alpha=0.25, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def straight_figure(rows):
    fig, axes = plt.subplots(len(rows), 4, figsize=(17, 3.6 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1, 1.3, 1.3, 1.3]))
    for ax, (name, c, g) in zip(axes, rows):
        herr, speed, dist_p, arc_g = decompose(c, g)
        a = ax[0]
        a.plot(g[:, 0, 3], g[:, 2, 3], color=ORANGE, lw=2, label="GT")
        a.plot(c[:, 0, 3], c[:, 2, 3], color=BLUE, lw=2, label="predicted")
        t = np.linspace(0, 2 * np.pi, 200)
        a.plot(297.94 * np.sin(t), 297.94 * np.cos(t), color=RED, lw=0.8, ls=":", label="298 m from frame 0")
        lim = max(np.abs(g[:, [0, 2], 3]).max(), 320) * 1.05
        a.set_xlim(-lim / 2, lim / 2)
        a.set_ylim(-20, lim)
        a.set_aspect("equal")
        a.set_title(f"{name}: top-down (x, z), m", fontsize=9)
        a.legend(frameon=False, fontsize=7, loc="lower right")
        a = ax[1]
        a.plot(arc_g, dist_p, color=BLUE, lw=2, label="predicted")
        a.plot(arc_g, np.linalg.norm(g[:, :3, 3], axis=1), color=ORANGE, lw=2, label="GT")
        a.axhline(297.94, color=RED, lw=0.8, ls=":")
        a.set_title("distance from frame 0 (m)", fontsize=9)
        a.legend(frameon=False, fontsize=7)
        a = ax[2]
        a.plot(arc_g, herr, color=BLUE, lw=1.5)
        a.axhline(0, color=INK2, lw=0.8, ls=":")
        a.set_title("heading error, predicted − GT (deg)", fontsize=9)
        a = ax[3]
        a.plot(arc_g[:-LAG], speed, color=BLUE, lw=1.2)
        a.axhline(1, color=INK2, lw=0.8, ls=":")
        a.set_ylim(-0.6, 1.8)
        a.set_title(f"forward speed, predicted / GT ({LAG}-frame windows)", fontsize=9)
        for a in ax[1:]:
            a.set_xlabel("GT path length travelled (m)")
        for a in ax:
            style(a)
    fig.tight_layout()
    fig.savefig(f"{OUT}/straight_segments.png", dpi=110)
    plt.close(fig)


def full_figure(seqs):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for a, (name, c, g) in zip(axes.flat, seqs):
        a.plot(g[:, 0, 3], g[:, 2, 3], color=ORANGE, lw=1.5, label="GT")
        a.plot(c[:, 0, 3], c[:, 2, 3], color=BLUE, lw=1.5, label="predicted")
        t = np.linspace(0, 2 * np.pi, 200)
        a.plot(297.94 * np.sin(t), 297.94 * np.cos(t), color=RED, lw=0.8, ls=":")
        a.set_aspect("equal", adjustable="datalim")
        a.set_title(f"KITTI {name}", fontsize=9)
        style(a)
    axes.flat[0].legend(frameon=False, fontsize=8)
    a = axes.flat[-1]
    a.axis("off")
    a.text(0, 0.5, "top-down (x, z) in frame-0 coordinates, metres\n"
           "dotted red: 298 m from frame 0\nforward pass of the palindrome run", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/kitti_full_sequences.png", dpi=100)
    plt.close(fig)


def table(seqs):
    print("\nPooled over KITTI 00-10 forward passes, binned by predicted distance from frame 0:")
    print("| pred dist (m) | n | median |heading err| (deg) | median forward speed ratio | frac. windows moving backward |")
    print("|---|---|---|---|---|")
    H, S, D = [], [], []
    for _, c, g in seqs:
        herr, speed, dist_p, _ = decompose(c, g)
        H.append(np.abs(herr[:-LAG])); S.append(speed); D.append(dist_p[:-LAG])
    H, S, D = map(np.concatenate, (H, S, D))
    for lo, hi in [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 280), (280, 300)]:
        s = (D >= lo) & (D < hi) & np.isfinite(S)
        if s.sum():
            print(f"| [{lo}, {hi}) | {s.sum()} | {np.median(H[s]):.1f} | {np.median(S[s]):.2f} | {(S[s] < 0).mean():.2f} |")
    print("\nPer sequence: final |heading err| and median forward speed while predicted dist < 100 m vs >= 150 m")
    print("| seq | GT path (m) | final |heading err| (deg) | speed <100 m | speed >=150 m |")
    print("|---|---|---|---|---|")
    for name, c, g in seqs:
        herr, speed, dist_p, arc = decompose(c, g)
        d = dist_p[:-LAG]
        lo = np.nanmedian(speed[d < 100]) if np.isfinite(speed[d < 100]).any() else np.nan
        hi = np.nanmedian(speed[d >= 150]) if np.isfinite(speed[d >= 150]).any() else np.nan
        print(f"| {name} | {arc[-1]:.0f} | {abs(herr[-1]):.0f} | {lo:.2f} | {hi:.2f} |")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for seg in ("06_f731-1100", "01_f214-484"):
        m, c, g = load(f"{C}/cut3r_pub_repeat4_s1/kitti/{seg}")
        rows.append((f"KITTI {seg} pass 1", c, g))
        herr, speed, dist_p, arc = decompose(c, g)
        print(f"{seg}: max |heading err| {np.abs(herr).max():.1f} deg; forward speed median "
              f"first 100 m of GT path {np.nanmedian(speed[arc[:-LAG] < 100]):.2f}, "
              f"after GT 300 m {np.nanmedian(speed[arc[:-LAG] > 300]):.2f}")
    straight_figure(rows)
    seqs = []
    for d in sorted(glob.glob(f"{C}/cut3r_pub_palindrome_s1/kitti/*/")):
        m, c, g = load(d.rstrip("/"))
        seqs.append((m["scene"], c, g))
    full_figure(seqs)
    table(seqs)


if __name__ == "__main__":
    main()
