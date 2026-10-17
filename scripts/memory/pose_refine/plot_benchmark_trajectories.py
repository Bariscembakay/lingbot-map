#!/usr/bin/env python3
"""Top-down plots of lingbot's trajectory against GT for every scene of the benchmark workspace.

Per scene: lingbot's camera centres Sim3-aligned (Umeyama) to GT on the frames both have,
projected onto GT's two principal axes (the ground plane for driving, the dominant plane
indoors), with start/end markers. Title: benchmark, scene, ATE, ATE / path length, Sim3 scale.
Writes <out>/<benchmark>/<scene>.png for all scenes, copies the failures to <out>/fails/, and
an index sorted by relative ATE. A failure is relative ATE (ATE / GT path length) above
--fail-rel, or in the worst quarter of its benchmark.
Usage: plot_benchmark_trajectories.py --workspace /group/.../lingbot_map/default --out DIR
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def load(p):
    a = np.loadtxt(p, comments="#")
    a = a[None] if a.ndim == 1 else a
    return {int(r[0]): r[1:].reshape(3, 4) for r in a}


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / max((S ** 2).sum() / len(src), 1e-12)
    return s, R, md - s * R @ ms


def scene_plot(bench, scene, gt_p, lb_p, out):
    ids = sorted(set(gt_p) & set(lb_p))
    G = np.array([gt_p[i][:, 3] for i in ids])
    L = np.array([lb_p[i][:, 3] for i in ids])
    s, R, t = umeyama(L, G)
    La = s * L @ R.T + t
    ate = float(np.sqrt((np.linalg.norm(La - G, axis=1) ** 2).mean()))
    path = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    c = G.mean(0)
    _, _, Vt = np.linalg.svd(G - c, full_matrices=False)
    ax2 = Vt[:2]
    g2, l2 = (G - c) @ ax2.T, (La - c) @ ax2.T
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(g2[:, 0], g2[:, 1], "-", color="black", lw=1.6, label="GT")
    ax.plot(l2[:, 0], l2[:, 1], "-", color="tab:red", lw=1.2, label="lingbot (Sim3-aligned)")
    ax.plot(*g2[0], "o", color="tab:green", ms=8, label="start")
    ax.plot(*g2[-1], "s", color="black", ms=7, label="end (GT)")
    ax.plot(*l2[-1], "s", color="tab:red", ms=7, label="end (lingbot)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    ax.set_xlabel("GT principal axis 1 (m or scene units)")
    ax.set_ylabel("GT principal axis 2")
    ax.set_title(f"{bench} / {scene}: {len(ids)} frames\nATE {ate:.3f}, ATE/path {ate / max(path, 1e-9):.2%}, "
                 f"path {path:.1f}, Sim3 scale {s:.3f}", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"benchmark": bench, "scene": scene, "frames": len(ids), "ate": ate, "path": path,
            "rel_ate": ate / max(path, 1e-9), "sim3_scale": s, "png": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("/group/compact-3dmem/campaigns/lingbot_map/default"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fail-rel", type=float, default=0.05)
    args = ap.parse_args()
    rows = []
    for bdir in sorted(p for p in args.workspace.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")) and p.name not in ("logs", "eval")):
        for sdir in sorted(bdir.iterdir()):
            lb, gt = sdir / "lingbot_map/traj.txt", sdir / "gt/traj.txt"
            if not (lb.exists() and gt.exists()):
                continue
            try:
                rows.append(scene_plot(bdir.name, sdir.name, load(gt), load(lb), args.out / bdir.name / f"{sdir.name}.png"))
            except Exception as e:  # one malformed trajectory must not stop the rest
                print(f"[skip] {bdir.name}/{sdir.name}: {e}", flush=True)
    fails_dir = args.out / "fails"
    shutil.rmtree(fails_dir, ignore_errors=True)
    fails_dir.mkdir(parents=True)
    for b in sorted({r["benchmark"] for r in rows}):
        rb = [r for r in rows if r["benchmark"] == b]
        q75 = np.quantile([r["rel_ate"] for r in rb], 0.75)
        for r in rb:
            r["fail"] = bool(r["rel_ate"] > args.fail_rel or (len(rb) >= 4 and r["rel_ate"] >= q75))
            if r["fail"]:
                shutil.copy(r["png"], fails_dir / f"{b}__{r['scene']}.png")
    rows.sort(key=lambda r: -r["rel_ate"])
    with open(args.out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[plots] {len(rows)} scenes, {sum(r['fail'] for r in rows)} failures -> {args.out}", flush=True)
    for r in rows[:15]:
        print(f"  {r['benchmark']}/{r['scene']}: ATE {r['ate']:.3f} rel {r['rel_ate']:.1%} scale {r['sim3_scale']:.3f}"
              + ("  FAIL" if r["fail"] else ""), flush=True)


if __name__ == "__main__":
    main()
