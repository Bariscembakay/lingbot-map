#!/usr/bin/env python3
"""Render what the state remembers, as point clouds you can open in a viewer.

Two modes:

  --viz DIR    fast, CPU-only, from the `probe_*.npz` a training run already
               dumped. Nothing to load, nothing to schedule.
  --ckpt PT    full re-render: roll the clip through the write to get s_T, then
               query EVERY k-th past camera against it and fuse the answers into
               one cloud. This is the design's "final-state probe" -- the query is
               not sequential, so the whole scene comes back in one batched pass.

Colouring is by per-point error rather than RGB, because the question these
clouds answer is "is it aligned", not "is it pretty" -- and the cache stores taps,
not images, so there is no RGB to use anyway.

  pred_*.ply   prediction, blue -> red by distance to the GT point
  gt_*.ply     ground truth, grey
  both_*.ply   the two overlaid, so a rigid offset is visible at a glance
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# blue (good) -> red (bad); a plain two-stop ramp reads more honestly than a
# perceptual colormap when the eye is looking for outliers.
def heat(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)[:, None]
    return (np.array([[0, 90, 255]]) * (1 - t) + np.array([[255, 40, 0]]) * t).astype(np.uint8)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    assert xyz.shape[0] == rgb.shape[0], (xyz.shape, rgb.shape)
    hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
           "property float x\nproperty float y\nproperty float z\n"
           "property uchar red\nproperty uchar green\nproperty uchar blue\n"
           "end_header\n")
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                    ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(hdr.encode())
        f.write(rec.tobytes())


def emit(out: Path, tag: str, pred: np.ndarray, gt: np.ndarray,
        valid: np.ndarray, conf: np.ndarray | None, conf_pct: float) -> dict:
    """One probe (or a fused set) -> pred/gt/both PLYs plus its error stats."""
    m = valid.reshape(-1)
    p, g = pred.reshape(-1, 3)[m], gt.reshape(-1, 3)[m]
    if p.size == 0:
        return {}
    err = np.linalg.norm(p - g, axis=-1)
    if conf is not None and conf_pct > 0:
        c = conf.reshape(-1)[m]
        keep = c >= np.percentile(c, conf_pct)
        p, g, err = p[keep], g[keep], err[keep]
    lo, hi = float(np.percentile(err, 5)), float(np.percentile(err, 95))
    write_ply(out / f"pred_{tag}.ply", p, heat(err, lo, hi))
    write_ply(out / f"gt_{tag}.ply", g, np.full((len(g), 3), 160, np.uint8))
    write_ply(out / f"both_{tag}.ply", np.concatenate([p, g]),
              np.concatenate([heat(err, lo, hi), np.full((len(g), 3), 160, np.uint8)]))
    # Centroid offset separates a rigid misalignment from diffuse noise: a large
    # offset with small spread means the cloud is in the wrong place, not the
    # wrong shape.
    off = float(np.linalg.norm(p.mean(0) - g.mean(0)))
    return {"tag": tag, "n": int(len(p)), "err_mean": float(err.mean()),
            "err_med": float(np.median(err)), "err_p95": hi,
            "centroid_offset": off}


def from_viz(viz_dir: Path, out: Path, conf_pct: float, frame: str = "world") -> None:
    files = sorted(viz_dir.glob("probe_*.npz"))
    if not files:
        raise SystemExit(f"no probe_*.npz in {viz_dir}")
    d = np.load(files[-1], allow_pickle=True)
    print(f"[render] {files[-1].name}  scene={d['scene']}  t={d['t']}  "
          f"lags={list(d['lags'])}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    pk, gk, ck = f"pred_{frame}", f"gt_{frame}", (
        "conf_world" if frame == "world" else "conf_self")
    for i, lag in enumerate(d["lags"]):
        r = emit(out, f"lag{int(lag):03d}", d[pk][i], d[gk][i],
                 d["valid"][i], d[ck][i], conf_pct)
        if r:
            rows.append(r)
    # Fused: every probe unioned into one cloud -- the whole-scene view.
    r = emit(out, "fused", d[pk], d[gk], d["valid"], d[ck], conf_pct)
    if r:
        rows.append(r)
    print(f"\n{'tag':>10s} {'points':>9s} {'err mean':>9s} {'err med':>9s} "
          f"{'err p95':>9s} {'centroid':>9s}")
    for r in rows:
        print(f"{r['tag']:>10s} {r['n']:9d} {r['err_mean']:9.4f} "
              f"{r['err_med']:9.4f} {r['err_p95']:9.4f} {r['centroid_offset']:9.4f}")
    print(f"\nwrote {len(rows)*3} PLYs to {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", type=Path, help="a training run's viz/ directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frame", default="world", choices=["world", "self"],
                    help="self = the queried camera's own frame, so no pose is "
                         "applied; comparing the two separates a pose error from "
                         "a geometry error")
    ap.add_argument("--conf-pct", type=float, default=0.0,
                    help="drop points below this confidence percentile (0 = keep all)")
    args = ap.parse_args()
    if not args.viz:
        raise SystemExit("--viz is required (checkpoint mode not wired yet)")
    from_viz(args.viz, args.out, args.conf_pct, args.frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
