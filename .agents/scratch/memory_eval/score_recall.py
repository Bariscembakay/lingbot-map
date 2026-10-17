#!/usr/bin/env python3
"""Score recall predictions (ours + CUT3R) against cached GT for one scene.

Both models ingested the same 16 frames and were queried with the same 16
past cameras from their final state. Ours predicts at the cache resolution;
CUT3R at its own -- CUT3R pointmaps are bilinearly resized to the GT grid.

Metrics per query view, in the query camera frame:
  L21        mean per-pixel L2 on 3D points (our training/val metric, raw scale)
  absrel     |z - z_gt| / z_gt on depth
  *_medal    same after ONE global median depth alignment per model (fair to
             CUT3R, whose metric scale is its own; ours is trained on this GT
             scale so alignment should barely move it)
"""
import argparse
import numpy as np
from pathlib import Path

D = Path(__file__).parent


def to_grid(pts, h, w):
    """Bilinear resize [B,Hs,Ws,C] -> [B,h,w,C], align_corners=False."""
    b, hs, ws, c = pts.shape
    ys = (np.arange(h) + 0.5) * hs / h - 0.5
    xs = (np.arange(w) + 0.5) * ws / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, hs - 1); y1 = np.clip(y0 + 1, 0, hs - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, ws - 1); x1 = np.clip(x0 + 1, 0, ws - 1)
    wy = np.clip(ys - y0, 0, 1)[None, :, None, None]
    wx = np.clip(xs - x0, 0, 1)[None, None, :, None]
    p00 = pts[:, y0][:, :, x0]; p01 = pts[:, y0][:, :, x1]
    p10 = pts[:, y1][:, :, x0]; p11 = pts[:, y1][:, :, x1]
    return (p00 * (1-wy) * (1-wx) + p01 * (1-wy) * wx
            + p10 * wy * (1-wx) + p11 * wy * wx)


def score(name, pts_self, gt_self, valid, align):
    z, zg = pts_self[..., 2], gt_self[..., 2]
    s = 1.0
    if align:
        m = valid & (z > 1e-6)
        s = float(np.median(zg[m]) / np.median(z[m]))
    p = pts_self * s
    l21 = np.linalg.norm(p - gt_self, axis=-1)
    absrel = np.abs(p[..., 2] - zg) / np.clip(zg, 1e-6, None)
    rows = []
    for i in range(len(p)):
        v = valid[i]
        rows.append((i, float(l21[i][v].mean()), float(absrel[i][v].mean())))
    mean_l21 = float(np.mean([r[1] for r in rows]))
    mean_ar = float(np.mean([r[2] for r in rows]))
    tag = "medal" if align else "raw  "
    print(f"{name:22s} [{tag}] scale={s:6.3f}  L21={mean_l21:.4f}  absrel={mean_ar:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path,
                    default=D / "ours_fullbptt_recall_210f741378.npz")
    ap.add_argument("--cut3r", type=Path,
                    default=D / "cut3r_recall_210f741378.npz")
    ap.add_argument("--label", default="scene 210f741378")
    ap.add_argument("--ours-name", default="ours (fullbptt@4000)")
    args = ap.parse_args()

    ours = np.load(args.ours)
    cut = np.load(args.cut3r)
    gt_self, valid = ours["gt_pts_self"], ours["valid"]
    h, w = gt_self.shape[1:3]
    n = len(gt_self)

    if "pred_pts_self" in cut and cut["pred_pts_self"].size:
        c_self = cut["pred_pts_self"]
    else:  # derive self from world via the model's own predicted query pose
        pw, c2w = cut["pred_pts_world"], cut["pred_query_c2w"]
        R, t = c2w[:, :3, :3], c2w[:, :3, 3]
        c_self = np.einsum("bji,bhwj->bhwi", R, pw - t[:, None, None, :])
    c_self = to_grid(c_self, h, w)

    print(f"{args.label}, {n} frames ingested, all {n} cameras queried "
          f"from final state; {int(valid.sum()) // n} valid GT px/view avg\n")
    for align in (False, True):
        ro = score(args.ours_name, ours["pred_pts_self"], gt_self, valid, align)
        rc = score("CUT3R (official 512)", c_self, gt_self, valid, align)
        if align:
            print(f"\nper-view L21 (median-aligned), lag = {n-1}-i frames since seen:")
            print("view  lag   ours   cut3r")
            for (i, lo, _), (_, lc, _) in zip(ro, rc):
                print(f"{i:4d}  {n-1-i:3d}  {lo:.4f}  {lc:.4f}")
        print()


if __name__ == "__main__":
    main()
