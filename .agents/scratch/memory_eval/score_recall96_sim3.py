#!/usr/bin/env python3
"""Score the 96-frame recall probes (ours + zero-shot CUT3R) on all 8 held-out
ScanNet++ clips.

Per scene x model:
  1. per-view L21 in the query camera frame, raw + ONE global median-depth
     scale per scene x model (same convention as score_recall.py), with a
     coarse lag profile (lag = 95 - view index, in subsample-2 steps);
  2. fused-cloud eval: predicted world-frame cloud from all 96 query views
     (valid GT pixels, 4x pixel subsample each way, lowest-conf 10% per view
     dropped) aligned to the GT fused cloud with RANSAC-Umeyama Sim(3) on the
     pixel-wise correspondences, refined with ICP, then accuracy /
     completeness / chamfer. Units: canonical cache units (metres / gt_scale);
     the metre versions are also reported. The Sim(3) absorbs each model's own
     world frame + scale (CUT3R: camera-0 ~metric; ours: anchor-0 canonical),
     so the identical pipeline is fair to both.

If ICP makes chamfer worse than the Umeyama-only alignment, the scene is
flagged and the Umeyama-only numbers are used as primary (never dropped).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmark"))

from benchmark.evaluation.points import (accuracy, chamfer_distance,  # noqa: E402
                                         completeness)
from benchmark.geometry.registration import (apply_transform,  # noqa: E402
                                             icp_registration,
                                             umeyama_registration,
                                             umeyama_registration_ransac)

LAG_BUCKETS = ((0, 15), (16, 47), (48, 95))
SUB = 4          # pixel subsample, each axis
CONF_DROP = 0.10  # drop this fraction of lowest-confidence pixels per view
MAX_CORR = 200_000
RNG = np.random.default_rng(0)


def to_grid(pts, h, w):
    """Bilinear resize [B,Hs,Ws,C] -> [B,h,w,C] (same as score_recall.py)."""
    if pts.ndim == 3:
        return to_grid(pts[..., None], h, w)[..., 0]
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


def l21_metrics(pred_self, gt_self, valid):
    """Raw + median-aligned per-view L21 means and the aligned lag profile."""
    z, zg = pred_self[..., 2], gt_self[..., 2]
    m = valid & (z > 1e-6)
    s = float(np.median(zg[m]) / np.median(z[m]))
    res = {}
    for tag, scale in (("raw", 1.0), ("medal", s)):
        l21 = np.linalg.norm(pred_self * scale - gt_self, axis=-1)
        per_view = np.array([float(l21[i][valid[i]].mean())
                             for i in range(len(l21))])
        res[f"l21_{tag}"] = float(per_view.mean())
        if tag == "medal":
            n = len(per_view)
            lags = (n - 1) - np.arange(n)
            res["l21_medal_by_lag"] = {
                f"{lo}-{hi}": float(per_view[(lags >= lo) & (lags <= hi)].mean())
                for lo, hi in LAG_BUCKETS}
            res["median_scale"] = s
    return res


def fused_clouds(pred_world, gt_world, valid, conf):
    """Subsampled, conf-filtered pred cloud + GT cloud + correspondences."""
    pw = pred_world[:, ::SUB, ::SUB]
    gw = gt_world[:, ::SUB, ::SUB]
    v = valid[:, ::SUB, ::SUB]
    c = conf[:, ::SUB, ::SUB]
    keep = np.zeros_like(v)
    for i in range(len(v)):
        if v[i].sum() == 0:
            continue
        thr = np.quantile(c[i][v[i]], CONF_DROP)
        keep[i] = v[i] & (c[i] >= thr)
    pred_cloud = pw[keep].reshape(-1, 3)
    corr_gt = gw[keep].reshape(-1, 3)
    gt_cloud = gw[v].reshape(-1, 3)
    return pred_cloud, corr_gt, gt_cloud


def cloud_metrics(pred, gt):
    acc = accuracy(pred, gt)
    comp = completeness(pred, gt)
    return {"accuracy": acc, "completeness": comp,
            "chamfer": chamfer_distance(acc, comp)}


def fused_eval(pred_world, gt_world, valid, conf):
    pred_cloud, corr_gt, gt_cloud = fused_clouds(pred_world, gt_world,
                                                 valid, conf)
    n = len(pred_cloud)
    sel = (RNG.choice(n, MAX_CORR, replace=False) if n > MAX_CORR
           else np.arange(n))
    note = None
    try:
        T, inl = umeyama_registration_ransac(pred_cloud[sel], corr_gt[sel],
                                             inlier_threshold=0.1)
        inlier_frac = float(inl.mean())
        if inl.sum() < 1000:
            raise RuntimeError(f"only {int(inl.sum())} RANSAC inliers")
    except Exception as e:  # degenerate RANSAC: plain Umeyama on all corr.
        note = f"ransac failed ({e}); plain umeyama fallback"
        T = umeyama_registration(pred_cloud[sel], corr_gt[sel])
        inlier_frac = float("nan")
    pred_sim3 = apply_transform(pred_cloud, T)
    m_umeyama = cloud_metrics(pred_sim3, gt_cloud)

    T_icp = icp_registration(pred_sim3, gt_cloud, icp_threshold=0.1)
    pred_icp = apply_transform(pred_sim3, T_icp)
    m_icp = cloud_metrics(pred_icp, gt_cloud)

    icp_diverged = m_icp["chamfer"] > m_umeyama["chamfer"]
    primary = m_umeyama if icp_diverged else m_icp
    return {"umeyama": m_umeyama, "icp": m_icp, "primary": primary,
            "icp_diverged": bool(icp_diverged), "ransac_inlier_frac": inlier_frac,
            "n_pred": int(len(pred_cloud)), "n_gt": int(len(gt_cloud)),
            "note": note}


def score_scene(ours_npz, cut3r_npz, gt_scale):
    ours = np.load(ours_npz)
    cut = np.load(cut3r_npz)
    gt_self, gt_world = ours["gt_pts_self"], ours["gt_pts_world"]
    valid = ours["valid"]
    h, w = gt_self.shape[1:3]

    out = {}
    # conf for the fused-cloud filter is the WORLD pointmap's confidence
    # (CUT3R has a separate one; ours has a single conf for both outputs).
    for model, (pself, pworld, conf) in {
        "ours": (ours["pred_pts_self"], ours["pred_pts_world"],
                 ours["pred_conf"]),
        "cut3r": (to_grid(cut["pred_pts_self"], h, w),
                  to_grid(cut["pred_pts_world"], h, w),
                  to_grid(cut["pred_conf_other"], h, w)),
    }.items():
        r = l21_metrics(pself, gt_self, valid)
        r["fused"] = fused_eval(pworld, gt_world, valid, conf)
        for k in ("accuracy", "completeness", "chamfer"):
            r["fused"]["primary_m"] = r["fused"].get("primary_m", {})
            r["fused"]["primary_m"][k] = r["fused"]["primary"][k] * gt_scale
        out[model] = r
    out["gt_scale"] = gt_scale
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall-dir", required=True, type=Path,
                    help="dir holding ours_<scene>.npz / cut3r_<scene>.npz")
    ap.add_argument("--caches", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    results, splits = {}, {}
    for cache in args.caches:
        meta = json.loads((cache / "meta.json").read_text())
        scene = meta["scene"]
        splits[scene] = cache.parent.name  # val_top / val_median
        t0 = time.time()
        results[scene] = score_scene(args.recall_dir / f"ours_{scene}.npz",
                                     args.recall_dir / f"cut3r_{scene}.npz",
                                     float(meta["gt_scale"]))
        print(f"[scored] {scene} ({splits[scene]}) in {time.time()-t0:.0f}s",
              flush=True)

    # ---------------- aggregate ----------------
    def collect(model, key, scenes):
        vals = []
        for s in scenes:
            r = results[s][model]
            v = r
            for k in key.split("/"):
                v = v[k]
            vals.append(v)
        return float(np.mean(vals))

    scene_names = list(results)
    groups = {"all8": scene_names,
              "val_top": [s for s in scene_names if splits[s] == "val_top"],
              "val_median": [s for s in scene_names if splits[s] == "val_median"]}
    keys = ["l21_raw", "l21_medal",
            "l21_medal_by_lag/0-15", "l21_medal_by_lag/16-47",
            "l21_medal_by_lag/48-95",
            "fused/primary/accuracy", "fused/primary/completeness",
            "fused/primary/chamfer",
            "fused/primary_m/accuracy", "fused/primary_m/completeness",
            "fused/primary_m/chamfer"]
    agg = {g: {m: {k: collect(m, k, ss) for k in keys}
               for m in ("ours", "cut3r")}
           for g, ss in groups.items()}

    # ---------------- tables ----------------
    def fmt(v):
        return f"{v:8.4f}"

    print("\n" + "=" * 132)
    print("Per-scene (96 frames ingested, all 96 past cameras queried from the"
          " final state). Distances in CANONICAL cache units unless '_m'.")
    print("=" * 132)
    hdr = (f"{'scene':<12}{'split':<11}{'model':<7}{'L21raw':>9}{'L21med':>9}"
           f"{'lag0-15':>9}{'lag16-47':>9}{'lag48-95':>9}"
           f"{'acc':>9}{'comp':>9}{'chamfer':>9}"
           f"{'acc_m':>9}{'comp_m':>9}{'cham_m':>9}  align")
    print(hdr); print("-" * len(hdr))
    for s in scene_names:
        for m in ("ours", "cut3r"):
            r = results[s][m]
            f = r["fused"]
            tag = "umeyama-only(ICP diverged)" if f["icp_diverged"] else "sim3+icp"
            print(f"{s:<12}{splits[s]:<11}{m:<7}"
                  f"{fmt(r['l21_raw'])}{fmt(r['l21_medal'])}"
                  f"{fmt(r['l21_medal_by_lag']['0-15'])}"
                  f"{fmt(r['l21_medal_by_lag']['16-47'])}"
                  f"{fmt(r['l21_medal_by_lag']['48-95'])}"
                  f"{fmt(f['primary']['accuracy'])}"
                  f"{fmt(f['primary']['completeness'])}"
                  f"{fmt(f['primary']['chamfer'])}"
                  f"{fmt(f['primary_m']['accuracy'])}"
                  f"{fmt(f['primary_m']['completeness'])}"
                  f"{fmt(f['primary_m']['chamfer'])}  {tag}")
    print("-" * len(hdr))
    for g in ("all8", "val_top", "val_median"):
        for m in ("ours", "cut3r"):
            a = agg[g][m]
            print(f"{'MEAN':<12}{g:<11}{m:<7}"
                  f"{fmt(a['l21_raw'])}{fmt(a['l21_medal'])}"
                  f"{fmt(a['l21_medal_by_lag/0-15'])}"
                  f"{fmt(a['l21_medal_by_lag/16-47'])}"
                  f"{fmt(a['l21_medal_by_lag/48-95'])}"
                  f"{fmt(a['fused/primary/accuracy'])}"
                  f"{fmt(a['fused/primary/completeness'])}"
                  f"{fmt(a['fused/primary/chamfer'])}"
                  f"{fmt(a['fused/primary_m/accuracy'])}"
                  f"{fmt(a['fused/primary_m/completeness'])}"
                  f"{fmt(a['fused/primary_m/chamfer'])}")
    diverged = [(s, m) for s in scene_names for m in ("ours", "cut3r")
                if results[s][m]["fused"]["icp_diverged"]]
    notes = [(s, m, results[s][m]["fused"]["note"]) for s in scene_names
             for m in ("ours", "cut3r") if results[s][m]["fused"]["note"]]
    print(f"\nICP diverged (umeyama-only reported): {diverged or 'none'}")
    if notes:
        print(f"alignment notes: {notes}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"protocol": "96-frame subsample-2 ingest (cache idx 0..190 step 2), "
                     "final-state raymap queries at all 96 past cameras",
         "units": "canonical cache units (metres / gt_scale); *_m keys are "
                  "metres (x gt_scale)",
         "splits": splits, "per_scene": results, "aggregate": agg},
        indent=1, default=float))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
