#!/usr/bin/env python3
"""CPU-only scorer for nrgbd_recall_s2: raw predictions -> metrics + clouds.

Pairs with `run_cut3r_family.py --dump-dir`. The GPU phase of a cell is minutes
(ingest is ~3 min even at n500); the hours go into GT unprojection and the
KDTree passes, which need no GPU at all. Splitting them means the long half can
run on the ~1000 idle CPU cores the GPU nodes carry, instead of occupying an
H200 that nothing else can use meanwhile.

Numerically identical to the fused path: the dump stores pw[::3], and this
scorer's own [::3] of that reproduces the old pw[::9] lag-curve subset exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.memory.render_state import write_ply, heat  # noqa: E402


def load_poses(p: Path) -> np.ndarray:
    lines = [l for l in Path(p).read_text().splitlines() if l.strip()]
    n = len(lines) // 4
    return np.array([[list(map(float, lines[4 * i + j].split()))
                      for j in range(4)] for i in range(n)], np.float64)


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = (D.T @ S) / len(src)
    U, sig, Vt = np.linalg.svd(C)
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    var = (S ** 2).sum() / len(src)
    s = float((sig * np.diag(E)).sum() / var) if var > 0 else 1.0
    return s, R, dst.mean(0) - s * (R @ mu_s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path,
                    default=REPO / "scripts/memory/recall_bench/manifests")
    ap.add_argument("--posed", action="store_true",
                    help="ours: predictions are already in the GT frame at metric "
                         "scale, so no Sim(3) is fitted and the single result is "
                         "reported under both mode keys.")
    args = ap.parse_args()

    od = args.out / args.method / f"{args.scene}_n{args.tier}"
    if (od / "metrics.json").exists():
        print(f"[skip] {args.scene}_n{args.tier} already scored")
        return 0
    f = args.dump_dir / args.method / f"{args.scene}_n{args.tier}.npz"
    if not f.exists():
        print(f"[abort] no dump at {f}")
        return 1

    man = json.loads((args.manifest_dir / f"{args.scene}.json").read_text())
    root = Path(man["root"])
    ids = man["ingest_frames"][:args.tier]
    focal = float(Path(root / man["focal"]).read_text().split()[0])
    poses = load_poses(root / man["poses"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_gt = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]
    dmin, dmax = man["depth_valid_range_m"]

    d = np.load(f)
    H, W = (int(x) for x in d["hw"])
    c2w_pred = d["c2w_pred"]
    n = args.tier

    Hn, Wn = cv2.imread(str(root / f"images/img{ids[0]}.png")).shape[:2]
    K = np.array([[focal, 0, Wn / 2], [0, focal, Hn / 2], [0, 0, 1]], np.float64)
    jj, ii = np.meshgrid(np.arange(Wn), np.arange(Hn), indexing="xy")

    # GT is mode-independent, so it is built once and reused by both modes.
    gt_sub, gtrgb_sub, gd_all, col_all = [], [], [], []
    for q in range(n):
        i = ids[q]
        dep = cv2.imread(str(root / f"depth/depth{i}.png"), cv2.IMREAD_UNCHANGED)
        dep = np.nan_to_num(dep.astype(np.float64)) / man["depth_scale_mm"]
        valid = (dep > dmin) & (dep < dmax)
        pts_c = np.stack([(jj - K[0, 2]) / K[0, 0] * dep,
                          (ii - K[1, 2]) / K[1, 1] * dep, dep], -1)
        c = c2w_gt[q]
        pw = pts_c @ c[:3, :3].T + c[:3, 3]
        rgb = cv2.cvtColor(cv2.imread(str(root / f"images/img{i}.png")),
                           cv2.COLOR_BGR2RGB)
        gt_sub.append(pw.reshape(-1, 3)[valid.reshape(-1)][::4])
        gtrgb_sub.append(rgb.reshape(-1, 3)[valid.reshape(-1)][::4])
        gd_all.append(cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST))
        # Prediction colours are the query image too, and like the GT they do
        # not depend on the mode -- build them here rather than re-reading the
        # same PNG once per mode.
        col_all.append(cv2.resize(rgb, (W, H),
                                  interpolation=cv2.INTER_AREA).reshape(-1, 3)[::3])
    G = np.concatenate(gt_sub); GC = np.concatenate(gtrgb_sub)
    print(f"[gt] {len(G):,} points", flush=True)

    res, clouds = {}, {}
    for mode in ("selfpose", "gtpose"):
        if f"pw_{mode}" not in d:
            continue
        pw_all = d[f"pw_{mode}"].astype(np.float64)      # [n, K, 3]
        zs_all = d[f"zs_{mode}"].astype(np.float64)      # [n, H, W]
        absrel, d125 = [], []
        for q in range(n):
            gd = gd_all[q]
            m = (gd > dmin) & (gd < dmax)
            zs = zs_all[q]
            if m.sum() > 100:
                absrel.append(float((np.abs(zs[m] - gd[m]) / gd[m]).mean()))
                d125.append(float((np.maximum(zs[m] / gd[m],
                                              gd[m] / zs[m]) < 1.25).mean()))
        P = np.concatenate(list(pw_all))
        C = np.concatenate(col_all)[:len(P)]
        per_q = [pw_all[q][::3] for q in range(n)]

        align = None
        if mode == "selfpose" and not args.posed:
            sA, RA, tA = umeyama_sim3(c2w_pred[:n, :3, 3], c2w_gt[:n, :3, 3])
            P = (sA * (RA @ P.T)).T + tA
            per_q = [(sA * (RA @ pq.T)).T + tA for pq in per_q]
            align = {"scale": float(sA)}

        tree = cKDTree(G[::2])
        d_acc, _ = tree.query(P, workers=-1)
        tree_p = cKDTree(P[::2])
        d_comp, _ = tree_p.query(G[::4], workers=-1)
        lag = [{"q": q, "age": n - 1 - q,
                "acc_mean": float(tree.query(pq, workers=-1)[0].mean())}
               for q, pq in enumerate(per_q)]
        res[mode] = {
            "acc_mean": float(d_acc.mean()), "acc_med": float(np.median(d_acc)),
            "comp_mean": float(d_comp.mean()),
            "comp_med": float(np.median(d_comp)),
            "chamfer": float((d_acc.mean() + d_comp.mean()) / 2),
            "depth_absrel": float(np.mean(absrel)),
            "depth_d125": float(np.mean(d125)),
            "n_queries": n, "lag": lag, "align": align,
        }
        clouds[mode] = (P, C, d_acc)
        print(f"[{args.method}|{args.scene}|n{n}|{mode}] acc {d_acc.mean():.4f} "
              f"comp {d_comp.mean():.4f} absrel {np.mean(absrel):.4f}", flush=True)

    if args.posed and "gtpose" in res:
        # aggregate.py reads whichever mode it was asked for; a posed system has
        # only one, so both keys carry it rather than leaving the table blank.
        res["selfpose"] = dict(res["gtpose"], posed=True)
        clouds.setdefault("selfpose", clouds["gtpose"])
    od.mkdir(parents=True, exist_ok=True)
    (od / "metrics.json").write_text(json.dumps(
        {"method": args.method, "scene": args.scene, "tier": n,
         "query_all_past": True, "stride": man["stride"],
         "scored_from_dump": True, **res}, indent=1))
    P, C, d_acc = clouds.get("selfpose", clouds[next(iter(clouds))])
    write_ply(od / "pred_cloud_rgb.ply", P, C.astype(np.uint8))
    lo, hi = np.percentile(d_acc, 5), np.percentile(d_acc, 95)
    write_ply(od / "pred_cloud_err.ply", P, heat(d_acc, float(lo), float(hi)))
    write_ply(od / "gt_cloud_rgb.ply", G, GC.astype(np.uint8))
    print(f"[viz] -> {od}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
