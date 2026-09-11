#!/usr/bin/env python3
"""Ingestion ("predicted") cloud for a recall cell -- the upper bound.

The benchmark scores the *recall* cloud: query the state with a camera, images
withheld. This writes the cloud the same model produces on the same frames
*with* the images, so the two can be viewed and scored side by side. The gap
between them is the cost of going through memory rather than looking.

Written next to the cell as ingest_cloud_rgb.ply / ingest_cloud_err.ply, using
the identical colouring, subsampling, Sim(3) and metric code as the recall
cloud -- otherwise the comparison would measure the pipeline, not the model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-rule", choices=["cut3r", "ttt3r"], required=True)
    ap.add_argument("--repo", default=None,
                    help="vendored tree; defaults to CUT3R/ for cut3r, TTT3R/ for ttt3r")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--manifest-dir", type=Path,
                    default=REPO / "scripts/memory/recall_bench/manifests")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tree = REPO / (args.repo or ("TTT3R" if args.update_rule == "ttt3r" else "CUT3R"))
    sys.path.insert(0, str(tree))
    from add_ckpt_path import add_path_to_dust3r
    ckpt = str(tree / "src/cut3r_512_dpt_4_64.pth")
    add_path_to_dust3r(ckpt)
    from src.dust3r.inference import inference
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from run_cut3r_recall import preprocess_like_demo
    import cv2
    import torch
    from scipy.spatial import cKDTree
    from scripts.memory.render_state import write_ply, heat
    from scripts.memory.recall_bench.run_recall_score import load_poses, umeyama_sim3

    man = json.loads((args.manifest_dir / f"{args.scene}.json").read_text())
    root = Path(man["root"])
    ids = man["ingest_frames"][:args.tier]
    focal = float(Path(root / man["focal"]).read_text().split()[0])
    poses = load_poses(root / man["poses"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_gt = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]
    dmin, dmax = man["depth_valid_range_m"]

    frames = np.stack([cv2.cvtColor(cv2.imread(str(root / f"images/img{i}.png")),
                                    cv2.COLOR_BGR2RGB) for i in ids])
    Hn, Wn = frames.shape[1:3]
    K = np.array([[focal, 0, Wn / 2], [0, focal, Hn / 2], [0, 0, 1]], np.float64)
    imgs, K_final, H, W = preprocess_like_demo(
        frames, np.repeat(K[None], len(ids), 0))

    model = ARCroco3DStereo.from_pretrained(ckpt).to("cuda")
    model.config.model_update_type = args.update_rule
    model.eval()

    views = [{
        "img": imgs[i:i + 1], "ray_map": torch.full((1, 6, H, W), torch.nan),
        "true_shape": torch.from_numpy(np.int32([[H, W]])),
        "idx": i, "instance": str(i),
        "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32))[None],
        "img_mask": torch.tensor(True)[None], "ray_mask": torch.tensor(False)[None],
        "update": torch.tensor(True)[None], "reset": torch.tensor(False)[None],
    } for i in range(len(ids))]
    outputs, _ = inference(views, model, "cuda")
    c2w_pred = np.stack([pose_encoding_to_camera(o["camera_pose"]).squeeze(0).numpy()
                         for o in outputs["pred"]])

    preds, cols, gts, gcols = [], [], [], []
    jj, ii = np.meshgrid(np.arange(Wn), np.arange(Hn), indexing="xy")
    for q in range(len(ids)):
        pw = outputs["pred"][q]["pts3d_in_other_view"][0].numpy().reshape(-1, 3)
        preds.append(pw[::3])
        cols.append(cv2.resize(frames[q], (W, H),
                               interpolation=cv2.INTER_AREA).reshape(-1, 3)[::3])
        dep = cv2.imread(str(root / f"depth/depth{ids[q]}.png"), cv2.IMREAD_UNCHANGED)
        dep = np.nan_to_num(dep.astype(np.float64)) / man["depth_scale_mm"]
        v = ((dep > dmin) & (dep < dmax)).reshape(-1)
        pts_c = np.stack([(jj - K[0, 2]) / K[0, 0] * dep,
                          (ii - K[1, 2]) / K[1, 1] * dep, dep], -1)
        c = c2w_gt[q]
        gts.append((pts_c @ c[:3, :3].T + c[:3, 3]).reshape(-1, 3)[v][::4])
        gcols.append(frames[q].reshape(-1, 3)[v][::4])

    P = np.concatenate(preds); C = np.concatenate(cols)
    G = np.concatenate(gts); GC = np.concatenate(gcols)
    # same single Sim(3) on camera centres the recall cloud gets
    sA, RA, tA = umeyama_sim3(c2w_pred[:, :3, 3], c2w_gt[:, :3, 3])
    P = (sA * (RA @ P.T)).T + tA

    tree_g = cKDTree(G[::2])
    d_acc, _ = tree_g.query(P, workers=-1)
    tree_p = cKDTree(P[::2])
    d_comp, _ = tree_p.query(G[::4], workers=-1)
    print(f"[ingest|{args.update_rule}|{args.scene}|n{args.tier}] "
          f"acc {d_acc.mean():.4f} comp {d_comp.mean():.4f} "
          f"chamfer {(d_acc.mean()+d_comp.mean())/2:.4f}", flush=True)

    od = args.out / args.update_rule / f"{args.scene}_n{args.tier}"
    od.mkdir(parents=True, exist_ok=True)
    write_ply(od / "ingest_cloud_rgb.ply", P, C.astype(np.uint8))
    lo, hi = np.percentile(d_acc, 5), np.percentile(d_acc, 95)
    write_ply(od / "ingest_cloud_err.ply", P, heat(d_acc, float(lo), float(hi)))
    (od / "ingest_metrics.json").write_text(json.dumps(
        {"method": args.update_rule, "scene": args.scene, "tier": args.tier,
         "pass": "ingestion (images visible) -- upper bound, not a benchmark row",
         "acc_mean": float(d_acc.mean()), "comp_mean": float(d_comp.mean()),
         "chamfer": float((d_acc.mean() + d_comp.mean()) / 2),
         "align": {"scale": float(sA)}}, indent=1))
    print(f"[viz] -> {od}/ingest_cloud_rgb.ply", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
