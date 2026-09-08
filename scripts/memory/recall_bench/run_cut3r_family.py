#!/usr/bin/env python3
"""nrgbd_recall_s2 benchmark runner for the CUT3R family (cut3r | ttt3r rule).

Protocol (frozen in scripts/memory/recall_bench/manifests/): ingest the
manifest's stride-2 frame prefix (n100/n300/n500), then -- images withheld --
query EVERY ingested frame's camera as a raymap and decode geometry.

Two query-pose sources are computed side by side:
  selfpose: the model's own per-frame pose estimates (self-revisit; primary --
            measures memory, uncontaminated by localisation error)
  gtpose:   GT poses expressed relative to frame 0 (the model's world anchor;
            meaningful for this family because it is ~metric and frame-0
            anchored -- matches our earlier zero-shot evals)

Outputs per (scene, tier): metrics.json + RGB-coloured pred/GT clouds +
error-heat cloud, under <out>/<method>/<scene>_n<tier>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3] / "lingbot-map"
sys.path.insert(0, str(Path.home() / "lingbot-map"))
sys.path.insert(0, str(Path.home() / "lingbot-map/.agents/scratch/memory_eval"))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) minimising ||s R src + t - dst||^2."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def load_poses(p: Path) -> np.ndarray:
    vals = np.loadtxt(p).reshape(-1, 4, 4)
    return vals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-rule", choices=["cut3r", "ttt3r"], required=True)
    ap.add_argument("--repo", choices=["CUT3R", "TTT3R"], default="TTT3R",
                    help="which vendored tree provides src.dust3r; TTT3R's "
                         "carries both update rules")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest-dir", type=Path, default=Path.home() /
                    "lingbot-map/scripts/memory/recall_bench/manifests")
    ap.add_argument("--tiers", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # the vendored tree must be importable as `src.dust3r`; TTT3R's fork keeps
    # CUT3R's API and adds model.config.model_update_type.
    tree = Path.home() / "lingbot-map" / args.repo
    sys.path.insert(0, str(tree))
    from add_ckpt_path import add_path_to_dust3r  # noqa: E402
    ckpt = str(tree / "src/cut3r_512_dpt_4_64.pth")
    add_path_to_dust3r(ckpt)
    from src.dust3r.inference import inference, inference_step  # noqa: E402
    from src.dust3r.model import ARCroco3DStereo  # noqa: E402
    from src.dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
    # image/raymap conventions verbatim from the validated zero-shot eval
    from run_cut3r_recall import get_ray_map, preprocess_like_demo  # noqa: E402
    import cv2  # noqa: E402
    from scipy.spatial import cKDTree  # noqa: E402
    from scripts.memory.render_state import write_ply, heat  # noqa: E402

    man = json.loads((args.manifest_dir / f"{args.scene}.json").read_text())
    root = Path(man["root"])
    ids = man["ingest_frames"]
    focal = float(Path(root / man["focal"]).read_text().split()[0])
    poses = load_poses(root / man["poses"])
    # NRGBD poses.txt is OpenGL c2w (verified: cross-frame depth consistency
    # is 1.9mm with the gl->cv flip vs 137mm raw). Convert to OpenCV before
    # anything touches geometry -- the tap-cache pipeline (gt.py
    # OPENGL_TO_OPENCV) always did; the first runner revision did not, which
    # left CUT3R self-consistently mirrored and threw ZipMap off by metres.
    poses = poses @ np.diag([1.0, -1.0, -1.0, 1.0])

    frames = np.stack([cv2.cvtColor(cv2.imread(str(root / f"images/img{i}.png")),
                                    cv2.COLOR_BGR2RGB) for i in ids])
    Hn, Wn = frames.shape[1:3]
    K = np.array([[focal, 0, Wn / 2], [0, focal, Hn / 2], [0, 0, 1]], np.float64)
    imgs, K_final, H, W = preprocess_like_demo(frames, np.repeat(K[None], len(ids), 0))
    # un-normalised RGB (0..255) at model resolution, for colouring pred points
    rgb_small = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
                          for f in frames])
    c2w_gt = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]  # frame-0 relative

    dmax, dmin = man["depth_valid_range_m"][1], man["depth_valid_range_m"][0]

    def gt_points(i, c2w_rel):
        d = cv2.imread(str(root / f"depth/depth{i}.png"), cv2.IMREAD_UNCHANGED)
        d = np.nan_to_num(d.astype(np.float64)) / man["depth_scale_mm"]
        valid = (d > dmin) & (d < dmax)
        jj, ii = np.meshgrid(np.arange(Wn), np.arange(Hn), indexing="xy")
        pts_c = np.stack([(jj - K[0, 2]) / K[0, 0] * d,
                          (ii - K[1, 2]) / K[1, 1] * d, d], -1)
        pw = pts_c @ c2w_rel[:3, :3].T + c2w_rel[:3, 3]
        rgb = cv2.cvtColor(cv2.imread(str(root / f"images/img{i}.png")),
                           cv2.COLOR_BGR2RGB)
        return pw.reshape(-1, 3), valid.reshape(-1), rgb.reshape(-1, 3), \
            d  # native depth map for per-view metric

    model = ARCroco3DStereo.from_pretrained(ckpt).to("cuda")
    model.config.model_update_type = args.update_rule
    model.eval()
    method = args.update_rule

    for tier in args.tiers:
        n = tier
        if (args.out / method / f"{args.scene}_n{tier}" / "metrics.json").exists():
            print(f"[skip] {args.scene}_n{tier} already done", flush=True)
            continue
        views = [{
            "img": imgs[i:i + 1], "ray_map": torch.full((1, 6, H, W), torch.nan),
            "true_shape": torch.from_numpy(np.int32([[H, W]])),
            "idx": i, "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32))[None],
            "img_mask": torch.tensor(True)[None],
            "ray_mask": torch.tensor(False)[None],
            "update": torch.tensor(True)[None], "reset": torch.tensor(False)[None],
        } for i in range(n)]
        t0 = time.time()
        outputs, state_args = inference(views, model, "cuda")
        state = state_args[-1]
        # model's own pose estimate per ingested frame (self-revisit queries)
        c2w_pred = np.stack([
            pose_encoding_to_camera(o["camera_pose"]).squeeze(0).numpy()
            for o in outputs["pred"]])
        print(f"[{method}|{args.scene}|n{tier}] ingest {time.time()-t0:.0f}s",
              flush=True)

        res = {"selfpose": {}, "gtpose": {}}
        clouds = {}
        for mode in ("selfpose", "gtpose"):
            preds, gts, cols, gcols, per_q = [], [], [], [], []
            depth_absrel, depth_d125 = [], []
            for q in range(n):
                if mode == "selfpose":
                    rm = get_ray_map(np.eye(4), c2w_pred[q], K_final[q], H, W)
                else:
                    rm = get_ray_map(c2w_gt[0], c2w_gt[q], K_final[q], H, W)
                view = {
                    "img": torch.full((1, 3, H, W), torch.nan),
                    "ray_map": torch.from_numpy(rm)[None].float(),
                    "true_shape": torch.from_numpy(np.int32([[H, W]])),
                    "idx": n + q, "instance": str(n + q),
                    "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32))[None],
                    "img_mask": torch.tensor(False)[None],
                    "ray_mask": torch.tensor(True)[None],
                    "update": torch.tensor(False)[None],
                    "reset": torch.tensor(False)[None],
                }
                out = inference_step(view, state, model, "cuda")["pred"]
                pw = out["pts3d_in_other_view"][0].numpy().reshape(-1, 3)
                zs = out["pts3d_in_self_view"][0].numpy()[..., 2]
                gp, gv, grgb, gdep = gt_points(ids[q], c2w_gt[q])
                # per-view depth: pred z vs GT depth, both at model res
                gd = cv2.resize(gdep, (W, H), interpolation=cv2.INTER_NEAREST)
                m = (gd > dmin) & (gd < dmax)
                if m.sum() > 100:
                    r = np.abs(zs[m] - gd[m]) / gd[m]
                    depth_absrel.append(float(r.mean()))
                    depth_d125.append(float(
                        (np.maximum(zs[m] / gd[m], gd[m] / zs[m]) < 1.25).mean()))
                preds.append(pw[::3])
                cols.append(rgb_small[q].reshape(-1, 3)[::3])
                gts.append(gp[gv][::4])
                gcols.append(grgb[gv][::4])
                per_q.append(pw[::9])
            P = np.concatenate(preds); C = np.concatenate(cols)
            G = np.concatenate(gts); GC = np.concatenate(gcols)
            align = None
            if mode == "selfpose":
                # the cloud lives in the model's own (drifted) frame; fused
                # metrics need it in the GT frame. One Sim(3) per scene-tier,
                # fit on camera centres -- the per-scene alignment the
                # benchmark spec prescribes for every method uniformly.
                sA, RA, tA = umeyama_sim3(c2w_pred[:n, :3, 3], c2w_gt[:n, :3, 3])
                P = (sA * (RA @ P.T)).T + tA
                per_q = [(sA * (RA @ pq.T)).T + tA for pq in per_q]
                align = {"scale": sA}
            tree = cKDTree(G[::2])
            d_acc, _ = tree.query(P, workers=-1)
            tree_p = cKDTree(P[::2])
            d_comp, _ = tree_p.query(G[::4], workers=-1)
            # lag curve: per-query accuracy vs age
            lag = []
            for q, pq in enumerate(per_q):
                dq, _ = tree.query(pq, workers=-1)
                lag.append({"q": q, "age": n - 1 - q, "acc_mean": float(dq.mean())})
            res[mode] = {
                "acc_mean": float(d_acc.mean()), "acc_med": float(np.median(d_acc)),
                "comp_mean": float(d_comp.mean()),
                "comp_med": float(np.median(d_comp)),
                "chamfer": float((d_acc.mean() + d_comp.mean()) / 2),
                "depth_absrel": float(np.mean(depth_absrel)),
                "depth_d125": float(np.mean(depth_d125)),
                "n_queries": n, "lag": lag, "align": align,
            }
            clouds[mode] = (P, C, d_acc, G, GC)
            print(f"[{method}|{args.scene}|n{tier}|{mode}] acc {d_acc.mean():.4f} "
                  f"comp {d_comp.mean():.4f} absrel {np.mean(depth_absrel):.4f}",
                  flush=True)

        od = args.out / method / f"{args.scene}_n{tier}"
        od.mkdir(parents=True, exist_ok=True)
        (od / "metrics.json").write_text(json.dumps(
            {"method": method, "scene": args.scene, "tier": tier,
             "query_all_past": True, "stride": man["stride"], **res}, indent=1))
        P, C, d_acc, G, GC = clouds["selfpose"]
        write_ply(od / "pred_cloud_rgb.ply", P, C.astype(np.uint8))
        lo, hi = np.percentile(d_acc, 5), np.percentile(d_acc, 95)
        write_ply(od / "pred_cloud_err.ply", P, heat(d_acc, float(lo), float(hi)))
        write_ply(od / "gt_cloud_rgb.ply", G, GC.astype(np.uint8))
        print(f"[viz] -> {od}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
