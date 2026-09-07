#!/usr/bin/env python3
"""nrgbd_recall_s2 benchmark runner for ZipMap (state-query checkpoint).

Caveat carried openly in the results: ZipMap's released state-query model is a
fine-tune of the stage-2 (bidirectional, ref-view, NOT affine-invariant)
model, so ingestion here is one bidirectional pass over the tier's frames --
an easier setting than the strictly sequential CUT3R/TTT3R/ours ingestion.

Queries are self-revisit: rays are built from the model's own per-frame camera
estimates (pose_enc -> extrinsics), 9-ch plucker_and_common_origin rays, fed to
model.render() against the cached TTT state. ZipMap is up-to-scale, so fused
clouds are median-depth-scale aligned per (scene, tier) before metric eval.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.home() / "lingbot-map"))
sys.path.insert(0, str(Path.home() / "lingbot-map/ZipMap"))
sys.path.insert(0, str(Path.home() / "lingbot-map/ZipMap/training"))

MODEL_CONFIG = {  # verbatim from training/config/default_finetune_state_query.yaml
    "img_size": 518, "patch_size": 14, "embed_dim": 1024,
    "enable_camera": True, "enable_depth": True, "enable_local_point": True,
    "enable_nvs": True,
    "ttt_config": {"ttt_mode": True,
                   "params": {"bias": True, "head_dim": 1024, "inter_multi": 2,
                              "base_lr": 0.01, "muon_update_steps": 5,
                              "use_gate_fn": True}},
    "nvs_config": {"nvs_output_type": "rgbd",
                   "nvs_ray_cond_type": "plucker_and_common_origin",
                   "nvs_ray_cond_dim": 9},
    "other_config": {"use_gradient_checkpointing_local_point": False,
                     "use_gradient_checkpointing_depth": False,
                     "affine_invariant": False},
}
CKPT = Path.home() / "lingbot-map/ZipMap/checkpoints/checkpoint_state_query.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest-dir", type=Path, default=Path.home() /
                    "lingbot-map/scripts/memory/recall_bench/manifests")
    ap.add_argument("--tiers", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from zipmap.models.ZipMap import ZipMap  # noqa: E402
    from zipmap.utils.load_fn import load_and_preprocess_images  # noqa: E402
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402
    from zipmap.utils.geometry import closed_form_inverse_se3  # noqa: E402
    from train_utils.normalization import get_ray_conditions  # noqa: E402
    from zipmap.layers.ttt import TTTOperator  # noqa: E402
    import cv2  # noqa: E402
    from scipy.spatial import cKDTree  # noqa: E402
    from scripts.memory.render_state import write_ply, heat  # noqa: E402
    from scripts.memory.recall_bench.run_cut3r_family import umeyama_sim3  # noqa: E402

    man = json.loads((args.manifest_dir / f"{args.scene}.json").read_text())
    root = Path(man["root"])
    ids = man["ingest_frames"]
    focal = float(Path(root / man["focal"]).read_text().split()[0])
    poses = np.loadtxt(root / man["poses"]).reshape(-1, 4, 4)
    # NRGBD poses.txt is OpenGL c2w (verified: cross-frame depth consistency
    # is 1.9mm with the gl->cv flip vs 137mm raw). Convert to OpenCV before
    # anything touches geometry -- the tap-cache pipeline (gt.py
    # OPENGL_TO_OPENCV) always did; the first runner revision did not, which
    # left CUT3R self-consistently mirrored and threw ZipMap off by metres.
    poses = poses @ np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_gt = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]
    dmin, dmax = man["depth_valid_range_m"]

    model = ZipMap(**MODEL_CONFIG)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] state_query ckpt: {len(missing)} missing / "
          f"{len(unexpected)} unexpected", flush=True)
    model = model.to("cuda").eval()

    paths = [str(root / f"images/img{i}.png") for i in ids]
    images_all = load_and_preprocess_images(paths)  # [S,3,H,W] on cpu
    S_all, _, H, W = images_all.shape
    # native-res GT helpers
    probe = cv2.imread(paths[0]); Hn, Wn = probe.shape[:2]
    K = np.array([[focal, 0, Wn / 2], [0, focal, Hn / 2], [0, 0, 1]])

    def gt_points(i, c2w_rel):
        d = cv2.imread(str(root / f"depth/depth{i}.png"), cv2.IMREAD_UNCHANGED)
        d = np.nan_to_num(d.astype(np.float64)) / man["depth_scale_mm"]
        valid = (d > dmin) & (d < dmax)
        jj, ii = np.meshgrid(np.arange(Wn), np.arange(Hn), indexing="xy")
        pc = np.stack([(jj - K[0, 2]) / K[0, 0] * d,
                       (ii - K[1, 2]) / K[1, 1] * d, d], -1)
        pw = pc @ c2w_rel[:3, :3].T + c2w_rel[:3, 3]
        rgb = cv2.cvtColor(cv2.imread(str(root / f"images/img{i}.png")),
                           cv2.COLOR_BGR2RGB)
        return pw.reshape(-1, 3), valid.reshape(-1), rgb.reshape(-1, 3), d

    dtype = torch.bfloat16
    for tier in args.tiers:
        n = tier
        images = images_all[:n].to("cuda")
        t0 = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            preds = model(images, store_state=True)
        extri, intri = pose_encoding_to_extri_intri(
            preds["pose_enc"].float(), (H, W))
        extri = extri[0].cpu().numpy()  # [S,3,4] cam-from-world
        intri = intri[0].cpu().numpy()
        eh = np.concatenate([extri, np.repeat(
            np.array([[[0, 0, 0, 1.0]]]), n, 0)], 1)
        c2w_pred = closed_form_inverse_se3(eh)  # [S,4,4]
        print(f"[zipmap|{args.scene}|n{tier}] ingest {time.time()-t0:.1f}s",
              flush=True)

        ray_cond = get_ray_conditions(
            torch.from_numpy(c2w_pred[:, :3, :]).float()[None],
            torch.from_numpy(intri).float()[None], (H, W),
            type=MODEL_CONFIG["nvs_config"]["nvs_ray_cond_type"]).to("cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            # pure query on the cached state: apply-only, no TTT update --
            # the same op the forward path uses for its query tokens.
            out = model.render(
                info={"state_list": preds["state_list"],
                      "ttt_op_order": [TTTOperator(start=0, end=-1,
                                                   update=False, apply=True)]},
                ray_conditions=ray_cond, chunksize=25)
        nvs = out["nvs_pred"].float().cpu().numpy()[0]        # [S,H,W,4] rgb+d
        depth_pred = nvs[..., 3]

        # unproject predicted depth at the query cameras -> world (model frame)
        jj, ii = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
        preds_w, cols, per_q, scale_num, scale_den = [], [], [], [], []
        gts, gcols = [], []
        depth_absrel, depth_d125 = [], []
        rgb_small = None
        for q in range(n):
            Kq = intri[q]
            d = depth_pred[q]
            pc = np.stack([(jj - Kq[0, 2]) / Kq[0, 0] * d,
                           (ii - Kq[1, 2]) / Kq[1, 1] * d, d], -1)
            pw = pc @ c2w_pred[q, :3, :3].T + c2w_pred[q, :3, 3]
            gp, gv, grgb, gdep = gt_points(ids[q], c2w_gt[q])
            gd = cv2.resize(gdep, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (gd > dmin) & (gd < dmax)
            if m.sum() > 100:
                scale_num.append(np.median(gd[m]))
                scale_den.append(np.median(d[m]))
            img = cv2.cvtColor(cv2.imread(paths[q]), cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            preds_w.append(pw.reshape(-1, 3)[::3])
            cols.append(rgb_small.reshape(-1, 3)[::3])
            per_q.append((q, pw.reshape(-1, 3)[::9], d, gd, m))
            gts.append(gp[gv][::4]); gcols.append(grgb[gv][::4])

        # ZipMap's frame is its own AND up-to-scale: one Sim(3) per
        # scene-tier, fit on camera centres (the uniform per-scene alignment
        # of the benchmark spec). The per-view depth scale uses the same s.
        sA, RA, tA = umeyama_sim3(c2w_pred[:n, :3, 3], c2w_gt[:n, :3, 3])
        s = sA
        med = float(np.median(np.array(scale_num) / np.array(scale_den)))
        print(f"[zipmap|{args.scene}|n{tier}] sim3 scale {s:.4f} "
              f"(median-depth scale {med:.4f})", flush=True)
        P = (sA * (RA @ np.concatenate(preds_w).T)).T + tA
        C = np.concatenate(cols)
        G = np.concatenate(gts); GC = np.concatenate(gcols)
        for q, pq, d, gd, m in per_q:
            if m.sum() > 100:
                ds = d * s
                r = np.abs(ds[m] - gd[m]) / gd[m]
                depth_absrel.append(float(r.mean()))
                depth_d125.append(float(
                    (np.maximum(ds[m] / gd[m], gd[m] / ds[m]) < 1.25).mean()))
        tree = cKDTree(G[::2])
        d_acc, _ = tree.query(P, workers=-1)
        tree_p = cKDTree(P[::2])
        d_comp, _ = tree_p.query(G[::4], workers=-1)
        lag = [{"q": q, "age": n - 1 - q,
                "acc_mean": float(tree.query(
                    (sA * (RA @ pq.T)).T + tA, workers=-1)[0].mean())}
               for q, pq, _, _, _ in per_q]
        res = {"acc_mean": float(d_acc.mean()), "acc_med": float(np.median(d_acc)),
               "comp_mean": float(d_comp.mean()),
               "comp_med": float(np.median(d_comp)),
               "chamfer": float((d_acc.mean() + d_comp.mean()) / 2),
               "depth_absrel": float(np.mean(depth_absrel)),
               "depth_d125": float(np.mean(depth_d125)),
               "scale": s, "n_queries": n, "lag": lag,
               "ingestion": "bidirectional (state-query ckpt is a stage-2 "
                            "fine-tune; no streaming variant released)"}
        od = args.out / "zipmap" / f"{args.scene}_n{tier}"
        od.mkdir(parents=True, exist_ok=True)
        (od / "metrics.json").write_text(json.dumps(
            {"method": "zipmap", "scene": args.scene, "tier": tier,
             "query_all_past": True, "stride": man["stride"],
             "selfpose": res}, indent=1))
        write_ply(od / "pred_cloud_rgb.ply", P, C.astype(np.uint8))
        lo, hi = np.percentile(d_acc, 5), np.percentile(d_acc, 95)
        write_ply(od / "pred_cloud_err.ply", P, heat(d_acc, float(lo), float(hi)))
        write_ply(od / "gt_cloud_rgb.ply", G, GC.astype(np.uint8))
        print(f"[zipmap|{args.scene}|n{tier}] acc {d_acc.mean():.4f} comp "
              f"{d_comp.mean():.4f} absrel {np.mean(depth_absrel):.4f} -> {od}",
              flush=True)
        del preds, out
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
