#!/usr/bin/env python3
"""Official CUT3R recall probe on a 7-Scenes clip (image-file variant of
run_cut3r_recall.py -- same protocol, same raymap convention, same query API;
see cut3r_recall_notes.md). Frames come from frame-NNNNNN.color.png instead of
a video; GT c2w + intrinsics come from the 7-Scenes v4 cache built by
build_cache_7scenes.py (metric, gt_scale=1.0, relative to frame 0).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

CUT3R_SRC = Path("/home/baris_bakay/lingbot-map/CUT3R/src")
sys.path.insert(0, str(CUT3R_SRC))

from dust3r.inference import inference, inference_step          # noqa: E402
from dust3r.model import ARCroco3DStereo                        # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera         # noqa: E402
from dust3r.utils.image import ImgNorm, _resize_pil_image       # noqa: E402

CKPT = "/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth"
SIZE = 512


def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    """Verbatim from CUT3R base_multiview_dataset.py -- see cut3r_recall_notes.md
    (translation-contaminated direction is the trained-on convention)."""
    c2w = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = c2w[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    return np.concatenate([ro, rd], axis=-1)


def preprocess_like_demo(frames: np.ndarray, K_native: np.ndarray):
    """load_images(size=512) replica: long-edge lanczos to 512, centre crop to
    multiples of 16, ImgNorm; intrinsics get the same scale + crop."""
    imgs, Ks = [], []
    H = W = None
    for f, K in zip(frames, K_native):
        img = PIL.Image.fromarray(f)
        W1, H1 = img.size
        img = _resize_pil_image(img, SIZE)
        W_r, H_r = img.size
        sx, sy = W_r / W1, H_r / H1
        cx, cy = W_r // 2, H_r // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if W_r == H_r:
            halfh = int(3 * halfw / 4)
        left, top = cx - halfw, cy - halfh
        img = img.crop((left, top, cx + halfw, cy + halfh))
        W2, H2 = img.size
        K2 = K.copy()
        K2[0, 0] *= sx; K2[0, 2] = K2[0, 2] * sx - left
        K2[1, 1] *= sy; K2[1, 2] = K2[1, 2] * sy - top
        if H is None:
            H, W = H2, W2
        assert (H, W) == (H2, W2)
        imgs.append(ImgNorm(img)[None])
        Ks.append(K2)
    return torch.cat(imgs, dim=0), np.stack(Ks), H, W


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path,
                    default=Path("/data/7-Scenes/chess/seq-03"))
    ap.add_argument("--cache", type=Path, default=Path(
        "/group/compact-3dmem/campaigns/spatial_memory/eval_7scenes_cache/chess_seq03_c0"))
    ap.add_argument("--out", type=Path,
                    default=Path("/group/compact-3dmem/campaigns/spatial_memory/_recall_dumps/cut3r_recall_7scenes_chess03.npz"))
    ap.add_argument("--frame-start", type=int, default=0)
    ap.add_argument("--frame-step", type=int, default=40)
    ap.add_argument("--num-frames", type=int, default=16)
    args = ap.parse_args()
    frame_ids = [args.frame_start + i * args.frame_step
                 for i in range(args.num_frames)]

    t0 = time.time()
    device = "cuda"
    meta = json.loads((args.cache / "meta.json").read_text())
    stride = meta["stride"]
    gt_scale = float(meta["gt_scale"])
    assert meta["gt_convention"] == "opencv_c2w" and meta["gt_pose_trusted"]
    cache_idx = [f // stride for f in frame_ids]
    assert all(f % stride == 0 for f in frame_ids)

    # gt_c2w is relative-to-frame-0; for 7-Scenes it is already metric
    # (gt_scale=1.0), the multiply is kept for protocol parity with ScanNet++.
    c2w = np.load(args.cache / "gt_c2w.npy")[cache_idx].astype(np.float64)
    assert np.allclose(c2w[0], np.eye(4), atol=1e-5)
    c2w_metric = c2w.copy()
    c2w_metric[:, :3, 3] *= gt_scale
    K_cache = np.load(args.cache / "gt_intrinsics.npy")[cache_idx].astype(np.float64)

    frames = np.stack([
        np.array(PIL.Image.open(
            args.frames_dir / f"frame-{f:06d}.color.png").convert("RGB"),
            dtype=np.uint8)
        for f in frame_ids])
    nh, nw = frames.shape[1:3]
    K_native = K_cache.copy()
    K_native[:, 0, :] *= nw / meta["width"]
    K_native[:, 1, :] *= nh / meta["height"]
    imgs, K_final, H, W = preprocess_like_demo(frames, K_native)
    print(f"[pre] {len(imgs)} frames {nw}x{nh} -> {W}x{H}; K[0]=\n{K_final[0]}")

    print(f"[model] loading {CKPT}")
    model = ARCroco3DStereo.from_pretrained(CKPT).to(device)
    model.eval()

    # ---- Ingest pass: images only, state updating, as in demo.prepare_input.
    views = []
    for i in range(len(imgs)):
        views.append({
            "img": imgs[i:i + 1],
            "ray_map": torch.full((1, 6, H, W), torch.nan),
            "true_shape": torch.from_numpy(np.int32([[H, W]])),
            "idx": i,
            "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        })
    t1 = time.time()
    outputs, state_args = inference(views, model, device)
    print(f"[ingest] {len(views)} frames in {time.time() - t1:.1f}s")

    ingest_pts_self = torch.cat(
        [p["pts3d_in_self_view"] for p in outputs["pred"]], 0).numpy()
    ingest_pts_world = torch.cat(
        [p["pts3d_in_other_view"] for p in outputs["pred"]], 0).numpy()
    ingest_conf = torch.cat([p["conf_self"] for p in outputs["pred"]], 0).numpy()
    ingest_c2w = torch.cat(
        [pose_encoding_to_camera(p["camera_pose"]) for p in outputs["pred"]],
        0).numpy()

    # ---- Query pass: raymap-only views against the FINAL state.
    final_state = state_args[-1]
    state_snapshot = [t.clone() for t in final_state]

    q_self, q_world, q_conf, q_conf_other, q_rgb, q_pose = [], [], [], [], [], []
    t2 = time.time()
    for i in range(len(imgs)):
        raymap = get_ray_map(c2w_metric[0], c2w_metric[i], K_final[i], H, W)
        view = {
            "img": torch.full((1, 3, H, W), torch.nan),
            "ray_map": torch.from_numpy(raymap)[None].float(),
            "true_shape": torch.from_numpy(np.int32([[H, W]])),
            "idx": len(imgs) + i,
            "instance": str(len(imgs) + i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(False).unsqueeze(0),
            "ray_mask": torch.tensor(True).unsqueeze(0),
            "update": torch.tensor(False).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
        }
        out = inference_step(view, final_state, model, device)["pred"]
        q_self.append(out["pts3d_in_self_view"].numpy())
        q_world.append(out["pts3d_in_other_view"].numpy())
        q_conf.append(out["conf_self"].numpy())
        q_conf_other.append(out["conf"].numpy())
        q_rgb.append(out["rgb"].numpy())
        q_pose.append(pose_encoding_to_camera(out["camera_pose"]).numpy())
    print(f"[query] {len(imgs)} raymap-only queries in {time.time() - t2:.1f}s")

    for a, b in zip(state_snapshot, final_state):
        assert torch.equal(a, b), "state changed across queries!"
    print("[check] final state bitwise identical before/after all queries")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pred_pts_self=np.concatenate(q_self, 0).astype(np.float32),
        pred_pts_world=np.concatenate(q_world, 0).astype(np.float32),
        pred_conf=np.concatenate(q_conf, 0).astype(np.float32),
        pred_conf_other=np.concatenate(q_conf_other, 0).astype(np.float32),
        pred_rgb=np.concatenate(q_rgb, 0).astype(np.float32),
        pred_query_c2w=np.concatenate(q_pose, 0).astype(np.float32),
        ingest_depth=ingest_pts_self[..., 2].astype(np.float32),
        ingest_pts_world=ingest_pts_world.astype(np.float32),
        ingest_conf=ingest_conf.astype(np.float32),
        ingest_c2w=ingest_c2w.astype(np.float32),
        query_c2w_used=c2w_metric.astype(np.float32),
        intrinsics_used=K_final.astype(np.float32),
        gt_scale=np.float64(gt_scale),
        hw=np.array([H, W]),
        video_frame_ids=np.array(frame_ids),
        cache_indices=np.array(cache_idx),
    )
    print(f"[saved] {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
          f"total {time.time() - t0:.1f}s")

    d = np.load(args.out)
    for k in d.files:
        print(f"  {k}: {d[k].shape} {d[k].dtype}")

    # Sanity: model's own predicted query poses vs the fed metric GT.
    for i in (1, len(imgs) // 2, len(imgs) - 1):
        fed = np.linalg.norm(c2w_metric[i, :3, 3])
        got = float(np.linalg.norm(q_pose[i][0][:3, 3]))
        print(f"  view {i}: fed |t|={fed:.3f} m, pred |t|={got:.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
