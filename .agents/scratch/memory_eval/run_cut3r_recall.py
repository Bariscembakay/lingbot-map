#!/usr/bin/env python3
"""Official CUT3R recall probe: ingest 16 frames of an unseen ScanNet++ scene,
then query the FINAL recurrent state once per past camera with a raymap-only
view (img absent, ray_map from GT intrinsics + GT pose relative to camera 0).

Raw predictions only -- an external scorer (score_recall.py) does the metrics.

Runs against the vendored, patched CUT3R at lingbot-map/CUT3R (same upstream
commit 8bc15dc as the fresh clone at ~/CUT3R; the vendored copy carries the
three local patches documented in CUT3R/UPSTREAM.md that torch 2.8 needs:
weights_only=False, RoPE2D negative-position fallback, mv_recon --datasets).
Nothing in either CUT3R tree is modified by this script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

CUT3R_SRC = Path("/home/baris_bakay/lingbot-map/CUT3R/src")
sys.path.insert(0, str(CUT3R_SRC))  # upstream's add_ckpt_path pattern: makes `dust3r` importable

from dust3r.inference import inference, inference_step          # noqa: E402
from dust3r.model import ARCroco3DStereo                        # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera         # noqa: E402
from dust3r.utils.image import ImgNorm, _resize_pil_image       # noqa: E402

CKPT = "/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth"
SIZE = 512  # the 512-resolution DPT model


def video_resolution(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, check=True, text=True).stdout.strip()
    w, h = (int(x) for x in out.split(",")[:2])
    return w, h


def decode_frames(path: Path, frame_ids: list[int], w: int, h: int) -> np.ndarray:
    """Decode exactly `frame_ids` from the video (uniform stride), native scale.

    Copied from scripts/memory/build_cache.py so the decode path is identical
    to the one that produced the cached GT indexing.
    """
    start, stride = frame_ids[0], frame_ids[1] - frame_ids[0]
    end = frame_ids[-1]
    sel = (rf"select='between(n\,{start}\,{end})"
           rf"*not(mod(n-{start}\,{stride}))'")
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"{sel},scale={w}:{h}:flags=lanczos",
        "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    nbytes = len(frame_ids) * h * w * 3
    proc = subprocess.run(cmd, capture_output=True, check=True)
    if len(proc.stdout) != nbytes:
        raise RuntimeError(
            f"decoded {len(proc.stdout)} bytes, expected {nbytes} "
            f"({len(proc.stdout) / (h * w * 3):.1f} frames vs {len(frame_ids)})")
    return np.frombuffer(proc.stdout, np.uint8).reshape(len(frame_ids), h, w, 3)


def get_ray_map(c2w1, c2w2, intrinsics, h, w):
    """Verbatim from CUT3R src/dust3r/datasets/base/base_multiview_dataset.py
    (identical math to viser_utils.PointCloudViewer.get_ray_map). Note the
    direction is the K^-1 pixel transformed as a homogeneous POINT by the full
    relative c2w -- translation included -- then normalised. That is CUT3R's
    trained-on convention; do not 'fix' it."""
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
    """Replicate load_images(size=512): aspect-preserving long-edge resize to
    512, then a centre crop to multiples of 16 per side, then ImgNorm.
    Intrinsics get the same scale + crop applied. Returns (imgs [N,3,H,W],
    K_out [N,3,3], H, W)."""
    imgs, Ks = [], []
    H = W = None
    for f, K in zip(frames, K_native):
        img = PIL.Image.fromarray(f)
        W1, H1 = img.size
        img = _resize_pil_image(img, SIZE)  # long edge -> 512, lanczos
        W_r, H_r = img.size
        sx, sy = W_r / W1, H_r / H1
        cx, cy = W_r // 2, H_r // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if W_r == H_r:  # square_ok=False branch of load_images
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
    ap.add_argument("--cache", required=True, type=Path,
                    help="tapcache clip dir (meta.json + gt_*.npy)")
    ap.add_argument("--video", type=Path, default=None,
                    help="default: /data/ScanNetpp/data/<scene>/iphone/rgb.mkv")
    ap.add_argument("--frames", type=int, default=96,
                    help="N ingest frames = cache indices 0,2,...,2(N-1)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda"
    CACHE = args.cache
    meta = json.loads((CACHE / "meta.json").read_text())
    stride = meta["stride"]
    gt_scale = float(meta["gt_scale"])
    assert meta["gt_convention"] == "opencv_c2w" and meta["gt_pose_trusted"]
    # subsample-2 stream of the cache, fixed window start 0
    cache_idx = list(range(0, 2 * args.frames, 2))
    assert cache_idx[-1] < meta["num_frames"]
    VIDEO_FRAME_IDS = [i * stride for i in cache_idx]
    VIDEO = args.video or Path(
        f"/data/ScanNetpp/data/{meta['scene']}/iphone/rgb.mkv")
    OUT = args.out

    # GT poses: cached gt_c2w is relative-to-frame-0 with translations in
    # CANONICAL units (divided by gt_scale). CUT3R is trained on metric
    # raymaps, so restore metres before building queries.
    c2w = np.load(CACHE / "gt_c2w.npy")[cache_idx].astype(np.float64)
    assert np.allclose(c2w[0], np.eye(4), atol=1e-5)
    c2w_metric = c2w.copy()
    c2w_metric[:, :3, 3] *= gt_scale
    K_cache = np.load(CACHE / "gt_intrinsics.npy")[cache_idx].astype(np.float64)

    # Decode the ingest frames at native resolution, then apply CUT3R's own
    # demo preprocessing (PIL lanczos resize + centre crop + ImgNorm).
    W_nat, H_nat = video_resolution(VIDEO)
    frames = decode_frames(VIDEO, VIDEO_FRAME_IDS, W_nat, H_nat)
    # Cached K is for the cache grid (meta h,w); rescale to native first.
    K_native = K_cache.copy()
    K_native[:, 0, :] *= W_nat / meta["width"]
    K_native[:, 1, :] *= H_nat / meta["height"]
    imgs, K_final, H, W = preprocess_like_demo(frames, K_native)
    print(f"[pre] {len(imgs)} frames -> {H}x{W}; K[0]=\n{K_final[0]}")

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

    ingest_c2w = torch.cat(
        [pose_encoding_to_camera(p["camera_pose"]) for p in outputs["pred"]],
        0).numpy()

    # ---- Query pass: raymap-only views against the FINAL state.
    # img_mask=False makes update_mask = img_mask & update = False inside the
    # model, and inference_step never writes its returned state back anyway --
    # verified below by comparing the state before and after all queries.
    final_state = state_args[-1]  # (state_feat, state_pos, init_state_feat, mem, init_mem)
    state_snapshot = [t.clone() for t in final_state]

    q_self, q_world, q_conf, q_conf_other, q_pose = [], [], [], [], []
    t2 = time.time()
    for i in range(len(imgs)):
        raymap = get_ray_map(c2w_metric[0], c2w_metric[i], K_final[i], H, W)
        view = {
            "img": torch.full((1, 3, H, W), torch.nan),
            "ray_map": torch.from_numpy(raymap)[None].float(),  # [1,H,W,6]
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
        q_pose.append(pose_encoding_to_camera(out["camera_pose"]).numpy())
    print(f"[query] {len(imgs)} raymap-only queries in {time.time() - t2:.1f}s")

    for a, b in zip(state_snapshot, final_state):
        assert torch.equal(a, b), "state changed across queries!"
    print("[check] final state bitwise identical before/after all queries")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Ingest pointmaps/conf/rgb deliberately not saved at 96 frames: the scorer
    # only needs the query-side predictions, and the full dump would be >1 GB.
    np.savez_compressed(
        OUT,
        pred_pts_self=np.concatenate(q_self, 0).astype(np.float32),
        pred_pts_world=np.concatenate(q_world, 0).astype(np.float32),
        pred_conf=np.concatenate(q_conf, 0).astype(np.float32),
        pred_conf_other=np.concatenate(q_conf_other, 0).astype(np.float32),
        pred_query_c2w=np.concatenate(q_pose, 0).astype(np.float32),
        ingest_c2w=ingest_c2w.astype(np.float32),
        query_c2w_used=c2w_metric.astype(np.float32),
        intrinsics_used=K_final.astype(np.float32),
        gt_scale=np.float64(gt_scale),
        hw=np.array([H, W]),
        video_frame_ids=np.array(VIDEO_FRAME_IDS),
        cache_indices=np.array(cache_idx),
    )
    print(f"[saved] {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) "
          f"total {time.time() - t0:.1f}s")

    d = np.load(OUT)
    for k in d.files:
        print(f"  {k}: {d[k].shape} {d[k].dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
