#!/usr/bin/env python3
"""Precompute FROZEN CUT3R ViT-L encoder tokens for the 40 tap-cache clips.

The CUT3R control arm (train_cut3r_control.py) freezes the image encoder, so
its output per frame is a constant of the data and is computed once here: the
same 160-frame stride-40 stream train_state.py uses (Clip(subsample=2) over the
tap cache's meta.json frame_ids), decoded from the ScanNet++ mkv, preprocessed
exactly like CUT3R's demo `load_images(size=512)` (long-edge lanczos resize,
centre crop to multiples of 16 -- a no-op at 4:3 -> 384x512, ImgNorm 0.5/0.5),
then one `_encode_image` pass.

Output, one dir per clip under --out (flat, keyed by tap-cache basename):
  enc_tokens.npy  [T, 768, 1024] fp16   post-enc_norm tokens (no masked-token add)
  K_cut3r.npy     [T, 3, 3]  float32    per-frame intrinsics at 384x512
  meta.json                             frame ids, shapes, ckpt hash

The stored tokens are the ENCODER output only; `masked_ray_map_token` is a
trainable parameter and is added by the consumer at every use, so training it
does not invalidate this cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
# dust3r and croco import each other by bare top-level name (same pattern as
# lingbot_map/memory/cut3r_state.py).
sys.path.insert(0, str(REPO / "CUT3R" / "src"))
sys.path.insert(0, str(REPO / "CUT3R" / "src" / "croco"))

from lingbot_map.memory.cache_format import FP16_MAX, ClipMeta  # noqa: E402
from scripts.memory.build_cache import decode_frames, git_state, probe_video  # noqa: E402

SIZE = 512


def preprocess_like_demo(frames: np.ndarray, K_native: np.ndarray):
    """Replicate CUT3R load_images(size=512) + intrinsics bookkeeping.

    Copied from .agents/scratch/memory_eval/run_cut3r_recall.py (a scratch
    script, not importable from a package path), which itself replicates
    CUT3R/src/dust3r/utils/image.py:load_images. Returns
    (imgs [N,3,H,W] float32 in [-1,1], K_out [N,3,3], H, W).
    """
    from dust3r.utils.image import ImgNorm, _resize_pil_image
    imgs, Ks = [], []
    H = W = None
    for f, K in zip(frames, K_native):
        img = PIL.Image.fromarray(f)
        W1, H1 = img.size
        img = _resize_pil_image(img, SIZE)          # long edge -> 512, lanczos
        W_r, H_r = img.size
        sx, sy = W_r / W1, H_r / H1
        cx, cy = W_r // 2, H_r // 2
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if W_r == H_r:                              # square_ok=False branch
            halfh = int(3 * halfw / 4)
        left, top = cx - halfw, cy - halfh
        img = img.crop((left, top, cx + halfw, cy + halfh))
        W2, H2 = img.size
        K2 = K.copy()
        K2[0, 0] *= sx; K2[0, 2] = K2[0, 2] * sx - left   # noqa: E702
        K2[1, 1] *= sy; K2[1, 2] = K2[1, 2] * sy - top    # noqa: E702
        if H is None:
            H, W = H2, W2
        assert (H, W) == (H2, W2)
        imgs.append(ImgNorm(img)[None])
        Ks.append(K2)
    return torch.cat(imgs, dim=0), np.stack(Ks), H, W


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def clip_dirs(root: Path, splits, only):
    out = []
    for s in splits:
        for d in sorted((root / s).iterdir()):
            if d.is_dir() and (not only or d.name in only):
                out.append(d)
    return out


@torch.no_grad()
def encode_clip(model, clip_dir: Path, args, device) -> dict:
    meta = ClipMeta.load(clip_dir)
    assert meta.gt_convention == "opencv_c2w" and meta.gt_pose_trusted, clip_dir
    idx = list(range(0, meta.num_frames, args.subsample))
    if args.max_frames:
        idx = idx[: args.max_frames]
    frame_ids = [meta.frame_ids[i] for i in idx]

    video = Path(args.video_root) / meta.scene / "iphone" / "rgb.mkv"
    nw, nh, _ = probe_video(video)
    t0 = time.time()
    frames = decode_frames(video, frame_ids, nw, nh)
    t_dec = time.time() - t0

    # Cached K is for the tap-cache grid (meta.height x meta.width, an
    # anisotropic resize of the native frame); undo per axis, then run the
    # demo's own resize+crop math.
    K_cache = np.load(clip_dir / "gt_intrinsics.npy")[idx].astype(np.float64)
    K_native = K_cache.copy()
    K_native[:, 0, :] *= nw / meta.width
    K_native[:, 1, :] *= nh / meta.height
    imgs, K_out, H, W = preprocess_like_demo(frames, K_native)
    ph, pw = H // 16, W // 16

    t1 = time.time()
    toks = []
    for i in range(0, len(imgs), args.enc_batch):
        b = imgs[i : i + args.enc_batch].to(device)
        shape = torch.tensor([[H, W]], device=device).repeat(len(b), 1)
        (x,), _, _ = model._encode_image(b, shape)
        toks.append(x.cpu())
    toks = torch.cat(toks, 0)
    t_enc = time.time() - t1
    assert toks.shape == (len(idx), ph * pw, model.enc_embed_dim), toks.shape
    m = float(toks.abs().max())
    assert m < FP16_MAX, f"{clip_dir.name}: token max {m} overflows fp16"

    out_dir = args.out / clip_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(
        out_dir / "enc_tokens.npy", mode="w+", dtype=np.float16,
        shape=tuple(toks.shape))
    arr[:] = toks.numpy().astype(np.float16)
    arr.flush()
    np.save(out_dir / "K_cut3r.npy", K_out.astype(np.float32))
    git_sha, dirty = git_state(REPO)
    (out_dir / "meta.json").write_text(json.dumps({
        "format": "cut3r_enc_cache_v1",
        "scene": meta.scene,
        "src_cache": str(clip_dir),
        "subsample": args.subsample,
        "cache_indices": idx,
        "frame_ids": frame_ids,
        "num_frames": len(idx),
        "hw": [H, W],
        "patch_hw": [ph, pw],
        "enc_dim": model.enc_embed_dim,
        "native_wh": [nw, nh],
        "token_absmax": m,
        "ckpt": str(args.ckpt),
        "ckpt_sha256": args.ckpt_sha256,
        "git_sha": git_sha,
        "git_dirty": dirty,
    }, indent=2))
    print(f"[{clip_dir.name}] {len(idx)} frames {H}x{W} -> "
          f"{toks.numel() * 2 / 1e6:.0f} MB | decode {t_dec:.1f}s "
          f"enc {t_enc:.1f}s | tok absmax {m:.1f}", flush=True)
    return {"frames": len(idx)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=Path,
                    default=Path("/data/lingbot-tapcache-v4-40"))
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val_top", "val_median"])
    ap.add_argument("--only", nargs="*", default=[],
                    help="clip dir basenames; empty = all")
    ap.add_argument("--out", type=Path,
                    default=Path("/group/compact-3dmem/datasets/cut3r_enc_cache_v1"))
    ap.add_argument("--ckpt",
                    default="/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth")
    ap.add_argument("--video-root", default="/data/ScanNetpp/data")
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--enc-batch", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    dirs = clip_dirs(args.cache_root, args.splits, set(args.only))
    if args.only:
        missing = set(args.only) - {d.name for d in dirs}
        assert not missing, f"--only names not found: {missing}"
    print(f"[plan] {len(dirs)} clip(s) -> {args.out}", flush=True)

    device = torch.device("cuda")
    args.ckpt_sha256 = sha256_file(Path(args.ckpt))
    print(f"[ckpt] sha256 {args.ckpt_sha256}", flush=True)
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(args.ckpt).to(device).eval()

    t0, done = time.time(), 0
    for d in dirs:
        mp = args.out / d.name / "meta.json"
        if mp.exists() and not args.force:
            em = json.loads(mp.read_text())
            if em.get("ckpt_sha256") == args.ckpt_sha256 and \
                    (args.out / d.name / "enc_tokens.npy").exists():
                print(f"[{d.name}] exists, skipping", flush=True)
                continue
        encode_clip(model, d, args, device)
        done += 1
    print(f"[done] {done} built, {len(dirs) - done} skipped, "
          f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
