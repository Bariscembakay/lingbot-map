#!/usr/bin/env python3
"""A format-correct cache with no teacher, so the validation suite can run
before a GPU is free.

Depth and conf are produced by actually running the frozen DPT head on the
synthetic taps, so V1 exercises the memmap/slicing/dtype plumbing. It cannot
check that the *builder* stores what it claims -- that needs a real teacher run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory import frozen                              # noqa: E402
from lingbot_map.memory.cache_format import (                       # noqa: E402
    CONF, DEPTH, FORMAT_VERSION, GT_C2W, GT_DEPTH, GT_INTRINSICS, META, POSE,
    REVISIT, TAPS, ClipMeta,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--height", type=int, default=378)
    ap.add_argument("--width", type=int, default=518)
    ap.add_argument("--heads", default="/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ph, pw = a.height // 14, a.width // 14
    P = ph * pw + 6
    L, E = a.frames, 2048
    g = torch.Generator().manual_seed(0)

    # Smooth in space and time, so the DPT head produces a plausible depth field
    # rather than noise -- V7's stability check is meaningless on white noise.
    base = torch.randn(1, 4, P, E, generator=g) * 0.5
    drift = torch.randn(L, 1, 1, E, generator=g).cumsum(0) * 0.02
    taps = (base + drift).to(torch.float16)
    np.memmap(out / TAPS, dtype=np.float16, mode="w+",
              shape=(L, 4, P, E))[:] = taps.numpy()

    head, _ = frozen.load_frozen(a.heads, torch.device("cpu"), need_camera=False)
    depth = np.zeros((L, a.height, a.width), np.float16)
    conf = np.zeros_like(depth)
    with torch.no_grad():
        for i in range(L):
            t = [taps[i, k].float().unsqueeze(0).unsqueeze(0) for k in range(4)]
            d, c = head(t, images=torch.zeros(1, 1, 3, a.height, a.width),
                        patch_start_idx=6)
            depth[i] = d[0, 0, ..., 0].numpy().astype(np.float16)
            conf[i] = c[0, 0].numpy().astype(np.float16)
    np.memmap(out / DEPTH, dtype=np.float16, mode="w+", shape=depth.shape)[:] = depth
    np.memmap(out / CONF, dtype=np.float16, mode="w+", shape=conf.shape)[:] = conf

    pose = np.zeros((L, 9), np.float32)
    pose[:, 3] = 1.0                        # identity quaternion
    pose[:, 0] = np.linspace(0, 0.5, L)     # a little translation
    pose[:, 7:] = 0.9
    np.save(out / POSE, pose)

    # GT is the teacher's own depth plus a small bias, so a loss has something to
    # reduce without the target being unreachable.
    np.memmap(out / GT_DEPTH, dtype=np.float16, mode="w+",
              shape=depth.shape)[:] = (depth.astype(np.float32) * 1.05).astype(np.float16)
    c2w = np.tile(np.eye(4, dtype=np.float32), (L, 1, 1))
    c2w[:, 0, 3] = np.linspace(0, 0.5, L)
    # Rotate as well as translate: identity rotations would let a raymap bug
    # (c2w vs w2c, or a missing rotation) pass unnoticed.
    ang = np.linspace(0, 0.6, L)
    c2w[:, 0, 0] = np.cos(ang); c2w[:, 0, 2] = np.sin(ang)
    c2w[:, 2, 0] = -np.sin(ang); c2w[:, 2, 2] = np.cos(ang)
    c2w[:, 2, 3] = np.linspace(0, 0.3, L)
    np.save(out / GT_C2W, c2w)
    f = 0.7 * a.width
    K = np.tile(np.array([[f, 0, a.width / 2], [0, f, a.height / 2], [0, 0, 1]],
                         np.float32), (L, 1, 1))
    np.save(out / GT_INTRINSICS, K)
    np.save(out / REVISIT, np.linspace(0, 0.6, L).astype(np.float32))

    meta = ClipMeta(
        format_version=FORMAT_VERSION, scene="synthetic", clip_index=0,
        frame_ids=list(range(L)), stride=1, num_frames=L,
        height=a.height, width=a.width, patch_h=ph, patch_w=pw,
        num_tokens=P, patch_start_idx=6, tap_layers=[4, 11, 17, 23],
        embed_dim=E, tap_dtype="float16", scale_frames=8,
        kv_cache_sliding_window=16, keyframe_interval=1,
        model_sha256="synthetic", git_sha="synthetic", git_dirty=False,
        gt_scale=1.0, gt_convention="synthetic",
        gt_pose_trusted=True, gt_pose_residual_deg=0.0,
        stats={"synthetic": True},
    )
    (out / META).write_text(meta.to_json())
    print(f"wrote {out}  L={L} P={P} {a.width}x{a.height}")


if __name__ == "__main__":
    main()
