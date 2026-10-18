#!/usr/bin/env python3
"""Build the 1048-frame lingbot caches from a plan_clips.py plan (one npz per clip).

lingbot streams each clip as the NRGBD / ScanNet++ caches were built (GCTStream, sliding window
64 + 8 scale frames, every frame a keyframe, bf16 aggregator), with the benchmark's FlashInfer
attention (--sdpa for the SDPA path; the first clips of tartanair-v2 / matrixcity used it). Per clip:
  specials [T, 4, 6, 2048] fp16   camera / 4 registers / scale token at taps 4, 11, 17, 23
  pose_enc [T, 9]                 lingbot's causal camera-head output
  gt_c2w   [T, 4, 4]              OpenCV camera-to-world relative to the clip's first frame, divided
                                  by gt_scale (paper 3.2: mean anchor-cloud distance from the 8 scale
                                  frames' GT depth; 1 = metric for datasets without depth)
  K [T, 3, 3], hw                 at the size lingbot was fed
  lbdino_{cls,mean,gem} [T, 1024] lingbot's own DINOv2 (aggregator.patch_embed) per frame
  megaloc [T, 8448]               MegaLoc place descriptor per frame
  rel_rot_residual_deg            lingbot-vs-GT consecutive relative rotation (convention check)
Patch tokens are not stored: tap 23 alone is 4.1 MB per frame.
Usage: build_clip_cache.py --plan plan.json --out DIR [--shard i --nshards n]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
from clip_sources import source  # noqa: E402
from lingbot_map.memory.gt import canonical_scale  # noqa: E402
from scripts.memory.build_cache import build_model  # noqa: E402

GEM_P = 3.0
A = 8


def load_megaloc():
    """From the local hub checkout and the HF cache when present: dozens of shards starting together
    re-fetch github's zipball into the same ~/.cache/torch/hub path and collide."""
    local = Path(os.environ.get("MEGALOC_DIR", Path.home() / ".cache/torch/hub/gmberton_MegaLoc_main"))
    if (local / "hubconf.py").exists():
        os.environ["HF_HUB_OFFLINE"] = "1"
        return torch.hub.load(str(local), "get_trained_model", source="local")
    return torch.hub.load("gmberton/MegaLoc", "get_trained_model")


def rel_rot_residual(P, G):
    ra = np.einsum("nji,njk->nik", P[:-1, :3, :3], P[1:, :3, :3])
    rb = np.einsum("nji,njk->nik", G[:-1, :3, :3], G[1:, :3, :3])
    m = np.einsum("nji,njk->nik", ra, rb)
    return float(np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1))).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-megaloc", action="store_true")
    ap.add_argument("--sdpa", action="store_true", help="SDPA attention instead of FlashInfer (~4x slower here)")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    ds = plan["summary"]["dataset"]
    clips = plan["clips"][args.shard::args.nshards]
    src = source(ds)
    dev = torch.device("cuda")
    # 3D RoPE frequencies are a fixed table indexed by frame: 2048 rows reproduce the first 1024 exactly and cover 1048-frame clips
    margs = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=2048, kv_cache_sliding_window=64, num_scale_frames=A,
                            use_sdpa=args.sdpa, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(margs, dev)
    agg = model.aggregator
    psi = agg.patch_start_idx
    mega = None if args.no_megaloc else load_megaloc().to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[cache] {ds}: shard {args.shard}/{args.nshards}, {len(clips)} clips, attention {'sdpa' if args.sdpa else 'flashinfer'}", flush=True)
    def clip_name(c):
        return f"{c['seq'].replace('/', '__')}__f{c['frames'][0]:06d}_s{c['stride']}_n{len(c['frames'])}"

    todo = [c for c in clips if not (args.out / f"{clip_name(c)}.npz").exists()]
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)      # decode the next clip's images while lingbot runs on this one
    nxt = pool.submit(src.frames, todo[0]["seq"], todo[0]["frames"]) if todo else None
    for ci, c in enumerate(todo):
        name = clip_name(c)
        f = args.out / f"{name}.npz"
        t0 = time.time()
        ids = c["frames"]
        imgs, K = nxt.result()
        nxt = pool.submit(src.frames, todo[ci + 1]["seq"], todo[ci + 1]["frames"]) if ci + 1 < len(todo) else None
        t_load = time.time() - t0
        x = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))).float().div_(255)
        acc = {"spec": [], "cls": [], "mean": [], "gem": []}

        def on_agg(_m, _i, o):
            acc["spec"].append(torch.stack(o[0], 2)[0][:, :, :psi].half().cpu())

        def on_dino(_m, _i, o):
            p = o["x_norm_patchtokens"].float()
            acc["cls"].append(o["x_norm_clstoken"].half().cpu())
            acc["mean"].append(p.mean(1).half().cpu())
            acc["gem"].append(p.clamp(min=1e-6).pow(GEM_P).mean(1).pow(1 / GEM_P).half().cpu())

        hs = [agg.register_forward_hook(on_agg), agg.patch_embed.register_forward_hook(on_dino)]
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model.inference_streaming(x, num_scale_frames=A, keyframe_interval=1, output_device=torch.device("cpu"))
        finally:
            for h in hs:
                h.remove()
        t_inf = time.time() - t0 - t_load
        spec = torch.cat(acc["spec"]).numpy()
        pose = pred["pose_enc"][0].float()
        assert len(spec) == len(ids) == len(pose), (len(spec), len(ids), len(pose))
        out = {"specials": spec, "pose_enc": pose.numpy(),
               "lbdino_cls": torch.cat(acc["cls"]).numpy(), "lbdino_mean": torch.cat(acc["mean"]).numpy(),
               "lbdino_gem": torch.cat(acc["gem"]).numpy()}
        if mega is not None:
            m = []
            for a in range(0, len(x), 64):
                m.append(mega((x[a:a + 64].to(dev) - mean) / std).float().cpu())
            out["megaloc"] = torch.cat(m).half().numpy()
        T = src.poses(c["seq"])[ids]
        Trel = np.linalg.inv(T[0])[None] @ T
        scale = 1.0
        if not src.metric:
            d, Kd = src.depth(c["seq"], ids[:A])
            scale = canonical_scale(d.astype(np.float64), Kd.astype(np.float64), Trel[:A], A)
        G = Trel.copy()
        G[:, :3, 3] /= scale
        hw = imgs.shape[1:3]
        P = tp.pose_enc_to_c2w(pose, hw).double().numpy()
        res = rel_rot_residual(np.linalg.inv(P[0])[None] @ P, G)
        out.update(gt_c2w=G.astype(np.float32), K=K.astype(np.float32), hw=np.int32(hw), gt_scale=np.float64(scale),
                   dataset=ds, seq=c["seq"], frame_ids=np.int32(ids), stride=np.int32(c["stride"]),
                   rel_rot_residual_deg=np.float32(res), attn_backend="sdpa" if args.sdpa else "flashinfer")
        tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **out)
        tmp.rename(f)
        print(f"[cache] {ci + 1}/{len(todo)} {name}: {len(ids)} frames {hw}, scale {scale:.3f}, rel-rot {res:.3f} deg/frame, "
              f"{len(ids) / (time.time() - t0):.1f} fps (wait-for-images {t_load:.0f} s, lingbot {t_inf:.0f} s, rest {time.time() - t0 - t_load - t_inf:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
