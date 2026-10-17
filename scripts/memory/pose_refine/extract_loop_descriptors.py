#!/usr/bin/env python3
"""Per-frame place descriptors from lingbot itself, for the loop-retrieval probe.

lingbot runs as the benchmark runs it (build_bench_inputs.py: the benchmark's LingbotMapMethod,
BSS loader resize, auto keyframe interval), with SDPA attention; hooks keep per frame
  lbdino_{cls,mean,gem}   lingbot's own DINOv2 backbone (aggregator.patch_embed, fine-tuned):
                          CLS and mean / GeM(p=3) of its normed patch tokens
  t23_{mean,gem}          the aggregator's last-layer (tap 23, 2048-d) patch tokens, pooled
  k_L{l}, v_L{l}          mean over patch tokens of the global-attention keys (after k_norm,
                          before RoPE, so the pool does not mix in token positions) and values
                          at global layers 4/11/17/23: a pooled form of the patch K/V that
                          lingbot's KV cache holds and evicts after 64 frames
  kcam_L{l}, vcam_L{l}    the same layers' camera-token key / value (what survives eviction)
and, with --megaloc, MegaLoc (a dedicated VPR model, DINOv2-B + SALAD-style aggregation) on the
same frames as a reference. Stored unnormalised, fp16.
Usage: extract_loop_descriptors.py --scenes <BSS scene dir>... --out DIR [--megaloc]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
from benchmark.core.loader import BSSLoader  # noqa: E402
from benchmark.core.storage import BSSArtifact  # noqa: E402
from methods.lingbot_map import LingbotMapMethod  # noqa: E402

LAYERS = (4, 11, 17, 23)
GEM_P = 3.0


def gem(x, dim):
    return x.float().clamp(min=1e-6).pow(GEM_P).mean(dim).pow(1 / GEM_P)


@torch.no_grad()
def megaloc(images, dev, bs=32):
    m = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    out = []
    for a in range(0, len(images), bs):
        x = (images[a:a + bs].to(dev) - mean) / std
        out.append(m(x).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method-config", type=Path, default=REPO / "benchmark/configs/methods/lingbot_map.yaml")
    ap.add_argument("--megaloc", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.method_config.read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw["use_sdpa"] = True   # the FlashInfer path can run attention through a CUDA graph that skips module hooks
    method = LingbotMapMethod(**kw)
    model = method.model
    agg = model.aggregator
    psi = agg.patch_start_idx
    args.out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        name = f"{scene.parent.name}_{scene.name}"
        f = args.out / f"{name}.npz"
        if f.exists():
            continue
        loader = BSSLoader(BSSArtifact(scene / "gt"), resize_context=method.resize_context)
        images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in loader.load_rgb_list()]).float().div_(255)
        S = len(images)
        acc = {}

        def put(k, v):
            acc.setdefault(k, []).append(v.half().cpu())

        def on_dino(_m, _i, o):
            p = o["x_norm_patchtokens"]
            put("lbdino_cls", o["x_norm_clstoken"])
            put("lbdino_mean", p.float().mean(1))
            put("lbdino_gem", gem(p, 1))

        def on_agg(_m, _i, o):
            last = o[0][-1][0]                      # [S_chunk, P, 2048], tap 23
            put("t23_mean", last[:, psi:].float().mean(1))
            put("t23_gem", gem(last[:, psi:], 1))

        def on_qkv(layer, attn):
            def hook(_m, _i, o):
                B, N, C3 = o.shape
                C = C3 // 3
                o = o.reshape(B, N, 3, attn.num_heads, C // attn.num_heads)
                k = attn.k_norm(o[:, :, 1]).flatten(2)          # [B, N, C]
                v = o[:, :, 2].flatten(2)
                P = N // n_frames[0]
                k = k.reshape(-1, P, C)
                v = v.reshape(-1, P, C)
                put(f"k_L{layer}", k[:, psi:].float().mean(1))
                put(f"v_L{layer}", v[:, psi:].float().mean(1))
                put(f"kcam_L{layer}", k[:, 0])
                put(f"vcam_L{layer}", v[:, 0])
            return hook

        n_frames = [0]

        def on_agg_pre(_m, a, kw):
            imgs = a[0] if a else kw["images"]
            n_frames[0] = imgs.shape[1] if imgs.dim() == 5 else imgs.shape[0]

        hs = [agg.patch_embed.register_forward_hook(on_dino), agg.register_forward_hook(on_agg),
              agg.register_forward_pre_hook(on_agg_pre, with_kwargs=True)]
        for l in LAYERS:
            attn = agg.global_blocks[l].attn
            hs.append(attn.qkv.register_forward_hook(on_qkv(l, attn)))
        try:
            method._run_inference(images)
        finally:
            for h in hs:
                h.remove()
        out = {k: torch.cat(v).numpy() for k, v in acc.items()}
        for k, v in out.items():
            assert len(v) == S, (k, len(v), S)
        if args.megaloc:
            out["megaloc"] = megaloc(images, next(model.parameters()).device).half().numpy()
        tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **out, gem_p=np.float32(GEM_P), layers=np.int32(LAYERS))
        tmp.rename(f)
        print(f"[desc] {name}: {S} frames, " + ", ".join(f"{k} {v.shape[1]}" for k, v in out.items()), flush=True)


if __name__ == "__main__":
    main()
