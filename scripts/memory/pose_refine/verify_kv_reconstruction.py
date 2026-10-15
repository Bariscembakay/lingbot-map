#!/usr/bin/env python3
"""Check that LingbotKV rebuilds lingbot's cached special-token K/V from the stored taps.

Runs the benchmark's LingbotMapMethod (SDPA) on one BSS scene, keeping the taps' special
tokens (as build_bench_inputs.py) and the final KV cache of global blocks 4/11/17/23, then
compares the cache's special-token K/V (anchors, evicted frames, sliding window) against
LingbotKV on the stored taps.
Usage: verify_kv_reconstruction.py --scene <BSS scene dir>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
sys.path.insert(0, str(tp.REPO / "benchmark"))
from benchmark.core.loader import BSSLoader  # noqa: E402
from benchmark.core.storage import BSSArtifact  # noqa: E402
from methods.lingbot_map import LingbotMapMethod  # noqa: E402


class _Clip:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load((tp.REPO / "benchmark/configs/methods/lingbot_map.yaml").read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw["use_sdpa"] = True
    method = LingbotMapMethod(**kw)
    model = method.model
    agg = model.aggregator
    psi = agg.patch_start_idx
    loader = BSSLoader(BSSArtifact(args.scene / "gt"), resize_context=method.resize_context)
    rgb = loader.load_rgb_list()
    images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in rgb]).float().div_(255)
    chunks, caches = [], {}

    def capture(_m, _i, output):
        chunks.append(torch.stack(output[0], 2)[0][:, :, :psi].half())

    def grab(l):
        def h(_m, _a, kwargs, _o):
            kv = kwargs.get("kv_cache")
            if isinstance(kv, dict) and kv.get(f"k_{l}") is not None:
                caches[l] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kv.items()
                             if k.startswith((f"k_{l}", f"v_{l}")) and (k in (f"k_{l}", f"v_{l}") or k.endswith("_special"))}
        return h

    hs = [model.aggregator.register_forward_hook(capture)]
    hs += [agg.global_blocks[l].attn.register_forward_hook(grab(l), with_kwargs=True) for l in tp.LAYERS]
    try:
        method._run_inference(images)
    finally:
        for h in hs:
            h.remove()
    spec = torch.cat(chunks).cuda()
    T = len(spec)
    c = _Clip()
    kfi = -(-T // 320) if T > 320 else 1
    i = torch.arange(T)
    kf = (i < 8) | ((i - 8) % kfi == 0)
    rope_f = (torch.cumsum(kf.long(), 0) - kf.long()).cuda()
    lkv = tp.LingbotKV(tp.REPO / "ckpt/lingbot-map.pt").cuda()
    kf_idx = torch.nonzero(kf).squeeze(1).cuda()
    print(f"T={T} keyframe interval {kfi}, {len(kf_idx)} cached frames; cache keys "
          + str({l: {k: tuple(v.shape) for k, v in d.items() if torch.is_tensor(v)} for l, d in caches.items()}))
    for li, l in enumerate(tp.LAYERS):
        k, v = lkv(spec[None], rope_f[None], li)                       # [1, 16, T*6, 64]
        k = k.view(16, T, 6, 64)[:, kf_idx]
        v = v.view(16, T, 6, 64)[:, kf_idx]
        d = caches[l]
        dense_k, dense_v = d[f"k_{l}"][0][:, :, :6], d[f"v_{l}"][0][:, :, :6]   # [16, F, 6, 64]
        parts_k, parts_v = [dense_k[:, :8]], [dense_v[:, :8]]
        if d.get(f"k_{l}_special") is not None:
            parts_k.append(d[f"k_{l}_special"][0]); parts_v.append(d[f"v_{l}_special"][0])
        parts_k.append(dense_k[:, 8:]); parts_v.append(dense_v[:, 8:])
        ck, cv = torch.cat(parts_k, 1).float(), torch.cat(parts_v, 1).float()
        n = min(ck.shape[1], k.shape[1])
        ek = (ck[:, :n] - k[:, :n]).abs().max().item() / ck.abs().max().item()
        ev = (cv[:, :n] - v[:, :n]).abs().max().item() / cv.abs().max().item()
        cos = torch.nn.functional.cosine_similarity(ck[:, :n].flatten(), k[:, :n].flatten(), 0).item()
        print(f"layer {l}: cache frames {ck.shape[1]} vs rebuilt {k.shape[1]}; "
              f"max rel err K {ek:.4f} V {ev:.4f}, cos(K) {cos:.5f}", flush=True)


if __name__ == "__main__":
    main()
