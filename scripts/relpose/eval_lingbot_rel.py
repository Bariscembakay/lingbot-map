#!/usr/bin/env python3
"""Evaluate a train_lingbot_rel.py checkpoint on benchmark scenes (BSS scene dirs: KITTI, VBR, ...) at
full length.

The fine-tuned backbone streams every frame in chunks with its training context (rel: the last
`window` frames; abs: lingbot's anchors + window + trajectory memory); the head's poses are chained
(rel: every `ref_every`-th frame a reference) and scored against the scene's GT in metres: Sim3 ATE,
consecutive-frame RPE, AUC. Images are loaded and resized as the benchmark loads them for lingbot
(BSSLoader, the method yaml's area budget and alignment). Writes <out>/<bench>_<scene>.json with the
metrics and .npz with the predicted camera-to-world poses and the layer-23 camera tokens (refiner input).
--lingbot scores the untouched lingbot checkpoint the same way (abs context, its own camera head): the
reference for both arms.
Usage: eval_lingbot_rel.py --run DIR --scenes <BSS scene dir>... [--ckpt last.pt] [--out DIR]
       eval_lingbot_rel.py --lingbot --scenes <BSS scene dir>... --out DIR
       eval_lingbot_rel.py --summary DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_lingbot_rel as tl  # noqa: E402

sys.path.insert(0, str(tl.REPO / "benchmark"))
from benchmark.core.loader import BSSLoader  # noqa: E402
from benchmark.core.storage import BSSArtifact  # noqa: E402
from benchmark.geometry.resize import ResizeContext  # noqa: E402

rh, tp = tl.rh, tl.tp


def load_bss_traj(p):
    a = np.loadtxt(p, comments="#")
    c2w = np.tile(np.eye(4), (len(a), 1, 1))
    c2w[:, :3] = a[:, 1:].reshape(-1, 3, 4)
    return c2w


def load_run(run: Path, ckpt: str, dev, max_frames: int):
    if run is None:     # --lingbot: what the abs arm is at step 0
        lb = torch.load(tl.REPO / "ckpt/lingbot-map.pt", map_location="cpu", weights_only=False)
        ck = {"args": {"mode": "abs", "window": 64, "head_window": 128, "ref_every": 1}, "step": 0,
              "model": lb.get("model", lb), "head": None}
    else:
        ck = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    rel = a.mode == "rel"
    model = tl.GCTStream(enable_3d_rope=True, max_frame_num=max_frames, kv_cache_sliding_window=a.window if rel else 64,
                         kv_cache_scale_frames=tl.A, kv_cache_include_scale_frames=not rel,
                         kv_cache_cross_frame_special=not rel, use_sdpa=True, use_gradient_checkpoint=False)
    model.load_state_dict(ck["model"], strict=run is not None)
    head = rh.RelHead(tp.build_head(tl.REPO / "ckpt/lingbot-map.pt", rope_on=rel, max_frames=4096 if rel else 1024),
                      a.mode, a.head_window, prev_token=rel, init_identity=rel, ref_every=a.ref_every)
    if ck["head"] is not None:
        head.load_state_dict(ck["head"])
    return model.to(dev).eval(), head.to(dev).eval(), a, ck["step"]


@torch.no_grad()
def stream_cams(model, mode, images, chunk, window, dev):
    """[T,3,H,W] uint8 on the CPU -> [T, C] layer-23 camera tokens, as train_lingbot_rel streams."""
    T = len(images)
    starts = tl.chunk_starts(T, chunk)
    model.clean_kv_cache()
    kv = model.aggregator.kv_cache
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, f0 in enumerate(starts):
            f1 = starts[i + 1] if i + 1 < len(starts) else T
            x = images[f0:f1].to(dev, non_blocking=True).float().div_(255)[None]
            kv["_chunk_mask"] = None if i == 0 else tl.chunk_mask(model, mode, f0, f1 - f0, kv["k_0"].shape[3], window, dev)
            agg, _ = model._aggregate_features(x, num_frame_for_scale=tl.A, num_frame_per_block=f1 - f0)
            kv["_chunk_mask"] = None
            out.append(agg[-1][0, :, 0].float())
    return torch.cat(out)


def summary(d: Path):
    rows = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))]
    by = {}
    for r in rows:
        by.setdefault(r["bench"], []).append(r)
    for b, rs in by.items():
        for r in rs:
            m = r["metrics"]
            print(f"{b:>6} {r['scene']:>18} {r['frames']:>6} | ATE {m['ate']:8.3f} rpe_t {m['rpe_t']:.4f} "
                  f"rpe_r {m['rpe_r']:.3f} auc30 {m['auc30']:.3f}")
        print(f"{b:>6} {'mean (' + str(len(rs)) + ')':>18} {'':>6} | ATE "
              f"{np.mean([r['metrics']['ate'] for r in rs]):8.3f} rpe_r {np.mean([r['metrics']['rpe_r'] for r in rs]):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--scenes", nargs="+", type=Path, default=[])
    ap.add_argument("--out", type=Path, help="default: <run>/eval_<ckpt stem>")
    ap.add_argument("--chunk", type=int, default=16, help="frames per streamed chunk (outputs do not depend on it)")
    ap.add_argument("--method-config", type=Path, default=tl.REPO / "benchmark/configs/methods/lingbot_map.yaml")
    ap.add_argument("--max-frames", type=int, default=20480)
    ap.add_argument("--lingbot", action="store_true", help="evaluate the untouched lingbot checkpoint")
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    if args.summary:
        return summary(args.summary)
    dev = torch.device("cuda")
    cfg = yaml.safe_load(args.method_config.read_text())
    resize = ResizeContext(mode="area_budget", align=cfg["_align"], area_budget=cfg["_area_budget"])
    assert args.lingbot != (args.run is not None), "--run or --lingbot"
    model, head, a, step = load_run(args.run, args.ckpt, dev, args.max_frames)
    out = args.out or args.run / f"eval_{Path(args.ckpt).stem}"
    out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        bench, name = scene.parent.name, scene.name
        loader = BSSLoader(BSSArtifact(scene / "gt"), resize_context=resize)
        images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in loader.load_rgb_list()])
        hw = tuple(images.shape[-2:])
        G = load_bss_traj(scene / "gt/traj.txt")
        assert len(G) == len(images), (len(G), len(images))
        cam = stream_cams(model, a.mode, images, args.chunk, a.window, dev)
        clip = rh.EvalClip(name, cam.cpu().numpy(), G.astype(np.float32), np.zeros((len(G), 3, 3), np.float32), hw, 1.0,
                           np.zeros((len(G), 9), np.float32), dev)
        P = rh.predict_c2w(head, clip).cpu()
        m = tp.traj_metrics(P, clip.gt_c2w, 1.0)
        rec = {"bench": bench, "scene": name, "frames": len(G), "hw": hw, "run": str(args.run) if args.run else "lingbot", "ckpt": args.ckpt,
               "step": step, "mode": a.mode, "chunk": args.chunk, "metrics": {k: float(v) for k, v in m.items()}}
        tmp = out / f".{bench}_{name}.{os.getpid()}"
        np.savez(f"{tmp}.npz", c2w=P.float().numpy(), cam_tokens=cam.half().cpu().numpy(), hw=np.int32(hw))
        os.replace(f"{tmp}.npz", out / f"{bench}_{name}.npz")
        (out / f"{bench}_{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"[eval] {bench}/{name} {len(G)} frames {hw}: ATE {m['ate']:.3f} m rpe_t {m['rpe_t']:.4f} "
              f"rpe_r {m['rpe_r']:.3f} auc30 {m['auc30']:.3f}", flush=True)


if __name__ == "__main__":
    main()
