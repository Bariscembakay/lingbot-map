#!/usr/bin/env python3
"""Pose-refiner inputs for benchmark scenes (KITTI, VBR, ...), run exactly as the benchmark runs lingbot.

Builds the benchmark's own LingbotMapMethod from its method yaml (same model, BSS loader
resize, auto keyframe interval) and hooks the aggregator to keep the 6 special tokens of
taps 4/11/17/23 per frame. GT is the scene's BSS gt/traj.txt in metres (gt_scale = 1), so
ATE comes out in metres. Images stay on the CPU; inference_streaming moves one frame at a
time, which is what lets 19k-frame VBR scenes fit.
Output matches build_inputs.py, plus `bench_traj_dmax`: the largest camera-position gap to
the benchmark's saved lingbot trajectory (a reproduction check).
Usage: build_bench_inputs.py --scenes <BSS scene dir>... --out DIR [--use-sdpa]
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
from methods.lingbot_map import LingbotMapMethod  # noqa: E402  (benchmark/methods is not inside the package)
from lingbot_map.memory.losses import pose_enc_to_c2w  # noqa: E402


def load_bss_traj(p):
    a = np.loadtxt(p, comments="#")
    c2w = np.tile(np.eye(4), (len(a), 1, 1))
    c2w[:, :3] = a[:, 1:].reshape(-1, 3, 4)
    return a[:, 0].astype(int), c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method-config", type=Path, default=REPO / "benchmark/configs/methods/lingbot_map.yaml")
    ap.add_argument("--use-sdpa", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.method_config.read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    if args.use_sdpa:
        kw["use_sdpa"] = True
    method = LingbotMapMethod(**kw)
    model = method.model
    psi = model.aggregator.patch_start_idx
    out = args.out / "specials"
    out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        name = f"{scene.parent.name}_{scene.name}"
        f = out / f"{name}.npz"
        if f.exists():
            continue
        loader = BSSLoader(BSSArtifact(scene / "gt"), resize_context=method.resize_context)
        rgb = loader.load_rgb_list()
        images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in rgb]).float().div_(255)
        hw = tuple(images.shape[-2:])
        chunks = []

        def capture(_m, _i, output):
            chunks.append(torch.stack(output[0], 2)[0][:, :, :psi].half().cpu())

        h = model.aggregator.register_forward_hook(capture)
        try:
            pred = method._run_inference(images)
        finally:
            h.remove()
        spec = torch.cat(chunks).numpy()
        pose = pred["pose_enc"][0].float()
        assert len(spec) == len(rgb) == len(pose), (len(spec), len(rgb), len(pose))
        _, gt = load_bss_traj(scene / "gt/traj.txt")
        bidx, bench = load_bss_traj(scene / "lingbot_map/traj.txt")
        mine = pose_enc_to_c2w(pose, hw).numpy()
        dmax = float(np.abs(mine[bidx, :3, 3] - bench[:, :3, 3]).max())
        H, W = hw
        fov_h, fov_w = pose[:, 7].numpy(), pose[:, 8].numpy()
        K = np.zeros((len(pose), 3, 3), np.float32)
        K[:, 0, 0], K[:, 1, 1] = W / 2 / np.tan(fov_w / 2), H / 2 / np.tan(fov_h / 2)
        K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = W / 2, H / 2, 1
        tmp = f.with_suffix(f".tmp{os.getpid()}.npz")  # concurrent copies of a job must not share it
        np.savez(tmp, specials=spec, pose_enc=pose.numpy(), gt_c2w=gt.astype(np.float32), K=K,
                 hw=np.int32(hw), gt_scale=np.float64(1.0), source=str(scene),
                 bench_traj_dmax=np.float32(dmax))
        tmp.rename(f)
        print(f"[bench] {name}: {len(spec)} frames {hw}, |pos - benchmark lingbot| max {dmax:.3f} m",
              flush=True)


if __name__ == "__main__":
    main()
