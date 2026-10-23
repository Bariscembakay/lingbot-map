#!/usr/bin/env python3
"""Released lingbot in the benchmark's default setting (streaming, auto keyframe interval ceil(N/320), the
method yaml) on BSS scenes, scored exactly as eval_lingbot_rel.py / abot_bench.py score theirs: the same-env
reference for protocol checks such as KITTI crop vs pad. Images go through the benchmark's own loader and
stay on the CPU (inference_streaming moves one frame at a time).
Writes <out>/<bench>_<scene>.json (metrics) and .npz (camera-to-world poses).
Usage: lingbot_default_bench.py --scenes <BSS scene dir>... --out DIR [--tag NAME]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
sys.path.insert(0, str(REPO / "scripts/memory/pose_refine"))
from benchmark.core.loader import BSSLoader  # noqa: E402
from benchmark.core.storage import BSSArtifact  # noqa: E402
from methods.lingbot_map import LingbotMapMethod  # noqa: E402
from lingbot_map.memory.losses import pose_enc_to_c2w  # noqa: E402
import train_pose_refine as tp  # noqa: E402


def load_bss_traj(p):
    a = np.loadtxt(p, comments="#")
    c2w = np.tile(np.eye(4), (len(a), 1, 1))
    c2w[:, :3] = a[:, 1:].reshape(-1, 3, 4)
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="", help="name suffix, e.g. the framing")
    ap.add_argument("--method-config", type=Path, default=REPO / "benchmark/configs/methods/lingbot_map.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.method_config.read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw.update(checkpoint=str(REPO / "ckpt/lingbot-map.pt"), use_sdpa=True)   # no FlashInfer on Leonardo
    method = LingbotMapMethod(**kw)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        name = f"{scene.parent.name}_{scene.name}{args.tag}"
        rgb = BSSLoader(BSSArtifact(scene / "gt"), resize_context=method.resize_context).load_rgb_list()
        images = torch.stack([torch.from_numpy(r).permute(2, 0, 1) for r in rgb]).float().div_(255)
        hw = tuple(images.shape[-2:])
        pose = method._run_inference(images)["pose_enc"][0].float()
        P = pose_enc_to_c2w(pose, hw).double()
        G = torch.from_numpy(load_bss_traj(scene / "gt/traj.txt"))
        assert len(P) == len(G), (len(P), len(G))
        m = {k: float(v) for k, v in tp.traj_metrics(P, G, 1.0).items()}
        tmp = args.out / f".{name}.{os.getpid()}.npz"
        np.savez(tmp, c2w=P.float().numpy(), hw=np.int32(hw))
        os.replace(tmp, args.out / f"{name}.npz")
        (args.out / f"{name}.json").write_text(json.dumps(
            {"bench": scene.parent.name, "scene": scene.name, "tag": args.tag, "frames": len(G), "hw": hw,
             "method": "lingbot_default", "metrics": m}, indent=1))
        print(f"[lingbot_default] {name} {len(G)} frames {hw}: ATE {m['ate']:.3f} m rpe_t {m['rpe_t']:.4f} "
              f"rpe_r {m['rpe_r']:.3f}", flush=True)


if __name__ == "__main__":
    main()
