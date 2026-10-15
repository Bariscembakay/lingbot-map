#!/usr/bin/env python3
"""DINOv2 global descriptors per frame of a benchmark scene, for the loop-retrieval probe.

Frames are loaded exactly as the benchmark feeds lingbot (BSSLoader, area_budget 255000,
align 14 -> 504x280 on KITTI/VBR), so DINO sees the same views as the cached lingbot tokens.
Official DINOv2 ViT-L/14-reg4; per frame: CLS, mean of patch tokens, GeM (p=3) of patch
tokens, all from the final normed outputs, stored unnormalised.
Usage: extract_dino_globals.py --scenes <BSS scene dir>... --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
sys.path.insert(0, str(REPO / "scripts/memory"))
from benchmark.core.loader import BSSLoader  # noqa: E402
from benchmark.core.storage import BSSArtifact  # noqa: E402
from benchmark.geometry.resize import ResizeContext  # noqa: E402
from benchmark.io.image import load_rgb  # noqa: E402
from dino_global_scene import MEAN, STD, load_dino  # noqa: E402

GEM_P = 3.0


class Frames(torch.utils.data.Dataset):
    def __init__(self, loader):
        self.loader, self.n = loader, loader.get_num_frames()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # BSSLoader.load_rgb_list, one frame at a time
        rgb = load_rgb(self.loader.artifact.rgb_dir / f"{i:06d}.png")
        rgb = self.loader.resize_context.apply(rgb, cv2.INTER_LINEAR)
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = load_dino(dev, "official")
    m, s = torch.tensor(MEAN, device=dev)[:, None, None], torch.tensor(STD, device=dev)[:, None, None]
    out = args.out / "dino"
    out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        name = f"{scene.parent.name}_{scene.name}"
        f = out / f"{name}.npz"
        if f.exists():
            continue
        loader = BSSLoader(BSSArtifact(scene / "gt"),
                           resize_context=ResizeContext(mode="area_budget", align=14, area_budget=255000))
        dl = torch.utils.data.DataLoader(Frames(loader), batch_size=args.bs, num_workers=args.workers)
        cls, mean, gem, hw = [], [], [], None
        for x in dl:
            hw = tuple(x.shape[-2:])
            x = (x.to(dev).float() / 255 - m) / s
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.forward_features(x)
            p = o["x_norm_patchtokens"].float()
            cls.append(o["x_norm_clstoken"].float().cpu())
            mean.append(p.mean(1).cpu())
            gem.append(p.clamp(min=1e-6).pow(GEM_P).mean(1).pow(1 / GEM_P).cpu())
        tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, cls=torch.cat(cls).half().numpy(), mean=torch.cat(mean).half().numpy(),
                 gem=torch.cat(gem).half().numpy(), hw=np.int32(hw), gem_p=np.float32(GEM_P))
        tmp.rename(f)
        print(f"[dino] {name}: {loader.get_num_frames()} frames {hw}", flush=True)


if __name__ == "__main__":
    main()
