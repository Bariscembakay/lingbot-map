#!/usr/bin/env python3
"""Pose-refiner inputs, one small npz per cached clip.

From a lingbot tap cache (all frames, no subsampling, prefix from frame 0):
  specials  [T, 4, 6, 2048] fp16  camera/register/scale tokens of taps 4/11/17/23
  pose_enc  [T, 9]                lingbot's own causal camera-head output
  gt_c2w    [T, 4, 4], K [T, 3, 3], hw, gt_scale   (canonical, frame-0 relative)
With --state-ckpt, also a frozen StateMemory's state after writing frames
0..t-1, for t in stops(T) (every 16 frames, plus T):
  states    [n_stops, 768, 768] fp16, stops [n_stops]
Usage: build_inputs.py --clips list.txt --out DIR [--state-ckpt best.pt --state-label X]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402

LAYERS = (4, 11, 17, 23)
STOP_EVERY = 16


def stops(T):
    s = list(range(STOP_EVERY, T + 1, STOP_EVERY))
    return s if s and s[-1] == T else s + [T]


def load_state_model(ckpt_path, device):
    from lingbot_map.memory.cut3r_state import StateMemory
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    t = ck["args"]
    assert t.get("taps", "23") == "23", "state writer must read tap 23"
    model = StateMemory(patch_size=14, tap_dim=2048,
                        state_tokens=int(t.get("state_tokens", 768)),
                        dec_depth=int(t.get("dec_depth", 12)),
                        read_depth=int(t.get("read_depth", 2)),
                        write_oneway=bool(t.get("write_oneway", False)),
                        head_type=t.get("head", "dpt"), grad_ckpt=False)
    if any(".base.weight" in k for k in ck["model"]):
        from lingbot_map.memory.lora import inject_conv_lora
        inject_conv_lora(model.head.dpt, int(t.get("lora_rank", 16)),
                         float(t.get("lora_alpha", 16.0)))
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return model.to(device).eval(), int(t.get("subsample", 1))


@torch.no_grad()
def states_for(model, c, device):
    T = len(c)
    i23 = c.meta.tap_layers.index(23)
    st = set(stops(T))
    state, spos = model.init_state(1, device)
    out = []
    for i in range(T):
        tap = torch.from_numpy(np.ascontiguousarray(c.taps[i, i23]))[None].to(device).float()
        state = model.write(state, spos, tap, (c.meta.patch_h, c.meta.patch_w))
        if i + 1 in st:
            out.append(state[0].half().cpu().numpy())
    return np.stack(out), np.array(sorted(st), np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True, help="text file, one clip dir per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--state-ckpt", type=Path)
    ap.add_argument("--state-label")
    args = ap.parse_args()
    model = None
    if args.state_ckpt:
        dev = torch.device("cuda")
        model, sub = load_state_model(args.state_ckpt, dev)
        print(f"[state] {args.state_ckpt} (trained at subsample {sub}; written here at 1)")
    for line in args.clips.read_text().split():
        p = Path(line)
        c = ClipCache(p)
        assert tuple(c.meta.tap_layers) == LAYERS, f"{p}: taps {c.meta.tap_layers}"
        T, psi = len(c), c.meta.patch_start_idx
        sp_dir = args.out / "specials"
        sp_dir.mkdir(parents=True, exist_ok=True)
        f = sp_dir / f"{p.name}.npz"
        if not f.exists():
            spec = np.stack([np.ascontiguousarray(c.taps[:, k, :psi]) for k in range(4)], 1)
            np.savez(f, specials=spec.astype(np.float16),
                     pose_enc=c.pose_enc[:T].astype(np.float32),
                     gt_c2w=c.gt_c2w[:T].astype(np.float32),
                     K=c.gt_intrinsics[:T].astype(np.float32),
                     hw=np.int32([c.meta.height, c.meta.width]),
                     gt_scale=np.float64(c.meta.gt_scale), source=str(p))
            print(f"[specials] {p.name} T={T}", flush=True)
        if model is not None:
            st_dir = args.out / f"states_{args.state_label}"
            st_dir.mkdir(parents=True, exist_ok=True)
            g = st_dir / f"{p.name}.npz"
            if not g.exists():
                s, t = states_for(model, c, dev)
                np.savez(g, states=s, stops=t)
                print(f"[states] {p.name} {s.shape}", flush=True)


if __name__ == "__main__":
    main()
