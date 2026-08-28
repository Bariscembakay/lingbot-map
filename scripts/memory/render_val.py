#!/usr/bin/env python3
"""Dump probe viz for a list of clips from a TRAINED checkpoint.

Model config (head, taps, state size, raymap convention) is read from the
checkpoint's own saved args, so this cannot silently render with the wrong
architecture. One npz per scene, dump_viz's format -- render with
render_state.py afterwards.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory.cut3r_state import StateMemory  # noqa: E402
from scripts.memory.train_state import Clip, dump_viz, expand_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--clips", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=160)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    t = ck["args"]
    ns = SimpleNamespace(out=args.out, no_write=t.get("no_write", False),
                         zero_state=False, head=t.get("head", "dpt"),
                         raymap_convention=t.get("raymap_convention", "cut3r"))
    print(f"[ckpt] step {ck.get('step')} head={ns.head} "
          f"state_tokens={t.get('state_tokens', 768)} taps={t.get('taps', '23')}")

    paths = expand_paths(args.clips)
    clips = [Clip(p, args.subsample, device, args.max_frames, t.get("taps", "23"))
             for p in paths]
    model = StateMemory(patch_size=14, tap_dim=clips[0].taps.shape[-1],
                        state_tokens=t.get("state_tokens", 768),
                        head_type=ns.head, grad_ckpt=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    args.out.mkdir(parents=True, exist_ok=True)
    for p, c in zip(paths, clips):
        dump_viz(model, c, ns, device, step=int(ck.get("step", 0)), tag=p.name)
        print(f"[viz] {p.name}: {len(c)} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
