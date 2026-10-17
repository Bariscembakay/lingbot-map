#!/usr/bin/env python3
"""Recall probe of a trained StateMemory checkpoint on one unseen scene.

Ingest N frames, then query the FINAL state with every past camera's raymap
(anchor frame 0), saving raw predictions + GT for an external scorer. This is
the deployment shape the project cares about: summary + camera -> depth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from lingbot_map.memory.cut3r_state import StateMemory  # noqa: E402
from scripts.memory.train_state import Clip, run_probe  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--subsample", type=int, default=None,
                    help="override the checkpoint's cache subsample")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    t = ck["args"]
    ns = SimpleNamespace(head=t.get("head", "dpt"),
                         raymap_convention=t.get("raymap_convention", "cut3r"))
    sub = args.subsample if args.subsample else int(t.get("subsample", 2))
    clip = Clip(args.clip, sub, device, args.frames, t.get("taps", "23"))
    model = StateMemory(patch_size=14, tap_dim=clip.taps.shape[-1],
                        state_tokens=int(t.get("state_tokens", 768)),
                        dec_depth=int(t.get("dec_depth", 12)),
                        head_type=ns.head, grad_ckpt=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"[ckpt] step {ck.get('step')} head={ns.head} frames={len(clip)}")

    state, spos = model.init_state(1, device)
    for i in range(len(clip)):
        state = model.write(state, spos, clip.tap(i), clip.patch_hw)

    qs = list(range(len(clip)))
    rm, xs, xw, valid = clip.probe_inputs(qs, ns.raymap_convention, anchor=0)
    # unit-ray rule mirrors train_state.py: lingbot-trunk heads predict z-depth,
    # which pairs with z-scaled (unit=False) rays.
    rays = (clip.probe_rays(qs, anchor=0,
                            unit=ns.head not in ("lingbot", "smallread_lingbot"))
            if ns.head in ("raydepth", "lingbot", "smallread",
                           "smallread_lingbot") else None)
    # Chunked: 96 views at once through the DPT trunk would not fit an a6000.
    chunks = []
    for a in range(0, len(qs), 16):
        b = min(a + 16, len(qs))
        r = tuple(x[a:b] for x in rays) if rays else None
        chunks.append(run_probe(model, state.expand(b - a, -1, -1),
                                spos.expand(b - a, -1, -1), rm[a:b],
                                (clip.h, clip.w), r))
    out = {k: torch.cat([c[k] for c in chunks], 0) for k in chunks[0]}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pred_pts_self=out["pts3d_in_self_view"].cpu().numpy(),
        pred_pts_world=out["pts3d_in_other_view"].cpu().numpy(),
        pred_conf=out["conf_self"].cpu().numpy(),
        gt_pts_self=xs.cpu().numpy(), gt_pts_world=xw.cpu().numpy(),
        valid=valid.cpu().numpy(), hw=np.array([clip.h, clip.w]),
        ckpt_step=np.array(ck.get("step", -1)))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
