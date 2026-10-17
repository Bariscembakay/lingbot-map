#!/usr/bin/env python3
"""Render THE WHOLE MEMORY of an overfit checkpoint: ingest the scene's 96
frames, then query the final state with every past camera and emit one cloud
per query plus the fused union -- the state's entire recallable content."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from lingbot_map.memory.cut3r_state import StateMemory            # noqa: E402
from scripts.memory.train_state import Clip, run_probe           # noqa: E402
from scripts.memory.render_state import emit, _exact_pose        # noqa: E402

CKPT = Path("/group/compact-3dmem/campaigns/spatial_memory/overfit1scene_96f_smallread/last.pt")
CLIP = Path("/data/lingbot-tapcache-v4-40/train/285efbc7cf_c0")
OUT = Path("/group/compact-3dmem/campaigns/spatial_memory/"
           "overfit96f_write_MEMORY_285efbc7cf_step3250_all96lags")
CHUNK = 12

@torch.no_grad()
def main():
    device = torch.device("cuda")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    t = ck["args"]
    head = t.get("head", "dpt")
    clip = Clip(CLIP, int(t.get("subsample", 2)), device, 96, t.get("taps", "23"))
    model = StateMemory(patch_size=14, tap_dim=clip.taps.shape[-1],
                        state_tokens=int(t.get("state_tokens", 768)),
                        dec_depth=int(t.get("dec_depth", 12)),
                        head_type=head, grad_ckpt=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"[ckpt] step {ck.get('step')} head={head} frames={len(clip)}")

    state, spos = model.init_state(1, device)
    for i in range(len(clip)):
        state = model.write(state, spos, clip.tap(i), clip.patch_hw)

    T = len(clip) - 1
    OUT.mkdir(parents=True, exist_ok=True)
    rows, preds_all, gts_all = [], [], []
    conv = t.get("raymap_convention", "cut3r")
    unit = head not in ("lingbot", "smallread_lingbot")
    for s0 in range(0, len(clip), CHUNK):
        qs = list(range(s0, min(s0 + CHUNK, len(clip))))
        rm, xs, xw, valid = clip.probe_inputs(qs, conv, anchor=0)
        rays = (clip.probe_rays(qs, anchor=0, unit=unit)
                if head in ("raydepth", "lingbot", "smallread",
                            "smallread_lingbot") else None)
        out = run_probe(model, state.expand(len(qs), -1, -1),
                        spos.expand(len(qs), -1, -1), rm, (clip.h, clip.w), rays)
        ps = out["pts3d_in_self_view"].float().cpu().numpy()
        cs = out["conf_self"].float().cpu().numpy()
        xs, xw, valid = xs.cpu().numpy(), xw.cpu().numpy(), valid.cpu().numpy()
        for j, q in enumerate(qs):
            R, tt = _exact_pose(xs[j], xw[j], valid[j])
            pw = ps[j] @ R.T + tt
            r = emit(OUT, f"memory_lag{T - q:03d}_q{q:03d}", pw, xw[j],
                     valid[j], cs[j], 0.0, hidden=True)
            if r:
                rows.append(r)
            preds_all.append(pw[valid[j]][::4])   # 4x pixel subsample for fused
            gts_all.append(xw[j][valid[j]][::4])
    emit(OUT, "memory_fused_all96", np.concatenate(preds_all)[None],
         np.concatenate(gts_all)[None],
         np.ones(len(np.concatenate(preds_all)), bool)[None],
         np.ones(len(np.concatenate(preds_all)), np.float32)[None], 0.0,
         hidden=False)
    print(f"[done] {len(rows)+1} entries -> {OUT}")

if __name__ == "__main__":
    main()
