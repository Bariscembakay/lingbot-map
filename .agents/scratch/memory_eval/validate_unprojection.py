#!/usr/bin/env python3
"""Closed-loop unprojection validation, no model involved.

For query cameras of a cached clip:
  V1  unit rays (raydepth/smallread path): s_gt = ||gt_world - o||, rebuild
      p = o + s_gt * d. Residual = distance of the GT point OFF our ray.
  V2  non-unit rays (lingbot heads): z_gt = gt_self z, rebuild p = o + z_gt*d,
      compare in world AND map to self via the einsum used by run_probe.
  V3  rigid consistency: pose-map gt_self -> gt_world directly.
All three should be ~float-precision; structure means a convention bug."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.train_state import Clip  # noqa: E402

clip = Clip(Path("/data/lingbot-tapcache-v4-40/train/285efbc7cf_c0"), 2,
            torch.device("cpu"), 96, "23")
qs = list(range(0, 96, 12))
rm, xs, xw, valid = clip.probe_inputs(qs, "cut3r", anchor=0)
o_u, d_u, c2w = clip.probe_rays(qs, anchor=0, unit=True)
o_z, d_z, _ = clip.probe_rays(qs, anchor=0, unit=False)

def stat(name, err, v):
    e = err[v]
    print(f"{name:34s} mean {e.mean():.2e}  p99 {np.quantile(e,0.99):.2e}  max {e.max():.2e}")

xw_n, xs_n, v = xw.numpy(), xs.numpy(), valid.numpy()

# V1: off-ray residual of GT points w.r.t. unit rays
s_gt = np.linalg.norm(xw_n - o_u.numpy(), axis=-1)
p1 = o_u.numpy() + s_gt[..., None] * d_u.numpy()
stat("V1 unit-ray rebuild (world)", np.linalg.norm(p1 - xw_n, axis=-1), v)

# V2: z-depth rebuild on non-unit rays
z_gt = xs_n[..., 2]
p2 = o_z.numpy() + z_gt[..., None] * d_z.numpy()
stat("V2 z-ray rebuild (world)", np.linalg.norm(p2 - xw_n, axis=-1), v)
R = c2w[:, :3, :3]; t = c2w[:, :3, 3]
p2s = torch.einsum("bji,bhwj->bhwi", R, torch.from_numpy(p2).float()
                   - t[:, None, None, :]).numpy()
stat("V2 z-ray rebuild -> self", np.linalg.norm(p2s - xs_n, axis=-1), v)

# V3: rigid self->world with the probe's own pose
p3 = torch.einsum("bij,bhwj->bhwi", R, xs) + t[:, None, None, :]
stat("V3 pose-map gt_self -> gt_world", np.linalg.norm(p3.numpy() - xw_n, axis=-1), v)

scale = float(np.linalg.norm(xw_n[v], axis=-1).mean())
print(f"[context] mean |gt point| = {scale:.3f} canonical units; "
      f"queries {qs}, valid px/view ~{int(v.sum()/len(qs))}")
