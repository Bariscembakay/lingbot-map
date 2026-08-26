"""The recall objective: CUT3R's confidence-weighted regression, probe-only.

There is no current-frame token loss and no pose loss. Every term comes from a
raymap query -- see `.agents/spatial_memory_design.md`.
"""
from __future__ import annotations

import torch

ALPHA = 0.2   # the DUSt3R -> MASt3R -> CUT3R lineage constant; do not tune here


def conf_l21(pred: torch.Tensor, gt: torch.Tensor, conf: torch.Tensor,
             valid: torch.Tensor, alpha: float = ALPHA):
    """`mean( c * ||pred - gt||_2 - alpha * log c )` over valid pixels.

    Returns (loss, unweighted mean L21) -- the second is the number to read,
    since the weighted one is driven toward `alpha` by construction: the
    optimum is `c* = alpha / err`, so `mean(c * err) -> alpha` regardless of how
    good the prediction is.
    """
    if valid.sum() == 0:
        z = pred.sum() * 0.0
        return z, z.detach()
    err = (pred - gt).norm(dim=-1)[valid]
    c = conf[valid]
    return (c * err - alpha * torch.log(c)).mean(), err.detach().mean()


def probe_loss(out: dict, gt_self: torch.Tensor, gt_world: torch.Tensor,
               valid: torch.Tensor, alpha: float = ALPHA):
    """Both pointmap terms for one probe."""
    ls, es = conf_l21(out["pts3d_in_self_view"], gt_self, out["conf_self"], valid, alpha)
    lw, ew = conf_l21(out["pts3d_in_other_view"], gt_world, out["conf"], valid, alpha)
    return ls + lw, {"l21_self": float(es), "l21_world": float(ew)}
