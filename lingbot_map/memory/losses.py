"""lingbot-map's own training objective, reconstructed from the paper (§3.3).

    L = w_depth * L_depth + w_abs * L_abs_pose + w_rel * L_rel_pose

No training code ships with the repo, so this is a reconstruction. Two details
the paper is explicit about and that are easy to get wrong:

* poses are supervised **camera-to-world**, not world-to-camera -- in w2c
  "rotation and translation are inherently coupled, making translation estimation
  highly sensitive to rotation errors, particularly in long sequences";
* the depth term supervises **the spatial gradient of depth as well as depth**.

`Sigma` is the head's own `conf` channel (`expp1`, so `1 + exp(x)` >= 1): the loss
multiplies the error by it and subtracts `alpha log Sigma`, so the network is paid
for confidence only where it is right. That is DUSt3R's aleatoric form.

The paper gives no values for w_depth / w_abs / w_rel / w_trans / alpha / eps.
Defaults here start from VGGT's and are meant to be swept.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.rotation import mat_to_quat, standardize_quaternion
from lingbot_map.utils.geometry import closed_form_inverse_se3


@dataclass
class LossWeights:
    depth: float = 1.0
    abs_pose: float = 5.0
    rel_pose: float = 1.0
    trans: float = 1.0
    grad: float = 1.0
    alpha: float = 0.2
    huber_eps: float = 1.0
    fov: float = 0.0          # GT intrinsics are known and fixed here, so off by default


def _spatial_grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward differences, padded so shapes are preserved."""
    gx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1))
    gy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return gx, gy


def depth_loss(
    pred: torch.Tensor,
    conf: torch.Tensor,
    target: torch.Tensor,
    weights: LossWeights = LossWeights(),
    valid: Optional[torch.Tensor] = None,
    y_space: bool = False,
) -> Dict[str, torch.Tensor]:
    """pred/conf/target: [..., H, W]. `valid` masks missing GT (0 in the raw depth).

    `y_space` supervises the head's pre-activation instead of depth. The depth head
    uses `activation="exp"` (`models/gct_base.py:106`, not the `inv_log` default),
    so `depth = e^y` -- strictly positive, and an L1 on depth back-propagates
    `dL/dy = dL/d(depth) * depth`. Measured on this data canonical depth spans
    p1 0.30 to p99 1.58, so the gradient scaling spans ~5x. That is mild, which is
    why the paper's form (on depth) is the faithful default; the flag exists for
    datasets with a wide depth range.
    """
    if valid is None:
        valid = target > 0
    valid = valid.to(pred.dtype)
    denom = valid.sum().clamp_min(1.0)

    if y_space:
        pred = torch.log(pred.clamp_min(1e-6))
        target = torch.log(target.clamp_min(1e-6))

    err = (pred - target).abs()
    pgx, pgy = _spatial_grad(pred)
    tgx, tgy = _spatial_grad(target)
    gerr = (pgx - tgx).abs() + (pgy - tgy).abs()

    l_val = (conf * err * valid).sum() / denom
    l_grad = (conf * gerr * valid).sum() / denom
    l_reg = (torch.log(conf.clamp_min(1e-6)) * valid).sum() / denom

    total = l_val + weights.grad * l_grad - weights.alpha * l_reg
    return {"depth": total, "depth_val": l_val, "depth_grad": l_grad,
            "depth_conf_reg": l_reg}


def pose_enc_to_c2w(pose_enc: torch.Tensor, image_size_hw) -> torch.Tensor:
    """[..., 9] pose encoding -> [..., 4, 4] camera-to-world.

    `pose_encoding_to_extri_intri` builds `cat([R, T])` from `absT_quaR_FoV`, where
    `absT` is the camera's **absolute** translation -- its position in world. That
    is already camera-to-world, which is also what the paper says it supervises.

    An earlier version of this function inverted it, on the assumption that
    "extrinsics" meant world-to-camera. Measured cost of that assumption: relative
    rotations disagreed with ScanNet++ GT by 11.43 deg, against 0.41 deg without
    the inversion -- and it sent us looking for a nonexistent axis convention.
    """
    lead = pose_enc.shape[:-1]
    extri, _ = pose_encoding_to_extri_intri(
        pose_enc.reshape(1, -1, 9), image_size_hw=image_size_hw, build_intrinsics=False
    )
    n = extri.shape[1]
    m = torch.zeros(n, 4, 4, dtype=extri.dtype, device=extri.device)
    m[:, :3, :] = extri[0]
    m[:, 3, 3] = 1.0
    return m.reshape(*lead, 4, 4)


def _c2w_to_qt(c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quaternions are standardised, since q and -q are the same rotation and an
    unstandardised pair costs the loss a spurious 2*|q|."""
    q = standardize_quaternion(mat_to_quat(c2w[..., :3, :3]))
    return q, c2w[..., :3, 3]


def abs_pose_loss(pred_c2w: torch.Tensor, gt_c2w: torch.Tensor,
                  weights: LossWeights = LossWeights(),
                  pred_fov: Optional[torch.Tensor] = None,
                  gt_fov: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    qp, tp = _c2w_to_qt(pred_c2w)
    qg, tg = _c2w_to_qt(gt_c2w)
    l_rot = F.huber_loss(qp, qg, delta=weights.huber_eps)
    l_trans = F.huber_loss(tp, tg, delta=weights.huber_eps)
    out = {"abs_rot": l_rot, "abs_trans": l_trans}
    total = l_rot + weights.trans * l_trans
    if pred_fov is not None and gt_fov is not None and weights.fov > 0:
        l_fov = F.huber_loss(pred_fov, gt_fov, delta=weights.huber_eps)
        out["abs_fov"] = l_fov
        total = total + weights.fov * l_fov
    out["abs_pose"] = total
    return out


def _geodesic(ra: torch.Tensor, rb: torch.Tensor) -> torch.Tensor:
    m = ra.transpose(-1, -2) @ rb
    cos = ((m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]) - 1.0) / 2.0
    return torch.arccos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6))


def rel_pose_loss(pred_c2w: torch.Tensor, gt_c2w: torch.Tensor,
                  weights: LossWeights = LossWeights(),
                  anchor: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """Geodesic rotation + l1 translation on relative poses within the window.

    The paper averages over all k(k-1) ordered pairs. With the teacher camera
    bridge only one pose per step is differentiable, so pairs of two teacher poses
    contribute exactly zero gradient; passing `anchor=idx` restricts the sum to
    the pairs involving that frame, which is gradient-equivalent and k times
    cheaper.
    """
    inv_p = closed_form_inverse_se3(pred_c2w)
    inv_g = closed_form_inverse_se3(gt_c2w)
    if anchor is None:
        rel_p = inv_p[:, None] @ pred_c2w[None, :]
        rel_g = inv_g[:, None] @ gt_c2w[None, :]
        eye = torch.eye(len(pred_c2w), dtype=torch.bool, device=pred_c2w.device)
        keep = ~eye
    else:
        rel_p = inv_p[anchor:anchor + 1] @ pred_c2w
        rel_g = inv_g[anchor:anchor + 1] @ gt_c2w
        keep = torch.ones(rel_p.shape[:-2], dtype=torch.bool, device=pred_c2w.device)
        keep[..., anchor] = False

    l_rot = _geodesic(rel_p[..., :3, :3], rel_g[..., :3, :3])[keep].mean()
    l_trans = (rel_p[..., :3, 3] - rel_g[..., :3, 3]).abs().sum(-1)[keep].mean()
    return {"rel_pose": l_rot + weights.trans * l_trans,
            "rel_rot": l_rot, "rel_trans": l_trans}


def total_loss(parts: Dict[str, torch.Tensor],
               weights: LossWeights = LossWeights()) -> torch.Tensor:
    any_part = next(iter(parts.values()))
    total = any_part.new_zeros(())
    if "depth" in parts:
        total = total + weights.depth * parts["depth"]
    if "abs_pose" in parts:
        total = total + weights.abs_pose * parts["abs_pose"]
    if "rel_pose" in parts:
        total = total + weights.rel_pose * parts["rel_pose"]
    return total
