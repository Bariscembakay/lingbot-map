"""Raymaps and ground-truth pointmaps for the recall probe.

A probe asks the state "what did you see from here?" by handing it a **raymap**
-- per pixel, the camera's position and the direction that pixel looked -- and
nothing else. The answer is scored against the queried frame's GT pointmap.

Everything here is expressed relative to **camera 0** of the clip, which is what
CUT3R calls the world frame.
"""
from __future__ import annotations

import torch

# CUT3R's `get_ray_map` transforms the camera-frame direction as a homogeneous
# POINT, so the translation is applied to it: the stored "direction" is
# normalize(R @ d_cam + t), not normalize(R @ d_cam). Exact only when t == 0,
# i.e. camera 0; ~10 deg off at t ~ 2 canonical units. We match it because that
# is the distribution the released raymap encoder was trained on -- see the open
# item in .agents/AGENTS.md. "true" is the geometrically correct alternative,
# kept as a sweep axis.
CONVENTIONS = ("cut3r", "true")


def pixel_dirs(K: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`[B, H*W, 3]` camera-frame directions, row-major, z == 1 exactly.

    z == 1 is what makes a depth map directly usable as a scale: multiplying by
    z-depth lands on the point, no renormalisation.
    """
    b = K.shape[0]
    y, x = torch.meshgrid(
        torch.arange(height, device=K.device, dtype=K.dtype),
        torch.arange(width, device=K.device, dtype=K.dtype),
        indexing="ij",
    )
    grid = torch.stack([x.reshape(-1), y.reshape(-1),
                        torch.ones_like(x.reshape(-1))], dim=-1)   # [HW, 3]
    return torch.linalg.inv(K) @ grid.T.expand(b, 3, -1)           # [B, 3, HW]


def relative_c2w(c2w_q: torch.Tensor, c2w_0: torch.Tensor) -> torch.Tensor:
    """`inv(c2w_0) @ c2w_q` -- camera q expressed in camera 0's frame."""
    return torch.linalg.inv(c2w_0) @ c2w_q


def build_raymap(K: torch.Tensor, c2w_q: torch.Tensor, c2w_0: torch.Tensor,
                 height: int, width: int,
                 convention: str = "cut3r") -> torch.Tensor:
    """`[B, 6, H, W]` -- channels 0-2 ray origin, 3-5 ray direction.

    K, c2w_q: `[B, 3, 3]`, `[B, 4, 4]`. c2w_0: `[B, 4, 4]` or `[1, 4, 4]`.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {CONVENTIONS}")
    b = K.shape[0]
    c2w = relative_c2w(c2w_q, c2w_0.expand(b, -1, -1))
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    d_cam = pixel_dirs(K, height, width)                            # [B, 3, HW]

    if convention == "cut3r":
        d = R @ d_cam + t[:, :, None]        # the translation-contaminated form
    else:
        d = R @ d_cam
    d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)

    o = t[:, :, None].expand_as(d)
    rays = torch.cat([o, d], dim=1)                                 # [B, 6, HW]
    return rays.reshape(b, 6, height, width)


def gt_pointmaps(depth: torch.Tensor, K: torch.Tensor,
                 c2w_q: torch.Tensor, c2w_0: torch.Tensor):
    """(X_self, X_world, valid) for the queried frames.

    depth `[B, H, W]` is z-depth in canonical units (already divided by the
    clip's `gt_scale`), so no rescaling happens here -- adding one would
    double-normalise.
    """
    b, h, w = depth.shape
    d_cam = pixel_dirs(K, h, w)                                     # [B, 3, HW]
    z = depth.reshape(b, 1, h * w)
    x_self = d_cam * z                                              # [B, 3, HW]

    c2w = relative_c2w(c2w_q, c2w_0.expand(b, -1, -1))
    x_world = c2w[:, :3, :3] @ x_self + c2w[:, :3, 3][:, :, None]

    valid = (depth > 1e-6) & torch.isfinite(depth)
    to_img = lambda a: a.reshape(b, 3, h, w).permute(0, 2, 3, 1)    # noqa: E731
    return to_img(x_self), to_img(x_world), valid


def ray_depth(x_world: torch.Tensor, raymap: torch.Tensor) -> torch.Tensor:
    """Distance from the ray origin to each point -- the readable depth metric.

    Independent of the direction-channel convention, so it stays comparable
    across the `--raymap-convention` axis.
    """
    o = raymap[:, :3].permute(0, 2, 3, 1)
    return (x_world - o).norm(dim=-1)


def true_rays(K: torch.Tensor, c2w_q: torch.Tensor, c2w_0: torch.Tensor,
              h: int, w: int, unit: bool = True):
    """(origin, ray direction, relative pose) per pixel in the anchor frame --
    the OUTPUT parameterisation of the depth heads, independent of the raymap
    INPUT convention. unit=True: unit rays, prediction = distance along the ray
    (GT is ||x_world - o||; V7: GT points are on these rays to 3e-7).
    unit=False: z-scaled directions (camera-frame z = 1), so a z-DEPTH
    prediction (the frozen lingbot head) reconstructs as `o + z * d` exactly."""
    b = K.shape[0]
    c2w = relative_c2w(c2w_q, c2w_0.expand(b, -1, -1))
    d = c2w[:, :3, :3] @ pixel_dirs(K, h, w)
    if unit:
        d = d / d.norm(dim=1, keepdim=True)
    o = c2w[:, :3, 3][:, :, None].expand(-1, -1, h * w)
    sh = lambda a: a.reshape(b, 3, h, w).permute(0, 2, 3, 1)  # noqa: E731
    return sh(o).contiguous(), sh(d).contiguous(), c2w
