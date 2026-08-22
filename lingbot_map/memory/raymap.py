"""Camera -> query tokens.

The frozen stack has no pose-conditioned input anywhere, so this is the only
pathway by which a camera can be asked a question. One query token per output
patch, carrying that patch's ray and nothing else.

Row-major over the patch grid, matching `PositionGetter` (`layers/rope.py:26`,
`cartesian_prod(y, x)`) and `PatchEmbed` (`layers/patch_embed.py:74`,
`flatten(2).transpose(1,2)`). The DPT head's `permute -> reshape` depends on it.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from lingbot_map.utils.geometry import closed_form_inverse_se3
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


def patch_center_pixels(patch_h: int, patch_w: int, patch_size: int,
                        device, dtype) -> torch.Tensor:
    """[patch_h*patch_w, 2] pixel centres, row-major."""
    r = (torch.arange(patch_h, device=device, dtype=dtype) + 0.5) * patch_size
    c = (torch.arange(patch_w, device=device, dtype=dtype) + 0.5) * patch_size
    vv, uu = torch.meshgrid(r, c, indexing="ij")
    return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)


def plucker_rays(pose_enc: torch.Tensor, height: int, width: int,
                 patch_size: int = 14) -> torch.Tensor:
    """pose_enc [B, 9] -> [B, patch_h*patch_w, 9] as (direction, origin, moment).

    Directions are unit and world-frame; origins are camera centres in the
    model's canonical scale (paper §3.2: normalised so the anchor point cloud has
    mean norm 1), so no per-clip rescaling is needed or wanted.
    """
    B = pose_enc.shape[0]
    extri, intri = pose_encoding_to_extri_intri(
        pose_enc.unsqueeze(1), image_size_hw=(height, width), build_intrinsics=True
    )
    extri, intri = extri[:, 0], intri[:, 0]              # [B,3,4], [B,3,3]

    e44 = torch.zeros(B, 4, 4, device=extri.device, dtype=extri.dtype)
    e44[:, :3, :] = extri
    e44[:, 3, 3] = 1.0
    c2w = closed_form_inverse_se3(e44)
    rot, origin = c2w[:, :3, :3], c2w[:, :3, 3]          # [B,3,3], [B,3]

    ph, pw = height // patch_size, width // patch_size
    uv = patch_center_pixels(ph, pw, patch_size, extri.device, extri.dtype)
    uv1 = torch.cat([uv, torch.ones_like(uv[:, :1])], dim=-1)          # [P,3]

    d_cam = torch.einsum("bij,pj->bpi", torch.inverse(intri), uv1)
    d_world = torch.einsum("bij,bpj->bpi", rot, d_cam)
    d_world = d_world / d_world.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    o = origin.unsqueeze(1).expand_as(d_world)
    moment = torch.cross(o, d_world, dim=-1)
    return torch.cat([d_world, o, moment], dim=-1)                     # [B,P,9]


class RaymapEncoder(nn.Module):
    """Plücker rays -> D-dim query tokens.

    `num_freqs` sets the sinusoidal band. Directions are unit but origins and
    moments grow with scene extent (a corridor reaches ~20 canonical units, a
    desk scan stays under 1), so pick the band from the ray-origin histogram in
    the cache metadata rather than guessing.
    """

    def __init__(self, dim: int, num_freqs: int = 10, patch_size: int = 14,
                 include_raw: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.num_freqs = num_freqs
        self.include_raw = include_raw
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32),
            persistent=False,
        )
        in_dim = 9 * (2 * num_freqs + (1 if include_raw else 0))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def encode(self, rays: torch.Tensor) -> torch.Tensor:
        scaled = rays.unsqueeze(-1) * self.freqs * torch.pi
        feats = [scaled.sin(), scaled.cos()]
        flat = torch.cat([f.flatten(-2) for f in feats], dim=-1)
        if self.include_raw:
            flat = torch.cat([rays, flat], dim=-1)
        return flat

    def forward(self, pose_enc: torch.Tensor, height: int, width: int) -> torch.Tensor:
        rays = plucker_rays(pose_enc, height, width, self.patch_size)
        return self.proj(self.encode(rays.to(self.freqs.dtype)))
