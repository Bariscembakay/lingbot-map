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

    # `pose_encoding_to_extri_intri` already returns **camera-to-world**: `absT`
    # is the camera's absolute position. Upstream's docstring says "camera from
    # world" and that is what misled the pose target for weeks -- inverting cost
    # 11.43 deg of relative-rotation error against GT, 0.41 deg without
    # (commit 0f6f878, `losses.pose_enc_to_c2w`). That fix never reached this
    # file. Note the trap: ||t_w2c|| == ||R^T t_c2w|| == ||t_c2w||, so the
    # ray-origin *magnitude* percentiles in the cache metadata are identical
    # under both conventions and cannot detect the error -- only positions and
    # directions differ.
    c2w = torch.zeros(B, 4, 4, device=extri.device, dtype=extri.dtype)
    c2w[:, :3, :] = extri
    c2w[:, 3, 3] = 1.0
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
    """Plücker rays -> query tokens for the read transformer.

    `num_freqs` sets the sinusoidal band. Measured ray-origin norms on an indoor
    clip are p50 0.48 / p100 0.97 -- O(1), for which `num_freqs=10` is exactly
    NeRF's L=10 and appropriate. An earlier note here guessed "a corridor reaches
    ~20 canonical units"; that was retracted. Re-derive the band from the cache's
    ray-origin histogram before using this on a large outdoor scene (Oxford).

    `num_query_streams` is the read-sweep axis: one query per half-tap (8), per tap
    (4), or a single query later expanded by a linear map (1). Each stream emits
    `patch_start_idx + P_patch` tokens so the layout matches a cached tap exactly,
    and all streams are concatenated along the token axis -- the reader stays at
    dim 1024 whatever the variant.

    Two init choices make the Loss-2 term start on-manifold, which is what removes
    the need for a separate warmup stage:

    * the ray projection's **last layer is zero-init**, so at step 0 a patch query
      is exactly its stream embedding and carries no pose yet;
    * `set_stream_init` sets those embeddings to the cache's mean of the
      corresponding half-tap, so the query enters the reader **centred and scaled
      like a real half-tap**. The query path's gate is open at init (see `read.py`),
      so the output is not exactly the mean tap -- it is the mean tap plus a
      same-magnitude read of the state -- but it starts on the head's manifold and
      at the right scale, which is what keeps the Loss-2 term finite from update 0
      and removes the need for a warmup stage.
    """

    def __init__(self, dim: int, num_freqs: int = 10, patch_size: int = 14,
                 include_raw: bool = True, num_query_streams: int = 1,
                 patch_start_idx: int = 6):
        super().__init__()
        self.patch_size = patch_size
        self.num_freqs = num_freqs
        self.include_raw = include_raw
        self.num_query_streams = num_query_streams
        self.patch_start_idx = patch_start_idx
        self.register_buffer(
            "freqs", 2.0 ** torch.arange(num_freqs, dtype=torch.float32),
            persistent=False,
        )
        in_dim = 9 * (2 * num_freqs + (1 if include_raw else 0))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)
        self.stream_emb = nn.Parameter(torch.zeros(num_query_streams, dim))
        self.special_emb = nn.Parameter(torch.randn(patch_start_idx, dim) * 0.02)

    @torch.no_grad()
    def set_stream_init(self, means: torch.Tensor) -> None:
        """means: [num_query_streams, dim] -- see the class docstring."""
        if means.shape != self.stream_emb.shape:
            raise ValueError(f"expected {tuple(self.stream_emb.shape)}, got {tuple(means.shape)}")
        self.stream_emb.copy_(means.to(self.stream_emb.dtype))

    def encode(self, rays: torch.Tensor) -> torch.Tensor:
        scaled = rays.unsqueeze(-1) * self.freqs * torch.pi
        feats = [scaled.sin(), scaled.cos()]
        flat = torch.cat([f.flatten(-2) for f in feats], dim=-1)
        if self.include_raw:
            flat = torch.cat([rays, flat], dim=-1)
        return flat

    def forward(self, pose_enc: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """pose_enc [B, 9] -> [B, num_query_streams * (patch_start_idx + P), dim]."""
        rays = plucker_rays(pose_enc, height, width, self.patch_size)
        patch = self.proj(self.encode(rays.to(self.freqs.dtype)))       # [B,P,D]
        B, _, D = patch.shape
        spec = self.special_emb.unsqueeze(0).expand(B, -1, -1)          # [B,6,D]
        base = torch.cat([spec, patch], dim=1)                          # [B,6+P,D]
        out = base.unsqueeze(1) + self.stream_emb.view(1, -1, 1, D)     # [B,S,6+P,D]
        return out.reshape(B, -1, D)
