"""Constructing the frozen heads exactly as the model does.

Centralised because the defaults are wrong for this model and the failure is
silent-ish: `DPTHead`'s defaults are `output_dim=4, activation="inv_log"`, which
belong to the *point* heads. `GCTBase._build_depth_head` uses
`output_dim=2, activation="exp"` (`models/gct_base.py:106`), so depth is
`exp(y)` -- strictly positive, and the pre-activation inverse is `log(depth)`,
not the signed `log1p` that `inv_log` would need.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from lingbot_map.heads.camera_head import CameraCausalHead
from lingbot_map.heads.dpt_head import DPTHead

EMBED_DIM = 1024
TAP_DIM = 2 * EMBED_DIM
DEPTH_ACTIVATION = "exp"
CONF_ACTIVATION = "expp1"


def build_depth_head() -> DPTHead:
    return DPTHead(
        dim_in=TAP_DIM, patch_size=14, output_dim=2,
        activation=DEPTH_ACTIVATION, conf_activation=CONF_ACTIVATION,
    )


def build_camera_head() -> CameraCausalHead:
    return CameraCausalHead(dim_in=TAP_DIM)


def load_frozen(path: Path | str, device, need_camera: bool = True
                ) -> Tuple[DPTHead, Optional[CameraCausalHead]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    depth = build_depth_head()
    depth.load_state_dict(payload["depth_head"], strict=True)
    depth = depth.to(device).eval()
    camera = None
    if need_camera:
        camera = build_camera_head()
        camera.load_state_dict(payload["camera_head"], strict=True)
        camera = camera.to(device).eval()
    for m in (depth, camera):
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(False)
    return depth, camera


def depth_to_preactivation(depth: torch.Tensor) -> torch.Tensor:
    """Exact inverse of the head's `exp` activation."""
    return torch.log(depth.clamp_min(1e-6))
