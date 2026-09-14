"""Low-rank adapters for the frozen lingbot DPT trunk.

The alternative to `--unfreeze-head`: instead of making all 32.7 M trunk
parameters trainable, keep them frozen and learn a rank-r residual beside each
convolution. Note this buys optimiser state, not activation memory -- the
backward pass already traverses the trunk to reach `LingbotFrozenHead.adapters`,
so peak VRAM is unchanged (~122 GB at 96 frames, H200 only).

The trunk is pure convolution (DPT: 1x1 projections, transposed-conv resizes,
3x3 refinenets), so the Linear formulation does not apply. For a Conv2d with
weight (C_out, C_in, k, k) the low-rank factorisation is two convolutions -- A
carries the spatial kernel down to r channels, B mixes r -> C_out pointwise:

    y = conv(x) + (alpha / r) * B(A(x))

B is zero-initialised, so an injected model is bit-identical to the frozen one
at step 0 and `--init-from` stays a valid starting point.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.Conv2d(base.in_channels, rank, base.kernel_size,
                           stride=base.stride, padding=base.padding,
                           dilation=base.dilation, bias=False)
        self.B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(x))


class LoRAConvTranspose2d(nn.Module):
    """A carries the upsampling, B the channel mix -- so B stays a 1x1."""

    def __init__(self, base: nn.ConvTranspose2d, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.A = nn.ConvTranspose2d(base.in_channels, rank, base.kernel_size,
                                    stride=base.stride, padding=base.padding,
                                    dilation=base.dilation, bias=False)
        self.B = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.B(self.A(x))


def inject_conv_lora(root: nn.Module, rank: int, alpha: float) -> dict:
    """Wrap every ungrouped conv under `root` in place; freeze everything else.

    Grouped convolutions are skipped: a shared rank-r bottleneck would mix
    channels the base layer deliberately keeps separate.
    """
    root.requires_grad_(False)
    n_wrapped, n_skipped = 0, 0
    for parent in list(root.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
                if child.groups != 1:
                    n_skipped += 1
                    continue
                cls = (LoRAConv2d if isinstance(child, nn.Conv2d)
                       else LoRAConvTranspose2d)
                setattr(parent, name, cls(child, rank, alpha))
                n_wrapped += 1
    n_lora = sum(p.numel() for n, p in root.named_parameters()
                 if p.requires_grad)
    return {"wrapped": n_wrapped, "skipped_grouped": n_skipped,
            "lora_params": n_lora}
