"""CUT3R's persistent state, fed by lingbot-map's aggregator.

Architecture record: `.agents/spatial_memory_design.md`. The short version:
lingbot-map (frozen, heads removed, KV cache kept) replaces CUT3R's ViT-L
encoder; its tap-23 tokens drive CUT3R's two interconnected decoders and the
768x768 state. Training supervises **recall only** -- a raymap query at a camera
we already visited, scored against that frame's ground truth.

The classes here compose CUT3R's blocks rather than subclassing
`ARCroco3DStereo`, because that class builds the full 302 M ViT-L encoder in
`__init__` and we would carry it on the GPU without ever calling it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def cut3r_src() -> Path:
    """CUT3R's `src/`, which must be on sys.path before its modules import."""
    return Path(__file__).resolve().parents[2] / "CUT3R" / "src"


def _ensure_path() -> None:
    # dust3r and croco import each other by bare top-level name, so `src` itself
    # has to be on the path -- importing `CUT3R.src.dust3r` does not work.
    p = str(cut3r_src())
    if p not in sys.path:
        sys.path.insert(0, p)
    croco = str(cut3r_src() / "croco")
    if croco not in sys.path:
        sys.path.insert(0, croco)


_ensure_path()

from croco.models.blocks import PatchEmbed  # noqa: E402
from croco.models.pos_embed import RoPE2D  # noqa: E402
from dust3r.blocks import Block, DecoderBlock  # noqa: E402
from dust3r.heads.dpt_head import DPTOutputAdapter_fix  # noqa: E402
from dust3r.heads.postprocess import postprocess  # noqa: E402
from dust3r.blocks import ConditionModulationBlock  # noqa: E402

HALF = 1024          # aggregator stream width; a tap is two of these
TAP_DIM = 2 * HALF   # 2048
DEC_DIM = 768
STATE_TOKENS = 768
DEC_DEPTH = 12
SPECIAL_POS = -1.0   # RoPE position for every non-patch token; see below


def grid_positions(patch_h: int, patch_w: int, n_special: int,
                   batch: int, device, dtype=torch.long) -> torch.Tensor:
    """`[B, n_special + patch_h*patch_w, 2]` RoPE positions, row-major patches.

    Every non-patch token gets the SAME position, (-1, -1). That is CUT3R's
    choice for its one pose token, so the loaded decoder weights stay in
    distribution. lingbot's `zeros` would alias each special onto patch (0, 0);
    giving the six specials distinct negative positions would put the loaded
    weights at RoPE phases they never saw in training, and they are already
    separated by content.
    """
    y, x = torch.meshgrid(
        torch.arange(patch_h, device=device),
        torch.arange(patch_w, device=device),
        indexing="ij",
    )
    patch = torch.stack([y.reshape(-1), x.reshape(-1)], dim=-1)
    if n_special:
        special = torch.full((n_special, 2), int(SPECIAL_POS), device=device)
        patch = torch.cat([special, patch], dim=0)
    return patch.to(dtype)[None].expand(batch, -1, -1).contiguous()


class RaymapEncoder(nn.Module):
    """6-channel raymap -> `[B, 1 + P, DEC_DIM]`, with a leading mod token.

    The mod token is not extra pose information -- the raymap already carries the
    pose in full. It is the single vector `dpt_cross`'s adaLN needs, and CUT3R
    derives it the same way (`global_img_feat_i = mean(feat_i)`), which is
    leak-free: mean-pooling a raymap cannot reveal anything about the queried
    frame's image.
    """

    def __init__(self, patch_size: int = 14, depth: int = 2):
        super().__init__()
        self.patch_size = patch_size
        # in_chans=6: ray origin (3) + unit direction (3), per pixel.
        self.patch_embed_ray_map = PatchEmbed(
            img_size=518, patch_size=patch_size, in_chans=6, embed_dim=HALF
        )
        self.rope = RoPE2D(freq=100.0)
        self.enc_blocks_ray_map = nn.ModuleList(
            [Block(HALF, 16, 4, qkv_bias=True,
                   norm_layer=lambda d: nn.LayerNorm(d, eps=1e-6), rope=self.rope)
             for _ in range(depth)]
        )
        self.enc_norm_ray_map = nn.LayerNorm(HALF, eps=1e-6)
        # 1024 -> 768 is CUT3R's own `decoder_embed`, loadable from the checkpoint.
        self.decoder_embed = nn.Linear(HALF, DEC_DIM, bias=True)
        self.mod_proj = nn.Linear(DEC_DIM, DEC_DIM, bias=True)

    def forward(self, raymap: torch.Tensor):
        """raymap `[B, 6, H, W]` -> (tokens `[B, 1+P, D]`, pos `[B, 1+P, 2]`)."""
        b, _, h, w = raymap.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        x = self.patch_embed_ray_map.proj(raymap).flatten(2).transpose(1, 2)
        pos = grid_positions(ph, pw, 0, b, raymap.device)
        for blk in self.enc_blocks_ray_map:
            x = blk(x, pos)
        x = self.decoder_embed(self.enc_norm_ray_map(x))
        mod = self.mod_proj(x.mean(dim=1, keepdim=True))
        tokens = torch.cat([mod, x], dim=1)
        full_pos = grid_positions(ph, pw, 1, b, raymap.device)
        return tokens, full_pos


class ProbeHead(nn.Module):
    """Two DPT heads at patch 14, sharing four decoder taps.

    Mirrors CUT3R's `DPTPts3dPose` rather than reusing it, for two reasons that
    both come from patch size. `DPTPts3dPose` never forwards `patch_size` to the
    adapter, so it silently builds a patch-16 head; and croco's DPT has no
    terminal resize -- its upsampling is a fixed 16x (four fusion blocks at 2x,
    the head at 2x, tap 0 pre-upsampled 4x). 14 = 2*7 is unreachable with conv
    strides, so the output lands at 16/14 of the target and must be resized. The
    original DPT paper resizes at the end too; only croco's variant relies on the
    factors happening to multiply to the patch size.
    """

    def __init__(self, patch_size: int = 14, dec_num_heads: int = 12, rope=None,
                 depth_mode=("exp", -float("inf"), float("inf")),
                 conf_mode=("exp", 1, float("inf")), grad_ckpt: bool = False):
        super().__init__()
        self.grad_ckpt = grad_ckpt
        self.depth_mode, self.conf_mode = depth_mode, conf_mode
        args = dict(
            num_channels=4,               # xyz + conf
            feature_dim=256, last_dim=128,
            hooks=[0, 1, 2, 3],
            dim_tokens=[DEC_DIM] * 4,     # tap 0 is seeded with PROJECTED tokens,
            head_type="regression",       # so all four taps are decoder-width
            output_width_ratio=1,
            patch_size=patch_size,
        )
        self.dpt_self = DPTOutputAdapter_fix(**args)
        self.dpt_self.init(dim_tokens_enc=[DEC_DIM] * 4)
        self.dpt_cross = DPTOutputAdapter_fix(**args)
        self.dpt_cross.init(dim_tokens_enc=[DEC_DIM] * 4)
        # Zero the last 1x1 conv of each head. depth_mode exponentiates the raw
        # output, so random init sends a few logits through exp() and produces
        # astronomic pointmaps: the first real step logged |grad| = 5.5e12, which
        # clipping keeps alive but leaves pointing in a direction set by outliers.
        # CUT3R never sees this because its heads are pretrained; ours are not.
        # Small-but-nonzero, so exp(~0) = 1 and the initial prediction is O(1).
        # NOT zero: zeroing the weight kills the gradient outright, which V1 and
        # V3 caught the moment it was tried (grad@q and the live-vs-dead state
        # difference both went to exactly 0.0). A zero weight gives a plausible
        # loss curve with no gradient to the state, read, or write at all.
        for h in (self.dpt_self, self.dpt_cross):
            last = h.head[-1]
            nn.init.normal_(last.weight, std=1e-4)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
        # Only the world head is pose-conditioned, and only at its deepest tap:
        # camera-frame points do not need to know the camera.
        self.final_transform = nn.ModuleList(
            [ConditionModulationBlock(DEC_DIM, dec_num_heads, mlp_ratio=4.0,
                                      qkv_bias=True, rope=rope)
             for _ in range(2)]
        )

    def forward(self, taps: list[torch.Tensor], mod: torch.Tensor,
                pos: torch.Tensor, hw: tuple[int, int]) -> dict:
        h, w = hw
        cross_last = taps[-1]
        for blk in self.final_transform:
            cross_last = blk(cross_last, mod, pos)
        out = {}
        ck = self.grad_ckpt and self.training and torch.is_grad_enabled()
        run = ((lambda f, *a, **k: checkpoint(f, *a, use_reentrant=False, **k))
               if ck else (lambda f, *a, **k: f(*a, **k)))
        with torch.amp.autocast("cuda", enabled=False):
            self_raw = run(self.dpt_self, [t.float() for t in taps], image_size=(h, w))
            cross_raw = run(self.dpt_cross,
                            [t.float() for t in taps[:-1]] + [cross_last.float()],
                            image_size=(h, w))
            # croco's DPT ends at 16*N; resize DOWN to the patch-14 target rather
            # than dropping the head's final 2x and upsampling from 8*N, which
            # would invent detail instead of decimating it.
            self_raw = F.interpolate(self_raw, size=(h, w), mode="bilinear",
                                     align_corners=False)
            cross_raw = F.interpolate(cross_raw, size=(h, w), mode="bilinear",
                                      align_corners=False)
            s = postprocess(self_raw, self.depth_mode, self.conf_mode)
            c = postprocess(cross_raw, self.depth_mode, self.conf_mode)
        out["pts3d_in_self_view"] = s["pts3d"]
        out["conf_self"] = s["conf"]
        out["pts3d_in_other_view"] = c["pts3d"]
        out["conf"] = c["conf"]
        return out


class RayDepthHead(nn.Module):
    """The read as a depth renderer: one ray distance and one confidence per
    pixel, reconstructed as `o + s*d` by the caller. Deliberately DPT-free and
    tiny (~0.8 M): the smallest head that can read the state, so the scene
    memorisation capacity that lives in the head is minimal. Also collapses
    the self/world distinction -- a rigid map preserves L2, so one geometry
    serves both frames (design doc: "The world head is dominated")."""

    def __init__(self, patch_size: int = 14):
        super().__init__()
        self.patch_size = patch_size
        # exp() output head: small-but-nonzero init, the DPT lesson -- zero
        # weights are a dead path, large ones explode through the exp.
        self.proj = nn.Linear(DEC_DIM, patch_size * patch_size * 2)
        nn.init.normal_(self.proj.weight, std=1e-4)
        nn.init.zeros_(self.proj.bias)

    def forward(self, tokens: torch.Tensor, patch_hw: tuple[int, int],
                hw: tuple[int, int]):
        b, _, _ = tokens.shape
        ph, pw = patch_hw
        p = self.patch_size
        with torch.amp.autocast("cuda", enabled=False):
            x = self.proj(tokens.float())
            x = x.view(b, ph, pw, p, p, 2).permute(0, 5, 1, 3, 2, 4)
            x = x.reshape(b, 2, ph * p, pw * p)
            if x.shape[-2:] != tuple(hw):
                x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
            s = torch.exp(x[:, 0].clamp(-10.0, 10.0))
            conf = 1.0 + torch.exp(x[:, 1].clamp(max=10.0))
        return s, conf


class LingbotFrozenHead(nn.Module):
    """lingbot-map's own trained depth head, FROZEN, behind four trainable
    Linear(768 -> 2048) adapters. The head decodes exactly one feature
    language -- aggregator taps -- so the probe is forced to translate the
    state into "the taps the encoder would have produced for the queried
    view". Zero trainable capacity in the trunk; the adapters are the whole
    learnable read interface. Known inherited artifact, accepted by decision
    2026-08-28: the checkpoint's deepest DPT branch is ReLU-dead
    (d(depth)/d(tap23) = 0), so hook 3 ignores its adapter."""

    CKPT = "/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt"

    def __init__(self, patch_size: int = 14, ckpt: str | None = None):
        super().__init__()
        from lingbot_map.heads.dpt_head import DPTHead
        self.adapters = nn.ModuleList(
            [nn.Linear(DEC_DIM, 2 * HALF) for _ in range(4)])
        # Mirrors gct_base._build_depth_head exactly, so the export loads 1:1.
        self.dpt = DPTHead(dim_in=2 * HALF, patch_size=patch_size, output_dim=2,
                           activation="exp", conf_activation="expp1")
        sd = torch.load(ckpt or self.CKPT, map_location="cpu",
                        weights_only=False)
        self.dpt.load_state_dict(sd["depth_head"])
        self.dpt.requires_grad_(False)

    def forward(self, taps: list[torch.Tensor], hw: tuple[int, int]):
        h, w = hw
        imgs = torch.zeros(taps[0].shape[0], 1, 3, h, w,
                           device=taps[0].device, dtype=torch.float32)
        with torch.amp.autocast("cuda", enabled=False):
            # Adapters must also run fp32: under the loop's bf16 autocast a
            # Linear outside this block emits bf16, which the frozen fp32
            # trunk rejects.
            toks = [a(t.float())[:, None] for a, t in zip(self.adapters, taps)]
            # The frozen trunk still needs its activations for backprop THROUGH
            # it to the adapters/decoder; checkpoint them instead of storing.
            if torch.is_grad_enabled() and self.training:
                z, conf = checkpoint(
                    lambda *ts: self.dpt(list(ts[:-1]), ts[-1], patch_start_idx=0),
                    *toks, imgs, use_reentrant=False)
            else:
                z, conf = self.dpt(toks, imgs, patch_start_idx=0)
        b = taps[0].shape[0]
        return z.reshape(b, h, w), conf.reshape(b, h, w)


class StateMemory(nn.Module):
    """The whole trainable model: write path, probe path, heads.

    The encoder is absent by design -- lingbot-map is frozen and its tap-23
    tokens arrive precomputed from the clip cache, so nothing here ever imports
    the aggregator.
    """

    def __init__(self, patch_size: int = 14, tap_dim: int = TAP_DIM,
                 state_tokens: int = STATE_TOKENS, dec_depth: int = DEC_DEPTH,
                 dec_num_heads: int = 12, state_dec_num_heads: int = 16,
                 patch_start_idx: int = 6, grad_ckpt: bool = False,
                 head_type: str = "dpt", read_depth: int = 2):
        super().__init__()
        # Full BPTT over 160 frames is required by the objective (a probe at t
        # must credit the write at q), and 954 decoder passes of activations do
        # not fit without recomputation: ~439 GB against ~31-48 GB checkpointed.
        self.grad_ckpt = grad_ckpt
        self.patch_size = patch_size
        self.patch_start_idx = patch_start_idx
        self.state_tokens = state_tokens
        self.dec_depth = dec_depth
        self.rope = RoPE2D(freq=100.0)
        norm = lambda d: nn.LayerNorm(d, eps=1e-6)  # noqa: E731

        # Tap 23 is 2048-d (frame stream ++ global stream); CUT3R's own
        # decoder_embed is 1024 -> 768 and belongs to the raymap path instead.
        self.in_proj = nn.Linear(tap_dim, DEC_DIM, bias=True)

        self.register_tokens = nn.Embedding(state_tokens, HALF)
        self.decoder_embed_state = nn.Linear(HALF, DEC_DIM, bias=True)

        self.dec_blocks = nn.ModuleList(
            [DecoderBlock(DEC_DIM, dec_num_heads, mlp_ratio=4.0, qkv_bias=True,
                          norm_layer=norm, norm_mem=True, rope=self.rope)
             for _ in range(dec_depth)]
        )
        self.dec_blocks_state = nn.ModuleList(
            [DecoderBlock(DEC_DIM, state_dec_num_heads, mlp_ratio=4.0,
                          qkv_bias=True, norm_layer=norm, norm_mem=True,
                          rope=self.rope)
             for _ in range(dec_depth)]
        )
        self.dec_norm = norm(DEC_DIM)
        self.dec_norm_state = norm(DEC_DIM)

        self.raymap = RaymapEncoder(patch_size=patch_size)
        self.head_type = head_type
        if head_type == "smallread":
            # Decoupled 2-block read over the RAW stored state + linear
            # ray-depth head: the smallest read that can query the memory, so
            # what it recalls measures the WRITE, not the read. The write path
            # (the interconnected pair) is untouched. Blocks init from
            # dec_blocks[:2] via load_cut3r_weights.
            self.read_blocks = nn.ModuleList(
                [DecoderBlock(DEC_DIM, dec_num_heads, mlp_ratio=4.0,
                              qkv_bias=True, norm_layer=norm, norm_mem=True,
                              rope=self.rope) for _ in range(read_depth)]
            )
            self.read_norm = norm(DEC_DIM)
            self.head = RayDepthHead(patch_size=patch_size)
        elif head_type == "raydepth":
            self.head = RayDepthHead(patch_size=patch_size)
        elif head_type == "lingbot":
            self.head = LingbotFrozenHead(patch_size=patch_size)
        else:
            self.head = ProbeHead(patch_size=patch_size,
                                  dec_num_heads=dec_num_heads,
                                  rope=self.rope, grad_ckpt=grad_ckpt)

    # -- state ---------------------------------------------------------------
    def init_state(self, batch: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.register_tokens(torch.arange(self.state_tokens, device=device))
        s = self.decoder_embed_state(s)[None].expand(batch, -1, -1)
        # CUT3R's 'state_pe=2d': the state sits on a square pseudo-grid, which is
        # unrelated to the image grid but is what the loaded weights were trained
        # against.
        w = int(self.state_tokens ** 0.5)
        w = w + 1 if w % 2 else w
        pos = torch.tensor([[i // w, i % w] for i in range(self.state_tokens)],
                           device=device)[None].expand(batch, -1, -1)
        return s.contiguous(), pos.contiguous()

    def _decode(self, state, state_pos, tokens, tok_pos):
        """CUT3R's interconnected decoders: write and read run SIMULTANEOUSLY.

        Both stacks consume layer l-1's pair, so neither sees the other's layer-l
        output. Returns (new_state, taps) where taps are the four the head reads.
        """
        pairs = [(state, tokens)]
        use_ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
        for blk_s, blk_i in zip(self.dec_blocks_state, self.dec_blocks):
            s_prev, i_prev = pairs[-1]
            if use_ckpt:
                # Both stacks must read the SAME layer l-1 pair, so they are
                # checkpointed separately rather than as one fused step.
                s_new = checkpoint(lambda a, b, bs=blk_s: bs(a, b, state_pos, tok_pos)[0],
                                   s_prev, i_prev, use_reentrant=False)
                i_new = checkpoint(lambda a, b, bi=blk_i: bi(a, b, tok_pos, state_pos)[0],
                                   i_prev, s_prev, use_reentrant=False)
            else:
                s_new, _ = blk_s(s_prev, i_prev, state_pos, tok_pos)   # write
                i_new, _ = blk_i(i_prev, s_prev, tok_pos, state_pos)   # read
            pairs.append((s_new, i_new))
        state_out = self.dec_norm_state(pairs[-1][0])
        img_out = self.dec_norm(pairs[-1][1])
        d = self.dec_depth
        # Tap 0 is seeded with the PROJECTED tokens, not pre-projection ones as
        # CUT3R does: at full aggregator width it would be a second, state-free
        # path into the head.
        taps = [pairs[0][1], pairs[d * 2 // 4][1], pairs[d * 3 // 4][1], img_out]
        return state_out, taps

    # -- write ---------------------------------------------------------------
    def write(self, state, state_pos, tap, patch_hw):
        """Consume one frame's cached tap 23. `tap` is `[B, 1005, 2048]`."""
        b = tap.shape[0]
        tokens = self.in_proj(tap)
        pos = grid_positions(*patch_hw, self.patch_start_idx, b, tap.device)
        new_state, _ = self._decode(state, state_pos, tokens, pos)
        return new_state

    # -- probe ---------------------------------------------------------------
    def probe(self, state, state_pos, raymap, hw):
        """Query a past camera. READ ONLY -- the state is never updated here."""
        tokens, pos = self.raymap(raymap)
        if self.head_type == "smallread":
            # No state-stack processing at probe time: the state AS STORED must
            # be readable, or the write did not do its job.
            h, w = hw
            x = tokens
            use_ckpt = self.grad_ckpt and self.training and torch.is_grad_enabled()
            for blk in self.read_blocks:
                if use_ckpt:
                    x = checkpoint(lambda a, b, bk=blk: bk(a, b, pos, state_pos)[0],
                                   x, state, use_reentrant=False)
                else:
                    x, _ = blk(x, state, pos, state_pos)
            x = self.read_norm(x)[:, 1:]        # strip the mod/pose token
            sd, conf = self.head(x, (h // self.patch_size, w // self.patch_size), hw)
            return {"ray_depth": sd, "conf": conf}
        _, taps = self._decode(state, state_pos, tokens, pos)
        # ModLN.modulate does `scale.unsqueeze(1)`, so the conditioning vector is
        # [B, D], not [B, 1, D] -- CUT3R slices `x[-1][:, 0]` for the same reason.
        mod = taps[-1][:, 0]
        taps = [t[:, 1:] for t in taps]      # strip the mod token; leaves P patches
        if self.head_type == "raydepth":
            h, w = hw
            s, conf = self.head(taps[-1],
                                (h // self.patch_size, w // self.patch_size), hw)
            return {"ray_depth": s, "conf": conf}
        if self.head_type == "lingbot":
            z, conf = self.head(taps, hw)
            return {"ray_depth": z, "conf": conf}   # z pairs with unit=False rays
        return self.head(taps, mod, pos[:, 1:], hw)


def load_cut3r_weights(model: StateMemory, ckpt_path: str | Path,
                       strict_report: bool = True) -> dict:
    """Load every CUT3R tensor whose role survives the architecture change.

    Deliberately NOT loaded: `enc_blocks` (lingbot replaces it),
    `pose_retriever` (152 M of learned ego-motion memory the aggregator's KV
    cache already covers), `downstream_head` (patch-16 shaped; ours is 14 and
    random-init), and `patch_embed_ray_map` (6*16*16 vs 6*14*14).
    """
    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = raw["model"] if "model" in raw else raw
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}

    want = {
        "dec_blocks": "dec_blocks",
        "dec_blocks_state": "dec_blocks_state",
        "dec_norm": "dec_norm",
        "dec_norm_state": "dec_norm_state",
        "decoder_embed_state": "decoder_embed_state",
        "register_tokens": "register_tokens",
        "enc_blocks_ray_map": "raymap.enc_blocks_ray_map",
        "enc_norm_ray_map": "raymap.enc_norm_ray_map",
        "decoder_embed": "raymap.decoder_embed",
    }
    # One source can feed several destinations (dict keys must be unique --
    # a duplicate "dec_norm" key here once silently UNLOADED dec_norm).
    extra = [("dec_blocks.0", "read_blocks.0"), ("dec_blocks.1", "read_blocks.1"),
             ("dec_norm", "read_norm")]
    own = model.state_dict()
    new, loaded, skipped = {}, [], []
    for src_pre, dst_pre in list(want.items()) + extra:
        for k, v in sd.items():
            if not (k == src_pre or k.startswith(src_pre + ".")):
                continue
            dst = dst_pre + k[len(src_pre):]
            if (dst == "register_tokens.weight" and dst in own
                    and own[dst].shape[0] > v.shape[0]
                    and own[dst].shape[0] % v.shape[0] == 0
                    and own[dst].shape[1] == v.shape[1]):
                # A widened state (state_tokens > 768) tiles the learned prior
                # rather than random-initialising the extra rows: stays in the
                # loaded decoders' distribution; tiny noise breaks the symmetry
                # so tied rows can specialise.
                reps = own[dst].shape[0] // v.shape[0]
                new[dst] = v.repeat(reps, 1) + 1e-4 * torch.randn(
                    own[dst].shape, generator=torch.Generator().manual_seed(0))
                loaded.append(dst)
                print(f"[load_cut3r_weights] {dst}: tiled x{reps} to "
                      f"{tuple(own[dst].shape)}")
                continue
            if dst in own and own[dst].shape == v.shape:
                new[dst] = v
                loaded.append(dst)
            else:
                skipped.append((dst, tuple(v.shape),
                                tuple(own[dst].shape) if dst in own else None))
    missing = [k for k in own if k not in new]
    model.load_state_dict(new, strict=False)
    report = {"loaded": len(loaded), "skipped": skipped,
              "random_init": sorted({k.split(".")[0] for k in missing})}
    if strict_report:
        n = sum(own[k].numel() for k in loaded) / 1e6
        print(f"[load_cut3r_weights] {len(loaded)} tensors, {n:.1f} M params")
        print(f"[load_cut3r_weights] random-init roots: {report['random_init']}")
        if skipped:
            print(f"[load_cut3r_weights] shape-skipped: {skipped}")
    return report
