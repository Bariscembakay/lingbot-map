#!/usr/bin/env python3
"""Retroactive pose refinement over lingbot's camera tokens.

At any time t: tap-23 camera tokens of frames 0..t-1 -> bidirectional refiner
(self-attention over frames, cross-attention to a memory) -> poses of all t
frames. No recurrence: every (clip, t) sample is independent.

Output variants (--variant):
  tokres_bidirhead     token residual (zero-init) -> lingbot's camera head run
                       WITHOUT its causal mask (`num_frame_for_scale=t` makes
                       every frame a "scale frame"), initialised from lingbot,
                       trained
  direct_pose          refiner -> new linear pose output (random start)
  pose_residual        lingbot's pose also embedded as input; zero-init pose
                       residual added to lingbot's pose
  tokres_frozencausal  token residual -> lingbot's camera head, frozen, causal
                       (step 0 reproduces lingbot exactly)

Memory arms (--memory):
  none      no cross-attention
  specials  camera/register/scale tokens of taps 4/11/17/23 of frames 0..t-1
  state     a frozen StateMemory's 768-token state after frame t-1
  kvcache   lingbot's own global-attention K/V of the 6 special tokens of every cached
            (key)frame at layers 4/11/17/23, rebuilt exactly from the stored taps (the
            global block's input is the frame block's output = the tap's first 1024
            dims) and rotated by lingbot's 3D RoPE as its cache stores them; refiner
            block j cross-attends to layer j's K/V through a learned query in that
            layer's head space. Dense anchor/window patch K/V are not stored.
Inputs come from build_inputs.py. --eval-only scores the untrained model
(zero residual) in --head-mode causal | bidir, which with causal should
reproduce lingbot's cached poses.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(1, str(REPO / "benchmark"))
from lingbot_map.heads.camera_head import CameraCausalHead  # noqa: E402
from lingbot_map.heads.head_act import activate_pose  # noqa: E402
from lingbot_map.memory.losses import pose_enc_to_c2w  # noqa: E402
from lingbot_map.utils.pose_enc import extri_intri_to_pose_encoding  # noqa: E402

TAP23, CAM = 3, 0          # index of tap 23 in the specials stack, camera token


# -- data ----------------------------------------------------------------------
class ClipData:
    def __init__(self, f: Path, state_dir: Path | None, device):
        z = np.load(f)
        self.name = f.stem
        self.spec = torch.from_numpy(z["specials"]).to(device)          # [T,4,6,C] fp16
        self.T = self.spec.shape[0]
        self.hw = tuple(int(x) for x in z["hw"])
        self.scale = float(z["gt_scale"])
        self.lingbot = torch.from_numpy(z["pose_enc"]).to(device)
        self.gt_c2w = torch.from_numpy(z["gt_c2w"]).to(device)
        K = torch.from_numpy(z["K"]).to(device)
        self.gt_enc = extri_intri_to_pose_encoding(self.gt_c2w[None, :, :3], K[None], self.hw)[0]
        self.gt_enc[:, 3:7] = self.gt_enc[:, 3:7] * torch.sign(self.gt_enc[:, 6:7] + 1e-12)
        # benchmark-built inputs (driving) ran the benchmark's auto keyframing; the tap caches interval 1
        auto = "bench_traj_dmax" in z.files or "rel_rot_residual_deg" in z.files
        self.kfi = -(-self.T // 320) if auto and self.T > 320 else 1
        i = torch.arange(self.T)
        kf = (i < 8) | ((i - 8) % self.kfi == 0)
        self.kf = kf.to(device)
        # lingbot's RoPE frame index: frames processed so far that were appended to the cache
        self.rope_f = (torch.cumsum(kf.long(), 0) - kf.long()).to(device)
        dz = f.resolve().parent.parent / "dino" / f.name
        self.gem = torch.from_numpy(np.load(dz)["gem"]).to(device) if dz.exists() else None
        self.states = self.stops = None
        if state_dir is not None:
            s = np.load(state_dir / f.name)
            self.states = torch.from_numpy(s["states"]).to(device)
            self.stops = [int(x) for x in s["stops"]]

    def state_at(self, t):
        return self.states[self.stops.index(t)]


def load_split(root: Path, state_label, device, names=None):
    sd = root / f"states_{state_label}" if state_label else None
    fs = sorted(f for f in (root / "specials").glob("*.npz") if ".tmp" not in f.stem)
    if names is not None:
        fs = [f for f in fs if f.stem in names]
    return [ClipData(f, sd, device) for f in fs]


# -- model ---------------------------------------------------------------------
def rope(x, pos, theta=10000.0):
    """1-D rotary embedding on the last dim of x [B, H, N, D] at positions pos [B, N]."""
    d = x.shape[-1] // 2
    freq = theta ** (-torch.arange(d, device=x.device, dtype=torch.float32) / d)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :d], x[..., d:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


def anchor_block(n, a, device):
    m = torch.zeros(n, n, dtype=torch.bool, device=device)
    m[:a, :a] = True
    return m


def anchor_mask(n, a, device):
    m = torch.ones(n, n, dtype=torch.bool, device=device)
    m[:a, a:] = False
    return m


LAYERS = (4, 11, 17, 23)
HEAD_DIM = 64


def rope3d_table(max_frames=1024):
    from lingbot_map.layers.rope import WanRotaryPosEmbed
    r = WanRotaryPosEmbed(attention_head_dim=HEAD_DIM, patch_size=(1, 14, 14), max_seq_len=max_frames)
    t, h, w = r.fhw_dim
    return r.freqs.split_with_sizes([t // 2, h // 2, w // 2], dim=1)   # complex, per axis


def rope3d_freqs(table, f, hw):
    """complex freqs [..., 32] for frame index f [...] and spatial index hw (int)."""
    ft, fh, fw = (x.to(f.device) for x in table)
    return torch.cat([ft[f], fh[hw].expand(*f.shape, -1), fw[hw].expand(*f.shape, -1)], -1)


def apply_rope3d(x, freqs):
    """x [..., 64] (interleaved pairs, as lingbot's apply_rotary_emb), freqs [..., 32] complex."""
    cos, sin = freqs.real.to(x.dtype), freqs.imag.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).reshape(x.shape)


class LingbotKV(nn.Module):
    """Frozen: special-token K/V of lingbot's global blocks 4/11/17/23, as its KV cache holds them."""

    def __init__(self, ckpt, adapter=False):
        super().__init__()
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        self.norm = nn.ModuleList(nn.LayerNorm(1024) for _ in LAYERS)
        self.kv = nn.ModuleList(nn.Linear(1024, 2048) for _ in LAYERS)
        self.k_norm = nn.ModuleList(nn.LayerNorm(HEAD_DIM) for _ in LAYERS)
        self.q_norm_init = []
        for i, l in enumerate(LAYERS):
            pre = f"aggregator.global_blocks.{l}."
            self.norm[i].load_state_dict({"weight": sd[pre + "norm1.weight"], "bias": sd[pre + "norm1.bias"]})
            W, b = sd[pre + "attn.qkv.weight"], sd[pre + "attn.qkv.bias"]
            self.kv[i].load_state_dict({"weight": W[1024:], "bias": b[1024:]})
            self.k_norm[i].load_state_dict({"weight": sd[pre + "attn.k_norm.weight"], "bias": sd[pre + "attn.k_norm.bias"]})
            self.q_norm_init.append({"weight": sd[pre + "attn.q_norm.weight"].clone(), "bias": sd[pre + "attn.q_norm.bias"].clone()})
        self.requires_grad_(False)
        self.table = rope3d_table()
        self.adapter = None
        if adapter:
            # zero-init residual on the frozen (pre-RoPE) K and on V, per layer
            self.adapter = nn.ModuleList(nn.Linear(2048, 2048) for _ in LAYERS)
            for a in self.adapter:
                nn.init.zeros_(a.weight)
                nn.init.zeros_(a.bias)

    def forward(self, spec, rope_f, i):
        """spec [B, t, 4, 6, 2048], rope_f [B, t] -> K, V [B, 16, t*6, 64] of layer LAYERS[i]."""
        B, t = spec.shape[:2]
        x = self.norm[i](spec[:, :, i, :, :1024].float())
        kv = self.kv[i](x)
        if self.adapter is not None:
            kv = kv + self.adapter[i](kv)
        k, v = kv.view(B, t, 6, 2, 16, HEAD_DIM).unbind(3)
        k = self.k_norm[i](k)
        fr = torch.stack([rope3d_freqs(self.table, rope_f, j) for j in range(6)], 2)   # [B, t, 6, 32]
        k = apply_rope3d(k, fr[:, :, :, None])
        return (k.reshape(B, t * 6, 16, HEAD_DIM).transpose(1, 2),
                v.reshape(B, t * 6, 16, HEAD_DIM).transpose(1, 2))


class GemMemory(nn.Module):
    """One extra memory token per cached frame per block: its DINOv2 GeM descriptor, projected
    into the block's K/V head space; keys get lingbot's 3D RoPE at (frame, 6, 6), the spatial
    slot after the 6 special tokens."""

    def __init__(self, n_blocks, dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.k = nn.ModuleList(nn.Linear(dim, 16 * HEAD_DIM) for _ in range(n_blocks))
        self.v = nn.ModuleList(nn.Linear(dim, 16 * HEAD_DIM) for _ in range(n_blocks))
        self.k_norm = nn.ModuleList(nn.LayerNorm(HEAD_DIM) for _ in range(n_blocks))

    def forward(self, gem, rope_f, table, i):
        B, t = gem.shape[:2]
        g = self.norm(gem.float())
        k = self.k_norm[i](self.k[i](g).view(B, t, 16, HEAD_DIM))
        k = apply_rope3d(k, rope3d_freqs(table, rope_f, 6)[:, :, None])
        return k.transpose(1, 2), self.v[i](g).view(B, t, 16, HEAD_DIM).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, d, heads, cross):
        super().__init__()
        self.h = heads
        self.n1, self.qkv, self.o1 = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.cross = cross
        if cross == "kvcache":
            self.nc = nn.LayerNorm(d)
            self.q, self.q_norm, self.oc = nn.Linear(d, 16 * HEAD_DIM), nn.LayerNorm(HEAD_DIM), nn.Linear(16 * HEAD_DIM, d)
        elif cross:
            self.nc, self.nm = nn.LayerNorm(d), nn.LayerNorm(d)
            self.q, self.kv, self.oc = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def split(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.h, D // self.h).transpose(1, 2)

    def forward(self, x, pos, mem=None, mem_pos=None):
        B, N, D = x.shape
        q, k, v = (self.split(t) for t in self.qkv(self.n1(x)).chunk(3, -1))
        q, k = rope(q, pos), rope(k, pos)
        x = x + self.o1(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        if self.cross == "kvcache":
            k, v, mask, qfreq = mem
            q = self.q_norm(self.q(self.nc(x)).view(B, N, 16, HEAD_DIM))
            q = apply_rope3d(q, qfreq[:, :, None]).transpose(1, 2)        # the camera token's position
            o = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype), attn_mask=mask)
            x = x + self.oc(o.transpose(1, 2).reshape(B, N, -1))
        elif self.cross:
            q = self.split(self.q(self.nc(x)))
            k, v = (self.split(t) for t in self.kv(self.nm(mem)).chunk(2, -1))
            if mem_pos is not None:
                q, k = rope(q, pos), rope(k, mem_pos)
            x = x + self.oc(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))


class PoseRefiner(nn.Module):
    def __init__(self, memory, variant, head, d=1024, depth=4, heads=16, tok_dim=2048,
                 state_dim=768, state_tokens=768, anchors=0, pose_emb=False, kv_adapter=False,
                 lingbot_ckpt=None, gem_mem=False):
        super().__init__()
        self.memory, self.variant, self.anchors = memory, variant, anchors
        self.inp = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
        if variant == "pose_residual":
            self.pose_in = nn.Linear(9, d)
        self.pose_emb = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d)) if pose_emb else None
        cross = "kvcache" if memory == "kvcache" else memory != "none"
        self.blocks = nn.ModuleList(Block(d, heads, cross) for _ in range(depth))
        if memory == "kvcache":
            assert depth == len(LAYERS)
            self.lkv = LingbotKV(lingbot_ckpt, kv_adapter)
            for blk, qn in zip(self.blocks, self.lkv.q_norm_init):
                blk.q_norm.load_state_dict(qn)
        self.gem_mem = GemMemory(depth) if gem_mem else None
        self.out_norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 9 if variant in ("direct_pose", "pose_residual") else tok_dim)
        if variant != "direct_pose":
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        if memory == "specials":
            self.mem_in = nn.Sequential(nn.LayerNorm(tok_dim), nn.Linear(tok_dim, d))
            self.tap_emb = nn.Parameter(torch.zeros(4, 1, d))
            self.kind_emb = nn.Parameter(torch.zeros(1, 6, d))
        elif memory == "state":
            self.mem_in = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, d))
            self.slot_emb = nn.Parameter(torch.zeros(state_tokens, d))
        self.head = head
        self.dense_causal_eval = False
        if variant == "tokres_frozencausal":
            head.requires_grad_(False)
        self.head_mode = "causal" if variant == "tokres_frozencausal" else "bidir"
        if anchors and self.head_mode == "bidir":
            # lingbot's anchor rule inside an otherwise bidirectional head: the first
            # `anchors` frames attend only to each other (they fix origin and scale),
            # every later frame attends to all.
            for blk in head.trunk:
                blk._prepare_blockwise_causal_attn_mask = (
                    lambda device, num_frames, frame_seqlen, num_frame_per_block, a=anchors:
                    anchor_mask(num_frames, a, device))

    def memory_tokens(self, spec, state):
        if self.memory == "specials":
            B, t = spec.shape[:2]
            m = self.mem_in(spec.float()) + self.tap_emb + self.kind_emb      # [B,t,4,6,d]
            pos = torch.arange(t, device=spec.device)[None, :, None, None].expand(B, t, 4, 6)
            return m.reshape(B, t * 24, -1), pos.reshape(B, -1)
        if self.memory == "state":
            return self.mem_in(state.float()) + self.slot_emb, None
        return None, None

    @property
    def outputs_poses(self):
        return self.variant in ("direct_pose", "pose_residual")

    def refine(self, spec, state=None, lingbot_pose=None, kf=None, rope_f=None, gem=None):
        """-> refined camera tokens [B, t, C], or [poses] for the pose-output variants.
        kvcache memory: kf [B, t] marks the frames lingbot cached, rope_f [B, t] their RoPE index."""
        B, t = spec.shape[:2]
        cam = spec[:, :, TAP23, CAM].float()
        pos = torch.arange(t, device=spec.device)[None].expand(B, t)
        x = self.inp(cam)
        if self.variant == "pose_residual":
            x = x + self.pose_in(lingbot_pose.float())
        if self.pose_emb is not None:
            x = x + self.pose_emb(lingbot_pose.float())
        if self.memory == "kvcache":
            mask = kf[:, None, None, :, None].expand(B, 1, 1, t, 6).reshape(B, 1, 1, t * 6)
            qfreq = rope3d_freqs(self.lkv.table, rope_f, 0)
            if self.gem_mem is not None:
                mask = torch.cat([mask, kf[:, None, None, :]], -1)
            for i, blk in enumerate(self.blocks):
                k, v = self.lkv(spec, rope_f, i)
                if self.gem_mem is not None:
                    gk, gv = self.gem_mem(gem, rope_f, self.lkv.table, i)
                    k, v = torch.cat([k, gk.to(k.dtype)], 2), torch.cat([v, gv.to(v.dtype)], 2)
                x = blk(x, pos, (k, v, mask, qfreq))
        else:
            mem, mem_pos = self.memory_tokens(spec, state)
            for blk in self.blocks:
                x = blk(x, pos, mem, mem_pos)
        y = self.out(self.out_norm(x))
        if self.anchors:
            y = y * (torch.arange(t, device=y.device) >= self.anchors)[None, :, None]
        self.last_delta = y if not self.outputs_poses else None
        if self.variant == "direct_pose":
            return [activate_pose(y, trans_act="linear", quat_act="linear", fl_act="relu")]
        if self.variant == "pose_residual":
            return [lingbot_pose.float() + y]
        return cam + y

    def forward(self, spec, state=None, lingbot_pose=None, head_mode=None, refine=True, kf=None, rope_f=None,
                gem=None):
        """spec [B, t, 4, 6, C] -> list of pose encodings [B, t, 9] (the last is final)."""
        if refine:
            cam = self.refine(spec, state, lingbot_pose, kf, rope_f, gem)
            if self.outputs_poses:
                return cam
        elif self.outputs_poses:
            return [lingbot_pose.float()]
        else:
            cam = spec[:, :, TAP23, CAM].float()
        return self.decode(cam, head_mode, lingbot_pose)

    def decode(self, cam, head_mode=None, lingbot_pose=None):
        """camera tokens [B, t, C] -> lingbot camera head (per this variant's mode)."""
        t = cam.shape[1]
        head_mode = head_mode or self.head_mode
        nfs = (self.anchors or t) if head_mode == "bidir" else 8
        restore = None
        if head_mode == "causal" and self.dense_causal_eval and not torch.is_grad_enabled():
            # the causal head's FlexAttention block mask is rebuilt by a per-frame python loop
            # and compiled per sequence length: minutes per full-length driving clip. Same mask,
            # dense: the 8 anchors attend among themselves, every later frame to itself and the past.
            restore = [blk._prepare_blockwise_causal_attn_mask for blk in self.head.trunk]
            for blk in self.head.trunk:
                blk._prepare_blockwise_causal_attn_mask = (
                    lambda device, num_frames, frame_seqlen, num_frame_per_block:
                    torch.ones(num_frames, num_frames, dtype=torch.bool, device=device).tril()
                    | anchor_block(num_frames, 8, device))
            nfs = -1   # as head_context_probe.py, whose dense "full" mask reproduced lingbot's poses
        preds = self.head.trunk_fn(self.head.token_norm(cam), None, self.head.num_iterations,
                                   num_frame_per_block=1, num_frame_for_scale=nfs,
                                   sliding_window_size=-1)
        if restore is not None:
            for blk, f in zip(self.head.trunk, restore):
                blk._prepare_blockwise_causal_attn_mask = f
        if self.anchors and lingbot_pose is not None:
            # a trainable head would otherwise drift the anchors' poses, and with them the gauge
            a = self.anchors
            preds = [torch.cat([lingbot_pose[:, :a].to(q.dtype), q[:, a:]], 1) for q in preds]
        return preds


def build_head(ckpt, rope_on, max_frames=1024):
    head = CameraCausalHead(dim_in=2048, sliding_window_size=-1, attend_to_scale_frames=False,
                            num_iterations=4, enable_3d_rope=rope_on, max_frame_num=max_frames,
                            rope_theta=10000.0)
    if rope_on and max_frames > 1024:
        # CameraCausalHead leaves WanRotaryPosEmbed's table at max_seq_len=1024; positions
        # below 1024 get identical frequencies in a longer table.
        from lingbot_map.layers.rope import WanRotaryPosEmbed
        r = head.rope3d
        head.rope3d = WanRotaryPosEmbed(attention_head_dim=r.attention_head_dim,
                                        patch_size=(max_frames, 1, 1), max_seq_len=max_frames,
                                        theta=10000.0, fhw_dim=list(r.fhw_dim))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    hs = {k[len("camera_head."):]: v for k, v in sd.items() if k.startswith("camera_head.")}
    missing, unexpected = head.load_state_dict(hs, strict=False)
    assert not unexpected, unexpected[:5]
    assert all("rope" in k for k in missing), missing[:5]
    print(f"[head] loaded {len(hs)} lingbot camera-head tensors, rope={rope_on}, "
          f"missing (rope buffers) {len(missing)}", flush=True)
    return head


def wandb_run(args, **kw):
    """wandb run named after the output dir, or None (history.json has everything regardless)."""
    import os
    # wandb's default dir is ./wandb; in the repo root that directory then shadows the wandb
    # package for every later run (REPO is on sys.path), which silently disabled logging.
    os.environ.setdefault("WANDB_DIR", "/group/compact-3dmem/wandb"
                          if Path("/group/compact-3dmem/wandb").is_dir() else f"/scratch/{os.environ.get('USER')}")
    try:
        import wandb
        return wandb.init(project="spatial_memory", name=args.out.name,
                          config={k: str(v) for k, v in vars(args).items()}, **kw)
    except Exception as e:  # no egress / no login on this node
        print(f"[wandb] disabled: {e}", flush=True)
        return None


# -- loss / metrics ------------------------------------------------------------
def pose_loss(preds, gt, gamma=0.6, w_fov=0.5, skip=0, scale_free=False, norm_trans=False,
              rel_w=0.0, hw=None, rel_steps=(1, 2, 4, 8, 16, 32, 64), trans_floor=1e-3):
    """scale_free: metric GT (driving) has no canonical scale, so its translations are
    first rescaled per sample by the closed-form factor that best fits the (detached)
    prediction; frame 0 is the origin of both, so a scale is the only freedom left.
    norm_trans: translation errors are divided by the sample's mean GT camera distance from
    frame 0 (at least trans_floor), so a 2 km drive and a 3 m room weigh alike in a mixed batch.
    rel_w: adds, on the final iteration, L1 on relative poses (translation in camera i's frame
    and rotation-matrix entries) for pairs (i, i+d), d in rel_steps, at least one past `skip`."""
    total, n = 0.0, len(preds)
    gtt_all = gt[..., :3]
    k = torch.ones(gt.shape[0], 1, 1, device=gt.device)
    if scale_free:
        pd, g = preds[-1][:, skip:, :3].detach(), gtt_all[:, skip:]
        k = (pd * g).sum((1, 2), keepdim=True) / (g * g).sum((1, 2), keepdim=True).clamp(min=1e-8)
    gtt_all = gtt_all * k
    unit = (gtt_all.norm(dim=-1).mean(1, keepdim=True).clamp(min=trans_floor) if norm_trans
            else torch.ones(gt.shape[0], 1, device=gt.device))
    for i, p in enumerate(preds):
        p = p[:, skip:]
        gtt = gtt_all[:, skip:]
        lt = (p[..., :3] - gtt).abs().sum(-1) / unit
        q, g = p[..., 3:7], gt[:, skip:, 3:7]
        lq = torch.minimum((q - g).abs().sum(-1), (q + g).abs().sum(-1))
        lf = (p[..., 7:] - gt[:, skip:, 7:]).abs().sum(-1)
        total = total + gamma ** (n - 1 - i) * (lt + lq + w_fov * lf).mean()
    if rel_w:
        P = pose_enc_to_c2w(preds[-1], hw)
        g2 = gt.clone()
        g2[..., :3] = gtt_all
        G = pose_enc_to_c2w(g2, hw)
        t = gt.shape[1]
        lr = []
        for d in rel_steps:
            if d >= t:
                break
            j = torch.arange(max(0, skip - d), t - d, device=gt.device)
            Ri, Rj, Gi, Gj = P[:, j, :3, :3], P[:, j + d, :3, :3], G[:, j, :3, :3], G[:, j + d, :3, :3]
            tp = (Ri.transpose(-1, -2) @ (P[:, j + d, :3, 3] - P[:, j, :3, 3])[..., None])[..., 0]
            tg = (Gi.transpose(-1, -2) @ (G[:, j + d, :3, 3] - G[:, j, :3, 3])[..., None])[..., 0]
            rp, rg = Ri.transpose(-1, -2) @ Rj, Gi.transpose(-1, -2) @ Gj
            lr.append(((tp - tg).abs().sum(-1) / unit).mean() + (rp - rg).abs().sum((-1, -2)).mean())
        total = total + rel_w * torch.stack(lr).mean()
    return total


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    S, D = src - ms, dst - md
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / ((S ** 2).sum() / len(src))
    return s, R, md - s * R @ ms


def pose_auc(P, G, max_frames=3000):
    """The benchmark's pairwise relative-pose AUC (benchmark/evaluation/auc.py), in %.
    Long clips are subsampled evenly: all-pairs on 19k-frame VBR would be 180M pairs."""
    from benchmark.evaluation.auc import AUCEvaluator
    if len(P) > max_frames:
        idx = np.linspace(0, len(P) - 1, max_frames).round().astype(int)
        P, G = P[idx], G[idx]
    ev = AUCEvaluator()
    pe = ev._align_to_first_camera(np.linalg.inv(P).astype(np.float32))
    ge = ev._align_to_first_camera(np.linalg.inv(G).astype(np.float32))
    r, t = ev._se3_to_relative_pose_error(pred_se3=pe, gt_se3=ge, num_frames=len(P))
    return {f"auc{th}": float(ev._calculate_auc_np(r, t, max_threshold=th)[0] * 100) for th in (3, 5, 15, 30)}


def traj_metrics(pred_c2w, gt_c2w, scale):
    """ATE (Sim3 on centres) and consecutive-frame RPE, in metres / degrees."""
    P, G = pred_c2w.double().cpu().numpy(), gt_c2w.double().cpu().numpy()
    s, R, t = umeyama(P[:, :3, 3], G[:, :3, 3])
    ate = np.sqrt((np.linalg.norm(s * P[:, :3, 3] @ R.T + t - G[:, :3, 3], axis=1) ** 2).mean())
    rp = np.linalg.inv(P[:-1]) @ P[1:]
    rg = np.linalg.inv(G[:-1]) @ G[1:]
    rpe_t = np.linalg.norm(s * rp[:, :3, 3] - rg[:, :3, 3], axis=1).mean()
    m = np.einsum("nji,njk->nik", rp[:, :3, :3], rg[:, :3, :3])
    rpe_r = np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1))).mean()
    # AUC compares all frame pairs: on whole trajectories (up to ~9k frames) use 1500 evenly spaced frames
    sub = np.linspace(0, len(P) - 1, min(len(P), 1500)).round().astype(int)
    return dict(ate=ate * scale, rpe_t=rpe_t * scale, rpe_r=rpe_r, **pose_auc(P[sub], G[sub]))


@torch.no_grad()
def evaluate(model, clips, ts, head_mode=None, refine=True):
    out = {}
    model.eval()
    for c in clips:
        for t in ts:
            t = c.T if t is None else min(t, c.T)
            st = c.state_at(t)[None] if model.memory == "state" else None
            # fp32: on NRGBD (~0.2 deg/frame) bf16 jitter alone inflated RPE-rot 30%
            with torch.autocast("cuda", enabled=False):
                gem = c.gem[None, :t] if model.gem_mem is not None else None
                p = model(c.spec[None, :t], st, c.lingbot[None, :t], head_mode, refine,
                          c.kf[None, :t], c.rope_f[None, :t], gem)[-1][0].float()
            r = traj_metrics(pose_enc_to_c2w(p, c.hw), c.gt_c2w[:t], c.scale)
            r["lingbot"] = traj_metrics(pose_enc_to_c2w(c.lingbot[:t].float(), c.hw),
                                        c.gt_c2w[:t], c.scale)
            r["vs_cached_lingbot"] = float((p - c.lingbot[:t]).abs().max())
            out[f"{c.name}@{t}"] = r
    model.train()
    return out


@torch.no_grad()
def online_poses(model, c, k):
    """Stream the clip, refining the whole prefix every k frames. Each frame's ONLINE pose
    is decoded when it arrives, by the variant's head, from the refined tokens up to the
    last refinement plus raw tokens since -- so new frames move with the corrected past.
    The pose-output variants have no head to feed back into: their online pose is lingbot's."""
    T, spec, lp = c.T, c.spec[None], c.lingbot[None].float()
    raw = spec[:, :, TAP23, CAM].float()
    out = torch.empty(T, 9, device=raw.device)
    hist = raw[:, :0]
    for b0 in range(0, T, k):
        b1 = min(b0 + k, T)
        if b0 and not model.outputs_poses:
            st = c.state_at(b0)[None] if model.memory == "state" else None
            hist = model.refine(spec[:, :b0], st, lp[:, :b0], c.kf[None, :b0], c.rope_f[None, :b0])
        if model.outputs_poses:
            out[b0:b1] = lp[0, b0:b1]
        elif model.head_mode == "causal":
            out[b0:b1] = model.decode(torch.cat([hist, raw[:, b0:b1]], 1), "causal",
                                      lp[:, :b1])[-1][0, b0:b1]
        else:
            for i in range(b0, b1):
                if i < model.anchors:   # fixed to lingbot; a 1-frame mask also breaks the head's attention
                    out[i] = lp[0, i]
                    continue
                out[i] = model.decode(torch.cat([hist, raw[:, b0:i + 1]], 1), "bidir",
                                      lp[:, :i + 1])[-1][0, i]
    st = c.state_at(T)[None] if model.memory == "state" else None
    final = model(spec, st, lp, kf=c.kf[None], rope_f=c.rope_f[None])[-1][0]
    return out, final


def evaluate_online(model, clips, k):
    res = {}
    model.eval()
    for c in clips:
        with torch.autocast("cuda", enabled=False):
            on, fin = online_poses(model, c, k)
        r = {"online": traj_metrics(pose_enc_to_c2w(on, c.hw), c.gt_c2w, c.scale),
             "final": traj_metrics(pose_enc_to_c2w(fin, c.hw), c.gt_c2w, c.scale),
             "lingbot": traj_metrics(pose_enc_to_c2w(c.lingbot.float(), c.hw), c.gt_c2w, c.scale)}
        res[f"{c.name}@{c.T}"] = r
        print(f"[online k={k}] {c.name}: ATE online {r['online']['ate']*100:.2f} final "
              f"{r['final']['ate']*100:.2f} lingbot {r['lingbot']['ate']*100:.2f} cm", flush=True)
    summ = {m: {q: float(np.mean([r[m][q] for r in res.values()])) for q in METRICS}
            for m in ("online", "final", "lingbot")}
    return res, summ


METRICS = ("ate", "rpe_t", "rpe_r", "auc3", "auc5", "auc15", "auc30")


def summarise(res):
    keys = METRICS
    agg = {k: float(np.mean([r[k] for r in res.values()])) for k in keys}
    agg.update({f"lingbot_{k}": float(np.mean([r["lingbot"][k] for r in res.values()])) for k in keys})
    return agg


# -- main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True, help="build_inputs.py root")
    ap.add_argument("--memory", choices=["none", "specials", "state", "kvcache"])
    ap.add_argument("--variant", choices=["tokres_bidirhead", "direct_pose",
                                                         "pose_residual", "tokres_frozencausal"])
    ap.add_argument("--state-label")
    ap.add_argument("--lingbot-ckpt", type=Path, default=REPO / "ckpt/lingbot-map.pt")
    ap.add_argument("--no-head-rope", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr-mult", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--t-min", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=320)
    ap.add_argument("--t-step", type=int, default=16)
    ap.add_argument("--eval-sets", default="val24:320,nrgbd:320+500")
    ap.add_argument("--scale-free-loss", action="store_true",
                    help="metric GT (driving): fit the GT translation scale per sample")
    ap.add_argument("--max-frames-rope", type=int, default=1024,
                    help="head temporal-RoPE table size; must exceed the longest eval clip")
    ap.add_argument("--train-dirs", default="train", help="comma list of input subdirs")
    ap.add_argument("--train-store", choices=["gpu", "cpu"], default="gpu",
                    help="cpu: train clips stay in host RAM (944 clips = 29 GB of specials)")
    ap.add_argument("--anchor-frames", type=int, default=0,
                    help="keep the first N frames' tokens and poses exactly lingbot's "
                         "(they fix the coordinate frame); 0 = off")
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", type=Path, help="eval-only: load this checkpoint (its args define the model)")
    ap.add_argument("--online-k", type=int, default=0,
                    help="eval-only: streaming evaluation, refining every k frames")
    ap.add_argument("--head-mode", choices=["causal", "bidir"],
                    help="eval-only override of the variant's head mode")
    ap.add_argument("--pose-emb", action="store_true", help="add MLP(lingbot pose) to each query")
    ap.add_argument("--kv-adapter", action="store_true", help="kvcache: zero-init learned residual on frozen K/V")
    ap.add_argument("--gem-mem", action="store_true",
                    help="kvcache: add each cached frame's DINOv2 GeM descriptor (<split>/dino/) as a memory token")
    ap.add_argument("--reg-w", type=float, default=0.0, help="weight of mean(dc^2)/mean(c^2)")
    ap.add_argument("--rel-w", type=float, default=0.0, help="weight of the local relative-pose loss")
    ap.add_argument("--norm-trans", action="store_true", help="translation loss / mean GT distance")
    ap.add_argument("--group-first", action="store_true",
                    help="draw the --train-dirs group by weight first, then t within its clip lengths "
                         "(otherwise t first, so short-clip groups only appear for small t)")
    ap.add_argument("--metric-groups", default="",
                    help="comma list of --train-dirs groups with metric GT (driving): only their batches get "
                         "the scale-free fit and distance normalisation; others keep the plain loss")
    ap.add_argument("--frames-per-batch", type=int, default=0,
                    help="batch = min(--batch, this // t): long prefixes take smaller batches")
    ap.add_argument("--t-max-start", type=int, default=0,
                    help="curriculum: t_max grows linearly from this to --t-max over 60%% of training")
    ap.add_argument("--select", default="",
                    help="comma list of eval sets; best.pt minimises their mean ATE / lingbot ATE "
                         "(default: the first eval set's ATE)")
    ap.add_argument("--tag", default="", help="eval-only: suffix of the output json name")
    ap.add_argument("--flex-causal-eval", action="store_true",
                    help="evaluate the causal head with its own FlexAttention mask instead of the "
                         "equivalent dense one (bit-identical on val24/NRGBD, but minutes per long clip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        for key, dflt in (("memory", None), ("variant", None), ("anchor_frames", 0),
                          ("no_head_rope", False), ("state_label", None), ("pose_emb", False),
                          ("kv_adapter", False), ("gem_mem", False)):
            setattr(args, key, ck["args"].get(key, dflt))
    assert args.memory and args.variant, "--memory/--variant (or --ckpt) required"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    label = args.state_label if args.memory == "state" else None

    head = build_head(args.lingbot_ckpt, not args.no_head_rope, args.max_frames_rope)
    assert args.anchor_frames == 0 or args.variant != "direct_pose", "direct_pose has no lingbot anchors"
    model = PoseRefiner(args.memory, args.variant, head, anchors=args.anchor_frames, pose_emb=args.pose_emb,
                        kv_adapter=args.kv_adapter, lingbot_ckpt=args.lingbot_ckpt, gem_mem=args.gem_mem).to(dev)
    # "name:len+len" per set; "full" = whole clip. The first set picks best.pt.
    evsets = []
    for item in args.eval_sets.split(","):
        name, lens = item.split(":")
        ts = [None if x == "full" else int(x) for x in lens.split("+")]
        evsets.append((name, load_split(args.inputs / name, label, dev), ts))
    print("[data] " + " ".join(f"{n} {len(c)}" for n, c, _ in evsets), flush=True)

    def run_eval():
        # eval-only without a checkpoint scores the untrained passthrough (lingbot itself)
        refine = not args.eval_only or args.ckpt is not None
        r = {n: evaluate(model, c, ts, args.head_mode, refine) for n, c, ts in evsets}
        return r, {k: summarise(v) for k, v in r.items()}

    model.dense_causal_eval = not args.flex_causal_eval
    if args.ckpt:
        model.load_state_dict(ck["model"])
        print(f"[ckpt] {args.ckpt} step {ck['step']}", flush=True)
    if args.eval_only and args.online_k:
        out = {}
        for name, clips, _ in evsets:
            r, s = evaluate_online(model, clips, args.online_k)
            out[name] = {"summary": s, "per_clip": r}
            print(f"[online k={args.online_k}] {name}: {json.dumps(s)}", flush=True)
        tag = args.ckpt.parent.name if args.ckpt else "untrained"
        (args.out / f"online_k{args.online_k}_{tag}.json").write_text(json.dumps(out, indent=1))
        return
    if args.eval_only:
        r, s = run_eval()
        tag = f"final_{args.ckpt.stem}{args.tag}" if args.ckpt else f"eval0_{args.head_mode or model.head_mode}{'_norope' if args.no_head_rope else ''}"
        (args.out / f"{tag}.json").write_text(
            json.dumps({"summary": s, "per_clip": r}, indent=1))
        print(json.dumps(s, indent=1))
        return

    tdev = dev if args.train_store == "gpu" else torch.device("cpu")
    # "dir[+dir...][:weight],...": each group is drawn with its weight among the groups that
    # have a clip at least t long, then a clip uniformly within the group
    groups_tr = []
    for item in args.train_dirs.split(","):
        names, _, w = item.partition(":")
        clips = [c for d in names.split("+") for c in load_split(args.inputs / d, label, tdev)]
        groups_tr.append((names, clips, float(w or len(clips)), np.array([c.T for c in clips])))
    Tmax = min(args.t_max, max(c.T for _, cl, _, _ in groups_tr for c in cl))
    print("[data] train " + ", ".join(f"{n}: {len(cl)} clips w {w:g}" for n, cl, w, _ in groups_tr)
          + f"; t in {args.t_min}..{Tmax}", flush=True)
    head_p = [p for p in model.head.parameters() if p.requires_grad]
    hid = {id(p) for p in model.head.parameters()}
    ref_p = [p for p in model.parameters() if id(p) not in hid]
    groups = [{"params": ref_p, "lr": args.lr}]
    if head_p:
        groups.append({"params": head_p, "lr": args.lr * args.head_lr_mult})
    opt = torch.optim.AdamW(groups,
                            weight_decay=args.wd, betas=(0.9, 0.95))
    base = [g["lr"] for g in opt.param_groups]
    print(f"[model] refiner {sum(p.numel() for p in ref_p)/1e6:.1f} M, head "
          f"{sum(p.numel() for p in head_p)/1e6:.1f} M", flush=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))
    run = None if args.no_wandb else wandb_run(args)

    hist, best = [], float("inf")
    sel = args.select.split(",") if args.select else []

    def score(s):
        if not sel:
            return s[evsets[0][0]]["ate"]
        return float(np.mean([s[n]["ate"] / s[n]["lingbot_ate"] for n in sel]))
    r, s = run_eval()
    hist.append({"step": 0, **{f"{k}/{m}": v for k, d in s.items() for m, v in d.items()}})
    print(f"[val 0] {hist[-1]}", flush=True)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        tm = Tmax
        if args.t_max_start:
            tm = int(args.t_max_start + (Tmax - args.t_max_start) * min(1.0, step / (0.6 * args.steps)))
        if args.group_first:
            w = np.array([g[2] for g in groups_tr])
            gname, clips, _, lens = groups_tr[rng.choice(len(groups_tr), p=w / w.sum())]
            t = int(rng.choice(np.arange(args.t_min, min(tm, lens.max()) + 1, args.t_step)))
        else:
            t = int(rng.choice(np.arange(args.t_min, tm + 1, args.t_step)))
            avail = [g for g in groups_tr if (g[3] >= t).any()]
            w = np.array([g[2] for g in avail])
            gname, clips, _, lens = avail[rng.choice(len(avail), p=w / w.sum())]
        ok = np.flatnonzero(lens >= t)
        B = min(args.batch, max(1, args.frames_per_batch // t)) if args.frames_per_batch else args.batch
        bi = rng.choice(ok, B, replace=len(ok) < B)
        sample = [clips[i] for i in bi]
        spec = torch.stack([c.spec[:t] for c in sample]).to(dev, non_blocking=True)
        gt = torch.stack([c.gt_enc[:t] for c in sample]).to(dev)
        st = (torch.stack([c.state_at(t) for c in sample]).to(dev)
              if args.memory == "state" else None)
        lp = torch.stack([c.lingbot[:t] for c in sample]).to(dev)
        kf = torch.stack([c.kf[:t] for c in sample]).to(dev)
        rf = torch.stack([c.rope_f[:t] for c in sample]).to(dev)
        gem = torch.stack([c.gem[:t] for c in sample]).to(dev) if args.gem_mem else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(spec, st, lp, kf=kf, rope_f=rf, gem=gem)
        metric = gname in args.metric_groups.split(",") if args.metric_groups else None
        loss = pose_loss([p.float() for p in preds], gt, skip=args.anchor_frames,
                         scale_free=args.scale_free_loss if metric is None else metric,
                         norm_trans=args.norm_trans if metric is None else metric,
                         rel_w=args.rel_w, hw=sample[0].hw)
        if args.reg_w and model.last_delta is not None:
            c = spec[:, :, TAP23, CAM].float()
            loss = loss + args.reg_w * model.last_delta.float().pow(2).mean() / c.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[{step:6d}] loss {loss.item():.4f} t {t} |g| {gn.item():.3f} "
                  f"{(time.time()-t0)/step:.2f}s/it", flush=True)
            if run:
                run.log({"loss": loss.item(), "grad_norm": gn.item(), "t": t}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            r, s = run_eval()
            row = {"step": step, **{f"{k}/{m}": v for k, d in s.items() for m, v in d.items()}}
            hist.append(row)
            print(f"[val {step}] {row}", flush=True)
            if run:
                run.log({k: v for k, v in row.items() if k != "step"}, step=step)
            ck = {"model": model.state_dict(), "args": vars(args), "step": step, "val": row}
            torch.save(ck, args.out / "last.pt")
            if score(s) < best:
                best = score(s)
                torch.save(ck, args.out / "best.pt")
                (args.out / "best_eval.json").write_text(json.dumps(r, indent=1))
            (args.out / "history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
