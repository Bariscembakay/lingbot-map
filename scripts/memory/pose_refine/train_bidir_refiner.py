#!/usr/bin/env python3
"""Bidirectional transformer on top of lingbot's aggregator, decoded by lingbot's frozen head.

Input at prefix length t (frames 0..t-1), all from lingbot's last layer (tap 23, 2048-d):
  specials   camera + 4 registers + scale token of every frame          -> 6t tokens
  --patches  also the patch tokens of the 8 anchor frames and of the last 64 frames,
             the frames whose full tokens lingbot's KV cache holds at time t
Four pre-norm self-attention blocks at width 2048 (no in/out projection), bidirectional,
(--width 1024: in-projection 2048->1024 and final norm + out-projection 1024->2048, so the
input token cannot reach the output through the residual stream unchanged),
1-D RoPE on frame index, learned token-type (and patch-position) embeddings. The outputs at
the camera-token positions ARE the new camera tokens (no added residual), decoded by
lingbot's camera head, frozen and causal as lingbot runs it.
--anchors fixed: frames 0..7 keep lingbot's own camera tokens and poses and are left out of
the loss; free: they are refined and supervised like every other frame.
Loss, lingbot paper Sec. 3.3 (training code unreleased; lambdas not given, so 1): absolute pose
as VGGT on the camera-to-world encoding over the head iterations, plus a relative-pose loss
over all frame pairs within a 64-frame window (geodesic rotation + L1 translation).
GT is the caches' anchor-scale-normalised pose. --norm extent divides translation errors by the
sample's mean GT camera distance from frame 0, floored at --trans-floor (1 = the anchor scene
depth), so long outdoor prefixes do not dominate and short/indoor ones keep the plain loss.
--packs trains on lingbot-cache1048 datasets packed by build_tap23_pack.py, up to 1048 frames,
sampling the dataset first, then t, then clips at least t long; --longval holds out whole
environments/scenes per dataset and evaluates them at 320/640/1048 (best.pt by their mean ATE
ratio to lingbot). --head-mode bidir runs lingbot's camera head without its causal mask.
--arch posetok trains PoseTokenRefiner instead (pose tokens + special tokens, no camera head).
Usage: train_bidir_refiner.py --out DIR [--patches] [--anchors fixed|free]
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
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402

D, HEADS, A, WINDOW, NPATCH = 2048, 16, 8, 64, 999
INPUTS = Path("/data/lingbot-poserefine-v1")
PACK_ROOT = Path("/data/lingbot-tap23pack-1048")
DESC_DIM = 8448                                                    # MegaLoc (SALAD aggregation) per frame
# scan_gt_jumps.py output, kept in the repo so msp3 (whose /group is a different filesystem) has it
GT_JUMPS = Path(__file__).resolve().parent / "data/gt_jump_scan.json"
# held out from --packs training, evaluated at long t: whole environments / scenes
LONGVAL_ENVS = {"hospital", "seasidetown", "gascola"}            # TartanAir V1, V2, TartanGround
VAL24 = Path(__file__).resolve().parents[3] / ".agents/scratch/memory_eval/val24.txt"


class Clip:
    """Tap-23 specials (in RAM), lingbot + GT poses, and a lazy memmap of tap-23 patch tokens."""

    def __init__(self, f: Path, device, patches: bool):
        z = np.load(f)
        self.name = f.stem
        # poserefine-v1 inputs carry the tap-cache path; cache1048 clips carry dataset/seq instead
        self.src = str(z["source"]) if "source" in z.files else f"{z['dataset']}/{z['seq']}"
        self.spec = torch.from_numpy(np.ascontiguousarray(z["specials"][:, tp.TAP23])).to(device)   # [T,6,D]
        self.T = len(self.spec)
        self.hw = tuple(int(x) for x in z["hw"])
        self.scale = float(z["gt_scale"])
        self.lingbot = torch.from_numpy(z["pose_enc"]).to(device)
        self.gt_c2w = torch.from_numpy(z["gt_c2w"]).to(device)
        K = torch.from_numpy(z["K"]).to(device)
        self.gt_enc = tp.extri_intri_to_pose_encoding(self.gt_c2w[None, :, :3], K[None], self.hw)[0]
        self._taps = None
        self.patches = patches

    def get_spec(self, t):
        return self.spec[:t]

    def patch_tokens(self, frames):
        if self._taps is None:
            c = ClipCache(self.src)
            self._taps, self._ti, self._psi = c.taps, c.meta.tap_layers.index(23), c.meta.patch_start_idx
        return torch.from_numpy(np.asarray(self._taps[frames, self._ti, self._psi:]))      # [F,999,D] fp16


class PackClip:
    """One clip of a build_tap23_pack.py pack: small arrays in RAM, specials read from the memmap per batch."""

    def __init__(self, ds, mm, idx, i, device="cpu", desc_mm=None):
        o, n = int(idx["offset"][i]), int(idx["length"][i])
        self.ds, self.name, self.seq = ds, str(idx["name"][i]), str(idx["seq"][i])
        self.src, self.T, self.hw = f"{ds}/{self.seq}", n, tuple(int(x) for x in idx["hw"][i])
        self.scale = float(idx["gt_scale"][i])
        self._mm, self._o = mm, o
        self.lingbot = torch.from_numpy(idx["pose_enc"][o:o + n]).to(device)
        self.gt_c2w = torch.from_numpy(idx["gt_c2w"][o:o + n]).to(device)
        K = torch.from_numpy(idx["K"][o:o + n]).to(device)
        self.gt_enc = tp.extri_intri_to_pose_encoding(self.gt_c2w[None, :, :3], K[None], self.hw)[0]
        self.spec = None
        self.patches = False
        self._dmm, self.desc = desc_mm, None

    def get_desc(self, t):
        if self._dmm is None:
            return None
        if self.desc is not None:
            return self.desc[:t]
        return torch.from_numpy(np.ascontiguousarray(self._dmm[self._o:self._o + t]))

    def get_spec(self, t):
        if self.spec is not None:
            return self.spec[:t]
        return torch.from_numpy(np.ascontiguousarray(self._mm[self._o:self._o + t]))

    def to(self, device):
        self.spec = self.get_spec(self.T).to(device)
        if self._dmm is not None:
            self.desc = self.get_desc(self.T).to(device)
        self.lingbot, self.gt_c2w, self.gt_enc = self.lingbot.to(device), self.gt_c2w.to(device), self.gt_enc.to(device)
        return self


def longval_split(ds, seq):
    """True if the sequence is held out for long validation"""
    if ds.startswith("tartan"):
        return seq.split("/")[0].lower() in LONGVAL_ENVS
    if ds == "matrixcity":
        return int(seq.replace("route", "")) % 10 == 0
    if ds == "scannet-v2":
        return int(seq[5:9]) % 20 == 0
    if ds == "kitti360":
        return "drive_0009" in seq
    return False


def patch_frames(t):
    return sorted(set(range(min(A, t))) | set(range(max(0, t - WINDOW), t)))


def rope1d(x, pos, theta=10000.0):
    d = x.shape[-1] // 2
    freq = theta ** (-torch.arange(d, device=x.device, dtype=torch.float32) / d)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :d], x[..., d:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Block(nn.Module):
    def __init__(self, w=D):
        super().__init__()
        self.h = w // (D // HEADS)
        self.n1, self.qkv, self.o = nn.LayerNorm(w), nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.n2, self.mlp = nn.LayerNorm(w), nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))

    def forward(self, x, pos):
        B, N, W = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.h, W // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope1d(q, pos), rope1d(k, pos)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, W))
        return x + self.mlp(self.n2(x))


class BidirRefiner(nn.Module):
    def __init__(self, head, anchors_fixed, depth=4, ckpt=False, width=D, head_mode="causal", desc_dim=0):
        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(7, width))      # 6 special kinds + patch
        # --desc: the frame's place descriptor (MegaLoc) as a 7th token of the frame, at the frame's position
        self.desc_in = nn.Sequential(nn.LayerNorm(desc_dim), nn.Linear(desc_dim, width)) if desc_dim else None
        self.desc_emb = nn.Parameter(torch.zeros(width)) if desc_dim else None
        self.patch_pos = nn.Parameter(torch.zeros(NPATCH, width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.inp = self.out = None
        if width != D:
            self.inp = nn.Linear(D, width)
            self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, D))
        self.head, self.anchors_fixed, self.ckpt = head.requires_grad_(False), anchors_fixed, ckpt
        self.head_mode = head_mode

    def forward(self, spec, lingbot_pose, patches=None, pframes=None, desc=None):
        """spec [B,t,6,D]; desc [B,t,Dd] or None; patches [B,F,999,D] of frames pframes [F] -> list of [B,t,9]."""
        B, t = spec.shape[:2]
        dev = spec.device
        proj = self.inp if self.inp is not None else (lambda z: z)
        x = proj(spec.float()) + self.type_emb[:6]
        if self.desc_in is not None:
            x = torch.cat([x, (self.desc_in(desc.float()) + self.desc_emb)[:, :, None]], 2)
        nf = x.shape[2]                                          # tokens per frame
        x = x.reshape(B, t * nf, -1)
        pos = torch.arange(t, device=dev).repeat_interleave(nf)
        if patches is not None:
            p = (proj(patches.float()) + self.type_emb[6] + self.patch_pos).reshape(B, -1, x.shape[-1])
            x = torch.cat([x, p], 1)
            pos = torch.cat([pos, torch.as_tensor(pframes, device=dev).repeat_interleave(NPATCH)])
        pos = pos[None].expand(B, -1)
        for blk in self.blocks:
            x = checkpoint(blk, x, pos, use_reentrant=False) if self.ckpt and self.training else blk(x, pos)
        cam = x[:, : t * nf].reshape(B, t, nf, -1)[:, :, 0]
        if self.out is not None:
            cam = self.out(cam)
        if self.anchors_fixed:
            cam = torch.cat([spec[:, :A, 0].to(cam.dtype), cam[:, A:]], 1)
        # lingbot's head on lingbot's camera head inputs: frozen, causal, 8 scale frames
        # (--head-mode bidir: every frame is a "scale frame", so the head attends both ways).
        # Always the equivalent dense mask (bit-identical, see train_pose_refine.py decode): the
        # FlexAttention one recompiles for every new length, and building it eagerly over t x t
        # frames in every block and iteration cost ~25 s/step at t ~ 1000.
        bidir = self.head_mode == "bidir"
        restore = [b._prepare_blockwise_causal_attn_mask for b in self.head.trunk]
        for b in self.head.trunk:
            b._prepare_blockwise_causal_attn_mask = (
                lambda device, num_frames, frame_seqlen, num_frame_per_block:
                torch.ones(num_frames, num_frames, dtype=torch.bool, device=device) if bidir else
                torch.ones(num_frames, num_frames, dtype=torch.bool, device=device).tril()
                | tp.anchor_block(num_frames, A, device))
        preds = self.head.trunk_fn(self.head.token_norm(cam), None, self.head.num_iterations,
                                   num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=-1)
        for b, f in zip(self.head.trunk, restore):
            b._prepare_blockwise_causal_attn_mask = f
        if self.anchors_fixed:
            preds = [torch.cat([lingbot_pose[:, :A].to(q.dtype), q[:, A:]], 1) for q in preds]
        return preds


class PoseTokenRefiner(nn.Module):
    """--arch posetok: lingbot's 9-d pose per frame (translation divided by lingbot's own extent, mean
    distance from frame 0 floored at 1) through an MLP to a pose token, self-attending bidirectionally with
    that frame's last-layer special tokens (camera, 4 registers, scale; tok_part full = the 2048-d
    frame+global tap projected to the width, global = its global-attention half as is) and every other
    frame's. Only the pose tokens are read out, by an MLP to 9 numbers -- lingbot's camera head is not used.
    pose_out residual: a zero-initialised correction to lingbot's pose (step 0 = lingbot);
    absolute: the pose itself."""

    def __init__(self, anchors_fixed, depth=4, width=1024, tok_part="full", pose_out="residual", ckpt=False, desc_dim=0):
        super().__init__()
        self.desc_in = nn.Sequential(nn.LayerNorm(desc_dim), nn.Linear(desc_dim, width)) if desc_dim else None
        self.desc_emb = nn.Parameter(torch.zeros(width)) if desc_dim else None
        self.anchors_fixed, self.tok_part, self.pose_out, self.ckpt = anchors_fixed, tok_part, pose_out, ckpt
        din = D if tok_part == "full" else D // 2
        self.tok_in = nn.Sequential(nn.LayerNorm(din), nn.Linear(din, width)) if tok_part == "full" else nn.LayerNorm(din)
        assert tok_part == "full" or din == width, "global-half tokens enter at their own width (1024)"
        self.pose_in = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.type_emb = nn.Parameter(torch.zeros(7, width))      # pose token + 6 special kinds
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 9))
        if pose_out == "residual":
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)

    def forward(self, spec, lingbot_pose, patches=None, pframes=None, desc=None):
        assert patches is None, "posetok reads special tokens only"
        B, t = spec.shape[:2]
        lp = lingbot_pose.float()
        unit = (lp[:, :, :3] - lp[:, :1, :3]).norm(dim=-1).mean(1).clamp(min=1.0)[:, None, None]
        pin = torch.cat([lp[:, :, :3] / unit, lp[:, :, 3:]], -1)
        tok = spec.float() if self.tok_part == "full" else spec[..., D // 2:].float()
        x = torch.cat([self.pose_in(pin)[:, :, None], self.tok_in(tok)], 2) + self.type_emb   # [B,t,7,W]
        if self.desc_in is not None:
            x = torch.cat([x, (self.desc_in(desc.float()) + self.desc_emb)[:, :, None]], 2)
        nf = x.shape[2]
        x = x.reshape(B, t * nf, -1)
        pos = torch.arange(t, device=spec.device).repeat_interleave(nf)[None].expand(B, -1)
        for blk in self.blocks:
            x = checkpoint(blk, x, pos, use_reentrant=False) if self.ckpt and self.training else blk(x, pos)
        y = self.out(x.reshape(B, t, nf, -1)[:, :, 0])
        if self.pose_out == "residual":
            pred = torch.cat([lp[..., :3] + unit * y[..., :3], lp[..., 3:7] + y[..., 3:7], lp[..., 7:] + y[..., 7:]], -1)
        else:
            pred = torch.cat([unit * y[..., :3], y[..., 3:7], F.relu(y[..., 7:])], -1)
        if self.anchors_fixed:
            pred = torch.cat([lp[:, :A], pred[:, A:]], 1)
        return [pred]


def geodesic(Ra, Rb):
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def loss_fn(preds, gt, hw, skip, norm=False, floor=1.0):
    """lingbot Sec. 3.3: VGGT absolute-pose loss (over head iterations) + relative-pose loss over
    all pairs within a 64-frame window (final iteration), frames < skip excluded. norm: both
    translation terms divided by the sample's mean GT distance from frame 0, at least floor."""
    l_abs = tp.pose_loss(preds, gt, skip=skip, norm_trans=norm, trans_floor=floor)
    unit = (gt[..., :3].norm(dim=-1).mean(1).clamp(min=floor) if norm
            else torch.ones(gt.shape[0], device=gt.device))[:, None]
    P, G = tp.pose_enc_to_c2w(preds[-1], hw), tp.pose_enc_to_c2w(gt, hw)
    t = gt.shape[1]
    i, j = torch.meshgrid(torch.arange(t, device=gt.device), torch.arange(t, device=gt.device), indexing="ij")
    m = (i != j) & ((i - j).abs() < WINDOW) & (i >= skip) & (j >= skip)
    i, j = i[m], j[m]
    Rp_i, Rp_j, Rg_i, Rg_j = P[:, i, :3, :3], P[:, j, :3, :3], G[:, i, :3, :3], G[:, j, :3, :3]
    rel_p = Rp_i.transpose(-1, -2) @ Rp_j
    rel_g = Rg_i.transpose(-1, -2) @ Rg_j
    tp_ = (Rp_i.transpose(-1, -2) @ (P[:, j, :3, 3] - P[:, i, :3, 3])[..., None])[..., 0]
    tg_ = (Rg_i.transpose(-1, -2) @ (G[:, j, :3, 3] - G[:, i, :3, 3])[..., None])[..., 0]
    l_rel = geodesic(rel_p, rel_g).mean() + ((tp_ - tg_).abs().sum(-1) / unit).mean()
    return l_abs + l_rel, l_abs.detach(), l_rel.detach()


def batch_cpu(clips, idx, t, patches):
    spec = torch.stack([clips[i].get_spec(t) for i in idx])
    d0 = clips[idx[0]].get_desc(t) if hasattr(clips[idx[0]], "get_desc") else None
    desc = torch.stack([clips[i].get_desc(t) for i in idx]) if d0 is not None else None
    gt = torch.stack([clips[i].gt_enc[:t] for i in idx])
    lp = torch.stack([clips[i].lingbot[:t] for i in idx])
    pf = patch_frames(t) if patches else None
    pt = torch.stack([clips[i].patch_tokens(pf) for i in idx]).pin_memory() if patches else None
    return spec, gt, lp, pt, pf, desc


def to_dev(b, dev):
    spec, gt, lp, pt, pf, desc = b
    return (spec.to(dev, non_blocking=True), gt.to(dev), lp.to(dev),
            pt.to(dev, non_blocking=True) if pt is not None else None, pf,
            desc.to(dev, non_blocking=True) if desc is not None else None)


def batch(clips, idx, t, dev, patches):
    return to_dev(batch_cpu(clips, idx, t, patches), dev)


@torch.no_grad()
def evaluate(model, sets, dev, patches):
    model.eval()
    out = {}
    for name, clips, ts in sets:
        res = {}
        for c in clips:
            for t_req in ts:
                t = c.T if t_req == 0 else min(t_req, c.T)       # 0 = the whole clip
                spec, gt, lp, pt, pf, desc = batch([c], [0], t, dev, patches)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(spec, lp, pt, pf, desc=desc)[-1][0].float()
                r = tp.traj_metrics(tp.pose_enc_to_c2w(p, c.hw), c.gt_c2w[:t], c.scale)
                r["lingbot"] = tp.traj_metrics(tp.pose_enc_to_c2w(c.lingbot[:t].float(), c.hw), c.gt_c2w[:t], c.scale)
                res[f"{c.name}@{t_req or 'full'}"] = r
        summ = {}
        for t in ts:
            rows = [r for k, r in res.items() if k.endswith(f"@{t or 'full'}")]
            summ[str(t or "full")] = {m: float(np.mean([r[m] for r in rows])) for m in tp.METRICS}
            summ[str(t or "full")].update({f"lingbot_{m}": float(np.mean([r["lingbot"][m] for r in rows])) for m in tp.METRICS})
        out[name] = {"summary": summ, "per_clip": res}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--anchors", choices=["fixed", "free"], required=True)
    ap.add_argument("--width", type=int, default=D)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--t-min", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=320)
    ap.add_argument("--t-step", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--nrgbd-lengths", default="64,128,256,320,500")
    ap.add_argument("--indoor-val", action="store_true", help="also validate on ScanNet++ val24 @320 and NRGBD")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--max-train", type=int, default=0, help="smoke: first N train clips per dir")
    ap.add_argument("--caches", default="",
                    help="train on lingbot-cache1048 datasets instead of poserefine-v1 train/train_v5, e.g. "
                         "scannetpp,tartanair-v2,matrixcity (ScanNet++ val24 scenes are left out)")
    ap.add_argument("--cache-root", type=Path, default=Path("/data"))
    ap.add_argument("--packs", default="",
                    help="train on build_tap23_pack.py packs of lingbot-cache1048 datasets, e.g. "
                         "tartanair-v1,tartanair-v2,tartanground,matrixcity,scannetpp,scannet-v2")
    ap.add_argument("--pack-root", type=Path, default=PACK_ROOT)
    ap.add_argument("--ds-weights", default="", help="dataset-first sampling weights, e.g. matrixcity:2 (default 1 each)")
    ap.add_argument("--t-weight", choices=["uniform", "linear"], default="uniform",
                    help="linear: P(t) proportional to t")
    ap.add_argument("--longval", action="store_true", help="hold out LONGVAL_ENVS / every 10th route / 20th scene")
    ap.add_argument("--longval-per-ds", type=int, default=8)
    ap.add_argument("--longval-lengths", default="320,640,1048")
    ap.add_argument("--keep-bad-gt", action="store_true",
                    help="keep the sequences GT_JUMPS flags (dropped whole by default)")
    ap.add_argument("--norm", choices=["none", "extent"], default="none")
    ap.add_argument("--trans-floor", type=float, default=1.0)
    ap.add_argument("--head-mode", choices=["causal", "bidir"], default="causal")
    ap.add_argument("--arch", choices=["tokens", "posetok"], default="tokens",
                    help="tokens: refined camera tokens through lingbot's head; posetok: PoseTokenRefiner")
    ap.add_argument("--pose-out", choices=["residual", "absolute"], default="residual")
    ap.add_argument("--tok-part", choices=["full", "global"], default="full")
    ap.add_argument("--desc", action="store_true", help="the frame's MegaLoc descriptor as an extra token (packs need desc.npy)")
    ap.add_argument("--frames-per-batch", type=int, default=0,
                    help="batch = min(--batch, this // t): long prefixes get fewer clips (0 = always --batch)")
    ap.add_argument("--overfit", action="store_true",
                    help="train on one ScanNet++ clip (the first with >= t-max frames), also evaluated as set 'train1'")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    head = tp.build_head(tp.REPO / "ckpt/lingbot-map.pt", rope_on=False)
    if args.arch == "posetok":
        model = PoseTokenRefiner(args.anchors == "fixed", width=args.width, tok_part=args.tok_part,
                                 pose_out=args.pose_out, ckpt=args.grad_ckpt, desc_dim=DESC_DIM if args.desc else 0).to(dev)
    else:
        model = BidirRefiner(head, args.anchors == "fixed", ckpt=args.grad_ckpt, width=args.width,
                             head_mode=args.head_mode, desc_dim=DESC_DIM if args.desc else 0).to(dev)

    def load(d, store):
        fs = sorted(f for f in (args.inputs / d / "specials").glob("*.npz") if ".tmp" not in f.stem)
        fs = fs[: args.max_train] if args.max_train and d.startswith("train") else fs
        return [Clip(f, store, args.patches) for f in fs]

    val_scenes = {l.strip().split("/")[-1].split("_c")[0] for l in VAL24.read_text().split()}
    longval = {}
    if args.packs:
        jumps = {} if args.keep_bad_gt else json.loads(GT_JUMPS.read_text())
        train = []
        for ds in args.packs.split(","):
            if ds == "kitti360" and args.pack_root == PACK_ROOT:
                # the 1048 pack keeps KITTI-360's metric GT; the whole-trajectory caches scale it to lingbot's units
                print("[data] kitti360 skipped: metric GT, not in lingbot's anchor scale", flush=True)
                continue
            idx = dict(np.load(args.pack_root / ds / "index.npz"))
            mm = np.load(args.pack_root / ds / "spec23.npy", mmap_mode="r")
            dmm = np.load(args.pack_root / ds / "desc.npy", mmap_mode="r") if args.desc else None
            bad = {b[0].rsplit("__f", 1)[0] for b in jumps.get(ds, [])}
            n_bad = n_val = 0
            lv = []
            for i in range(len(idx["name"])):
                name = str(idx["name"][i])
                seq_key = name.rsplit("__f", 1)[0]
                if seq_key in bad:
                    n_bad += 1
                    continue
                c = PackClip(ds, mm, idx, i, desc_mm=dmm)
                held = (ds == "scannetpp" and name.split("__")[0] in val_scenes) or \
                       (args.longval and longval_split(ds, c.seq))
                if held:
                    n_val += 1
                    lv.append(c)
                else:
                    train.append(c)
            if args.longval and lv:
                lv.sort(key=lambda c: (-c.T, c.name))
                longval[ds] = [c.to(dev) for c in lv[: args.longval_per_ds]]
            print(f"[data] {ds}: {sum(c.ds == ds for c in train)} train clips, {n_val} held out, "
                  f"{n_bad} dropped (bad-GT sequences)", flush=True)
    elif args.caches:
        train = []
        for ds in args.caches.split(","):
            if ds == "kitti360":
                # metric GT (no depth for the canonical scale) against lingbot's canonical-unit
                # predictions: this trainer's loss has no scale-free branch for it yet
                print("[data] kitti360 skipped: metric GT needs a scale-free loss", flush=True)
                continue
            fs = sorted((args.cache_root / f"lingbot-cache1048-{ds}").glob("*.npz"))
            if ds == "scannetpp":
                fs = [f for f in fs if f.name.split("__")[0] not in val_scenes]
            fs = fs[: args.max_train] if args.max_train else fs
            train += [Clip(f, "cpu", False) for f in fs]
            print(f"[data] {ds}: {len(fs)} clips", flush=True)
    else:
        train = load("train", "cpu") + load("train_v5", "cpu")
    lens = [int(x) for x in args.nrgbd_lengths.split(",")]
    sets = [("val24", load("val24", dev), [320]), ("nrgbd", load("nrgbd", dev), lens)] if args.indoor_val else []
    lv_lens = [0 if x == "full" else int(x) for x in args.longval_lengths.split(",")]   # full = whole clip
    sets += [(f"longval_{ds}", cl, lv_lens) for ds, cl in longval.items()]
    if args.overfit:
        train = [next(c for c in train if c.T >= args.t_max)]
        sets.append(("train1", train, [args.t_max]))
        print(f"[overfit] clip {train[0].name} ({train[0].T} frames)", flush=True)
    print(f"[data] longval: " + ", ".join(f"{ds} {len(cl)}" for ds, cl in longval.items()), flush=True)
    print(f"[data] train {len(train)} clips; val " + ", ".join(f"{n} {len(c)}" for n, c, _ in sets) +
          f"; patches={args.patches} anchors={args.anchors}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] refiner {sum(p.numel() for p in params)/1e6:.1f} M (head frozen)", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    skip = A if args.anchors == "fixed" else 0
    tchoices = np.arange(args.t_min, args.t_max + 1, args.t_step)
    tprob = tchoices / tchoices.sum() if args.t_weight == "linear" else None

    def summarise(r):
        row = {}
        for s, v in r.items():
            for t, m in v["summary"].items():
                for k in ("ate", "rpe_t", "rpe_r", "auc3", "auc30",
                          "lingbot_ate", "lingbot_rpe_t", "lingbot_rpe_r", "lingbot_auc3", "lingbot_auc30"):
                    row[f"{s}@{t}/{k}"] = m[k]
        return row

    run = None if args.no_wandb else tp.wandb_run(args, group="bidir_refiner")

    def wlog(row):
        """val/<set>@<t>/<metric>: the refiner only (comparable across runs); ratio/: refined / lingbot ATE;
        vs_lingbot/: refined and lingbot in one chart, from the whole validation history (lingbot's own
        numbers are a constant per set, so they get no chart of their own)"""
        if run is None:
            return
        import wandb
        log = {f"val/{k}": v for k, v in row.items() if k != "step" and "lingbot" not in k}
        log.update({f"ratio/{k[:-4]}": v / row[k[:-4] + "/lingbot_ate"] for k, v in row.items()
                    if k.endswith("/ate") and "lingbot" not in k})
        steps = [h["step"] for h in hist]
        for k in row:
            head, _, metric = k.rpartition("/")
            lk = f"{head}/lingbot_{metric}"
            if metric in ("ate", "rpe_t", "rpe_r", "auc3", "auc30") and lk in row:
                log[f"vs_lingbot/{k}"] = wandb.plot.line_series(
                    xs=steps, ys=[[h.get(k) for h in hist], [h.get(lk) for h in hist]],
                    keys=["refined", "lingbot"], title=k, xname="step")
        run.log(log, step=row["step"])

    hist, best = [], float("inf")
    r = evaluate(model, sets, dev, args.patches)
    hist.append({"step": 0, **summarise(r)})
    print(f"[val 0] {hist[-1]}", flush=True)
    wlog(hist[-1])
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(2)   # patch tokens are ~150 MB per sample of /data memmap reads

    lengths = np.array([c.T for c in train])
    group = np.array([getattr(c, "ds", "all") for c in train])
    gnames = sorted(set(group))
    w = dict(x.split(":") for x in args.ds_weights.split(",") if x)
    gw = np.array([float(w.get(g, 1)) for g in gnames])
    gw = gw / gw.sum()
    print(f"[data] sampling: " + ", ".join(f"{g} {p:.2f}" for g, p in zip(gnames, gw)) + f"; t {args.t_min}-{args.t_max} "
          f"step {args.t_step} {args.t_weight}", flush=True)

    def submit(force_t=None):
        g = gnames[rng.choice(len(gnames), p=gw)]     # dataset first, then t, then clips at least t long
        t = force_t or int(rng.choice(tchoices, p=tprob))
        ing = group == g
        pool_idx = np.flatnonzero(ing & (lengths >= t))
        if len(pool_idx) == 0:
            t = int(lengths[ing].max())
            pool_idx = np.flatnonzero(ing & (lengths == t))
        bs = max(1, min(args.batch, args.frames_per_batch // t)) if args.frames_per_batch else args.batch
        idx = rng.choice(pool_idx, bs, replace=len(pool_idx) < bs)
        return t, train[idx[0]].hw, pool.submit(batch_cpu, train, idx, t, args.patches)

    queue = [submit(force_t=int(tchoices.max())), submit()]   # the longest prefix first: an OOM shows at step 1
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        t, hw, fut = queue.pop(0)
        queue.append(submit())
        spec, gt, lp, pt, pf, desc = to_dev(fut.result(), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(spec, lp, pt, pf, desc=desc)
        loss, la, lr_ = loss_fn([p.float() for p in preds], gt, hw, skip, norm=args.norm == "extent",
                                floor=args.trans_floor)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[{step:6d}] loss {loss.item():.4f} abs {la.item():.4f} rel {lr_.item():.4f} t {t} "
                  f"|g| {gn.item():.3f} {(time.time()-t0)/step:.2f}s/it", flush=True)
            if run is not None:
                run.log({"train/loss": loss.item(), "train/abs": la.item(), "train/rel": lr_.item(),
                         "train/t": t, "train/grad_norm": gn.item(), "train/lr": lr}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            r = evaluate(model, sets, dev, args.patches)
            row = {"step": step, **summarise(r)}
            hist.append(row)
            print(f"[val {step}] {row}", flush=True)
            wlog(row)
            ck = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("head.")},
                  "args": vars(args), "step": step, "val": row}
            torch.save(ck, args.out / "last.pt")
            lv_t = "full" if 0 in lv_lens else max(lv_lens)
            score = (float(np.mean([row[f"longval_{ds}@{lv_t}/ate"] / row[f"longval_{ds}@{lv_t}/lingbot_ate"]
                                    for ds in longval])) if longval else row["val24@320/ate"])
            row["select"] = score
            if score < best:
                best = score
                torch.save(ck, args.out / "best.pt")
                (args.out / "best_eval.json").write_text(json.dumps(r, indent=1))
            (args.out / "history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
