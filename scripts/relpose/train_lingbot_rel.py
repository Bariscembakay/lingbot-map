#!/usr/bin/env python3
"""Option B of the CUT3R-rel transfer: fine-tune lingbot's backbone for moving-frame poses.

--mode rel: lingbot (GCTStream, released weights) streams over a clip with a SLIDING-WINDOW
  context -- every frame cached, the last --window frames only, no anchor frames once the window
  has passed them, no trajectory memory -- so attention distances and token count stay bounded at
  any stream length. Its tap-23 camera tokens feed train_rel_head's RelHead: each frame's pose
  relative to a reference frame that moves every --ref-every frames, chained into the trajectory.
  The depth head is kept and supervised.
--mode abs: the matched control -- lingbot's own context (anchor frames + window + trajectory
  memory), lingbot's camera head on absolute poses (train_bidir_refiner.loss_fn), same data,
  schedule and depth loss.

Truncated backprop by chunk recomputation: (1) the clip streams chunk by chunk without gradients,
with a KV-cache snapshot (to CPU) wherever a trained chunk starts; (2) the head and the pose loss
run on all camera tokens, which gives the trained frames' token gradients; (3) every trained chunk
re-runs from its snapshot with gradients and back-propagates its depth loss plus those token
gradients. One chunk's graph is in memory at a time, and the computation is exactly step (1)'s.
The trained window is --train-chunks consecutive chunks at a random position (CUT3R scale2's
option C), so trained frames see anything from a fresh stream to a full window of history.

Data: CUT3R scale2's datasets and hold-outs through CUT3R's own loaders (--cut3r), images centre-
cropped to multiples of 14 (lingbot's patch size), GT in frame 0's camera divided by the first
frames' mean point distance (lingbot Sec. 3.2). Multi-GPU: torchrun; gradients are all-reduced by
hand because the chunked forward never goes through DDP.
Usage: torchrun --nproc_per_node N train_lingbot_rel.py --out DIR [--mode rel|abs] [...]
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/relpose"))
sys.path.insert(0, str(REPO / "scripts/memory/pose_refine"))
import train_bidir_refiner as br  # noqa: E402
import train_pose_refine as tp  # noqa: E402
import train_rel_head as rh  # noqa: E402
from lingbot_map.memory.losses import depth_loss  # noqa: E402
from lingbot_map.models.gct_stream import GCTStream  # noqa: E402
from lingbot_map.utils.rotation import quat_to_mat  # noqa: E402

A = 8          # lingbot's scale (anchor) frames, processed jointly at the start of every stream
TAPS = [4, 11, 17, 23]


# -- data ----------------------------------------------------------------------
# CUT3R scale2's resolutions without 512x144: centre-cropped to 504x140 its aspect (0.28) is below
# lingbot's training range (0.33-1)
LINGBOT_RES = "[(512, 384), (512, 336), (512, 288), (512, 256), (512, 208), (384, 512), (336, 512), (288, 512), (256, 512)]"


def cut3r_datasets(cut3r: Path, data_root: str, num_views: int, which: str):
    """CUT3R scale2's train (2000 clips per dataset per epoch) or validation datasets, built from its
    own hydra config so splits, hold-outs and loaders are exactly CUT3R's."""
    sys.path.insert(0, str(cut3r / "src"))
    os.environ["DATA_ROOT"] = data_root             # scale2's data_root is ${oc.env:DATA_ROOT,/data}
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(cut3r / "config"), version_base=None):
        cfg = compose(config_name="scale2_frame0", overrides=[f"num_views={num_views}", f"num_test_views={num_views}",
                                                              f"pilot_res='{LINGBOT_RES}'"])
    import dust3r.datasets as dd
    from dust3r.datasets.utils import transforms
    ns = {**vars(dd), **vars(transforms)}
    if which == "train":
        return eval(cfg.train_dataset, ns)
    return [eval(s, ns) for s in cfg.test_dataset.split("+")]


class Shard(torch.utils.data.Sampler):
    """Every world-th batch of a CUT3R batch sampler (which seeds by epoch only, so ranks agree)."""

    def __init__(self, inner, rank, world):
        self.inner, self.rank, self.world = inner, rank, world

    def set_epoch(self, e):
        self.inner.set_epoch(e)

    def __iter__(self):
        for i, b in enumerate(self.inner):
            if i % self.world == self.rank:
                yield b

    def __len__(self):
        return len(self.inner) // self.world


def to_lingbot(views, dev):
    """CUT3R views (list over frames of batched dicts) -> lingbot inputs and normalised GT:
    imgs [B,T,3,H,W] in [0,1]; depth/valid [B,T,H,W]; c2w [B,T,4,4] in frame 0's camera; K [B,T,3,3];
    hw; per-clip scale s (mean distance of the first A frames' valid GT points from frame 0)."""
    img = torch.stack([v["img"] for v in views], 1).to(dev) * 0.5 + 0.5          # ImgNorm -> [0,1]
    depth = torch.stack([v["depthmap"] for v in views], 1).to(dev).float()
    valid = torch.stack([v["valid_mask"] for v in views], 1).to(dev).bool() & (depth > 0)
    pose = torch.stack([v["camera_pose"] for v in views], 1).to(dev).float()
    K = torch.stack([v["camera_intrinsics"] for v in views], 1).to(dev).float().clone()
    pts = torch.stack([v["pts3d"] for v in views], 1).to(dev).float()
    H, W = img.shape[-2:]
    h, w = H // 14 * 14, W // 14 * 14
    dy, dx = (H - h) // 2, (W - w) // 2
    img, depth, valid, pts = (img[..., dy:dy + h, dx:dx + w], depth[..., dy:dy + h, dx:dx + w],
                              valid[..., dy:dy + h, dx:dx + w], pts[:, :, dy:dy + h, dx:dx + w])
    K[..., 0, 2] -= dx
    K[..., 1, 2] -= dy
    c2w = rh.rigid_inv(pose[:, :1]) @ pose
    p0 = torch.einsum("bij,btnmj->btnmi", rh.rigid_inv(pose[:, 0])[:, :3, :3], pts) + rh.rigid_inv(pose[:, 0])[:, None, None, None, :3, 3]
    m = valid[:, :A]
    s = torch.stack([p0[b, :A][m[b]].norm(dim=-1).mean() if m[b].any() else torch.tensor(1.0, device=dev)
                     for b in range(len(m))]).clamp_min(1e-3)
    depth = depth / s[:, None, None, None]
    c2w = c2w.clone()
    c2w[..., :3, 3] /= s[:, None, None]
    return img, depth, valid, c2w, K, (h, w), s


# -- streaming -----------------------------------------------------------------
def chunk_mask(model, mode, f0, c, P, window, dev, anchors=False):
    """Dense key mask for chunk frames f0..f0+c-1 over [evicted specials | cached frames | chunk].
    rel: the last `window` frames up to the query (+ the A scale frames with anchors); abs: lingbot's
    own context, as in GCTStream.inference_chunked."""
    kv = model.aggregator.kv_cache
    n_cached = kv["k_0"].shape[2]
    q = f0 + torch.arange(c, device=dev)[:, None]      # one mask row per query frame, shared by its P tokens
    if mode == "rel" and anchors:
        n_a = min(A, n_cached)
        k = torch.cat([torch.arange(n_a, device=dev), torch.arange(f0 - (n_cached - n_a), f0 + c, device=dev)])
        k = k.repeat_interleave(P)[None]
        return additive_bias((k < A) | ((k >= q - window + 1) & (k <= q)), P)
    if mode == "rel":
        k = torch.arange(f0 - n_cached, f0 + c, device=dev).repeat_interleave(P)[None]
        return additive_bias((k >= q - window + 1) & (k <= q), P)
    blk = model.aggregator.global_blocks[0].attn
    w, kv_scale = blk.kv_cache_sliding_window, blk.kv_cache_scale_frames
    n_special = model.aggregator.patch_start_idx
    n_win = n_cached - min(kv_scale, n_cached)
    n_ev = f0 - kv_scale - n_win
    tok = torch.arange(P, device=dev)
    kf = torch.cat([kv_scale + torch.arange(n_ev, device=dev).repeat_interleave(n_special),
                    torch.cat([torch.arange(min(kv_scale, n_cached), device=dev),
                               torch.arange(f0 - n_win, f0, device=dev)]).repeat_interleave(P),
                    torch.arange(f0, f0 + c, device=dev).repeat_interleave(P)])
    kspec = torch.cat([torch.ones(n_ev * n_special, dtype=torch.bool, device=dev),
                       (tok < n_special).repeat(n_cached + c)])
    k = kf[None]
    return additive_bias((k < kv_scale) | ((k >= q - w + 1) & (k <= q)) | (kspec[None] & (k <= q)), P)


def additive_bias(m, P):
    """[c,S] bool per query frame -> [1,1,c*P,S] additive bf16 bias (each frame's row repeated for its
    P tokens) whose row stride is a multiple of 8. SDPA's memory-efficient kernel otherwise converts a
    bool mask (and pads an unaligned one) into a new tensor on every call, and autograd keeps one per
    global layer: 24 x ~1 GB at chunk 16."""
    c, S = m.shape
    b = torch.zeros(c, P, -(-S // 8) * 8, dtype=torch.bfloat16, device=m.device)[..., :S]
    return b.masked_fill_(~m[:, None], float("-inf")).flatten(0, 1)[None, None]


def chunk_context(model, mode, anchors, f0, c, P, window, hw, dev):
    """The kv-cache entries one chunk needs: its dense mask and, with relocated anchors, the time shift
    that moves the cached scale frames to just before the window. Keys are cached after RoPE, so plain
    anchors would sit at positions 0..A-1 forever and their distance to the query would grow with the
    stream, past lingbot's 320-frame RoPE range; shifted, it stays as in training at any length."""
    kv = model.aggregator.kv_cache
    kv["_chunk_mask"] = chunk_mask(model, mode, f0, c, P, window, dev, anchors)
    kv["_anchor_shift"] = None
    s = f0 - (kv["k_0"].shape[2] - A) - A               # first window frame - A
    if anchors and s > 0:
        agg = model.aggregator
        rot = lambda t: agg._get_3d_positions_streaming(1, hw[0], hw[1], dev, t, t + 1)
        kv["_anchor_shift"] = (rot(s) * rot(0).conj()).repeat(1, 1, A, 1)


def clear_chunk_context(model):
    model.aggregator.kv_cache["_chunk_mask"] = None
    model.aggregator.kv_cache["_anchor_shift"] = None


def run_block(model, x, nfpb):
    agg, psi = model._aggregate_features(x, num_frame_for_scale=A, num_frame_per_block=nfpb)
    pred = model._predict_depth(agg, x, psi)
    # clone: a view of the camera token would keep the whole chunk's token tensor alive in the callers' lists
    return agg[-1][:, :, 0].clone(), pred["depth"][..., 0], pred["depth_conf"]


def snapshot(model):
    kv = model.aggregator.kv_cache
    return ({k: (v.detach().to("cpu", non_blocking=True) if torch.is_tensor(v) else v) for k, v in kv.items()},
            model.aggregator.total_frames_processed)


def restore(model, snap, dev):
    kv = model.aggregator.kv_cache
    for k, v in snap[0].items():
        kv[k] = v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
    model.aggregator.total_frames_processed = snap[1]
    model.aggregator._cached_pos3d = None


def chunk_starts(T, c):
    return [0] + list(range(A, T, c))


def stream_block(model, mode, imgs, i, starts, window, anchors=False):
    """Run block i (0 = the scale frames) of the stream on the current cache."""
    T = imgs.shape[1]
    kv = model.aggregator.kv_cache
    f0 = starts[i]
    f1 = starts[i + 1] if i + 1 < len(starts) else T
    if i == 0:
        clear_chunk_context(model)
        return run_block(model, imgs[:, :A], A)
    chunk_context(model, mode, anchors, f0, f1 - f0, kv["k_0"].shape[3], window, imgs.shape[-2:], imgs.device)
    try:
        return run_block(model, imgs[:, f0:f1], f1 - f0)
    finally:
        clear_chunk_context(model)


# -- next-step forecast ----------------------------------------------------------
class ForecastHead(torch.nn.Module):
    """Frame t+1 in frame t's camera from the camera tokens of frames t-kernel+1..t only: a learned motion
    prior (the refiner fuses it with the observed step). Gated causal depthwise conv over time, then an
    MLP; translation in the normalised GT units, quaternion scalar-last. Starts at no motion."""

    def __init__(self, dim, width=512, kernel=16):
        super().__init__()
        self.kernel = kernel
        self.inp = torch.nn.Sequential(torch.nn.LayerNorm(dim), torch.nn.Linear(dim, width))
        self.conv = torch.nn.Conv1d(width, width, kernel, groups=width)
        self.gate = torch.nn.Conv1d(width, width, kernel, groups=width)
        self.mix = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, width), torch.nn.GELU(),
                                       torch.nn.Linear(width, width))
        self.out = torch.nn.Linear(width, 7)
        torch.nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.zero_()
            self.out.bias[6] = 1.0

    def forward(self, cam):
        """[B, T, dim] -> step matrices [B, T, 4, 4]: row t is the forecast of frame t+1 in frame t."""
        h = torch.nn.functional.pad(self.inp(cam).transpose(1, 2), (self.kernel - 1, 0))
        h = (self.conv(h) * torch.sigmoid(self.gate(h))).transpose(1, 2)
        o = self.out(h + self.mix(h)).float()
        M = torch.zeros(*o.shape[:2], 4, 4, device=o.device)
        M[..., :3, :3] = quat_to_mat(torch.nn.functional.normalize(o[..., 3:], dim=-1))
        M[..., :3, 3] = o[..., :3]
        M[..., 3, 3] = 1
        return M


def forecast_loss(F_, G, args):
    S = rh.rigid_inv(G[:, :-1]) @ G[:, 1:]
    P = F_[:, :-1]
    lt = (P[..., :3, 3] - S[..., :3, 3]).abs().sum(-1)
    lr = rh.angle(P[..., :3, :3], S[..., :3, :3])
    return (args.trans_w * lt + args.rot_w * lr).mean(), {"fcst_t": lt.mean().detach(),
                                                          "fcst_r_deg": torch.rad2deg(lr).mean().detach()}


# -- training step -------------------------------------------------------------
def train_step(model, head, args, batch, dev, rng, fcst=None):
    imgs, depth_gt, valid, G, K, hw, _ = batch
    T = imgs.shape[1]
    starts = chunk_starts(T, args.chunk)
    n = len(starts)
    first = int(rng.integers(0, max(1, n - args.train_chunks) + 1))
    trained = set(range(first, min(n, first + args.train_chunks)))
    cams, snaps = [None] * n, {}
    model.clean_kv_cache()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(n):
            if i in trained:
                snaps[i] = snapshot(model)
            cams[i] = stream_block(model, args.mode, imgs, i, starts, args.window, args.anchors)[0].float()
    bounds = starts + [T]
    cam = torch.cat(cams, 1)
    cam_leaf = cam.detach().requires_grad_(True)
    enc = tp.extri_intri_to_pose_encoding(G[:, :, :3], K, hw)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        preds = head(cam_leaf)
    preds = [p.float() for p in preds]
    if args.mode == "rel":
        lpose, parts = rh.rel_loss(preds, G, enc[..., 7:], hw, args)
    else:
        lpose, la, lpair = br.loss_fn(preds, enc, hw, 0, norm=True, floor=1.0)
        parts = {"abs": la, "pair": lpair}
    total = args.pose_w * lpose
    if fcst is not None:
        lf, pf = forecast_loss(fcst(cam_leaf), G, args)
        total = total + args.forecast_w * lf
        parts.update(pf)
    total.backward()
    g_cam = cam_leaf.grad
    ldep = []
    for i in sorted(trained):
        restore(model, snaps[i], dev)
        f0, f1 = bounds[i], bounds[i + 1]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            c_i, d_i, conf_i = stream_block(model, args.mode, imgs, i, starts, args.window, args.anchors)
        ld = depth_loss(d_i.float(), conf_i.float(), depth_gt[:, f0:f1], valid=valid[:, f0:f1])["depth"]
        (args.depth_w * ld / len(trained) + (c_i.float() * g_cam[:, f0:f1]).sum()).backward()
        ldep.append(ld.detach())
    parts.update(depth=torch.stack(ldep).mean())
    return lpose.detach(), parts


def all_reduce_grads(params, world, bucket=2 ** 26):
    """Average gradients over ranks in flat buckets (a parameter without a gradient on this rank
    counts as zero, so every rank reduces the same buckets)."""
    if world == 1:
        return
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    group, size = [], 0
    for p in params + [None]:
        if p is not None:
            group.append(p)
            size += p.numel()
        if group and (p is None or size >= bucket):
            flat = torch.cat([q.grad.reshape(-1) for q in group])
            dist.all_reduce(flat)
            flat /= world
            o = 0
            for q in group:
                q.grad.copy_(flat[o:o + q.numel()].view_as(q.grad))
                o += q.numel()
            group, size = [], 0


# -- evaluation ----------------------------------------------------------------
@torch.no_grad()
def eval_clip(model, head, args, batch):
    imgs, depth_gt, valid, G, K, hw, s = batch
    starts = chunk_starts(imgs.shape[1], args.chunk)
    model.clean_kv_cache()
    cams, deps = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(len(starts)):
            c_i, d_i, _ = stream_block(model, args.mode, imgs, i, starts, args.window, args.anchors)
            cams.append(c_i.float())
            deps.append(d_i.float())
        p = head(torch.cat(cams, 1))[-1].float()
    if args.mode == "rel":
        P = rh.chain(rh.step_matrices(p, hw), args.ref_every)
    else:
        P = tp.pose_enc_to_c2w(p, hw)
    d = torch.cat(deps, 1)
    rows = []
    for b in range(len(imgs)):
        r = tp.traj_metrics(P[b], G[b], float(s[b]))
        m = valid[b]
        k = (depth_gt[b][m] / d[b][m]).median()          # per-clip median scale, as depth benchmarks do
        r["abs_rel"] = float(((d[b][m] * k - depth_gt[b][m]).abs() / depth_gt[b][m]).mean())
        rows.append(r)
    return rows


def evaluate(model, head, args, val_sets, dev):
    model.eval()
    head.eval()
    out = {}
    for name, loader in val_sets:
        rows = []
        for views in loader:
            rows += eval_clip(model, head, args, to_lingbot(views, dev))
        out[name] = {k: float(np.mean([r[k] for r in rows])) for k in ("ate", "rpe_t", "rpe_r", "auc30", "abs_rel")}
    model.train()
    head.train()
    return out


# -- main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["rel", "abs"], default="rel")
    ap.add_argument("--cut3r", type=Path, default=Path.home() / "CUT3R")
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--ckpt", type=Path, default=REPO / "ckpt/lingbot-map.pt")
    ap.add_argument("--views", type=int, default=104, help="frames per training clip (8 scale frames + chunks)")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--train-chunks", type=int, default=2)
    ap.add_argument("--window", type=int, default=64, help="rel: frames of backbone context")
    ap.add_argument("--ref-every", type=int, default=64)
    ap.add_argument("--head-window", type=int, default=128)
    ap.add_argument("--rot-w", type=float, default=10.0)
    ap.add_argument("--trans-w", type=float, default=10.0)
    ap.add_argument("--pair-w", type=float, default=1.0)
    ap.add_argument("--pair-window", type=int, default=64)
    ap.add_argument("--long-w", type=float, default=1.0)
    ap.add_argument("--pose-w", type=float, default=1.0)
    ap.add_argument("--depth-w", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1, help="clips per GPU")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--val-clips", type=int, default=8, help="per validation dataset")
    ap.add_argument("--val-views", type=int, default=128)
    ap.add_argument("--max-steps-smoke", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchors", action="store_true", help="rel: keep the A scale frames in the backbone "
                    "context (relocated to just before the window), for scale")
    ap.add_argument("--forecast-w", type=float, default=0.0, help="weight of the causal next-step forecast head")
    ap.add_argument("--forecast-kernel", type=int, default=16, help="frames the forecast sees (t-k+1..t)")
    ap.add_argument("--init-head", type=Path, help="rel: start the head from a train_rel_head.py checkpoint "
                    "(same head config) instead of the identity")
    ap.add_argument("--resume", action="store_true", help="continue from --out/last.pt (Leonardo's 24 h walltime)")
    args = ap.parse_args()
    args.ref_every = args.ref_every if args.mode == "rel" else 1
    args.residual = False

    dist_on = "WORLD_SIZE" in os.environ
    if dist_on:
        # validation runs on rank 0 alone while the others wait in the next all-reduce
        dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist_on else (0, 1)
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    torch.manual_seed(args.seed + rank)
    rng = np.random.default_rng(args.seed + rank)
    main_proc = rank == 0
    if main_proc:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    rel = args.mode == "rel"
    model = GCTStream(enable_3d_rope=True, max_frame_num=4096, kv_cache_sliding_window=args.window if rel else 64,
                      kv_cache_scale_frames=A, kv_cache_include_scale_frames=not rel or args.anchors,
                      kv_cache_cross_frame_special=not rel, use_sdpa=True, use_gradient_checkpoint=False)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(sd.get("model", sd), strict=False)
    model.to(dev).train()
    model.aggregator.patch_embed.requires_grad_(False)          # DINO encoder stays frozen
    if model.camera_head is not None:
        model.camera_head.requires_grad_(False)                 # replaced by `head`
    head = rh.RelHead(tp.build_head(args.ckpt, rope_on=rel, max_frames=4096 if rel else 1024), args.mode,
                      args.head_window, prev_token=rel, init_identity=rel, ref_every=args.ref_every).to(dev)
    if args.init_head:
        head.load_state_dict(torch.load(args.init_head, map_location="cpu", weights_only=False)["model"])
    fcst = ForecastHead(rh.D, kernel=args.forecast_kernel).to(dev) if args.forecast_w > 0 else None
    bb = [p for p in model.parameters() if p.requires_grad]
    hp = [p for p in head.parameters() if p.requires_grad] + (list(fcst.parameters()) if fcst is not None else [])
    if main_proc:
        print(f"[model] mode={args.mode} backbone trainable {sum(p.numel() for p in bb) / 1e6:.0f} M, head "
              f"{sum(p.numel() for p in hp) / 1e6:.0f} M; missing {len(missing)} unexpected {len(unexpected)}", flush=True)
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": hp, "lr": args.head_lr}],
                            weight_decay=0.05, betas=(0.9, 0.95))

    train_ds = cut3r_datasets(args.cut3r, args.data_root, args.views, "train")
    sampler = Shard(train_ds.make_sampler(args.batch, shuffle=True, drop_last=True, world_size=1, fixed_length=True),
                    rank, world)
    loader = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_sets = []
    if main_proc:
        for ds in cut3r_datasets(args.cut3r, args.data_root, args.val_views, "val"):
            name = getattr(ds.dataset, "name", type(ds.dataset).__name__) if hasattr(ds, "dataset") else type(ds).__name__
            sub = torch.utils.data.Subset(ds, list(range(min(args.val_clips, len(ds)))))
            ds.set_epoch(0) if hasattr(ds, "set_epoch") else None
            val_sets.append((name, torch.utils.data.DataLoader(sub, batch_size=1, num_workers=2)))

    run = None
    if main_proc and not args.no_wandb:
        os.environ.setdefault("WANDB_DIR", str(args.out))
        try:
            import wandb
            # one wandb run across --resume jobs
            idf = args.out / "wandb_id"
            run_id = idf.read_text().strip() if idf.exists() else wandb.util.generate_id()
            idf.write_text(run_id)
            run = wandb.init(project="lingbot-rel", group="backbone", name=args.out.name, id=run_id, resume="allow",
                             config={k: str(v) for k, v in vars(args).items()})
        except Exception as e:
            print(f"[wandb] disabled: {e}", flush=True)

    hist, step0 = [], 0
    if args.resume and (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        head.load_state_dict(ck["head"])
        if fcst is not None:
            fcst.load_state_dict(ck["forecast"])
        opt.load_state_dict(ck["opt"])
        step0, hist = ck["step"], ck["hist"]
        del ck
        if main_proc:
            print(f"[resume] step {step0}", flush=True)

    def validate(step):
        if not main_proc:
            return
        r = evaluate(model, head, args, val_sets, dev)
        row = {"step": step, **{f"{k}/{m}": v for k, d in r.items() for m, v in d.items()}}
        hist.append(row)
        print(f"[val {step}] " + " ".join(f"{k} ate {d['ate']:.3f} rpe_t {d['rpe_t']:.3f}" for k, d in r.items()), flush=True)
        if run is not None:                      # wandb gets ATE and RPE-t only; history.json keeps every metric
            log = {f"val_{m}/{k}": d[m] for k, d in r.items() for m in ("ate", "rpe_t")}
            log.update({f"val_{m}/mean": float(np.mean([d[m] for d in r.values()])) for m in ("ate", "rpe_t")})
            run.log(log, step=step)
        tmp = args.out / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "forecast": fcst.state_dict() if fcst is not None else None,
                    "args": vars(args), "step": step, "hist": hist}, tmp)
        tmp.replace(args.out / "last.pt")
        (args.out / "history.json").write_text(json.dumps(hist, indent=1))

    if not hist:                                         # a resumed run already has its step-0 row
        validate(0)
    step, epoch, t0 = step0, step0 // len(sampler), time.time()
    params = bb + hp
    acc = None
    while step < args.steps:
        sampler.set_epoch(epoch)
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        for views in loader:
            step += 1
            f = min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = args.lr * f, args.head_lr * f
            opt.zero_grad(set_to_none=True)
            loss, parts = train_step(model, head, args, to_lingbot(views, dev), dev, rng, fcst)
            all_reduce_grads(params, world)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if torch.isfinite(gn):
                opt.step()
            # logged losses are means over the last 10 steps and all ranks: one rank's clip alone is too noisy
            v = torch.stack([torch.as_tensor(x, device=dev).detach().float() for x in (loss, *parts.values(), gn)]
                            + [torch.ones((), device=dev)])
            acc = torch.zeros_like(v) if acc is None else acc
            if torch.isfinite(v).all():
                acc += v
            if step % 10 == 0:
                if world > 1:
                    dist.all_reduce(acc)
                m = dict(zip(["pose", *parts, "grad_norm"], (acc[:-1] / acc[-1].clamp(min=1)).tolist()))
                acc.zero_()
                if main_proc:
                    msg = " ".join(f"{k} {v:.4f}" for k, v in m.items() if k not in ("pose", "grad_norm"))
                    print(f"[{step:6d}] pose {m['pose']:.4f} {msg} |g| {m['grad_norm']:.2f} "
                          f"{(time.time() - t0) / (step - step0):.1f}s/it {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                          flush=True)
                    if run is not None:
                        run.log({f"train/{k}": v for k, v in m.items()} | {"train/lr": opt.param_groups[0]["lr"]}, step=step)
            if step % args.val_every == 0 or step == args.steps or step == args.max_steps_smoke:
                validate(step)
            if step >= args.steps or step == args.max_steps_smoke:
                break
        epoch += 1
        if step == args.max_steps_smoke:
            break
    if dist_on:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
