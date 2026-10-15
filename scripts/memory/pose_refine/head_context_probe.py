#!/usr/bin/env python3
"""How much does lingbot's causal camera head use its cached past camera tokens?

Re-decodes lingbot's poses from the cached tap-23 camera tokens with the head's context
restricted by an attention mask, and records where its attention goes.
  full      what streaming lingbot sees: the 8 anchor frames as one bidirectional block,
            then every earlier KEYFRAME (interval ceil(N/320), as the benchmark's auto
            keyframing) and itself. Must reproduce the cached poses.
  recent64  anchors + the last 64 frames + itself
  anchors   anchors + itself
  self      itself only
Attention mass (full context, averaged over blocks, iterations, heads, query frames >= 8)
is split into self / anchors / recent 64 / older past.
Usage: head_context_probe.py --sets val24,nrgbd,kitti --out x.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402

A = 8


def allowed(n, mode, kf_interval, device):
    i = torch.arange(n, device=device)[:, None]
    j = torch.arange(n, device=device)[None, :]
    anchors = j < A
    selfm = i == j
    if mode == "self":
        return selfm
    if mode == "anchors":
        return selfm | anchors
    kf = (j < A) | ((j - A) % kf_interval == 0)
    past = (j < i) & kf
    if mode == "recent64":
        return selfm | anchors | past & (i - j <= 64)
    return selfm | anchors | past


class Recorder:
    """Recomputes each camera block's attention weights from its own q/k projections."""

    def __init__(self, head):
        self.mass, self.mask, self.hooks = [], None, []
        for blk in head.trunk:
            self.hooks.append(blk.attn.register_forward_hook(self.hook))

    def hook(self, attn, inp, _out):
        if self.mask is None:
            return
        x = inp[0]
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k = attn.q_norm(qkv[0]), attn.k_norm(qkv[1])
        w = (q @ k.transpose(-1, -2)) / math.sqrt(attn.head_dim)
        w = w.masked_fill(~self.mask, float("-inf")).softmax(-1).mean((0, 1))   # [N, N]
        i = torch.arange(N, device=x.device)
        rows = i >= A
        d = i[:, None] - i[None, :]
        parts = {"self": d == 0, "anchors": (i[None, :] < A) & (d > 0),
                 "recent64": (i[None, :] >= A) & (d > 0) & (d <= 64),
                 "older": (i[None, :] >= A) & (d > 64)}
        self.mass.append({k: float((w * m)[rows].sum(1).mean()) for k, m in parts.items()})


def decode(head, cam, mask):
    for blk in head.trunk:
        blk._prepare_blockwise_causal_attn_mask = lambda device, num_frames, frame_seqlen, num_frame_per_block, m=mask: m
    with torch.autocast("cuda", enabled=False):
        return head.trunk_fn(head.token_norm(cam), None, head.num_iterations, num_frame_per_block=1,
                             num_frame_for_scale=-1, sliding_window_size=-1)[-1][0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("/group/compact-3dmem/datasets/lingbot-poserefine-v1"))
    ap.add_argument("--sets", default="val24,nrgbd,kitti")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    head = tp.build_head(tp.REPO / "ckpt/lingbot-map.pt", rope_on=False).to(dev).eval()
    rec = Recorder(head)
    res = {}
    for name in args.sets.split(","):
        clips = tp.load_split(args.inputs / name, None, dev)
        per = {}
        for c in clips:
            n, cam = c.T, c.spec[None, :, tp.TAP23, tp.CAM].float()
            # benchmark scenes ran with auto keyframing; the ScanNet++/NRGBD caches with interval 1
            kfi = math.ceil(n / 320) if name in ("kitti", "vbr") and n > 320 else 1
            r = {"frames": n, "keyframe_interval": kfi}
            for mode in ("full", "recent64", "anchors", "self"):
                m = allowed(n, mode, kfi, dev)
                rec.mask, rec.mass = (m if mode == "full" else None), []
                p = decode(head, cam, m)
                c2w = tp.pose_enc_to_c2w(p, c.hw)
                r[mode] = tp.traj_metrics(c2w, c.gt_c2w, c.scale)
                r[mode]["dpos_vs_lingbot"] = float((p[:, :3] - c.lingbot[:, :3]).norm(dim=-1).mean() * c.scale)
                if mode == "full":
                    r["attention_mass"] = {k: float(np.mean([x[k] for x in rec.mass])) for k in rec.mass[0]}
            r["lingbot"] = tp.traj_metrics(tp.pose_enc_to_c2w(c.lingbot.float(), c.hw), c.gt_c2w, c.scale)
            per[c.name] = r
            print(f"[{name}] {c.name} n={n} kf={kfi} mass={ {k: round(v, 3) for k, v in r['attention_mass'].items()} } "
                  + " ".join(f"{m}:ATE {r[m]['ate']:.3f}" for m in ("lingbot", "full", "recent64", "anchors", "self")),
                  flush=True)
        keys = ("ate", "rpe_t", "rpe_r", "auc3", "auc30", "dpos_vs_lingbot")
        summ = {m: {k: float(np.mean([v[m][k] for v in per.values() if k in v[m]])) for k in keys if any(k in v[m] for v in per.values())}
                for m in ("lingbot", "full", "recent64", "anchors", "self")}
        summ["attention_mass"] = {k: float(np.mean([v["attention_mass"][k] for v in per.values()]))
                                  for k in ("self", "anchors", "recent64", "older")}
        res[name] = {"summary": summ, "per_clip": per}
        print(f"[{name}] SUMMARY {json.dumps(summ)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
