#!/usr/bin/env python3
"""How much does a frame's global DINOv2 feature change across one scene?

Per cached frame of a clip (decoded exactly as build_cache.py did, 518x378):
DINOv2 ViT-L/14-reg4 (official pretrained weights) CLS token and mean patch
token. Pairwise cosine is compared with how much two views actually see the same
surface (GT depth + GT poses: fraction of frame i's pixels that reproject into
frame j with consistent depth), camera distance (m), view angle and time gap.
A second clip gives the across-scene reference.

Writes <out>.png (figure) and <out>.json (numbers).
Usage: python dino_global_scene.py <clip_dir> --other <clip_dir> --out <path stem>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402

DINO_CKPT = "/group/compact-3dmem/checkpoints/DINOv2/dinov2_vitl14_reg4_pretrain.pth"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_dino(device, which: str):
    from functools import partial
    from lingbot_map.layers.vision_transformer import DinoVisionTransformer, Block, MemEffAttention
    # facebookresearch/dinov2 hubconf dinov2_vitl14_reg settings.
    m = DinoVisionTransformer(img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
                              mlp_ratio=4, block_fn=partial(Block, attn_class=MemEffAttention),
                              init_values=1.0, ffn_layer="mlp", block_chunks=0, num_register_tokens=4,
                              interpolate_antialias=True, interpolate_offset=0.0)
    if which == "official":
        sd = torch.load(DINO_CKPT, map_location="cpu")
    else:
        # lingbot trained its patch embed end to end (weights moved ~70-100% in
        # relative norm from DINOv2) and reads only its patch tokens; its CLS
        # output is never used.
        ck = torch.load(Path(__file__).resolve().parents[2] / "ckpt/lingbot-map.pt", map_location="cpu", weights_only=False)
        pre = "aggregator.patch_embed."
        sd = {k[len(pre):]: v for k, v in ck.get("model", ck).items() if k.startswith(pre)}
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def decode(clip: Path, c: ClipCache, frames: list[int]) -> np.ndarray:
    meta = json.loads((clip / "meta.json").read_text())
    video = Path(f"/data/ScanNetpp/data/{meta['scene']}/iphone/rgb.mkv")
    ids = [meta["frame_ids"][i] for i in frames]
    # build_cache.py's selector: frame_ids are a uniform stride, so one
    # between/mod expression picks them (a long eq() chain overflows ffmpeg).
    start, stride = ids[0], (ids[1] - ids[0]) if len(ids) > 1 else 1
    assert all(b - a == stride for a, b in zip(ids, ids[1:])), "frames must be uniformly strided"
    sel = rf"between(n\,{start}\,{ids[-1]})*not(mod(n-{start}\,{stride}))"
    w, h = c.meta.width, c.meta.height
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{sel}',scale={w}:{h}:flags=lanczos",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(len(ids), h, w, 3)


@torch.no_grad()
def features(model, rgb, device, bs=16):
    cls, mean = [], []
    m, s = torch.tensor(MEAN, device=device)[:, None, None], torch.tensor(STD, device=device)[:, None, None]
    for i in range(0, len(rgb), bs):
        x = torch.from_numpy(rgb[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255
        o = model.forward_features((x - m) / s)
        cls.append(o["x_norm_clstoken"].float()); mean.append(o["x_norm_patchtokens"].float().mean(1))
    return F.normalize(torch.cat(cls), dim=-1).cpu(), F.normalize(torch.cat(mean), dim=-1).cpu()


@torch.no_grad()
def overlap(c: ClipCache, frames, device, stride=8, tol=0.05):
    """o[i, j]: fraction of frame i's valid pixels (subsampled) that land inside
    frame j in front of it and agree with j's GT depth within tol."""
    d = torch.from_numpy(np.asarray(c.gt_depth[frames], np.float32)).to(device)
    K = torch.from_numpy(np.asarray(c.gt_intrinsics[frames], np.float32)).to(device)
    T = torch.from_numpy(np.asarray(c.gt_c2w[frames], np.float32)).to(device)
    n, H, W = d.shape
    v, u = torch.meshgrid(torch.arange(0, H, stride, device=device), torch.arange(0, W, stride, device=device), indexing="ij")
    u, v = u.reshape(-1).float(), v.reshape(-1).float()
    w2c = torch.linalg.inv(T)
    o = torch.zeros(n, n)
    for i in range(n):
        z = d[i, v.long(), u.long()]; ok = z > 0
        pc = torch.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * z, (v - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)[ok]
        pw = pc @ T[i, :3, :3].T + T[i, :3, 3]
        pj = torch.einsum("jab,pb->jpa", w2c[:, :3, :3], pw) + w2c[:, None, :3, 3]          # [n, P, 3]
        zj = pj[..., 2].clamp(min=1e-6)
        uj = K[:, None, 0, 0] * pj[..., 0] / zj + K[:, None, 0, 2]
        vj = K[:, None, 1, 1] * pj[..., 1] / zj + K[:, None, 1, 2]
        inb = (pj[..., 2] > 0) & (uj >= 0) & (uj <= W - 1) & (vj >= 0) & (vj <= H - 1)
        ui, vi = uj.round().long().clamp(0, W - 1), vj.round().long().clamp(0, H - 1)
        dj = d[torch.arange(n, device=device)[:, None], vi, ui]
        vis = inb & (dj > 0) & ((pj[..., 2] - dj).abs() < tol * dj)
        o[i] = vis.float().mean(1).cpu()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--other", type=Path, required=True, help="a clip from a different scene")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = ClipCache(args.clip); frames = list(range(len(c)))
    rgb = decode(args.clip, c, frames)
    co = ClipCache(args.other); of = list(range(0, len(co), 5))
    rgb_o = decode(args.other, co, of)
    F_in, F_out = {}, {}
    for which in ("official", "lingbot"):
        model = load_dino(dev, which)
        (ci, mi), (cx, mx) = features(model, rgb, dev), features(model, rgb_o, dev)
        F_in[f"{which}_cls"], F_in[f"{which}_meanpatch"] = ci, mi
        F_out[f"{which}_cls"], F_out[f"{which}_meanpatch"] = cx, mx
        del model
        torch.cuda.empty_cache()
    o = overlap(c, frames, dev); ov = 0.5 * (o + o.T)
    T = np.asarray(c.gt_c2w[frames], np.float64); s = float(c.meta.gt_scale)
    pos = T[:, :3, 3] * s
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    fwd = T[:, :3, 2]
    ang = np.degrees(np.arccos(np.clip(fwd @ fwd.T, -1, 1)))
    gap = np.abs(np.arange(len(frames))[:, None] - np.arange(len(frames))[None])

    S = {k: (v @ v.T).numpy() for k, v in F_in.items()}
    X = {k: (F_in[k] @ F_out[k].T).numpy() for k in F_in}
    iu = np.triu_indices(len(frames), 1)
    stats = {}
    for k in S:
        a = S[k][iu]; ovu = ov.numpy()[iu]
        stats[k] = dict(
            adjacent=float(np.mean(np.diag(S[k], 1))),
            overlap_gt_0p5=float(a[ovu > 0.5].mean()), overlap_0p1_0p5=float(a[(ovu > 0.1) & (ovu <= 0.5)].mean()),
            overlap_lt_0p02=float(a[ovu < 0.02].mean()), within_scene_min=float(a.min()),
            within_scene_p5=float(np.percentile(a, 5)), within_scene_mean=float(a.mean()),
            other_scene_mean=float(X[k].mean()), other_scene_max=float(X[k].max()),
            frac_within_below_other_max=float((a < X[k].max()).mean()))
    stats["pairs"] = dict(n=len(iu[0]), overlap_gt_0p5=int((ov.numpy()[iu] > 0.5).sum()),
                          overlap_lt_0p02=int((ov.numpy()[iu] < 0.02).sum()))
    stats["clip"], stats["other"] = args.clip.name, args.other.name
    stats["trajectory_extent_m"] = float(dist.max())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(stats, indent=1))

    keys = ["official_cls", "official_meanpatch", "lingbot_meanpatch", "lingbot_cls"]
    name = {"official_cls": "DINOv2 CLS", "official_meanpatch": "DINOv2 mean patch",
            "lingbot_meanpatch": "lingbot-DINO mean patch", "lingbot_cls": "lingbot-DINO CLS (unused by lingbot)"}
    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.7])
    for col, k in enumerate(["official_cls", "lingbot_meanpatch"]):
        ax = fig.add_subplot(gs[0, col]); im = ax.imshow(S[k], cmap="viridis"); fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{name[k]}: cosine, frame x frame"); ax.set_xlabel("frame"); ax.set_ylabel("frame")
    ax = fig.add_subplot(gs[0, 2]); im = ax.imshow(ov, cmap="magma", vmin=0, vmax=1); fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("view overlap (GT reprojection)")
    ax = fig.add_subplot(gs[0, 3])
    for k, colr in zip(keys[:3], ("C0", "C1", "C2")):
        ax.hist(S[k][iu], bins=60, alpha=0.4, density=True, color=colr, label=f"{name[k]} within")
        ax.hist(X[k].ravel(), bins=60, density=True, histtype="step", lw=2, color=colr, label=f"{name[k]} vs other scene")
    ax.set_xlabel("cosine"); ax.set_title("within-scene vs across-scene"); ax.legend(fontsize=7)
    sub = np.random.default_rng(0).choice(len(iu[0]), min(15000, len(iu[0])), replace=False)
    for col, k in enumerate(keys):
        ax = fig.add_subplot(gs[1, col])
        sc = ax.scatter(ov.numpy()[iu][sub], S[k][iu][sub], s=2, c=dist[iu][sub], cmap="coolwarm", alpha=0.5)
        ax.axhline(X[k].max(), color="k", ls="--", lw=1, label=f"other scene max {X[k].max():.2f}")
        ax.axhline(X[k].mean(), color="gray", ls=":", lw=1, label=f"other scene mean {X[k].mean():.2f}")
        ax.set_xlabel("view overlap"); ax.set_ylabel("cosine"); ax.set_title(f"{name[k]} vs overlap (colour = distance m)", fontsize=10)
        ax.legend(loc="lower right", fontsize=8); fig.colorbar(sc, ax=ax, fraction=0.046)
    for col, (k, x, lab) in enumerate((("official_cls", dist, "camera distance (m)"), ("official_cls", ang, "view angle diff (deg)"),
                                       ("lingbot_meanpatch", dist, "camera distance (m)"), ("lingbot_meanpatch", ang, "view angle diff (deg)"))):
        ax = fig.add_subplot(gs[2, col])
        ax.scatter(x[iu][sub], S[k][iu][sub], s=2, alpha=0.4)
        ax.axhline(X[k].max(), color="k", ls="--", lw=1)
        ax.set_xlabel(lab); ax.set_ylabel("cosine"); ax.set_title(f"{name[k]} vs {lab}", fontsize=10)
    pick = np.linspace(0, len(frames) - 1, 8).astype(int)
    g8 = gs[3, :].subgridspec(1, 8)
    for j, f in enumerate(pick):
        ax = fig.add_subplot(g8[0, j]); ax.imshow(rgb[f]); ax.axis("off")
        ax.set_title(f"f{f}  DINOv2 CLS cos\nto f{pick[0]}: {S['official_cls'][pick[0], f]:.2f}  to f{pick[-1]}: {S['official_cls'][pick[-1], f]:.2f}", fontsize=8)
    fig.suptitle(f"{args.clip.name}: global DINO features (official DINOv2 ViT-L/14-reg4 and lingbot's trained copy) over {len(frames)} frames "
                 f"(trajectory extent {dist.max():.1f} m); reference scene {args.other.name}", fontsize=14)
    fig.tight_layout(); fig.savefig(args.out.with_suffix(".png"), dpi=80); plt.close(fig)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
