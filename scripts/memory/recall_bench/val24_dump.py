#!/usr/bin/env python3
"""val24 recall comparison, GPU phase: per val clip, stream the SAME 96 frames
our validation streams (cache indices 0, 2, ..., 190 -- train_state's Clip at
subsample 2, window start 0), then query the final state at the SAME lag-ladder
cameras evaluate() uses (lags 0, 1, 2, 4, T//2, T), images withheld.

  --method ours   a train_state checkpoint; predictions on the cache grid in
                  canonical units, exactly as evaluate() computes valm_self
  --method cut3r | ttt3r
                  the official 512 model from the vendored TTT3R tree (it
                  carries both update rules), CUT3R's own preprocessing; GT
                  raymaps in metres (canonical x gt_scale), predictions saved
                  in metres on CUT3R's own grid with its intrinsics

Writes <dump>/<label>/<clip>.npz (pred_self, K_pred, hw, qs, unit); the CPU
scorer (val24_score.py) resamples and scores every method identically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.home() / "lingbot-map"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/memory"))
sys.path.insert(0, str(REPO / ".agents/scratch/memory_eval"))


def lag_queries(n):
    """evaluate()'s lag ladder over a stream of n frames."""
    T = n - 1
    lags = sorted({0, 1, min(2, T), min(4, T), T // 2, T})
    return [T - lag for lag in lags]


@torch.no_grad()
def dump_ours(args, clips):
    from train_state import Clip, run_probe
    from lingbot_map.memory.cut3r_state import StateMemory
    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = None
    for cp in clips:
        c = Clip(Path(cp), int(a.get("subsample", 2)), dev, int(a.get("max_frames", 160)),
                 a.get("taps", "23"), store="cpu")
        if model is None:
            model = StateMemory(patch_size=14, tap_dim=c.taps.shape[-1],
                                state_tokens=int(a.get("state_tokens", 768)),
                                dec_depth=int(a["dec_depth"]), read_depth=int(a.get("read_depth", 2)),
                                write_oneway=bool(a.get("write_oneway", False)),
                                head_type=a.get("head", "dpt"), grad_ckpt=False)
            if any(".base.weight" in k for k in ck["model"]):
                from lingbot_map.memory.lora import inject_conv_lora
                inject_conv_lora(model.head.dpt, int(a.get("lora_rank", 16)), float(a.get("lora_alpha", 16.0)))
            if a.get("fresh_head"):
                model.head.fresh_head(a.get("pts_frame"))
            elif a.get("pts_frame") or a.get("fresh_last_conv"):
                model.head.reset_output(a.get("pts_frame"))
            if any(k.startswith("sem.") for k in ck["model"]):
                from lingbot_map.memory.semantic import VoxelQuery
                model.sem = VoxelQuery()
            miss, unexp = model.load_state_dict(ck["model"], strict=False)
            assert not miss and not unexp, (miss[:3], unexp[:3])
            model.to(dev).eval()
            print(f"[ckpt] {args.ckpt} step {ck.get('step')}", flush=True)
        n = min(args.frames, len(c))
        state, spos = model.init_state(1, dev)
        for t in range(n):
            state = model.write(state, spos, c.tap(t), c.patch_hw)
        qs = lag_queries(n)
        rm, xs, xw, valid = c.probe_inputs(qs, a.get("raymap_convention", "cut3r"), anchor=0)
        rays = c.probe_rays(qs, anchor=0, unit=a["head"] not in ("lingbot", "smallread_lingbot"))
        out = run_probe(model, state.expand(len(qs), -1, -1), spos.expand(len(qs), -1, -1),
                        rm, (c.h, c.w), rays)
        pred = out["pts3d_in_self_view"].float().cpu().numpy()
        l21 = float((out["pts3d_in_self_view"] - xs).norm(dim=-1)[valid].mean())
        save(args, c.name, pred, c.K[qs].cpu().numpy(), qs, "canonical")
        print(f"{c.name}: valm_self-equivalent {l21:.4f}", flush=True)


@torch.no_grad()
def dump_cut3r_family(args, clips):
    import json
    tree = REPO / "TTT3R"
    sys.path.insert(0, str(tree))
    from add_ckpt_path import add_path_to_dust3r
    ckpt = str(tree / "src/cut3r_512_dpt_4_64.pth")
    add_path_to_dust3r(ckpt)
    from src.dust3r.inference import inference, inference_step
    from src.dust3r.model import ARCroco3DStereo
    from run_cut3r_recall import decode_frames, get_ray_map, preprocess_like_demo, video_resolution
    model = ARCroco3DStereo.from_pretrained(ckpt).to("cuda")
    model.config.model_update_type = args.method
    model.eval()
    for cp in clips:
        cp = Path(cp)
        meta = json.loads((cp / "meta.json").read_text())
        s = float(meta["gt_scale"])
        cache_idx = list(range(0, 2 * args.frames, 2))
        c2w = np.load(cp / "gt_c2w.npy")[cache_idx].astype(np.float64)
        c2w_m = c2w.copy(); c2w_m[:, :3, 3] *= s
        K_cache = np.load(cp / "gt_intrinsics.npy")[cache_idx].astype(np.float64)
        video = Path(f"/data/ScanNetpp/data/{meta['scene']}/iphone/rgb.mkv")
        Wn, Hn = video_resolution(video)
        frames = decode_frames(video, [i * meta["stride"] for i in cache_idx], Wn, Hn)
        K_nat = K_cache.copy(); K_nat[:, 0, :] *= Wn / meta["width"]; K_nat[:, 1, :] *= Hn / meta["height"]
        imgs, K_fin, H, W = preprocess_like_demo(frames, K_nat)
        views = [{"img": imgs[i:i + 1], "ray_map": torch.full((1, 6, H, W), torch.nan),
                  "true_shape": torch.from_numpy(np.int32([[H, W]])), "idx": i, "instance": str(i),
                  "camera_pose": torch.eye(4)[None], "img_mask": torch.tensor([True]),
                  "ray_mask": torch.tensor([False]), "update": torch.tensor([True]),
                  "reset": torch.tensor([False])} for i in range(len(imgs))]
        _, state_args = inference(views, model, "cuda")
        final = state_args[-1]
        qs = lag_queries(len(imgs))
        preds = []
        for q in qs:
            rmap = get_ray_map(c2w_m[0], c2w_m[q], K_fin[q], H, W)
            view = {"img": torch.full((1, 3, H, W), torch.nan),
                    "ray_map": torch.from_numpy(rmap)[None].float(),
                    "true_shape": torch.from_numpy(np.int32([[H, W]])), "idx": len(imgs) + q,
                    "instance": str(len(imgs) + q), "camera_pose": torch.eye(4)[None],
                    "img_mask": torch.tensor([False]), "ray_mask": torch.tensor([True]),
                    "update": torch.tensor([False]), "reset": torch.tensor([False])}
            preds.append(inference_step(view, final, model, "cuda")["pred"]["pts3d_in_self_view"].numpy())
        save(args, cp.name, np.concatenate(preds, 0), K_fin[qs], qs, "metres", gt_scale=s)
        print(f"{cp.name}: {args.method} done ({H}x{W})", flush=True)


def save(args, name, pred, K, qs, unit, gt_scale=1.0):
    d = args.dump / args.label
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"{name}.npz", pred_self=pred.astype(np.float32), K_pred=np.asarray(K, np.float32),
                        hw=np.array(pred.shape[1:3]), qs=np.array(qs), unit=unit, gt_scale=gt_scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ours", "cut3r", "ttt3r"], required=True)
    ap.add_argument("--label", required=True, help="dump subdir / table row name")
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--val-list", type=Path, default=REPO / ".agents/scratch/memory_eval/val24.txt")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--dump", type=Path, required=True)
    args = ap.parse_args()
    clips = args.val_list.read_text().split()
    (dump_ours if args.method == "ours" else dump_cut3r_family)(args, clips)


if __name__ == "__main__":
    main()
