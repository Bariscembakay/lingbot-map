#!/usr/bin/env python3
"""Relative-pose error against frame gap, for a train_rel_head.py checkpoint and for lingbot.

Error growing like sqrt(gap) is random per-step noise; growing like the gap is a per-step bias,
which the mean signed per-step rotation error (axis-angle of R_pred^T R_gt, in the camera frame,
averaged over all steps) then shows as clearly non-zero.
Usage: gap_error.py --run DIR [--sets val24,nrgbd,longval] [--device cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rel_head as rh  # noqa: E402

GAPS = (1, 4, 16, 64, 256)


def log_rot(R):
    """[n,3,3] -> [n,3] axis-angle (rad)."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.where(th > 1e-6, th / (2 * np.sin(np.maximum(th, 1e-6))), 0.5)
    return v * s[:, None]


def gap_stats(P, G):
    out = {}
    for k in GAPS:
        if len(P) <= k:
            continue
        rp, rg = np.linalg.inv(P[:-k]) @ P[k:], np.linalg.inv(G[:-k]) @ G[k:]
        E = np.einsum("nji,njk->nik", rp[:, :3, :3], rg[:, :3, :3])
        out[k] = (np.degrees(np.linalg.norm(log_rot(E), axis=1)).mean(),
                  np.linalg.norm(rp[:, :3, 3] - rg[:, :3, 3], axis=1).mean())
    sp, sg = np.linalg.inv(P[:-1]) @ P[1:], np.linalg.inv(G[:-1]) @ G[1:]
    bias = np.degrees(log_rot(np.einsum("nji,njk->nik", sp[:, :3, :3], sg[:, :3, :3])).mean(0))
    return out, bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    ck = torch.load(args.run / args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    rope = a.mode == "rel" and not a.no_rope
    head = rh.tp.build_head(rh.tp.REPO / "ckpt/lingbot-map.pt", rope_on=rope, max_frames=4096 if rope else 1024)
    model = rh.RelHead(head, a.mode, a.window, a.prev_token, residual=getattr(a, "residual", False))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    clips = [rh.EvalClip.from_npz(f, dev) for d in ("val24", "nrgbd")
             for f in sorted((rh.INPUTS / d / "specials").glob("*.npz")) if ".tmp" not in f.stem]
    res = {"model": {}, "lingbot": {}}
    for who in res:
        rows, biases = [], []
        for c in clips:
            G = c.gt_c2w.double().cpu().numpy()
            if who == "model":
                P = rh.predict_c2w(model, c).double().cpu().numpy()
            else:
                P = rh.tp.pose_enc_to_c2w(c.lingbot.float(), c.hw).double().cpu().numpy()
            s, _ = rh.tp.umeyama(P[:, :3, 3], G[:, :3, 3])[:2]
            P[:, :3, 3] *= s
            g, b = gap_stats(P, G)
            rows.append(g)
            biases.append(b)
        res[who] = {"gap": {k: [float(np.mean([r[k][0] for r in rows if k in r])),
                                float(np.mean([r[k][1] for r in rows if k in r]))] for k in GAPS},
                    "mean_signed_step_rot_err_deg_xyz": np.mean(biases, 0).tolist(),
                    "per_clip_bias_deg": [b.tolist() for b in biases]}
        print(f"{who:>8}: " + "  ".join(f"k={k} {v[0]:.2f}deg/{v[1]:.3f}" for k, v in res[who]["gap"].items())
              + f"  | mean signed step rot err (x,y,z) deg: {np.round(np.mean(biases, 0), 4)}", flush=True)
    (args.run / f"gap_error_{Path(args.ckpt).stem}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
