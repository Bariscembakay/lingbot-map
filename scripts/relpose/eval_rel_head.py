#!/usr/bin/env python3
"""Evaluate a train_rel_head.py checkpoint (and lingbot's own poses) on a directory of benchmark
inputs (build_bench_inputs.py output: <inputs>/specials/*.npz), at full length. Per clip and mean:
ATE (Sim3, metres), consecutive-frame RPE, AUC.
Usage: eval_rel_head.py --run DIR --inputs DIR [--ckpt best.pt] [--out FILE.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rel_head as rh  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    dev = torch.device("cuda")
    ck = torch.load(args.run / args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    rope = a.mode == "rel" and not a.no_rope
    head = rh.tp.build_head(rh.tp.REPO / "ckpt/lingbot-map.pt", rope_on=rope, max_frames=4096 if rope else 1024)
    model = rh.RelHead(head, a.mode, a.window, a.prev_token, residual=getattr(a, "residual", False),
                       ref_every=getattr(a, "ref_every", 1))
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    rows = {}
    for f in sorted((args.inputs / "specials").glob("*.npz")):
        c = rh.EvalClip.from_npz(f, dev)
        m = rh.tp.traj_metrics(rh.predict_c2w(model, c), c.gt_c2w, c.scale)
        l = rh.tp.traj_metrics(rh.tp.pose_enc_to_c2w(c.lingbot.float(), c.hw), c.gt_c2w, c.scale)
        rows[c.name] = {"model": m, "lingbot": l}
        print(f"{c.name:>26} {c.T:>6} frames | ATE model {m['ate']:8.3f} lingbot {l['ate']:8.3f} "
              f"ratio {m['ate'] / l['ate']:.3f} | rpe_r {m['rpe_r']:.3f} / {l['rpe_r']:.3f}", flush=True)
    mean = {who: {k: float(np.mean([r[who][k] for r in rows.values()])) for k in rh.tp.METRICS}
            for who in ("model", "lingbot")}
    print(f"{'mean':>26} {'':>6}        | ATE model {mean['model']['ate']:8.3f} lingbot {mean['lingbot']['ate']:8.3f} "
          f"ratio {mean['model']['ate'] / mean['lingbot']['ate']:.3f}", flush=True)
    if args.out:
        args.out.write_text(json.dumps({"run": str(args.run), "ckpt": args.ckpt, "inputs": str(args.inputs),
                                        "step": ck["step"], "per_clip": rows, "mean": mean}, indent=1))


if __name__ == "__main__":
    main()
