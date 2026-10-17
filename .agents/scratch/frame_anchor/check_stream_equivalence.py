#!/usr/bin/env python
"""Validates cut3r_anchor_dump.run_stream against upstream's batched
`model(views)` (`_forward_impl`, what demo.py and eval/ call) on a short clip.

Batched and per-frame execution pick different kernels, so they are only
expected to agree to fp32 noise -- which is only meaningful with TF32 off,
hence --strict-fp32. Reports per-frame median and 99th-percentile relative
differences, plus a streamed-vs-streamed rerun as the noise floor.
Passes if, in strict fp32, every output agrees to a median relative
difference < 1e-3 and a 99th percentile < 1e-2 (conf and points go through
expm1, which amplifies reordering noise in the tails)."""
import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cut3r_anchor_dump as D  # noqa: E402

KEYS = ("pts3d_in_self_view", "pts3d_in_other_view", "camera_pose", "conf", "conf_self")


def rel(a, b):
    d = ((a - b).abs() / b.abs().clamp(min=1e-6)).flatten().double()
    return float(d.median()), float(torch.quantile(d[:1_000_000], 0.99))


def compare(name, A, B):
    worst_med = worst = 0.0
    print(f"--- {name}")
    for k in KEYS:
        per = [rel(a[k], b[k]) for a, b in zip(A, B)]
        med = [p[0] for p in per]
        p99 = [p[1] for p in per]
        worst = max(worst, max(p99))
        worst_med = max(worst_med, max(med))
        print(f"{k:22s} median rel: frame0 {med[0]:.1e} max {max(med):.1e} | "
              f"p99 rel: frame0 {p99[0]:.1e} max {max(p99):.1e}")
    return worst_med, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/NRGBD")
    ap.add_argument("--scene", default="kitchen")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--strict-fp32", type=int, default=1)
    args = ap.parse_args()
    if args.strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(f"tf32 matmul={torch.backends.cuda.matmul.allow_tf32} cudnn={torch.backends.cudnn.allow_tf32}")

    frames = D.nrgbd_frames(args.root, args.scene)[: args.n * args.step : args.step]
    order = list(range(len(frames)))
    ds = D.StreamDataset(frames, order)
    model = D.ARCroco3DStereo.from_pretrained(D.CKPT).to("cuda").eval()

    def streamed():
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4)
        return [{k: v.clone() for k, v in res.items()} for _, res in D.run_stream(model, loader, "cuda")]

    S1, S2 = streamed(), streamed()

    views = []
    for i in order:
        it = ds[i]
        img = it["img"][None].cuda()
        views.append(dict(
            img=img, true_shape=it["true_shape"][None].cuda(),
            ray_map=torch.full((1, 6, *img.shape[-2:]), torch.nan, device="cuda"),
            idx=i, instance=str(i),
            camera_pose=torch.eye(4, device="cuda")[None],
            img_mask=torch.tensor([True], device="cuda"),
            ray_mask=torch.tensor([False], device="cuda"),
            update=torch.tensor([True], device="cuda"),
            reset=torch.tensor([False], device="cuda")))
    with torch.no_grad():
        B = model(views).ress

    compare("streamed vs streamed (noise floor)", S1, S2)
    worst_med, worst = compare("streamed vs batched model(views)", S1, B)
    ok = bool(args.strict_fp32) and worst_med < 1e-3 and worst < 1e-2
    print("PASS" if ok else "FAIL", f"worst median={worst_med:.2e} worst p99={worst:.2e}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
