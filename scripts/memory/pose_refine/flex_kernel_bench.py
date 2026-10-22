#!/usr/bin/env python3
"""Attention-kernel ceiling at inference_chunked's shapes: dense flash SDPA (no mask, upper bound) vs
FlexAttention with a frame-band mask (the chunked-streaming pattern) under several kernel_options.
Usage: flex_kernel_bench.py [--P 1302] [--chunk 64]
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


def bench(fn, n=5):
    fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", type=int, default=1302, help="tokens per frame")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--window", type=int, default=64)
    args = ap.parse_args()
    P, C, W, H, D = args.P, args.chunk, args.window, 16, 64
    dev = torch.device("cuda")
    nkf = 8 + W + C                       # scale + window + chunk frames as keys
    Q, K = C * P, nkf * P
    f0 = 8 + W
    q = torch.randn(1, H, Q, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, H, K, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, H, K, D, device=dev, dtype=torch.bfloat16)
    kf = torch.cat([torch.arange(8), torch.arange(f0 - W, f0), torch.arange(f0, f0 + C)]).to(dev).repeat_interleave(P)
    kspec = (torch.arange(P, device=dev) < 6).repeat(nkf)

    def mask_mod(b, h, qi, ki):
        qf = f0 + qi // P
        kfr = kf[ki]
        return (kfr < 8) | ((kfr >= qf - W + 1) & (kfr <= qf)) | (kspec[ki] & (kfr <= qf))

    bm = create_block_mask(mask_mod, None, None, Q, K, device=dev, _compile=True)
    dense_flops = 4 * Q * K * D * H
    print(f"{torch.cuda.get_device_name()} torch {torch.__version__}: Q {Q} K {K} heads {H} d {D}; "
          f"block-mask density {bm.sparsity():.1f}% sparse", flush=True)
    t = bench(lambda: F.scaled_dot_product_attention(q, k, v))
    print(f"[sdpa dense, no mask] {t * 1e3:.1f} ms, {dense_flops / t / 1e12:.0f} TFLOPS", flush=True)
    fx = torch.compile(flex_attention, dynamic=False)
    for opts in [None, {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
                 {"BLOCK_M": 128, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3},
                 {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 4, "num_stages": 3},
                 {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 8, "num_stages": 3}]:
        try:
            t = bench(lambda: fx(q, k, v, block_mask=bm, kernel_options=opts))
            print(f"[flex {opts}] {t * 1e3:.1f} ms, {dense_flops / t / 1e12:.0f} TFLOPS dense-equivalent", flush=True)
        except Exception as e:
            print(f"[flex {opts}] FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)


if __name__ == "__main__":
    main()
