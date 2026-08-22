#!/usr/bin/env python3
"""Extract the frozen heads from the full checkpoint.

Training needs the DPT head (to decode refined tokens into depth) and the camera
head (for Loss 1's pose terms, driven through `memory/camera_bridge.py`). Shipping
those instead of the 4.63 GB checkpoint also makes the frozen surface explicit:
anything not in this file cannot silently be trained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

PREFIXES = ("depth_head.", "camera_head.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/lingbot-map.pt")
    ap.add_argument("--out", default="ckpt/frozen_heads.pt")
    args = ap.parse_args()

    src = Path(args.ckpt)
    raw = torch.load(src, map_location="cpu", weights_only=False)
    state = raw.get("model", raw)

    heads = {}
    for prefix in PREFIXES:
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if not sub:
            raise SystemExit(f"no keys with prefix {prefix!r} in {src}")
        heads[prefix.rstrip(".")] = sub

    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    payload = {
        **heads,
        "source_checkpoint": str(src),
        "source_sha256": src_sha,
        "num_keys": {k: len(v) for k, v in heads.items()},
    }
    out = Path(args.out)
    torch.save(payload, out)

    print(json.dumps({
        "out": str(out),
        "keys": {k: len(v) for k, v in heads.items()},
        "params": {k: sum(t.numel() for t in v.values()) for k, v in heads.items()},
        "size_mb": round(out.stat().st_size / 1e6, 1),
        "source_sha256": src_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
