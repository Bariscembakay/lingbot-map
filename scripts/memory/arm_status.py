#!/usr/bin/env python3
"""Report progress and measured throughput for each training arm.

Reads `step` out of each arm's checkpoint and pairs it with the job's elapsed
time, which gives a measured seconds-per-update instead of an estimate.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

import torch


def elapsed_by_name() -> dict:
    out = {}
    try:
        r = subprocess.run(["sacct", "-X", "-n", "--format=JobName%30,Elapsed,State%20",
                            "-S", "today"], capture_output=True, text=True).stdout
        for line in r.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[0].startswith("mem_"):
                h, m, s = (parts[1].split(":") + ["0", "0"])[:3]
                out[parts[0][4:]] = (int(h) * 3600 + int(m) * 60 + int(s), parts[2])
    except FileNotFoundError:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root",
                    default="/group/compact-3dmem/campaigns/summary-memory/dev_v1")
    ap.add_argument("--target", type=int, default=20000)
    a = ap.parse_args()

    el = elapsed_by_name()
    root = Path(a.out_root)
    rows = sorted(p for p in root.iterdir() if (p / "last.pt").exists()) if root.exists() else []
    if not rows:
        raise SystemExit(f"no checkpoints under {root}")

    print(f"{'arm':18s} {'step':>7s} {'s/upd':>7s} {'elapsed':>9s} {'proj. total':>12s} "
          f"{'ckpt age':>9s}  config")
    for d in rows:
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=False)
        g = ck["args"]
        secs, state = el.get(d.name, (None, "?"))
        age = (time.time() - os.path.getmtime(d / "last.pt")) / 60
        rate = secs / ck["step"] if secs and ck["step"] else None
        proj = f"{rate * a.target / 3600:.1f} h" if rate else "-"
        print(f"{d.name:18s} {ck['step']:7d} {rate if rate else 0:7.2f} "
              f"{secs / 3600 if secs else 0:8.2f}h {proj:>12s} {age:7.0f}m  "
              f"taps={g['refine_taps']} slots={g['num_slots']} "
              f"camera={g['with_camera']} frozen={g['frozen_state']} [{state}]")


if __name__ == "__main__":
    main()
