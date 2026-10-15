#!/usr/bin/env python3
"""One table over pose-refiner arms: best step, and per eval set the final-mode ATE / AUC@3 of
best.pt against lingbot (from final_best.json when the post-training eval ran, else from
best_eval.json at the selection step).
Usage: summarize_runs.py <arm dir glob>... [--md out.md]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

SETS = ("val24", "kitti360_val", "nrgbd", "kitti", "vbr")
UNIT = {"val24": 100, "nrgbd": 100}   # cm indoors, m driving


def per_set(per_clip):
    """per_clip {name@t: metrics} -> {set: (ate, lingbot_ate, auc3, lingbot_auc3)} by clip-name prefix."""
    out = {}
    for s, rows in per_clip.items():
        rs = list(rows.values())
        out[s] = tuple(float(np.mean([r[k] if k != "l" else r["lingbot"][m] for r in rs]))
                       for k, m in (("ate", None), ("l", "ate"), ("auc3", None), ("l", "auc3")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    dirs = sorted({Path(p) for a in args.arms for p in glob.glob(a)})
    head = "| arm | steps run | best step | " + " | ".join(f"{s} ATE (lingbot) | {s} AUC@3 (lingbot)" for s in SETS) + " |"
    lines = [head, "|" + "---|" * (3 + 2 * len(SETS))]
    for d in dirs:
        hist = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
        fb = d / "final_best.json"
        src = fb if fb.exists() else d / "best_eval.json"
        if not src.exists():
            lines.append(f"| {d.name} | {hist[-1]['step'] if hist else 0} | – |" + " – | – |" * len(SETS))
            continue
        js = json.loads(src.read_text())
        pc = js["per_clip"] if "per_clip" in js else js
        ps = per_set(pc)
        best = json.loads((d / "best.pt.json").read_text())["step"] if (d / "best.pt.json").exists() else None
        if best is None and hist:
            # best.pt is the row whose selection score is lowest; recompute from history
            sel = [s for s in ("val24", "kitti360_val") if f"{s}/ate" in hist[0]]
            best = min(hist[1:] or hist, key=lambda r: np.mean([r[f"{s}/ate"] / r[f"{s}/lingbot_ate"] for s in sel]))["step"]
        cells = []
        for s in SETS:
            if s not in ps:
                cells += ["–", "–"]
                continue
            a, la, u, lu = ps[s]
            k = UNIT.get(s, 1)
            cells += [f"{a*k:.2f} ({la*k:.2f})", f"{u:.1f} ({lu:.1f})"]
        tag = "" if fb.exists() else " *"
        lines.append(f"| {d.name}{tag} | {hist[-1]['step'] if hist else 0} | {best} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines) + "\n\nATE: cm for val24/nrgbd, m for driving. * = no post-training eval yet (selection-step numbers).\n"
    print(txt)
    if args.md:
        args.md.write_text(txt)


if __name__ == "__main__":
    main()
