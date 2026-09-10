#!/usr/bin/env python3
"""Aggregate nrgbd_recall_s2 cells into the benchmark table.

Reports per (method, tier) means over the FIXED 7-scene set and refuses to
average a tier that is missing any scene -- a tier's number must always mean
the same scene set (campaign rule: never report a metric with scenes silently
excluded). Also emits the recall-vs-lag curve data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SCENES = ["whiteroom", "kitchen", "grey_white_room", "green_room",
          "complete_kitchen", "breakfast_room", "staircase"]
TIERS = [100, 300, 500]
METHODS = ["cut3r", "ttt3r", "zipmap", "ours"]
NOTE = {"zipmap": " [B]", "ours": " [P]"}   # [B] bidirectional, [P] posed
FOOT = ("[B] ZipMap: its released state-query checkpoint is a stage-2 "
        "(bidirectional) fine-tune; no streaming variant exists, so its "
        "ingestion sees all frames jointly -- an easier setting, i.e. an "
        "upper bound on a streaming ZipMap.")
FOOT += ("\n[P] Ours is a POSED system: it queries GT raymaps natively and its "
         "canonical scale is an input-side constant, so its cloud needs no "
         "Sim(3) fit (align.fitted=false) where the baselines get one. It is "
         "therefore a separate column, not a like-for-like row.")


def cell(root: Path, method: str, scene: str, tier: int, mode: str):
    p = root / method / f"{scene}_n{tier}" / "metrics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get(mode) or d.get("selfpose")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(
        "/group/compact-3dmem/campaigns/spatial_memory/recallbench"))
    ap.add_argument("--mode", default="selfpose", choices=["selfpose", "gtpose"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = [f"# nrgbd_recall_s2 results ({args.mode} queries)", ""]
    lines.append("| method | tier | Acc | Comp | Chamfer | AbsRel | d<1.25 |")
    lines.append("|---|---|---|---|---|---|---|")
    lag_rows = []
    for m in METHODS:
        for t in TIERS:
            cells = {s: cell(args.root, m, s, t, args.mode) for s in SCENES}
            have = [s for s, c in cells.items() if c]
            if len(have) < len(SCENES):
                missing = ",".join(s for s in SCENES if s not in have)
                lines.append(f"| {m}{NOTE.get(m,'')} | n{t} | "
                             f"INCOMPLETE {len(have)}/7 (missing {missing}) "
                             f"| | | | |")
                continue
            def mean(k):
                return sum(cells[s][k] for s in SCENES) / len(SCENES)
            lines.append(
                f"| {m}{NOTE.get(m,'')} | n{t} | {mean('acc_mean'):.4f} | "
                f"{mean('comp_mean'):.4f} | {mean('chamfer'):.4f} | "
                f"{mean('depth_absrel'):.4f} | {mean('depth_d125'):.4f} |")
            # lag curve: bucket per-query accuracy by age decile of the tier
            buckets = {}
            for s in SCENES:
                for e in cells[s]["lag"]:
                    b = int(10 * e["age"] / max(1, t))
                    buckets.setdefault(b, []).append(e["acc_mean"])
            lag_rows.append((m, t, {b: sum(v) / len(v)
                                    for b, v in sorted(buckets.items())}))
    lines += ["", FOOT, "", "## Recall vs frame age (Acc by age decile, 0=newest)", ""]
    lines.append("| method | tier | " + " | ".join(f"d{b}" for b in range(11)) + " |")
    lines.append("|---" * 13 + "|")
    for m, t, b in lag_rows:
        lines.append(f"| {m} | n{t} | " +
                     " | ".join(f"{b[i]:.3f}" if i in b else "-"
                               for i in range(11)) + " |")
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        args.out.write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
