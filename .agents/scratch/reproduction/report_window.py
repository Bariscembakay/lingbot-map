"""Does the register effect grow as the trajectory memory's share of context grows?

Each block is a self-contained protocol (its own anchor/window), so deltas are
computed within a block only. Comparing absolute numbers ACROSS blocks is
meaningless -- a smaller resident store is a worse model regardless of the memory.
"""
import json
from pathlib import Path

W = Path("/group/compact-3dmem/campaigns/lingbot_map/context_token_ablation/oxford/eval")
BLOCKS = [
    ("anchor 8 / window 64", "2.6%",  ""),
    ("anchor 8 / window  8", "12.7%", "_w8"),
    ("anchor 2 / window  4", "28.6%", "_w4"),
]
COMPOS = [("camera+register+scale", 6, "lingbot_map_traj6"),
          ("register",             4, "lingbot_map_traj_regonly"),
          ("camera+scale",         2, "lingbot_map_traj_noreg")]
auc = json.loads((W / "auc_macro.json").read_text())
traj = json.loads((W / "traj.json").read_text())

print(f"{'resident store':22s} {'mem share':>9s}  {'kept':22s} {'n':>2s} "
      f"{'AUC@15':>8s} {'dAUC':>8s} {'ATE':>8s} {'dATE':>8s}")
print("-" * 92)
summary = []
for label, share, suf in BLOCKS:
    ctrl_a = auc.get(f"lingbot_map_traj6{suf}", {}).get("AUC_15")
    ctrl_t = traj.get(f"lingbot_map_traj6{suf}", {}).get("ate")
    for kept, n, base in COMPOS:
        k = base + suf
        a = auc.get(k, {}).get("AUC_15"); t = traj.get(k, {}).get("ate")
        if a is None:
            print(f"{label:22s} {share:>9s}  {kept:22s} {n:>2d}   (pending)"); continue
        da = a - ctrl_a if ctrl_a is not None else float("nan")
        dt = t - ctrl_t if ctrl_t is not None else float("nan")
        mark = "" if base.endswith("traj6") else f"{da:+8.2f} "
        print(f"{label:22s} {share:>9s}  {kept:22s} {n:>2d} {a:8.2f} "
              f"{'  control' if base.endswith('traj6') else f'{da:+8.2f}'} {t:8.3f} "
              f"{'  control' if base.endswith('traj6') else f'{dt:+8.3f}'}")
        if base.endswith("noreg"):
            summary.append((label, share, da, dt))
    print()

print("noreg penalty (drop the 4 registers) as the memory's share of context grows:")
for label, share, da, dt in summary:
    print(f"  {label:22s} memory {share:>6s} of context -> AUC@15 {da:+6.2f}, ATE {dt:+6.3f}")
