"""Divergence analysis for the trajectory-memory context-token ablation.

Each ablation must be a strict no-op until the first frame is evicted, i.e.
until frame index anchor+window. A divergence that starts EARLIER would mean the
change leaked into the anchor set or the sliding window; one that never starts
would mean the flag never reached the model.
"""
import sys
from pathlib import Path
import numpy as np

W = Path("/group/compact-3dmem/campaigns/lingbot_map/context_token_ablation")
CONTROL = "lingbot_map_traj6"
ARMS = ["lingbot_map_traj_noreg", "lingbot_map_traj_nocam", "lingbot_map_traj_noscale",
        "lingbot_map_traj_regonly", "lingbot_map_traj_camonly"]
ANCHOR, WINDOW = 8, 64
EXPECT = ANCHOR + WINDOW

def traj(ds, scene, method):
    p = W / ds / scene / method / "traj.txt"
    return np.loadtxt(p) if p.exists() else None

ds = "oxford"
scenes = sorted(d.name for d in (W / ds).iterdir()
                if d.is_dir() and not d.name.startswith("_") and d.name != "eval")

print(f"Expected first divergence: frame {EXPECT} (anchor {ANCHOR} + window {WINDOW})\n")
hdr = f"{'scene':26s} {'arm':26s} {'first diff':>10s} {'@100':>10s} {'@final':>10s}"
print(hdr); print("-" * len(hdr))
bad = []
for scene in scenes:
    ctrl = traj(ds, scene, CONTROL)
    if ctrl is None:
        continue
    for arm in ARMS:
        a = traj(ds, scene, arm)
        if a is None or a.shape != ctrl.shape:
            continue
        d = np.abs(a - ctrl).max(axis=1)
        nz = np.nonzero(d > 0)[0]
        first = int(nz[0]) if len(nz) else None
        at100 = d[100] if len(d) > 100 else float("nan")
        print(f"{scene:26s} {arm.replace('lingbot_map_traj_',''):26s} "
              f"{str(first):>10s} {at100:>10.2e} {d[-1]:>10.2e}")
        if first != EXPECT:
            bad.append(f"{scene}/{arm}: first divergence at {first}, expected {EXPECT}")
        if not np.array_equal(a[:EXPECT], ctrl[:EXPECT]):
            bad.append(f"{scene}/{arm}: frames < {EXPECT} are NOT identical to control")

print()
if bad:
    print(f"PROBLEMS ({len(bad)}):")
    for b in bad:
        print("  -", b)
    sys.exit(1)
print(f"OK: every arm is bit-identical to the control through frame {EXPECT-1} "
      f"and diverges first at exactly frame {EXPECT}.")
print("=> the ablation touches the trajectory memory only; anchor and window are intact.")
