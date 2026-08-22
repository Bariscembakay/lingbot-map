"""Render the context-token ablation's eval JSONs as comparison tables."""
import json
from pathlib import Path

W = Path("/group/compact-3dmem/campaigns/lingbot_map/context_token_ablation")
ARMS = [
    ("lingbot_map",              "FlashInfer", "camera+register+scale", 6),
    ("lingbot_map_traj6",        "SDPA",       "camera+register+scale", 6),
    ("lingbot_map_traj_nocam",   "SDPA",       "register+scale",        5),
    ("lingbot_map_traj_noscale", "SDPA",       "camera+register",       5),
    ("lingbot_map_traj_regonly", "SDPA",       "register",              4),
    ("lingbot_map_traj_noreg",   "SDPA",       "camera+scale",          2),
    ("lingbot_map_traj_camonly", "SDPA",       "camera",                1),
]
CONTROL = "lingbot_map_traj6"

def load(ds, name):
    p = W / ds / "eval" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None

def table(ds, fname, metrics, higher_better):
    data = load(ds, fname)
    if not data:
        print(f"  [{ds}/{fname}.json not present yet]\n")
        return
    w = max(len(a[0]) for a in ARMS)
    head = f"{'arm':{w}s} {'backend':10s} {'kept':22s} {'n':>2s} " + \
           " ".join(f"{m:>10s}" for m in metrics)
    print(head); print("-" * len(head))
    ctrl = data.get(CONTROL)
    for name, backend, kept, n in ARMS:
        r = data.get(name)
        if not r:
            print(f"{name:{w}s} {backend:10s} {kept:22s} {n:>2d}  (missing)")
            continue
        cells = " ".join(f"{r.get(m, float('nan')):>10.4f}" for m in metrics)
        print(f"{name:{w}s} {backend:10s} {kept:22s} {n:>2d} {cells}")
    if ctrl:
        print()
        print(f"{'delta vs SDPA control':{w}s} {'':10s} {'':22s} {'':2s} " +
              " ".join(f"{m:>10s}" for m in metrics))
        for name, backend, kept, n in ARMS:
            r = data.get(name)
            if not r or name == CONTROL:
                continue
            cells = []
            for m in metrics:
                if m not in r or m not in ctrl:
                    cells.append(f"{'-':>10s}"); continue
                d = r[m] - ctrl[m]
                sign = "+" if d >= 0 else ""
                cells.append(f"{sign}{d:>9.4f}")
            print(f"{name:{w}s} {backend:10s} {kept:22s} {n:>2d} " + " ".join(cells))
    print(f"\n  ({'higher is better' if higher_better else 'see per-metric direction'})\n")

print("=" * 100)
print("OXFORD SPIRES (sparse, stride 12; 10 scenes x 320 frames) — pose")
print("=" * 100)
table("oxford", "auc_macro", ["AUC_03", "AUC_05", "AUC_15", "AUC_30"], True)
table("oxford", "auc_micro", ["AUC_03", "AUC_05", "AUC_15", "AUC_30"], True)
table("oxford", "traj", ["ate", "rpe_trans", "rpe_rot"], False)

print("=" * 100)
print("NEURAL RGB-D (stride 5; 9 scenes) — reconstruction")
print("=" * 100)
table("neural_rgbd", "points",
      ["accuracy", "completeness", "chamfer", "precision", "recall", "f1"], False)
