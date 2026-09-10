#!/usr/bin/env bash
# Prove the gt_for() caching is numerically identical: re-score an already
# finished cell into /scratch and diff every metric against the stored one.
set -uo pipefail
R=/group/compact-3dmem/campaigns/spatial_memory/recallbench
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
OUT=/scratch/$USER/regress_gtcache
rm -rf "$OUT"
"$P" scripts/memory/recall_bench/run_cut3r_family.py --update-rule cut3r \
    --scene breakfast_room --tiers 100 --out "$OUT" || exit 1
"$P" - "$R/cut3r/breakfast_room_n100/metrics.json" \
      "$OUT/cut3r/breakfast_room_n100/metrics.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
bad = []
for mode in ("selfpose", "gtpose"):
    for k, va in a[mode].items():
        if k in ("lag", "align"):
            continue
        vb = b[mode][k]
        if isinstance(va, float) and abs(va - vb) > 1e-9:
            bad.append(f"{mode}.{k}: stored {va!r} != new {vb!r}")
la, lb = a["selfpose"]["lag"], b["selfpose"]["lag"]
if len(la) != len(lb):
    bad.append(f"lag length {len(la)} != {len(lb)}")
else:
    d = max(abs(x["acc_mean"] - y["acc_mean"]) for x, y in zip(la, lb))
    if d > 1e-9:
        bad.append(f"lag max delta {d:.3e}")
print("REGRESSION FAIL:\n  " + "\n  ".join(bad) if bad
      else "REGRESSION PASS: every metric identical to <=1e-9")
PY
