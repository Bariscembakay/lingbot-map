#!/usr/bin/env bash
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail
PY="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
CAMP=/group/compact-3dmem/campaigns/spatial_memory
"$PY" .agents/scratch/memory_eval/run_ours_recall.py \
  --ckpt "$CAMP/scenes96_96f_b4_write4_read2_lingbothead/last.pt" \
  --clip /data/lingbot-tapcache-v4-40/val_top/2ab7bea148_c0 --frames 96 \
  --out /scratch/$USER/recall96/scenes96best_2ab7bea148.npz
"$PY" - <<'PYEOF'
import sys, numpy as np
sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import emit
d = np.load("/scratch/baris_bakay/recall96/scenes96best_2ab7bea148.npz")
import pathlib
OUT = pathlib.Path("/group/compact-3dmem/campaigns/spatial_memory/"
                   "scenes96_96f_b4_write4_read2_lingbothead/recall96_allqueries_bestckpt")
p, g = [], []
for i in range(len(d["pred_pts_world"])):
    v = d["valid"][i]
    p.append(d["pred_pts_world"][i][v][::4]); g.append(d["gt_pts_world"][i][v][::4])
P, G = np.concatenate(p), np.concatenate(g)
emit(OUT, "val_2ab7bea148", P, G, np.ones(len(P), bool), np.ones(len(P), np.float32), 0.0)
print("rendered ->", OUT)
PYEOF
