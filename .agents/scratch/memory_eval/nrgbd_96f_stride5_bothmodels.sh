#!/usr/bin/env bash
# NRGBD breakfast_room, 96 frames stride 5: build tap cache, run OUR best
# checkpoint + zero-shot CUT3R on the identical frames, render both recalls.
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
set +e +u +o pipefail
PY_LM="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
CACHE=/group/compact-3dmem/campaigns/spatial_memory/eval_nrgbd_cache/breakfast_room_96f_s5
CAMP=/group/compact-3dmem/campaigns/spatial_memory
dataset pull NRGBD >/dev/null 2>&1 || true
echo "=== 1. cache build (lingbot fwd) ==="
"$PY_LM" .agents/scratch/memory_eval/build_cache_nrgbd.py \
  --scene breakfast_room --stride 5 --clip-len 96 --out "$CACHE" --use-sdpa || exit 1
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail
PY_C="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
echo "=== 2. ours (scenes96 best ckpt) recall ==="
"$PY_C" .agents/scratch/memory_eval/run_ours_recall.py \
  --ckpt "$CAMP/scenes96_96f_b4_write4_read2_lingbothead/last.pt" \
  --clip "$CACHE" --frames 96 --subsample 1 \
  --out /scratch/$USER/recall96/ours_nrgbd_breakfast96s5.npz || exit 1
"$PY_C" - <<'PYEOF'
import sys, numpy as np, pathlib
sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import emit
d = np.load("/scratch/baris_bakay/recall96/ours_nrgbd_breakfast96s5.npz")
OUT = pathlib.Path("/group/compact-3dmem/campaigns/spatial_memory/"
                   "scenes96_96f_b4_write4_read2_lingbothead/nrgbd_breakfastroom_96f_stride5_recall")
p, g = [], []
for i in range(len(d["pred_pts_world"])):
    v = d["valid"][i]
    p.append(d["pred_pts_world"][i][v][::3]); g.append(d["gt_pts_world"][i][v][::3])
P, G = np.concatenate(p), np.concatenate(g)
err = np.linalg.norm(P - G, axis=-1)
print(f"[ours-nrgbd] fused {len(P)} pts, L21 mean {err.mean():.4f} med {np.median(err):.4f} (canonical)")
emit(OUT, "breakfast_room", P, G, np.ones(len(P), bool), np.ones(len(P), np.float32), 0.0)
print("rendered ->", OUT)
PYEOF
echo "=== 3. CUT3R zero-shot, same 96 stride-5 frames ==="
"$PY_C" - <<'PYEOF'
import sys
sys.argv = ["x"]
sys.path.insert(0, "/home/baris_bakay/lingbot-map/.agents/scratch/memory_eval")
import nrgbd_cut3r_recall_viz as M
M.STRIDE, M.SNAPSHOTS, M.QSTRIDE = 5, (96,), 1
import pathlib
M.OUTDIR = pathlib.Path("/group/compact-3dmem/campaigns/spatial_memory/"
                        "CUT3R_zeroshot_512dpt/nrgbd_breakfastroom_96f_stride5_recall")
M.main()
PYEOF
echo "=== all done ==="
