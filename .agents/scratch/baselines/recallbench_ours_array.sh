#!/usr/bin/env bash
# Score OUR best checkpoint on nrgbd_recall_s2. One array task per scene.
set -uo pipefail
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
CAMP=/group/compact-3dmem/campaigns/spatial_memory
CACHE=$CAMP/eval_nrgbd_cache/${S}_500f_s2
CKPT=${OURS_CKPT:-$CAMP/scenes96_96f_b4_write4_read2_lingbothead_unfrozenhead/best.pt}
if [ ! -f "$CACHE/meta.json" ]; then
  echo "[abort] no cache for $S at $CACHE (cache build must land first)"; exit 0
fi
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
trap 'kill $! 2>/dev/null' EXIT
"$P" scripts/memory/recall_bench/run_ours.py \
    --scene "$S" --ckpt "$CKPT" --cache "$CACHE" \
    --out "$CAMP/recallbench"
