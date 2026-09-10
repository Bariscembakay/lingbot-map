#!/usr/bin/env bash
# GPU phase for our checkpoint: ingest + query + dump. Minutes, not hours.
set -uo pipefail
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
CAMP=/group/compact-3dmem/campaigns/spatial_memory
CACHE=$CAMP/eval_nrgbd_cache/${S}_500f_s2
CKPT=${OURS_CKPT:-$CAMP/scenes96_96f_b4_write4_read2_lingbothead_unfrozenhead/best.pt}
[ -f "$CACHE/meta.json" ] || { echo "[abort] no cache for $S"; exit 0; }
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" scripts/memory/recall_bench/run_ours.py --scene "$S" --ckpt "$CKPT" \
    --cache "$CACHE" --out "$CAMP/recallbench" \
    --dump-dir "$CAMP/recallbench_dumps"
