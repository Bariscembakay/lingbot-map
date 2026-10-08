#!/usr/bin/env bash
# NRGBD recall benchmark (nrgbd_recall_s2) for one checkpoint, one scene per
# array task: ingest+query+dump on the GPU, then score n100/n300/n500 posed.
# Clouds (~1.5 GB/cell) only for PLY_SCENE at n500.
# Usage: sbatch --array=0-6 nrgbd_recall_eval.sh <method> <ckpt> [ply_scene]
set -uo pipefail
METHOD=$1; CKPT=$2; PLY_SCENE=${3:-whiteroom}
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
CAMP=/group/compact-3dmem/campaigns/spatial_memory
CACHE=$CAMP/eval_nrgbd_cache/${S}_500f_s2
[ -f "$CACHE/meta.json" ] || { echo "[abort] no cache for $S"; exit 1; }
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^/[gpu] /'
"$P" scripts/memory/recall_bench/run_ours.py --scene "$S" --ckpt "$CKPT" --method "$METHOD" \
    --cache "$CACHE" --out "$CAMP/recallbench" --dump-dir "$CAMP/recallbench_dumps" || exit 1
for T in 100 300 500; do
    ply=--no-ply; [ "$S" = "$PLY_SCENE" ] && [ "$T" = 500 ] && ply=""
    "$P" scripts/memory/recall_bench/run_recall_score.py --method "$METHOD" --posed $ply \
        --scene "$S" --tier "$T" --dump-dir "$CAMP/recallbench_dumps" --out "$CAMP/recallbench" || exit 1
done
