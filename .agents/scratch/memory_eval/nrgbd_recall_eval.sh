#!/usr/bin/env bash
# NRGBD recall benchmark (nrgbd_recall_s2), GPU phase only: ingest + query +
# dump for one scene per array task. Scoring is CPU-only and runs separately
# (nrgbd_recall_score.sh): scored inside this job, the GPU sat idle for up to an
# hour and the idle-GPU reaper killed it (1086779_0/2/3).
# Usage: sbatch --array=0-6 nrgbd_recall_eval.sh <method> <ckpt>
set -uo pipefail
METHOD=$1; CKPT=$2
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
