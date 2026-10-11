#!/usr/bin/env bash
# Chorus voxel features for a slice of ScanNet++ scenes (see chorus_extract.py).
# Usage: sbatch [--array=0-K] chorus_extract_job.sh <scene_list.txt> <out_dir> [per_task] [extra chorus_extract.py args]
# Array task i takes scenes [i*per_task, (i+1)*per_task) of the list.
set -euo pipefail
LIST=$1; OUT=$2; PER=${3:-1000000}
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_chorus_env.sh
dataset pull ScanNetppv2_gsplat >/dev/null
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
I=${SLURM_ARRAY_TASK_ID:-0}
mapfile -t SCENES < <(sed -n "$((I * PER + 1)),$(((I + 1) * PER))p" "$LIST")
[ ${#SCENES[@]} -gt 0 ] || { echo "no scenes for task $I"; exit 0; }
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/chorus/bin/python" scripts/memory/chorus_extract.py \
    --out "$OUT" "${@:4}" --scenes "${SCENES[@]}"
