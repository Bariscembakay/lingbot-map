#!/usr/bin/env bash
# Generic job wrapper for sof1 zone (and gcp-eu1, which shares its storage).
# Sets up envs, keeps the GPU minimally busy (see gpu_keep_alive.py --
# GPU_KEEP_ALIVE_FRACTION to override, GPU_KEEP_ALIVE=0 to disable), then
# runs whatever command you give it.
#
# Usage: srun ... bash submit_sof1.sh <command...>
#   e.g. bash submit_sof1.sh bash .agents/scratch/run_stage.sh prepare configs/oxford.yaml
set -euo pipefail
: "${MAMBA_ROOT_PREFIX:=/scratch/$USER/micromamba}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"

source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
source .agents/scratch/insait_cluster_files/setup_bench_env.sh

if [ "${GPU_KEEP_ALIVE:-1}" != "0" ]; then
    "$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python" \
        .agents/scratch/insait_cluster_files/gpu_keep_alive.py \
        "${GPU_KEEP_ALIVE_FRACTION:-0.4}" &
    KEEP_ALIVE_PID=$!
    trap 'kill $KEEP_ALIVE_PID 2>/dev/null' EXIT
fi

"$@"
