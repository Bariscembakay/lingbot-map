#!/usr/bin/env bash
# Generic job wrapper for msp3: dataset pull -> compute on /scratch -> ship
# small results back to sof1. Also sets up envs and keeps the GPU minimally
# busy (see gpu_keep_alive.py -- GPU_KEEP_ALIVE_FRACTION/GPU_KEEP_ALIVE=0).
#
# Usage: bash submit_msp3.sh <dataset_name|-> <local_results_dir|-> <sof1_dest_dir> <command...>
#   dataset_name: `dataset pull` target, or `-` to skip.
#   local_results_dir: shipped to sof1_dest_dir (eval/+logs/ only) after the
#     command exits; `-` to skip shipping anything back.
#
#   e.g. bash submit_msp3.sh oxford_spires_processed /scratch/$USER/ws \
#          /group/compact-3dmem/campaigns/lingbot_map/oxford_spires/sparse_s12 \
#          bash .agents/scratch/reproduction/run_benchmark.sh configs/oxford.yaml
set -euo pipefail

DATASET_NAME="$1"
LOCAL_RESULTS_DIR="$2"
SOF1_DEST_DIR="$3"
shift 3

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"

if [ "$DATASET_NAME" != "-" ]; then
    dataset pull "$DATASET_NAME"
fi

source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
source .agents/scratch/insait_cluster_files/setup_bench_env.sh

if [ "${GPU_KEEP_ALIVE:-1}" != "0" ]; then
    micromamba run -n lingbot_map python \
        .agents/scratch/insait_cluster_files/gpu_keep_alive.py \
        "${GPU_KEEP_ALIVE_FRACTION:-0.4}" &
    KEEP_ALIVE_PID=$!
    trap 'kill $KEEP_ALIVE_PID 2>/dev/null' EXIT
fi

"$@"

if [ "$LOCAL_RESULTS_DIR" != "-" ]; then
    ssh sof1 "mkdir -p $SOF1_DEST_DIR"
    rsync -a --include='*/' --include='eval/***' --include='logs/***' --exclude='*' \
        "$LOCAL_RESULTS_DIR"/ "sof1:$SOF1_DEST_DIR/"
fi
