#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
# Usage: sbatch ... batch_stream_test_job.sh DATASET "BATCHES"
set -euo pipefail
DS=$1 BS=$2
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
case $DS in scannetpp) dataset pull ScanNetpp;; scannet-v2) dataset pull scannet-v2-extracted;; *) dataset pull $DS;; esac >/dev/null
PLAN=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans/$DS.json
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!
trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/batch_stream_test.py --plan $PLAN --batches $BS --frames ${FRAMES:-256} 2>&1 | grep -v "Streaming inference"
