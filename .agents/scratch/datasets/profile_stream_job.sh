#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:40:00
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
dataset pull tartanair-v1 >/dev/null
PLAN=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans/tartanair-v1.json
for B in 1 4; do
  "$P" scripts/memory/pose_refine/profile_stream.py --plan $PLAN --frames 112 --profile-frames 96 --batch $B --skip-timing 2>&1 | grep -v "Streaming inference"
done
