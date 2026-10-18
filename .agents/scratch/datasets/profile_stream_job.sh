#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
set -euo pipefail
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull tartanair-v1 >/dev/null
"$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python" scripts/memory/pose_refine/profile_stream.py \
    --plan /group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans/tartanair-v1.json --frames ${FRAMES:-160} --profile-frames ${PFRAMES:-96} ${PROFILE_ARGS:-}
