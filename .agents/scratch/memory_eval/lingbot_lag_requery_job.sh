#!/usr/bin/env bash
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Usage: sbatch -J <name> lingbot_lag_requery_job.sh [eval_lag_requery.py args...]
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull lingbot-poserefine-v1 >/dev/null
dataset pull NRGBD >/dev/null
"$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python" scripts/memory/pose_refine/eval_lag_requery.py \
    --out /group/compact-3dmem/campaigns/spatial_memory/poserefine_lingbot_lag63_requery_nrgbd500s2/nrgbd.json "$@"
