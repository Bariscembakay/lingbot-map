#!/usr/bin/env bash
#SBATCH --job-name=poserefine_PROBE_lingbot_camerahead_past_context_usage_val24_nrgbd_kitti
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_headprobe.log
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" scripts/memory/pose_refine/head_context_probe.py --sets val24,nrgbd,kitti \
    --out /group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/head_context_probe.json
