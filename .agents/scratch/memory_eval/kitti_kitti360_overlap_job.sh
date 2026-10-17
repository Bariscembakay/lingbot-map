#!/usr/bin/env bash
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --gpus=a6000:1
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# KITTI odometry vs KITTI-360 place overlap by MegaLoc retrieval (VBR as the null).
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull kitti360 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/cross_dataset_overlap.py --desc /group/compact-3dmem/campaigns/spatial_memory/loop_descriptor_probe/_desc \
    --out /group/compact-3dmem/campaigns/spatial_memory/kitti_kitti360_overlap --stride 2
