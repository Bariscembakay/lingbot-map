#!/usr/bin/env bash
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-13
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%A_%a_dinoseq.log
# DINOv2 CLS/mean/GeM per frame of the KITTI-360 / VKITTI2 training runs (GeM-memory arm).
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
JOBS=(kitti360:2013_05_28_drive_00{00,02,03,04,05,06,07,09,10}_sync vkitti2:Scene{01,02,06,18,20})
J=${JOBS[$SLURM_ARRAY_TASK_ID]}
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/extract_dino_globals.py --seq $J --workers 10 \
    --out /group/compact-3dmem/datasets/lingbot-poserefine-v1/${J%%:*}
