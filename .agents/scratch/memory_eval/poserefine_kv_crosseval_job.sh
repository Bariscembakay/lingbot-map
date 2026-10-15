#!/usr/bin/env bash
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# best.pt of each given arm dir, final mode, on extra eval sets -> final_best<TAG>.json
# Usage: sbatch [--export=ALL,SETS=...,TAG=...] poserefine_kv_crosseval_job.sh <arm dir>...
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
for D in "$@"; do
    "$P" scripts/memory/pose_refine/train_pose_refine.py --inputs /group/compact-3dmem/datasets/lingbot-poserefine-v1 \
        --out "$D" --eval-only --ckpt "$D/best.pt" --max-frames-rope 20000 \
        --eval-sets ${SETS:-kitti:full,vbr:full} --tag ${TAG:-_driving}
done
