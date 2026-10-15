#!/usr/bin/env bash
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=07:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Usage: sbatch -J <name> poserefine_kv_train_job.sh <out_arm> <train_pose_refine.py args...>
# Trains, then scores best.pt in final mode on every eval set including full-length VBR.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb  # lingbot_map env ships without it
IN=/group/compact-3dmem/datasets/lingbot-poserefine-v1
CAMP=/group/compact-3dmem/campaigns/spatial_memory
ARM=$1; shift
"$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --out $CAMP/$ARM --wandb "$@"
"$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --out $CAMP/$ARM --eval-only \
    --ckpt $CAMP/$ARM/best.pt --max-frames-rope 20000 \
    --eval-sets ${FINAL_EVAL:-val24:320,kitti360_val:full,nrgbd:500,kitti:full,vbr:full}
