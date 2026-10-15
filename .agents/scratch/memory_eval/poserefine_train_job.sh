#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --constraint=zone-gcp-eu1
#SBATCH --gpus=a100-80g:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Usage: sbatch -J <name> poserefine_train_job.sh <out_arm> <train_pose_refine.py args...>
#        ARM=smoke runs every variant for a few steps instead.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb  # lingbot_map env ships without it
IN=/group/compact-3dmem/datasets/lingbot-poserefine-v1
CAMP=/group/compact-3dmem/campaigns/spatial_memory
ARM=$1; shift
if [ "$ARM" = smoke ]; then
    for v in tokres_frozencausal tokres_bidirhead direct_pose pose_residual; do
        rope=""; [ $v = tokres_frozencausal ] && rope=--no-head-rope
        "$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --memory specials \
            --variant $v $rope --out /scratch/$USER/poserefine_smoke/$v --steps 20 --val-every 10 --warmup 5
    done
    "$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --memory specials \
        --variant tokres_bidirhead --no-head-rope --eval-only --out $CAMP/_poserefine_logs/eval0
    "$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --memory specials \
        --variant tokres_bidirhead --eval-only --out $CAMP/_poserefine_logs/eval0
    "$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --memory specials \
        --variant tokres_frozencausal --no-head-rope --eval-only --out $CAMP/_poserefine_logs/eval0
    exit 0
fi
"$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --out $CAMP/$ARM --wandb "$@"
