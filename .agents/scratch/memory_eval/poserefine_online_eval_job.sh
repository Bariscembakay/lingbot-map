#!/usr/bin/env bash
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Usage: sbatch -J <name> poserefine_online_eval_job.sh <ckpt> <k> [<ckpt> <k> ...]
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull lingbot-poserefine-v1 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb  # lingbot_map env ships without it
while [ $# -ge 2 ]; do
    "$P" scripts/memory/pose_refine/train_pose_refine.py --inputs /data/lingbot-poserefine-v1 \
        --eval-only --ckpt "$1" --online-k "$2" --out "$(dirname "$1")"
    shift 2
done
