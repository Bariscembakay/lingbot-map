#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --constraint=zone-msp3
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --chdir=/home/baris_bakay/lingbot-map-rel
#SBATCH --output=/home/baris_bakay/relhead_logs/%x_%j.out
# msp3 twin of relhead_job.sh: code by git (the relpose/relative-head worktree, pulled), data by
# `dataset pull`, the run dir on /scratch mirrored to sof1's campaign dir every 10 min and at the end.
# Usage: sbatch -J <name> --time=.. relhead_msp3_job.sh <arm> [train_rel_head.py args]
set -euo pipefail
# Since the 2026-10-22 cluster update, batch steps get CUDA_VISIBLE_DEVICES="" (srun steps get the
# GPU UUIDs): CUDA then sees no GPU. The device cgroup already limits the job to its own GPUs.
[ -n "${CUDA_VISIBLE_DEVICES+x}" ] && [ -z "$CUDA_VISIBLE_DEVICES" ] && unset CUDA_VISIBLE_DEVICES
cd "$HOME/lingbot-map-rel"
git pull -q --ff-only origin relpose/relative-head || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "[relhead_msp3_job] torch sees no GPU on $(hostname)"; nvidia-smi; exit 4; }
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb
dataset pull lingbot-poserefine-v1 >/dev/null
dataset pull lingbot-tap23pack-1048 >/dev/null
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.01 & KA=$!
echo "[relhead_msp3_job] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) git $(git rev-parse --short HEAD)"
ARM=$1; shift
DEST=sof1:/group/compact-3dmem/campaigns/lingbot_rel/relhead/$ARM
WORK=/scratch/$USER/relhead/$ARM
mkdir -p "$WORK"
( while sleep 600; do rsync -a "$WORK/" "$DEST/" 2>/dev/null || true; done ) & SYNC=$!
trap 'kill $SYNC $KA 2>/dev/null; rsync -a "$WORK/" "$DEST/" || true' EXIT
"$P" scripts/relpose/train_rel_head.py --out "$WORK" "$@"
