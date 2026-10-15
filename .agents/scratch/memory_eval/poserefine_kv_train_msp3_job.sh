#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --constraint=zone-msp3
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/baris_bakay/lingbot-map
#SBATCH --output=/home/baris_bakay/poserefine_logs/%j_%x.log
# msp3 twin of poserefine_kv_train_job.sh: inputs via `dataset pull`, work on /scratch,
# results mirrored to sof1's campaign dir every 10 min and at the end.
# Usage: sbatch -J <name> poserefine_kv_train_msp3_job.sh <out_arm> <train_pose_refine.py args...>
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
dataset pull lingbot-poserefine-v1 >/dev/null
IN=/data/lingbot-poserefine-v1
ARM=$1; shift
DEST=sof1:/group/compact-3dmem/campaigns/spatial_memory/$ARM
WORK=/scratch/$USER/poserefine/$ARM
mkdir -p "$WORK"
export WANDB_MODE=offline   # msp3 has no wandb login; history.json carries every val row
( while sleep 600; do rsync -a "$WORK/" "$DEST/" 2>/dev/null || true; done ) &
SYNC=$!; trap 'kill $SYNC 2>/dev/null; rsync -a "$WORK/" "$DEST/" || true' EXIT
"$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --out $WORK "$@"
"$P" scripts/memory/pose_refine/train_pose_refine.py --inputs $IN --out $WORK --eval-only \
    --ckpt $WORK/best.pt --max-frames-rope 20000 \
    --eval-sets val24:320,kitti360_val:full,nrgbd:500,kitti:full,vbr:full
