#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --constraint=zone-msp3
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --chdir=/home/baris_bakay/lingbot-map
#SBATCH --output=/home/baris_bakay/poserefine_logs/%j_%x.log
# msp3 twin of poserefine_bidir_long_job.sh: code by git, packs via `dataset pull` staged to /scratch,
# the run dir on /scratch mirrored to sof1's campaign dir every 10 min and at the end.
# Usage: sbatch -J <name> --time=.. poserefine_bidir_long_msp3_job.sh <arm> <train_bidir_refiner.py args...>
set -euo pipefail
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb
dataset pull lingbot-poserefine-v1 >/dev/null
dataset pull lingbot-tap23pack-1048 >/dev/null
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!
PK=/scratch/$USER/tap23pack-1048
mkdir -p $PK
( flock 9; rsync -a /data/lingbot-tap23pack-1048/ $PK/ ) 9>$PK.lock
ARM=$1; shift
DEST=sof1:/group/compact-3dmem/campaigns/spatial_memory/$ARM
WORK=/scratch/$USER/poserefine/$ARM
mkdir -p "$WORK"
( while sleep 600; do rsync -a "$WORK/" "$DEST/" 2>/dev/null || true; done ) & SYNC=$!
trap 'kill $SYNC $KA 2>/dev/null; rsync -a "$WORK/" "$DEST/" || true' EXIT
"$P" scripts/memory/pose_refine/train_bidir_refiner.py --out $WORK --pack-root $PK "$@"
