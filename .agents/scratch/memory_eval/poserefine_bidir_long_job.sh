#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Long-horizon bidirectional refiner on the tap-23 packs of the 1048-frame caches (sof1-side zones:
# sof1, gcp-eu*). The packs (~120 GB) are staged to node-local /scratch first: /data is network-backed.
# Usage: sbatch -J <name> --gpus=<type>:1 --constraint=<zone> --time=.. poserefine_bidir_long_job.sh <arm|smoke> <args...>
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull lingbot-poserefine-v1 >/dev/null
dataset pull lingbot-tap23pack-1048 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!
trap 'kill $KA 2>/dev/null' EXIT
PK=/scratch/$USER/tap23pack-1048
mkdir -p $PK
( flock 9; rsync -a /data/lingbot-tap23pack-1048/ $PK/ ) 9>$PK.lock
CAMP=/group/compact-3dmem/campaigns/spatial_memory
ARM=$1; shift
if [ "$ARM" = smoke ]; then
    "$P" scripts/memory/pose_refine/train_bidir_refiner.py --out /scratch/$USER/bidir_long_smoke --pack-root $PK \
        --anchors fixed --steps 30 --val-every 15 --warmup 5 --longval-per-ds 2 --nrgbd-lengths 64,500 --no-wandb "$@"
    exit 0
fi
"$P" scripts/memory/pose_refine/train_bidir_refiner.py --out $CAMP/$ARM --pack-root $PK "$@"
