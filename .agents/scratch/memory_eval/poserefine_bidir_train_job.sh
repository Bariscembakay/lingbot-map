#!/usr/bin/env bash
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Usage: sbatch -J <name> poserefine_bidir_train_job.sh <out_arm|smoke> <train_bidir_refiner.py args...>
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
CAMP=/group/compact-3dmem/campaigns/spatial_memory
ARM=$1; shift
# patch tokens come from the /data tap caches; the specials-only runs read only /group npz inputs
if [ "$ARM" = smoke ] || [[ " $* " == *" --patches "* ]]; then
    for d in v4-40 v4-ext64 v5-tap23 v5-val16-4tap; do dataset pull lingbot-tapcache-$d >/dev/null; done
fi
if [ "$ARM" = smoke ]; then
    S=/scratch/$USER/bidir_smoke
    "$P" scripts/memory/pose_refine/train_bidir_refiner.py --out $S/spec_fixed --anchors fixed \
        --steps 20 --val-every 10 --warmup 5 --max-train 4 --nrgbd-lengths 64,500
    "$P" scripts/memory/pose_refine/train_bidir_refiner.py --out $S/patch_free --anchors free --patches \
        --batch 2 --steps 20 --val-every 10 --warmup 5 --max-train 4 --nrgbd-lengths 64,500
    exit 0
fi
"$P" scripts/memory/pose_refine/train_bidir_refiner.py --out $CAMP/$ARM "$@"
