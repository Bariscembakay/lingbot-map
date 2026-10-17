#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# One shard (SLURM_ARRAY_TASK_ID of NSHARDS) of build_clip_cache.py for DATASET. Results always end up
# in sof1's /group/compact-3dmem/datasets/_staging/lingbot-cache1048-<dataset>/ (registered once
# complete): sof1-zone jobs write there directly, msp3 jobs write to local /scratch and rsync to sof1
# every 10 minutes and at exit.
# Usage: sbatch --array=0-N ... build_clip_cache_job.sh DATASET NSHARDS
set -euo pipefail
DS=$1 NSH=$2 SH=${SLURM_ARRAY_TASK_ID:-0}
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import OpenEXR" 2>/dev/null || "$P" -m pip install -q OpenEXR
case $DS in scannetpp) dataset pull ScanNetpp;; matrixcity) dataset pull matrix_city;; scannet-v2) dataset pull scannet-v2-extracted;; *) dataset pull $DS;; esac >/dev/null
PLAN=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans/$DS.json
FINAL=/group/compact-3dmem/datasets/_staging/lingbot-cache1048-$DS
if findmnt -n -o SOURCE -T /group/compact-3dmem | grep -q msp3; then
    [ -f $PLAN ] || { mkdir -p $(dirname $PLAN); rsync -a sof1:$PLAN $PLAN; }
    OUT=/scratch/$USER/cache1048_$DS
    mkdir -p $OUT; ssh sof1 mkdir -p $FINAL
    ( while sleep 600; do rsync -a --include='*.npz' --exclude='*' $OUT/ sof1:$FINAL/; done ) & SY=$!
    trap 'kill $SY 2>/dev/null; rsync -a --include="*.npz" --exclude="*" $OUT/ sof1:$FINAL/; kill $KA 2>/dev/null' EXIT
else
    OUT=$FINAL; mkdir -p $OUT
    trap 'kill $KA 2>/dev/null' EXIT
fi
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!
"$P" scripts/memory/pose_refine/build_clip_cache.py --plan $PLAN --out $OUT --shard $SH --nshards $NSH
