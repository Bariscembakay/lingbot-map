#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=24:00:00
# One worker of the whole-trajectory default-mode cache build: goes through DATASETS in order and takes
# clips through claim files in sof1's staging dir, so any number of these, on any GPU type, share the
# work. sof1 /group zones only (sof1, gcp-eu1, gcp-eu9).
# Usage: sbatch ... cache_worker_job.sh "tartanair-v2 tartanground tartanair-v1 kitti360 matrixcity"
set -euo pipefail
cd "$HOME/lingbot-map"
sleep $(( RANDOM % 60 ))
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import OpenEXR" 2>/dev/null || "$P" -m pip install -q OpenEXR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.03 & KA=$!
trap 'kill $KA 2>/dev/null' EXIT
CACHE_NAME=${CACHE_NAME:-lingbot-cachefull}
for DS in $1; do
    case $DS in matrixcity) dataset pull matrix_city;; *) dataset pull $DS;; esac >/dev/null
    OUT=/group/compact-3dmem/datasets/_staging/$CACHE_NAME-$DS
    mkdir -p $OUT
    "$P" scripts/memory/pose_refine/build_clip_cache.py --plan ${PLAN_DIR:-/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans_full}/$DS.json \
        --out $OUT --claim --cache-name $CACHE_NAME --keyframes auto --spec23 ${CACHE_ARGS:-}
done
