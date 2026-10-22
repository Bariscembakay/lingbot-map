#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
# plan_clips.py for each dataset given, in the zone holding its data. Plans land in this zone's
# /group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans/ (msp3 plans are also copied to sof1).
#   sof1: sbatch --partition=rendering --qos=rendering --gpus=a6000:1 --constraint=zone-sof1 \
#         --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log plan_clips_job.sh scannetpp kitti360 matrixcity tartanair-v1
#   msp3: sbatch --partition=login --constraint=zone-msp3 --output=/scratch/baris_bakay/%j_%x.log plan_clips_job.sh tartanair-v2 tartanground scannet-v2
set -euo pipefail
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import OpenEXR" 2>/dev/null || "$P" -m pip install -q OpenEXR
if nvidia-smi -L >/dev/null 2>&1; then "$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!; trap 'kill $KA 2>/dev/null' EXIT; fi
D=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans
mkdir -p $D
for ds in "$@"; do
    case $ds in scannetpp) dataset pull ScanNetpp;; matrixcity) dataset pull matrix_city;; scannet-v2) dataset pull scannet-v2-extracted;; waymo) dataset pull waymo-cut3r-video;; *) dataset pull $ds;; esac >/dev/null
    "$P" scripts/memory/pose_refine/plan_clips.py --dataset $ds --out $D/$ds.json --workers 16
    [ "$(findmnt -n -o SOURCE -T /group/compact-3dmem | grep -c msp3)" = 1 ] && rsync -a $D/$ds.json sof1:$D/ || true
done
