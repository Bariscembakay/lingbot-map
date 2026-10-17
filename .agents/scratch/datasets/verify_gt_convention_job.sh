#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
# GT camera-convention check (verify_gt_convention.py) for each dataset given; run in the zone
# that holds the data. Usage:
#   sof1: sbatch --partition=rendering --qos=rendering --gpus=a6000:1 --constraint=zone-sof1 \
#         --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log verify_gt_convention_job.sh tartanair-v1 matrixcity
#   msp3: sbatch --partition=batch --gpus=h200:1 --constraint=zone-msp3 --output=/scratch/baris_bakay/%j_%x.log \
#         verify_gt_convention_job.sh tartanair-v2 tartanground
set -euo pipefail
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
for ds in "$@"; do
    case $ds in matrixcity) dataset pull matrix_city >/dev/null;; *) dataset pull $ds >/dev/null;; esac
    "$P" scripts/memory/pose_refine/verify_gt_convention.py --dataset $ds --seqs 3 --frames 400 \
        --out ${OUT_DIR:-/scratch/$USER}/verify_gt_convention_$ds.json
done
