#!/usr/bin/env bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
# Re-plan the 1048 clips of ScanNet++ (COLMAP-validated ARKit poses) and ScanNet v2 (pose jumps split)
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
dataset pull ScanNetpp >/dev/null; dataset pull scannet-v2-extracted >/dev/null
O=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans_gtfix
mkdir -p $O
cd scripts/memory/pose_refine
for d in scannetpp scannet-v2; do "$P" plan_clips.py --dataset $d --out $O/$d.json --workers 32; done
