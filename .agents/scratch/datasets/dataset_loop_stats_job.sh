#!/usr/bin/env bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:30:00
# Training-set survey for the CVPR plan: trajectories, length range, path length, revisit (loop) counts
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
for d in tartanair-v1 tartanair-v2 tartanground matrix_city ScanNetpp scannet-v2-extracted kitti360; do dataset pull $d >/dev/null; done
cd scripts/memory/pose_refine
"$P" dataset_loop_stats.py --out /group/compact-3dmem/campaigns/spatial_memory/_data_prep/dataset_loop_stats.json --workers 32
