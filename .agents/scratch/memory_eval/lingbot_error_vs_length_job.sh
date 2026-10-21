#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
for d in tartanair-v1 tartanair-v2 matrixcity kitti360 scannetpp scannet-v2; do dataset pull lingbot-cache1048-$d >/dev/null; done
"$P" scripts/memory/pose_refine/lingbot_error_vs_length.py --exclude /group/compact-3dmem/campaigns/spatial_memory/_data_prep/lingbot_error_vs_length/gt_jump_scan.json --out /group/compact-3dmem/campaigns/spatial_memory/_data_prep/lingbot_error_vs_length/result.json
