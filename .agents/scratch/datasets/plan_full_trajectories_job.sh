#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# plan_full_trajectories.py (whole loop trajectories, stride 1) for each dataset given; sof1 zone
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
D=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans_full
mkdir -p $D
for ds in "$@"; do
    case $ds in matrixcity) dataset pull matrix_city;; *) dataset pull $ds;; esac >/dev/null
    (cd scripts/memory/pose_refine && "$P" plan_full_trajectories.py --dataset $ds --out $D/$ds.json --workers 16)
done
