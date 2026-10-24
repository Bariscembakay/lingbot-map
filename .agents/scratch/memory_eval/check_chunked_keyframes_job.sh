#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Default-mode chunked inference vs frame-by-frame streaming (scripts/memory/pose_refine/check_chunked_keyframes.py).
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import OpenEXR" 2>/dev/null || "$P" -m pip install -q OpenEXR
for d in ${DATASETS:-kitti360}; do case $d in matrixcity) dataset pull matrix_city;; *) dataset pull $d;; esac >/dev/null; done
cd scripts/memory/pose_refine
for d in ${DATASETS:-kitti360}; do "$P" check_chunked_keyframes.py --dataset $d "$@"; done
