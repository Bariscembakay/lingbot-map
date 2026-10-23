#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# refiner_trajectories.py on sof1 (holds the tap-23 packs); GPU if the job has one: example train / val trajectories of every refiner
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
dataset pull lingbot-tap23pack-1048 >/dev/null
export OMP_NUM_THREADS=16
"$P" scripts/memory/pose_refine/refiner_trajectories.py \
    --runs "$(ls -d /group/compact-3dmem/campaigns/spatial_memory/bidirlong_6ds_* | paste -sd,)"
