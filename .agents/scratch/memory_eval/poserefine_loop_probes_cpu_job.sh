#!/usr/bin/env bash
#SBATCH --job-name=looprobe_RETRIEVAL_lingbot_specials_dino_r10m_gap200m_and_REVISITPOSE_mlp_kitti360_to_kitti_vbr_cpu
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=04:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_loopprobes.log
# PROBE=retrieval|pose|both. Retrieval picks up DINO descriptors wherever <set>/dino/ exists.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
O=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PROBE=${PROBE:-both}
if [[ $PROBE != pose ]]; then
    "$P" scripts/memory/pose_refine/loop_retrieval_probe.py --sets kitti,vbr --out $O/loop_retrieval_probe${TAG:-}.json
fi
if [[ $PROBE != retrieval ]]; then
    "$P" scripts/memory/pose_refine/revisit_pose_probe.py --out $O/revisit_pose_probe.json
fi
