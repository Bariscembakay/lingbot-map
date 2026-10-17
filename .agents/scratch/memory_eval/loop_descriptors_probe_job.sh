#!/usr/bin/env bash
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# Loop-retrieval probe over every descriptor (official DINOv2, lingbot specials, lingbot-side set, MegaLoc).
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull lingbot-poserefine-v1 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
D=/group/compact-3dmem/campaigns/spatial_memory/loop_descriptor_probe
SETS=${SETS:-kitti,vbr}
"$P" scripts/memory/pose_refine/loop_retrieval_probe.py --sets $SETS --desc $D/_desc --out $D/probe_${SETS//,/_}_r10_minpath200_neg50.json
