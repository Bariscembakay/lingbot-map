#!/usr/bin/env bash
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%A_%a_%x.log
# One benchmark scene per task: lingbot-side place descriptors (+ MegaLoc) for the loop-retrieval probe.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
B=/group/compact-3dmem/campaigns/lingbot_map/default
SCENES=($B/kitti/{00,01,02,03,04,05,06,07,08,09,10} $B/vbr/{campus_train0,campus_train1,ciampino_train1,colosseo_train0,diag_train0,pincio_train0,spagna_train0})
S=${SCENES[$SLURM_ARRAY_TASK_ID]}
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/extract_loop_descriptors.py --scenes $S --megaloc \
    --out /group/compact-3dmem/campaigns/spatial_memory/loop_descriptor_probe/_desc/$(basename $(dirname $S))
