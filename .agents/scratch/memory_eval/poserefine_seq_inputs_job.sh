#!/usr/bin/env bash
#SBATCH --job-name=poserefine_BUILD_driving_train_inputs_kitti360x9_vkitti2x5_runs4000_every1000_sdpa_a6000
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-13
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%A_%a_seqinputs.log
# Driving training inputs from data lingbot trained on: KITTI-360 (9 posed drives), VKITTI2 (5 scenes).
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
JOBS=(kitti360:2013_05_28_drive_00{00,02,03,04,05,06,07,09,10}_sync vkitti2:Scene{01,02,06,18,20})
I=$SLURM_ARRAY_TASK_ID; [ "${REVERSE:-0}" = 1 ] && I=$(( ${#JOBS[@]} - 1 - I ))  # a second copy on other GPUs works from the other end
DS=${JOBS[$I]%%:*}; SEQ=${JOBS[$I]#*:}
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/build_seq_inputs.py $DS --seq $SEQ --use-sdpa \
    --out /group/compact-3dmem/datasets/lingbot-poserefine-v1/$DS
