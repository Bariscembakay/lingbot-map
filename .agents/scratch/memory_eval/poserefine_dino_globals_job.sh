#!/usr/bin/env bash
#SBATCH --job-name=looprobe_EXTRACT_dinov2L_reg4_cls_mean_gem3_kitti00-10_vbr7_504x280_a6000
#SBATCH --partition=rendering
#SBATCH --qos=rendering
#SBATCH --constraint=zone-sof1
#SBATCH --gpus=a6000:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-17
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%A_%a_dinoglobals.log
# One benchmark scene per task: DINOv2 CLS / mean / GeM descriptors for the loop-retrieval probe.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
B=/group/compact-3dmem/campaigns/lingbot_map/default
SCENES=($B/kitti/{00,01,02,03,04,05,06,07,08,09,10} $B/vbr/{campus_train0,campus_train1,ciampino_train1,colosseo_train0,diag_train0,pincio_train0,spagna_train0})
S=${SCENES[$SLURM_ARRAY_TASK_ID]}
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/extract_dino_globals.py --scenes $S --workers 8 \
    --out /group/compact-3dmem/datasets/lingbot-poserefine-v1/$(basename $(dirname $S))
