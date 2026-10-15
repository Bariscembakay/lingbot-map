#!/usr/bin/env bash
#SBATCH --job-name=poserefine_RERUN_CONTINUE_lingbot_specials4tap_v5train848_leftovers_sdpa_rtx6000
#SBATCH --partition=batch
#SBATCH --constraint=zone-gcp-eu1
#SBATCH --gpus=rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%A_%a_rerun_continue.log
# Specials of taps 4/11/17/23 for the 848 tap-23-only v5 train clips, leftovers after the first array, 1/16 per task.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
for d in lingbot-tapcache-v5-tap23 ScanNetpp; do dataset pull $d >/dev/null 2>&1 || true; done
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
L=$(mktemp)
ls -d /data/lingbot-tapcache-v5-tap23/train/* | awk -v k=$SLURM_ARRAY_TASK_ID 'NR % 16 == k' > $L
echo "$(wc -l < $L) clips"
"$P" scripts/memory/pose_refine/rerun_specials.py --clips $L --use-sdpa \
    --out /group/compact-3dmem/datasets/lingbot-poserefine-v1/train_v5
