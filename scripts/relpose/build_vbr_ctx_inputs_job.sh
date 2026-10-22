#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --array=0-13%6
#SBATCH --output=/group/compact-3dmem/campaigns/_joblogs/%x_%A_%a.out
# VBR inputs (lingbot tap specials + poses + GT, build_bench_inputs.py) under two lingbot context
# configs, every frame a keyframe as in the training caches: task = config x scene.
#   kf1_sdpa              anchors + window 64 + trajectory memory
#   kf1_window64only_sdpa the last 64 frames only
# Usage: sbatch -J <name> --gpus=<type>:1 --constraint=<zone> --time=.. build_vbr_ctx_inputs_job.sh
set -euo pipefail
[ -n "${CUDA_VISIBLE_DEVICES+x}" ] && [ -z "$CUDA_VISIBLE_DEVICES" ] && unset CUDA_VISIBLE_DEVICES
cd "$HOME/lingbot-map-rel"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "torch sees no GPU"; exit 4; }
CONFIGS=(kf1_sdpa kf1_window64only_sdpa)
SCENES=(campus_train0 campus_train1 ciampino_train1 colosseo_train0 diag_train0 pincio_train0 spagna_train0)
C=${CONFIGS[$((SLURM_ARRAY_TASK_ID / 7))]}
S=${SCENES[$((SLURM_ARRAY_TASK_ID % 7))]}
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.01 & KA=$!
trap 'kill $KA 2>/dev/null' EXIT
echo "[build_vbr_ctx] $(hostname) config=$C scene=$S git $(git rev-parse --short HEAD)"
"$P" scripts/memory/pose_refine/build_bench_inputs.py --use-sdpa \
    --method-config benchmark/configs/methods/relpose/lingbot_map_$C.yaml \
    --scenes /group/compact-3dmem/campaigns/lingbot_map/default/vbr/$S \
    --out /group/compact-3dmem/datasets/lingbot-relpose-vbr-ctx/$C
