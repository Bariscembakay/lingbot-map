#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/group/compact-3dmem/campaigns/_joblogs/%x_%j.out
# One train_rel_head.py arm on a sof1-side node (sof1, hala, gcp-eu*: same /home and /group). The
# packs' camera tokens are extracted once per node to /scratch; the run dir is the campaign dir.
# Usage: sbatch -J <name> --gpus=<type>:1 --constraint=<zone> [--partition/--qos] --time=.. relhead_job.sh <arm|smoke> [args]
set -euo pipefail
cd "$HOME/lingbot-map-rel"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull lingbot-poserefine-v1 >/dev/null
dataset pull lingbot-tap23pack-1048 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import wandb" 2>/dev/null || "$P" -m pip install -q wandb
# evaluation is CPU-bound (Umeyama, AUC over long streams); keep the idle-GPU reaper off
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!
trap 'kill $KA 2>/dev/null' EXIT
echo "[relhead_job] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) git $(git rev-parse --short HEAD)"
ARM=$1; shift
if [ "$ARM" = smoke ]; then
    "$P" scripts/relpose/train_rel_head.py --out "/scratch/$USER/relhead_smoke" --steps 60 --val-every 30 \
        --warmup 10 --longval-per-ds 1 --nrgbd-lengths 64,500 --no-wandb "$@"
    exit 0
fi
"$P" scripts/relpose/train_rel_head.py --out "/group/compact-3dmem/campaigns/lingbot_rel/relhead/$ARM" "$@"
