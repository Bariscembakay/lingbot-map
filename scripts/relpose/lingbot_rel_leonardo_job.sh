#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=/leonardo_scratch/large/userexternal/bbakay00/logs/%x_%j.out
# One train_lingbot_rel.py arm (option B: lingbot backbone fine-tuned) on CINECA Leonardo, 4 A100 64 GB
# per node, one torchrun per node. Code (git), env, the CUT3R repo (dataset code + hold-out splits) and
# the CUT3R scale2 data all live in $CINECA_SCRATCH; compute nodes have no internet, so wandb is offline.
# Usage: sbatch -J <name> --nodes=N --time=.. lingbot_rel_leonardo_job.sh <arm> [train_lingbot_rel.py args]
set -euo pipefail
S=$CINECA_SCRATCH
P=$S/micromamba/envs/lingbot_map/bin/python
ARM=$1; shift
OUT=$S/runs/lingbot_rel/backbone/$ARM
mkdir -p "$OUT"
export WANDB_MODE=offline OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
echo "[lingbot_rel_leonardo_job] $SLURM_JOB_NUM_NODES nodes ($SLURM_JOB_NODELIST), master $MASTER, git $(git -C $S/lingbot-map-rel rev-parse --short HEAD)"
# CUT3R's hold-out lists are relative paths (rel/splits/...), so the trainer runs from the CUT3R repo
cd $S/CUT3R
srun --kill-on-bad-exit=1 "$P" -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=4 \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER:29517" --rdzv-id="$SLURM_JOB_ID" \
    $S/lingbot-map-rel/scripts/relpose/train_lingbot_rel.py --out "$OUT" --cut3r $S/CUT3R \
    --data-root $S/data/cut3r "$@"
