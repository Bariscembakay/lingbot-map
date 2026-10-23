#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/leonardo_scratch/large/userexternal/bbakay00/logs/%x_%j.out
# One train_lingbot_rel.py arm (option B: lingbot backbone fine-tuned) on CINECA Leonardo A100 64 GB.
# One Slurm task per GPU, launched by srun without torchrun, so the ranks can fill free GPUs on
# partly used nodes instead of waiting for whole ones. Code (git), env, the CUT3R repo (dataset code +
# hold-out splits) and the CUT3R scale2 data all live in $CINECA_SCRATCH; compute nodes have no
# internet, so wandb is offline. Waits for data/cut3r/READY, written once the copy from INSAIT is verified.
# Usage: sbatch -J <name> --ntasks=N --time=.. lingbot_rel_leonardo_job.sh <arm> [train_lingbot_rel.py args]
set -euo pipefail
S=$CINECA_SCRATCH
P=$S/micromamba/envs/lingbot_map/bin/python
ARM=$1; shift
OUT=$S/runs/lingbot_rel/backbone/$ARM
mkdir -p "$OUT"
until [ -f $S/data/cut3r/READY ]; do echo "[lingbot_rel_leonardo_job] $(date -u +%H:%M) waiting for data/cut3r/READY"; sleep 300; done
export WANDB_MODE=offline OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1) MASTER_PORT=29517 WORLD_SIZE=$SLURM_NTASKS
echo "[lingbot_rel_leonardo_job] $SLURM_NTASKS ranks on $SLURM_JOB_NODELIST, git $(git -C $S/lingbot-map-rel rev-parse --short HEAD)"
# CUT3R's hold-out lists are relative paths (rel/splits/...), so the trainer runs from the CUT3R repo
cd $S/CUT3R
srun --kill-on-bad-exit=1 bash -c '
    export RANK=$SLURM_PROCID
    # one GPU per task is the default binding; fall back to the node-local index if a task sees several
    case "${CUDA_VISIBLE_DEVICES:-}" in *,*) export LOCAL_RANK=$SLURM_LOCALID ;; *) export LOCAL_RANK=0 ;; esac
    exec "$0" "$@"' "$P" $S/lingbot-map-rel/scripts/relpose/train_lingbot_rel.py --out "$OUT" --cut3r $S/CUT3R \
    --data-root $S/data/cut3r "$@"
