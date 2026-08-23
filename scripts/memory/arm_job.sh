#!/usr/bin/env bash
set -euo pipefail
CACHE="$1"; OUT="$2"; ARM="$3"; TAPS="$4"; HEADS="$5"; FLAGS="$6"

mkdir -p "$OUT"
export TQDM_MININTERVAL=30
# wandb writes run data and artifacts; /home is repos and scripts only.
export WANDB_DIR="${WANDB_DIR:-/scratch/$USER/wandb}"
mkdir -p "$WANDB_DIR"
# Credentials are per zone and deliberately unsynced, so ~/.netrc may not exist
# here. Offline still records everything; `wandb sync` uploads it later.
[ -f "$HOME/.netrc" ] || export WANDB_MODE="${WANDB_MODE:-offline}"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh

# Use the interpreter directly, never `micromamba run`. `micromamba run` takes a
# lock under ~/.cache/mamba/proc, which lives on /home == CephFS, where locking is
# unreliable; a long-running job holds it for its whole lifetime and every later
# invocation blocks. Two timing jobs died at their walltime waiting on the lock
# held by two training arms.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"


# /scratch is per-node and the env is rebuilt per node, so installing wandb on a
# login node does not make it available here. Idempotent, and train.py survives
# without it either way.
"$PY_ENV" -c "import wandb" 2>/dev/null   || "$PY_ENV" -m pip install -q wandb

# Small kernels plus heavy cache I/O read as idle to the GPU reaper; 0.05 rather
# than the 0.4 default because the job needs its own VRAM.
"$PY_ENV" \
    .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
trap 'kill $! 2>/dev/null' EXIT

"$PY_ENV" scripts/memory/train.py \
    --cache "$CACHE" --out "$OUT" --arm "$ARM" --heads "$HEADS" \
    --refine-taps "$TAPS" $FLAGS
