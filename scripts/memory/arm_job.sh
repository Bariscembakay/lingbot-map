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

# Small kernels plus heavy cache I/O read as idle to the GPU reaper; 0.05 rather
# than the 0.4 default because the job needs its own VRAM.
micromamba run -n lingbot_map python \
    .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
trap 'kill $! 2>/dev/null' EXIT

micromamba run -n lingbot_map python scripts/memory/train.py \
    --cache "$CACHE" --out "$OUT" --arm "$ARM" --heads "$HEADS" \
    --refine-taps "$TAPS" $FLAGS
