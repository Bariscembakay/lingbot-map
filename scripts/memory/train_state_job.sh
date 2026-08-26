#!/usr/bin/env bash
# One training run of the CUT3R-state recall model.
# Usage: train_state_job.sh <out_dir> <clip_glob_or_dir> [extra train_state.py args]
set -euo pipefail
OUT="$1"; shift
CLIPS_SPEC="$1"; shift

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"

export TQDM_MININTERVAL=30
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh

# Never `micromamba run` in a job: its lock lives on CephFS and a long job holds
# it for its whole lifetime, wedging every later invocation on the node.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"

# The loop is bursty on the GPU (long CPU stretches building raymaps and GT
# pointmaps), which reads as idle to the deallocation monitor. 0.05 rather than
# the 0.4 default because the run needs its own VRAM.
"$PY_ENV" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
trap 'kill $! 2>/dev/null' EXIT

# Work on node-local /scratch, ship to sof1's /group at the end -- /group is the
# system of record and /scratch is wiped.
WORK="/scratch/$USER/train_state/$(basename "$OUT")"
mkdir -p "$WORK"

# shellcheck disable=SC2086
"$PY_ENV" scripts/memory/train_state.py --clips $CLIPS_SPEC --out "$WORK" "$@"

mkdir -p "$OUT"
rsync -a "$WORK/" "$OUT/"
echo "[train_state_job] shipped $WORK -> $OUT"
