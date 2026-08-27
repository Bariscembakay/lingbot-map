#!/usr/bin/env bash
# One training run of the CUT3R-state recall model.
# Usage: train_state_job.sh <out_dir> <clip_glob_or_dir> [extra train_state.py args]
set -euo pipefail
OUT="$1"; shift
CLIPS_SPEC="$1"; shift

# Slurm copies the batch script to /var/lib/slurm/slurmd/job<id>/slurm_script, so
# resolving the repo from ${BASH_SOURCE[0]} lands in Slurm's spool, not here.
# Every other job script in this repo relies on --chdir instead; do the same, and
# fail loudly rather than silently running from the wrong tree.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
[ -f .agents/scratch/insait_cluster_files/setup_cut3r_env.sh ] || {
    echo "not in the repo root: $PWD (pass --chdir to sbatch)" >&2; exit 1; }

export TQDM_MININTERVAL=30
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh

# Never `micromamba run` in a job: its lock lives on CephFS and a long job holds
# it for its whole lifetime, wedging every later invocation on the node.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"

# NO gpu_keep_alive here, deliberately. It exists for inference jobs that are
# bursty on the GPU and read as idle to the deallocation reaper. This loop is the
# opposite: ~954 decoder passes plus backward per update is near-continuous GPU
# work, so the reaper is not a risk. Measured cost of keeping it: 6.79 GiB on a
# 44.42 GiB a6000, which is what OOM-killed job 753366 at ~38 GiB of real use.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Work on node-local /scratch, ship to sof1's /group at the end -- /group is the
# system of record and /scratch is wiped.
WORK="/scratch/$USER/train_state/$(basename "$OUT")"
mkdir -p "$WORK"

# shellcheck disable=SC2086
"$PY_ENV" scripts/memory/train_state.py --clips $CLIPS_SPEC --out "$WORK" "$@"

mkdir -p "$OUT"
rsync -a "$WORK/" "$OUT/"
echo "[train_state_job] shipped $WORK -> $OUT"
