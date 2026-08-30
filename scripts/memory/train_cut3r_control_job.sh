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
# --chdir already put us in the repo; SLURM_SUBMIT_DIR is only a fallback and
# is wrong whenever sbatch ran from some other directory.
if [ ! -f .agents/scratch/insait_cluster_files/setup_cut3r_env.sh ]; then
    cd "${SLURM_SUBMIT_DIR:-$PWD}"
fi
[ -f .agents/scratch/insait_cluster_files/setup_cut3r_env.sh ] || {
    echo "not in the repo root: $PWD (pass --chdir to sbatch)" >&2; exit 1; }

export TQDM_MININTERVAL=30
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh

# Never `micromamba run` in a job: its lock lives on CephFS and a long job holds
# it for its whole lifetime, wedging every later invocation on the node.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"

# /data is an autofs registry mount that resolves only after `dataset pull`
# on the running node (idempotent; replicates cross-zone when needed).
# CLIPS_SPEC may name clips from several /data datasets; pull each once.
for _tok in $CLIPS_SPEC; do
    case "$_tok" in
        /data/*) dataset pull "$(echo "$_tok" | cut -d/ -f3)" >/dev/null ;;
    esac
done
# The control also needs the frozen-encoder token cache in-zone.
dataset pull cut3r-enc-cache-v1 >/dev/null

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
# Pre-seed from OUT so `--resume auto` survives requeues onto fresh nodes
# (rsync also accepts a remote "sof1:" OUT). No-op for a brand-new run.
rsync -a "$OUT/" "$WORK/" 2>/dev/null || true
# OUT may be a remote rsync destination ("sof1:/group/...") when running on
# msp3: results must ship to sof1's /group, never land on msp3's 300G one.
case "$OUT" in *:*) : ;; *) mkdir -p "$OUT" ;; esac

# A walltime kill would strand everything on this node's /scratch (bash does not
# run EXIT traps when SIGKILLed after KillWait). Mirror instead: history.json and
# last.pt are rewritten continuously, so a 10-min sync loses at most 10 min.
( while sleep 600; do rsync -a "$WORK/" "$OUT/" 2>/dev/null || true; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true' EXIT

# shellcheck disable=SC2086
"$PY_ENV" scripts/memory/train_cut3r_control.py --clips $CLIPS_SPEC --out "$WORK" "$@"

rsync -a "$WORK/" "$OUT/"
echo "[train_state_job] shipped $WORK -> $OUT"
