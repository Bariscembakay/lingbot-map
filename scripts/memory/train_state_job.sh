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
# Val clips can name a dataset the train globs do not (v5-tap23 val on a
# 96-scene run), so the extra args are scanned too; one pull per dataset, and
# no glob expansion (a mounted /data glob would expand to every clip).
set -f
for _ds in $(for _tok in $CLIPS_SPEC "$@"; do
                 case "$_tok" in /data/*) echo "$_tok" | cut -d/ -f3 ;; esac
             done | sort -u); do
    dataset pull "$_ds" >/dev/null
done
set +f

# NO gpu_keep_alive here, deliberately. It exists for inference jobs that are
# bursty on the GPU and read as idle to the deallocation reaper. This loop is the
# opposite: ~954 decoder passes plus backward per update is near-continuous GPU
# work, so the reaper is not a risk. Measured cost of keeping it: 6.79 GiB on a
# 44.42 GiB a6000, which is what OOM-killed job 753366 at ~38 GiB of real use.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# KEEPALIVE=1: opt-in reaper guard. The blanket "training never needs it"
# rationale above broke on the a100: the multi-minute clip-preload phase plus
# slower steps read as idle and job 829672 was reaped 1h in. Tiny VRAM slice.
if [ "${KEEPALIVE:-0}" = "1" ]; then
    "$PY_ENV" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.03 &
    KEEPALIVE_PID=$!
fi

# STAGE=1: read clips from a per-node /scratch copy instead of /data (see
# stage_data.sh). After the keep-alive starts: a first copy is hours of idle GPU.
if [ "${STAGE:-0}" = "1" ]; then
    source scripts/memory/stage_data.sh
    set -f   # rewrite the globs, do not expand them
    _spec=""
    for _tok in $CLIPS_SPEC; do _spec="$_spec $(stage_path "$_tok")"; done
    CLIPS_SPEC=${_spec# }
    _args=()
    for _a in "$@"; do _args+=("$(stage_path "$_a")"); done
    set -- "${_args[@]}"
    set +f
    echo "[stage] clips: $CLIPS_SPEC"
fi

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
# A --init-from on sof1's /group (e.g. the previous curriculum stage) does not
# exist on msp3, whose /group is a different filesystem: fetch it into WORK.
_args=(); _next=0
for _a in "$@"; do
    if [ "$_next" = 1 ] && [ ! -e "$_a" ]; then
        rsync -a "sof1:$_a" "$WORK/init_from.pt"
        _a="$WORK/init_from.pt"
    fi
    _next=0; [ "$_a" = "--init-from" ] && _next=1
    _args+=("$_a")
done
set -- "${_args[@]}"

( while sleep 600; do rsync -a "$WORK/" "$OUT/" 2>/dev/null || true; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; kill "${KEEPALIVE_PID:-}" 2>/dev/null || true' EXIT

# The keep-alive guards ONLY the idle clip preload. Left running, its burst
# (calibrated on the idle GPU) competes with training for the whole job --
# prime suspect in the 211 s/update A100 3D arms. Stop it at the first [data]
# line of THIS segment (train.log is appended across segments).
if [ -n "${KEEPALIVE_PID:-}" ]; then
    _n0=$( (wc -l < "$WORK/train.log") 2>/dev/null || echo 0)
    ( until tail -n +"$((_n0 + 1))" "$WORK/train.log" 2>/dev/null | grep -q '^\[data\]'; do
          sleep 30; done; kill "$KEEPALIVE_PID" 2>/dev/null ) &
fi

# shellcheck disable=SC2086
# Tee into $WORK so the periodic rsync carries the log to sof1 too. Slurm's own
# --output goes to the RUNNING zone's /home, so an msp3 job's log is unreadable
# from sof1 -- that is how run 865094 sat "warming up" for two hours when it had
# actually died in argparse after 46s.
"$PY_ENV" scripts/memory/train_state.py --clips $CLIPS_SPEC --out "$WORK" "$@" \
    2>&1 | tee -a "$WORK/train.log"

rsync -a "$WORK/" "$OUT/"
echo "[train_state_job] shipped $WORK -> $OUT"
