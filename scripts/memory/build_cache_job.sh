#!/usr/bin/env bash
# sbatch payload for one (scene, clip) cache build.
#
# A real script file rather than `sbatch --wrap`: --wrap runs its body under
# /bin/sh (dash), where `source` does not exist, so the env bootstrap died with
# exit 127 before doing anything (job 736831).
set -euo pipefail

SCANNETPP_ROOT="$1"; SCENE="$2"; CLIP="$3"; CLIP_LEN="$4"; STRIDE="$5"; OUT="$6"

mkdir -p "$OUT"
export TQDM_MININTERVAL=30   # a per-frame progress bar makes the log unreadable
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh

# Use the interpreter directly, never `micromamba run`. `micromamba run` takes a
# lock under ~/.cache/mamba/proc, which lives on /home == CephFS, where locking is
# unreliable; a long-running job holds it for its whole lifetime and every later
# invocation blocks. Two timing jobs died at their walltime waiting on the lock
# held by two training arms.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"


if [ "${GPU_KEEP_ALIVE:-1}" != "0" ]; then
    "$PY_ENV" \
        .agents/scratch/insait_cluster_files/gpu_keep_alive.py \
        "${GPU_KEEP_ALIVE_FRACTION:-0.05}" &
    trap 'kill $! 2>/dev/null' EXIT
fi

"$PY_ENV" scripts/memory/build_cache.py \
    --scannetpp-root "$SCANNETPP_ROOT" \
    --scene "$SCENE" \
    --clip-index "$CLIP" \
    --clip-len "$CLIP_LEN" \
    --stride "$STRIDE" \
    --out "$OUT"
