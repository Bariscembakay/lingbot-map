#!/usr/bin/env bash
# One array task: pick the (scene, clip) on line $SLURM_ARRAY_TASK_ID and build it.
set -euo pipefail
LIST="$1"; ROOT="$2"; CLIP_LEN="$3"; STRIDE="$4"; OUT_ROOT="$5"

# list lines are "scene:clip:split"; split becomes a subdirectory so the arm jobs
# can point --cache and --val-cache at disjoint roots.
spec=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
IFS=':' read -r scene clip split <<< "$spec"
out="$OUT_ROOT/${split}/${scene}_c${clip}"
echo "task $SLURM_ARRAY_TASK_ID -> $scene clip $clip split $split -> $out"

mkdir -p "$out"
export TQDM_MININTERVAL=30
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh

# Use the interpreter directly, never `micromamba run`. `micromamba run` takes a
# lock under ~/.cache/mamba/proc, which lives on /home == CephFS, where locking is
# unreliable; a long-running job holds it for its whole lifetime and every later
# invocation blocks. Two timing jobs died at their walltime waiting on the lock
# held by two training arms.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"


"$PY_ENV" \
    .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
trap 'kill $! 2>/dev/null' EXIT

# --use-sdpa, not FlashInfer. FlashInfer JIT-compiles its kernels on first use,
# and on nodes with glibc >= 2.41 that fails against CUDA 12.8's headers:
#   mathcalls.h(79): exception specification is incompatible ... "cospi"
# glibc declares cospi/sinpi noexcept(true) and CUDA's crt/math_functions.h
# disagrees. All 32 tasks of array 751246 died this way. SDPA is a supported
# path in gct_stream_window (use_flashinfer=not use_sdpa) and needs no nvcc at
# all, so it sidesteps the clash rather than fighting the toolchain.
"$PY_ENV" scripts/memory/build_cache.py \
    --scannetpp-root "$ROOT" --scene "$scene" --clip-index "$clip" \
    --clip-len "$CLIP_LEN" --stride "$STRIDE" --out "$out" --use-sdpa
