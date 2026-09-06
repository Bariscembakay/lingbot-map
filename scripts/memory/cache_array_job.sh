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

if [ -f "$out/meta.json" ]; then
    echo "already built, skipping"; exit 0
fi

# Adaptive stride: STRIDE is the MAXIMUM. A 320-frame clip at stride 20 needs
# 6400 source frames; 209 of the 864 big-run scenes are shorter, so the stride
# shrinks to the largest value that fits (all scenes have >=1600 frames, so it
# never drops below 5). meta.json records the stride actually used.
NPOSE=$(grep -c '"frame_' "$ROOT/data/$scene/iphone/pose_intrinsic_imu.json")
MAXS=$(( (NPOSE - 1) / (CLIP_LEN - 1) ))
EFF_STRIDE=$(( MAXS < STRIDE ? MAXS : STRIDE ))
if [ "$EFF_STRIDE" -lt 2 ]; then
    echo "scene too short even for stride 2 ($NPOSE frames), skipping"; exit 0
fi
echo "poses $NPOSE -> stride $EFF_STRIDE"

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
# CACHE_EXTRA_ARGS: optional extra build_cache flags (e.g. "--tap-layers 23"
# for the slim single-tap caches of the big run). sbatch propagates the env.
"$PY_ENV" scripts/memory/build_cache.py \
    --scannetpp-root "$ROOT" --scene "$scene" --clip-index "$clip" \
    --clip-len "$CLIP_LEN" --stride "$EFF_STRIDE" --out "$out" --use-sdpa \
    ${CACHE_EXTRA_ARGS:-}
