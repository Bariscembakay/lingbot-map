#!/usr/bin/env bash
# Array task: rebuild one tap-23-only val24 clip (v5-tap23/val) with all four
# taps, same scene/clip/length/stride, so --taps other than 23 can use val24.
set -euo pipefail
read -r scene clip len stride < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" .agents/scratch/memory_eval/val16_v5_clips.txt)
out=/group/compact-3dmem/datasets/lingbot-tapcache-v5-val16-4tap/val/${scene}_c${clip}
[ -f "$out/meta.json" ] && { echo "built: $out"; exit 0; }
export TQDM_MININTERVAL=60
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"
dataset pull ScanNetpp >/dev/null 2>&1 || true
tmp=/scratch/$USER/val16_4tap/${scene}_c${clip}
mkdir -p "$tmp"
"$PY_ENV" scripts/memory/build_cache.py --scene "$scene" --clip-index "$clip" --clip-len "$len" \
    --stride "$stride" --tap-layers 4,11,17,23 ${USE_SDPA:+--use-sdpa} --out "$tmp"
mkdir -p "$(dirname "$out")"
rsync -a "$tmp/" "$out.partial/" && mv "$out.partial" "$out" && rm -rf "$tmp"
echo "done: $out"
