#!/usr/bin/env bash
# Per-tap mean-ablation of lingbot's depth and camera heads on all 40 4-tap clips.
set -euo pipefail
export TQDM_MININTERVAL=60
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
OUT=/group/compact-3dmem/campaigns/spatial_memory/lingbot_tap_ablation
mkdir -p "$OUT"
"$PY_ENV" scripts/memory/lingbot_tap_ablation.py --out "$OUT/results_f96.json" /data/lingbot-tapcache-v4-40/*/*
