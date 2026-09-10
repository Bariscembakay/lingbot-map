#!/usr/bin/env bash
# Build v4 tap caches for the nrgbd_recall_s2 protocol: stride 2, first 1000
# raw frames -> 500 cached frames per scene. Pinned to zone-sof1 because the
# caches are ~8 GB each and land directly on the campaign filesystem; msp3's
# /group is the small zone-local store, and shipping 57 GB cross-zone would
# cost more than the build itself.
set -uo pipefail
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
OUT=/group/compact-3dmem/campaigns/spatial_memory/eval_nrgbd_cache/${S}_500f_s2
if [ -f "$OUT/meta.json" ]; then echo "[skip] $S already cached"; exit 0; fi
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
set +u
PY="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
echo "[build] scene=$S -> $OUT"
"$PY" .agents/scratch/memory_eval/build_cache_nrgbd.py \
    --scene "$S" --stride 2 --clip-len 500 --start 0 --out "$OUT" --use-sdpa
RC=$?
echo "[done] $S rc=$RC"
exit $RC
