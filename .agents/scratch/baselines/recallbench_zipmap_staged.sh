#!/usr/bin/env bash
# ZipMap is still single-phase (no --dump-dir), so it holds a GPU through its
# own scoring. Writes to a STAGING dir: the a100 cells stay in place and are
# only replaced if all 21 regenerate, so a partial run cannot leave the table
# without a zipmap column.
set -uo pipefail
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_zipmap_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/zipmap/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
trap 'kill $! 2>/dev/null' EXIT
"$P" scripts/memory/recall_bench/run_zipmap.py --scene "$S" \
    --out "$CAMP/recallbench_zipmap_h200"
