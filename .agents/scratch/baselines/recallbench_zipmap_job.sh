#!/usr/bin/env bash
# recall bench, ZipMap state-query: $@ = scenes
set -uo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/zipmap/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
trap 'kill $! 2>/dev/null' EXIT
cd "$HOME/lingbot-map"
RC=0
for SCENE in "$@"; do
  "$P" scripts/memory/recall_bench/run_zipmap.py \
    --scene "$SCENE" --out /group/compact-3dmem/campaigns/spatial_memory/recallbench \
    || RC=1
done
exit $RC
