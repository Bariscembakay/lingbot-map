#!/usr/bin/env bash
# ZipMap GPU phase only: ingest + query + dump. Minutes, not >10h.
set -uo pipefail
S=${1:?scene}; CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_zipmap_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/zipmap/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.10 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/recall_bench/run_zipmap.py --scene "$S" \
    --out "$CAMP/recallbench" --dump-dir "$CAMP/recallbench_dumps"
