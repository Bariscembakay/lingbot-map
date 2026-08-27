#!/usr/bin/env bash
# Register a completed tap cache with the cluster dataset registry.
#
# Must run ON the node that built it: /scratch is per-node local disk, so the
# tree is invisible from anywhere else until the registry has replicated it.
#
# Usage: register_cache_dataset.sh <cache_root> <dataset_name>
set -euo pipefail
ROOT="$1"; NAME="$2"

# Refuse to register an incomplete cache. `meta.json` is written LAST by
# build_cache.py, so its absence means that clip did not finish -- and
# build_cache pre-allocates a full-size sparse taps.npy before the first forward,
# so a failed clip leaves a 5 GB file that looks like data. A previous
# `dataset create` was moments from publishing 12 such shells as if they were
# real: "Failed to read file ./train/086f09d6e3_c0/taps.npy, creating empty file".
missing=0
for d in "$ROOT"/*/*_c*/; do
    [ -d "$d" ] || continue
    if [ ! -f "$d/meta.json" ]; then
        echo "INCOMPLETE: $d (no meta.json)" >&2
        missing=$((missing + 1))
    fi
done
n_ok=$(find "$ROOT" -name meta.json | wc -l)
echo "[register] $n_ok complete clips, $missing incomplete"
if [ "$missing" -gt 0 ]; then
    echo "refusing to register: clear or rebuild the incomplete clips first" >&2
    exit 1
fi
[ "$n_ok" -gt 0 ] || { echo "refusing: no complete clips found" >&2; exit 1; }

du -sh "$ROOT"
dataset create "$NAME" "$ROOT"
echo "[register] done -- pull it in any zone with: dataset pull $NAME"
