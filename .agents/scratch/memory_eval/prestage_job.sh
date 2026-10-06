#!/usr/bin/env bash
# CPU-only: copy datasets into this node's /scratch datacache ahead of the
# training segments that will read them (STAGE=1), then verify the copy.
# Usage: prestage_job.sh NAME [NAME ...]
set -euo pipefail
cd /home/baris_bakay/lingbot-map
source scripts/memory/stage_data.sh
for name in "$@"; do
    dataset pull "$name" >/dev/null
    t0=$(date +%s)
    dst=$(stage_dataset "$name")
    src_n=$(find "/data/$name" -type f | wc -l); dst_n=$(find "$dst" -type f ! -name .complete | wc -l)
    src_b=$(du -sb --apparent-size "/data/$name" | cut -f1); dst_b=$(du -sb --apparent-size --exclude=.complete "$dst" | cut -f1)
    echo "[prestage] $name: $(( $(date +%s) - t0 ))s  files $src_n -> $dst_n  bytes $src_b -> $dst_b  $([ "$src_n" = "$dst_n" ] && [ "$src_b" = "$dst_b" ] && echo OK || echo MISMATCH)"
done
df -h "$STAGE_ROOT" | tail -1
