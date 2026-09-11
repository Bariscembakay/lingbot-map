#!/usr/bin/env bash
# pred_* -> recalled_*, ingest_* -> predicted_*. Idempotent: safe to re-run
# while scoring tasks that still carry the old code are finishing.
set -uo pipefail
R=${1:-/group/compact-3dmem/campaigns/spatial_memory/recallbench}
n=0
for d in "$R"/*/*_n*/; do
  [ -d "$d" ] || continue
  for pair in "pred_cloud_rgb.ply:recalled_cloud_rgb.ply" \
              "pred_cloud_err.ply:recalled_cloud_err.ply" \
              "ingest_cloud_rgb.ply:predicted_cloud_rgb.ply" \
              "ingest_cloud_err.ply:predicted_cloud_err.ply" \
              "ingest_metrics.json:predicted_metrics.json"; do
    old=${pair%%:*}; new=${pair##*:}
    if [ -f "$d$old" ]; then
      if [ -f "$d$new" ]; then rm -f "$d$old"; else mv "$d$old" "$d$new"; fi
      n=$((n+1))
    fi
  done
done
echo "renamed/reconciled $n file(s) under $R"
