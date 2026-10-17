#!/usr/bin/env bash
# Re-register the four datasets whose registry copy is INCOMPLETE, so their
# /group copies can be deleted too.
#   nrgbd          registry missing eval_gt (29G) + eval_gt_cache (11G)
#   7-scenes       registry missing eval_gt (23G), used by seven_scenes.py
#   Long3D         registry missing one top-level entry
#   kitti_odometry sample check inconclusive
# Runs as a compute job: mksquashfs on a multi-hundred-GB tree is OOM-killed by
# the login-node memory cgroup (happened twice on this cluster).
set -uo pipefail
G=/group/compact-3dmem/datasets
declare -A M=( [nrgbd]=NRGBD [7-scenes]=7-Scenes [Long3D]=Long3D [kitti_odometry]=kitti_odometry )
for d in "${!M[@]}"; do
  reg=${M[$d]}
  [ -d "$G/$d" ] || { echo "[skip] $G/$d missing"; continue; }
  echo "=== $d -> $reg ($(du -sh $G/$d 2>/dev/null | cut -f1)) ==="
  dataset create "$reg" "$G/$d" --replace && echo "[ok] $reg re-registered" \
    || echo "[FAIL] $reg"
done
echo "=== verification ==="
for d in "${!M[@]}"; do
  reg=${M[$d]}
  dataset pull "$reg" >/dev/null 2>&1
  gc=$(ls "$G/$d" 2>/dev/null | wc -l); mc=$(ls "/data/$reg" 2>/dev/null | wc -l)
  echo "  $d: group=$gc registry=$mc $([ "$gc" = "$mc" ] && echo MATCH || echo MISMATCH)"
done
