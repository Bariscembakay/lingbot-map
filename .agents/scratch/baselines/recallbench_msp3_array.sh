#!/usr/bin/env bash
# Recall bench on msp3 H200. msp3's /group/compact-3dmem is the small (300G)
# zone-local filesystem, NOT the campaign record -- so results are built on
# node-local /scratch and shipped to sof1's /group, which stays the system of
# record.
#
# Array index -> (method, scene), read from the spec file given as $1: one
# "method,scene" per line. A bash array cannot be passed through --export, and a
# file also leaves an auditable record of exactly what each task was asked to do.
set -uo pipefail
SPECFILE="${1:?usage: recallbench_msp3_array.sh <specfile>}"
SOF1=sof1:/group/compact-3dmem/campaigns/spatial_memory/recallbench
WORK=/scratch/$USER/recallbench

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SPECFILE")
[ -n "$LINE" ] || { echo "no spec for task $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
IFS=',' read -r METHOD SCENE <<< "$LINE"
echo "[task $SLURM_ARRAY_TASK_ID] method=$METHOD scene=$SCENE host=$(hostname)"

mkdir -p "$WORK"
# Seed ONLY the metrics.json files (a few KB each): the runner's per-tier resume
# keys off them, so without this a fresh /scratch would recompute n100/n300 that
# are already finished on sof1. The heavy PLYs are deliberately not pulled back.
rsync -am --include='*/' --include='metrics.json' --exclude='*' \
    "$SOF1/" "$WORK/" 2>/dev/null || true
echo "[seed] $(find "$WORK" -name metrics.json | wc -l) existing cells known"

case "$METHOD" in
  zipmap) source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_env.sh"
          P="$MAMBA_ROOT_PREFIX/envs/zipmap/bin/python"
          RUNNER=(scripts/memory/recall_bench/run_zipmap.py) ;;
  *)      source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"
          P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
          RUNNER=(scripts/memory/recall_bench/run_cut3r_family.py --update-rule "$METHOD") ;;
esac

dataset pull NRGBD >/dev/null 2>&1 || true
# The n500 cells are CPU-bound for hours (KDTree over ~30M fused points), which
# reads as idle to the GPU reaper.
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
KA=$!
# Ship whatever finished even if walltime kills us mid-cell.
ship() { rsync -a "$WORK/" "$SOF1/" 2>/dev/null || true; }
trap 'kill $KA 2>/dev/null; ship' EXIT

cd "$HOME/lingbot-map"
( while sleep 900; do ship; done ) & SYNC=$!
trap 'kill $KA $SYNC 2>/dev/null; ship' EXIT

"$P" "${RUNNER[@]}" --scene "$SCENE" --out "$WORK"
RC=$?
kill $SYNC 2>/dev/null
ship
echo "[done] rc=$RC shipped $WORK -> $SOF1"
exit $RC
