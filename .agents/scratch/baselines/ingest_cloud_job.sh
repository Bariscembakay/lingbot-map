#!/usr/bin/env bash
# Ingestion ("predicted") clouds for the predicted-vs-recalled comparison.
set -uo pipefail
SPEC=$1
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:?} + 1))p" "$SPEC")
IFS=',' read -r METHOD SCENE TIER <<< "$LINE"
CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
# The GPU phase is ~60s; the KD-tree scoring that follows is hours of pure CPU,
# so the idle-GPU reaper kills the job mid-scoring (observed: job 884079 killed
# after 3190s idle). Hold the GPU busy for the job's lifetime.
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.10 &
KA=$!
trap 'kill $KA 2>/dev/null' EXIT

echo "[ingest] $METHOD/$SCENE n$TIER"
"$P" scripts/memory/recall_bench/dump_ingest_cloud.py --update-rule "$METHOD" \
    --scene "$SCENE" --tier "$TIER" --out "$CAMP/recallbench"
