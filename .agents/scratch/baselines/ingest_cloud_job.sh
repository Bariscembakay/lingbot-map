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
echo "[ingest] $METHOD/$SCENE n$TIER"
"$P" scripts/memory/recall_bench/dump_ingest_cloud.py --update-rule "$METHOD" \
    --scene "$SCENE" --tier "$TIER" --out "$CAMP/recallbench"
