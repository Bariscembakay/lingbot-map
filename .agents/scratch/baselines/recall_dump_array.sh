#!/usr/bin/env bash
# GPU phase: ingest + query only, dump raw predictions. Minutes per cell.
set -uo pipefail
SPEC=$1
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:?} + 1))p" "$SPEC")
IFS=',' read -r METHOD SCENE <<< "$LINE"
CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
echo "[dump] $METHOD/$SCENE on $(hostname)"
"$P" scripts/memory/recall_bench/run_cut3r_family.py --update-rule "$METHOD" \
    --scene "$SCENE" --out "$CAMP/recallbench" \
    --dump-dir "$CAMP/recallbench_dumps"
