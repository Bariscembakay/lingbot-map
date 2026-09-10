#!/usr/bin/env bash
# CPU phase: no GPU requested at all, so it schedules on the idle cores the
# GPU nodes carry while every GPU on them is busy.
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
echo "[score] $METHOD/$SCENE n$TIER on $(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
"$P" scripts/memory/recall_bench/run_recall_score.py \
    --method "$METHOD" --scene "$SCENE" --tier "$TIER" \
    --dump-dir "$CAMP/recallbench_dumps" --out "$CAMP/recallbench"
