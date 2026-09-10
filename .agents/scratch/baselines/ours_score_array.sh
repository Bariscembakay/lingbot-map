#!/usr/bin/env bash
# CPU phase for our checkpoint. --posed: no Sim(3) is fitted.
set -uo pipefail
SPEC=$1
LINE=$(sed -n "$((${SLURM_ARRAY_TASK_ID:?} + 1))p" "$SPEC")
IFS=',' read -r SCENE TIER <<< "$LINE"
CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" scripts/memory/recall_bench/run_recall_score.py --method ours --posed \
    --scene "$SCENE" --tier "$TIER" \
    --dump-dir "$CAMP/recallbench_dumps" --out "$CAMP/recallbench"
