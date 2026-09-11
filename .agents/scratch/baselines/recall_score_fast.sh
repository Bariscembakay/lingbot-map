#!/usr/bin/env bash
# Deadline mode: score only the modes named in $MODES (default selfpose).
# gtpose is recoverable later from the same dump.
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
EXTRA=""
[ "$METHOD" = "ours" ] && EXTRA="--posed --modes gtpose"
[ -z "$EXTRA" ] && EXTRA="--modes ${MODES:-selfpose}"
echo "[score-fast] $METHOD/$SCENE n$TIER $EXTRA"
"$P" scripts/memory/recall_bench/run_recall_score.py \
    --method "$METHOD" --scene "$SCENE" --tier "$TIER" $EXTRA \
    --dump-dir "$CAMP/recallbench_dumps" --out "$CAMP/recallbench"
