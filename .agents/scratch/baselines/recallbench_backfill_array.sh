#!/usr/bin/env bash
# Backfill the n300/n500 recall cells the 10h a100 runs never reached.
# Array index -> (update rule, scene); the runner itself skips tiers whose
# metrics.json already exists, so a task only spends its wall on what is
# genuinely missing.
set -uo pipefail
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen staircase)
I=${SLURM_ARRAY_TASK_ID:?}
if [ "$I" -lt 6 ]; then RULE=cut3r; S=${SCENES[$I]}; else RULE=ttt3r; S=${SCENES[$((I-6))]}; fi
echo "[task $I] rule=$RULE scene=$S host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}"
exec "$HOME/lingbot-map/.agents/scratch/baselines/recallbench_cut3r_job.sh" "$RULE" "$S"
