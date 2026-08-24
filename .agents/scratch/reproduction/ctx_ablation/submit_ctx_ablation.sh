#!/usr/bin/env bash
# Submits the trajectory-memory context-token ablation as a dependency chain on
# gcp-eu1 A100s (that zone shares sof1's /group and /home outright, so the
# workspace, datasets and checkpoint symlink all resolve with no zone handling).
#
#   stage 0  prepare       1 CPU-only job, both datasets
#   stage 1  run           10 jobs (2 datasets x 5 arms), 1 A100 each, in parallel
#   stage 2  evaluate      1 CPU-only job per dataset, gated on that dataset's 5 arms
#
# Evaluate is deliberately GPU-less: its ICP/KD-tree work is CPU-bound for hours,
# and a GPU-less job is also not a target for the idle-GPU reaper.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_DIR"
LOG_DIR="$REPO_DIR/.agents/logs/ctx_ablation"
mkdir -p "$LOG_DIR"

ZONE="zone-gcp-eu1"
GPU="a100-80g"
SUBMIT="bash .agents/scratch/insait_cluster_files/submit_sof1.sh"

METHODS=(lingbot_map lingbot_map_traj6 lingbot_map_traj_noreg
         lingbot_map_traj_nocam lingbot_map_traj_noscale)

# dataset-key : config
declare -A CFG=( [oxford]=configs/ctx_ablation/oxford_ctx_ablation.yaml
                 [neural_rgbd]=configs/ctx_ablation/neural_rgbd_ctx_ablation.yaml )
declare -A EVAL_TIME=( [oxford]=2:00:00 [neural_rgbd]=8:00:00 )

# ── stage 0: prepare ───────────────────────────────────────────────────────────
PREP=$(sbatch --parsable \
  --job-name=ctxa_prep --partition=batch --constraint="$ZONE" \
  --cpus-per-task=16 --mem=96G --time=1:00:00 \
  --output="$LOG_DIR/00_prepare_%j.log" \
  --wrap="GPU_KEEP_ALIVE=0 $SUBMIT bash .agents/scratch/reproduction/run_stage.sh prepare ${CFG[oxford]} ${CFG[neural_rgbd]}")
echo "stage 0  prepare              -> $PREP"

# ── stage 1: one job per (dataset, arm) ───────────────────────────────────────
declare -A RUN_IDS
for ds in oxford neural_rgbd; do
    ids=""
    for m in "${METHODS[@]}"; do
        jid=$(sbatch --parsable \
          --job-name="ctxa_${ds}_${m#lingbot_map}" \
          --partition=batch --constraint="$ZONE" \
          --gpus=$GPU:1 --cpus-per-task=12 --mem=64G --time=3:00:00 \
          --dependency=afterok:$PREP --kill-on-invalid-dep=yes \
          --output="$LOG_DIR/10_run_${ds}_${m}_%j.log" \
          --wrap="GPU_KEEP_ALIVE_FRACTION=0.10 $SUBMIT bash .agents/scratch/reproduction/run_arm.sh ${CFG[$ds]} $ds $m")
        ids="${ids:+$ids:}$jid"
        echo "stage 1  run $ds / $m -> $jid"
    done
    RUN_IDS[$ds]="$ids"
done

# ── stage 2: evaluate, gated on that dataset's five arms ──────────────────────
for ds in oxford neural_rgbd; do
    # afterany, not afterok: one failed arm should not block scoring the rest.
    jid=$(sbatch --parsable \
      --job-name="ctxa_eval_${ds}" --partition=batch --constraint="$ZONE" \
      --cpus-per-task=32 --mem=192G --time="${EVAL_TIME[$ds]}" \
      --dependency=afterany:"${RUN_IDS[$ds]}" \
      --output="$LOG_DIR/20_eval_${ds}_%j.log" \
      --wrap="GPU_KEEP_ALIVE=0 $SUBMIT bash .agents/scratch/reproduction/run_stage.sh evaluate ${CFG[$ds]}")
    echo "stage 2  evaluate $ds        -> $jid"
done
