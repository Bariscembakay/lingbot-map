#!/usr/bin/env bash
# Fused-eval the fullbptt DPT checkpoint (independently decoded world branch)
# against the same GT + CUT3R predictions as the recall96 run.
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail
PY="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
OUTD=/scratch/$USER/recall96
CKPT=/group/compact-3dmem/campaigns/spatial_memory/scenes32_16f_b4_fullbptt/last.pt
for c in /data/lingbot-tapcache-v4-40/val_top/*_c0 /data/lingbot-tapcache-v4-40/val_median/*_c0; do
  scene=$(basename "$c"); scene=${scene%_c0}
  echo "=== fullbptt $scene $(date +%T) ==="
  "$PY" .agents/scratch/memory_eval/run_ours_recall.py \
      --ckpt "$CKPT" --clip "$c" --frames 96 \
      --out "$OUTD/fullbptt/ours_${scene}.npz" || echo "FAILED $scene"
  ln -sf "$OUTD/cut3r_${scene}.npz" "$OUTD/fullbptt/cut3r_${scene}.npz"
done
"$PY" .agents/scratch/memory_eval/score_recall96_sim3.py \
  --recall-dir "$OUTD/fullbptt" \
  --caches /data/lingbot-tapcache-v4-40/val_top/*_c0 /data/lingbot-tapcache-v4-40/val_median/*_c0 \
  --out .agents/scratch/memory_eval/recall96/summary_fullbptt.json
