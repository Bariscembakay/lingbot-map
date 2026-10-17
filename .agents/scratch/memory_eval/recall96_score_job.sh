#!/usr/bin/env bash
# Recall96 scoring: per-view L21 + fused Sim(3)+ICP, both models, 8 scenes.
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail
PY="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
"$PY" .agents/scratch/memory_eval/score_recall96_sim3.py \
    --recall-dir /scratch/$USER/recall96 \
    --caches /data/lingbot-tapcache-v4-40/val_top/*_c0 \
             /data/lingbot-tapcache-v4-40/val_median/*_c0 \
    --out .agents/scratch/memory_eval/recall96/summary.json \
    2>&1 | tee .agents/scratch/memory_eval/recall96/score_table.txt
