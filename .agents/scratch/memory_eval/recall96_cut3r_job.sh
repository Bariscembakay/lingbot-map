#!/usr/bin/env bash
# Recall96 eval, model B: published CUT3R 512-dpt zero-shot, 8 val scenes.
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail
PY="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
dataset pull ScanNetpp >/dev/null
OUTD=/scratch/$USER/recall96
mkdir -p "$OUTD"
rc=0
for c in /data/lingbot-tapcache-v4-40/val_top/*_c0 /data/lingbot-tapcache-v4-40/val_median/*_c0; do
    scene=$(basename "$c"); scene=${scene%_c0}
    echo "=== cut3r $scene ($c) $(date +%T) ==="
    "$PY" .agents/scratch/memory_eval/run_cut3r_recall.py \
        --cache "$c" --frames 96 \
        --out "$OUTD/cut3r_${scene}.npz" || { echo "FAILED cut3r $scene"; rc=1; }
done
ls -la "$OUTD"
exit $rc
