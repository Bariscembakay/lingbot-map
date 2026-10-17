#!/usr/bin/env bash
# Recall96 eval, model A (ours): StateMemory smallread_lingbot ckpt, 8 val scenes.
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail
PY="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
OUTD=/scratch/$USER/recall96
mkdir -p "$OUTD"
CKPT=/group/compact-3dmem/campaigns/spatial_memory/scenes32_96f_b4_write4_read2_lingbothead/last.pt
rc=0
for c in /data/lingbot-tapcache-v4-40/val_top/*_c0 /data/lingbot-tapcache-v4-40/val_median/*_c0; do
    scene=$(basename "$c"); scene=${scene%_c0}
    echo "=== ours $scene ($c) $(date +%T) ==="
    "$PY" .agents/scratch/memory_eval/run_ours_recall.py \
        --ckpt "$CKPT" --clip "$c" --frames 96 \
        --out "$OUTD/ours_${scene}.npz" || { echo "FAILED ours $scene"; rc=1; }
done
ls -la "$OUTD"
exit $rc
