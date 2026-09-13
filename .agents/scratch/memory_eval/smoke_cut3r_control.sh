#!/usr/bin/env bash
# Smoke for the CUT3R control arm, all inside ONE srun on a gcp-eu1 rtx6000:
#   (c) injected-cache vs RGB equivalence check
#   (b) 2 updates at the real batch-4 config with val forced every step
#   (d) resume round-trip: 4 updates save-every 2, then rerun with --resume auto
# Prereq: smoke enc cache for TRAIN_CLIP and VAL_CLIP already built (smoke a).
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +e +u +o pipefail

PY="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
ENC=/data/cut3r-enc-cache-v1
TRAIN_CLIP=/data/lingbot-tapcache-v4-40/train/036bce3393_c0
VAL_CLIP=/data/lingbot-tapcache-v4-40/val_top/210f741378_c0
SMOKE=/scratch/$USER/cut3r_ctrl_smoke
rm -rf "$SMOKE"; mkdir -p "$SMOKE"

dataset pull lingbot-tapcache-v4-40 >/dev/null && dataset pull ScanNetpp >/dev/null

echo "=== (c) equivalence check ==="
"$PY" scripts/memory/train_cut3r_control.py --equiv-check \
    --clips "$TRAIN_CLIP" --enc-cache "$ENC" --out "$SMOKE/equiv" --wandb off
echo "equiv exit: $?"

echo "=== (b) 2 updates, batch 4 (same clip x4), val forced ==="
# The same clip listed 4x gives a true batch-4 peak/speed reading with only
# one smoke cache built; data variety is irrelevant to memory and s/step.
"$PY" scripts/memory/train_cut3r_control.py \
    --clips "$TRAIN_CLIP" "$TRAIN_CLIP" "$TRAIN_CLIP" "$TRAIN_CLIP" \
    --val-clips "$VAL_CLIP" --enc-cache "$ENC" \
    --out "$SMOKE/b" --updates 2 --val-every 1 --log-every 1 --wandb off
echo "train-smoke exit: $?"

echo "=== (d) resume round-trip ==="
"$PY" scripts/memory/train_cut3r_control.py \
    --clips "$TRAIN_CLIP" --enc-cache "$ENC" \
    --out "$SMOKE/d" --updates 4 --save-every 2 --log-every 1 --wandb off
echo "first-leg exit: $?"
"$PY" scripts/memory/train_cut3r_control.py \
    --clips "$TRAIN_CLIP" --enc-cache "$ENC" \
    --out "$SMOKE/d" --updates 6 --save-every 2 --log-every 1 --wandb off \
    --resume auto
echo "resume-leg exit: $?"
echo "=== smoke done ==="
