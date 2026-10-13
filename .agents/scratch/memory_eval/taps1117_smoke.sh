#!/usr/bin/env bash
set -euo pipefail
G=/group/compact-3dmem/campaigns/spatial_memory
OUT=/scratch/$USER/smoke_taps1117
rm -rf "$OUT" /scratch/$USER/train_state/smoke_taps1117
exec scripts/memory/train_state_job.sh "$OUT" '/data/lingbot-tapcache-v4-40/train/03*' \
    --head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head --taps 11,17 \
    --init-from $G/scenes96_96f_b4_write8oneway_read8_tap23_FRESHDPT_depth_scratch/best.pt \
    --val-clips /data/lingbot-tapcache-v4-40/val_median/0a7cc12c0e_c0 --val-every 2 --val-frames 32 \
    --updates 3 --patience 0 --frames 16 --max-frames 32 --batch 2 --lr 1e-4 --warmup 300 \
    --n-past 4 --probe-every 1 --tbptt 8 --clip-store cpu --save-every 100000 --viz-every 100000 --wandb off --log-every 1
