#!/usr/bin/env bash
# Smoke: frozen BIGRUN memory + fully unfrozen DPT (LoRA merged), fast recipe,
# DDP 4 GPUs, worst-case N=160, async validator.
set -euo pipefail
G=/group/compact-3dmem/campaigns/spatial_memory
VA=$(ls -d /data/lingbot-tapcache-v4-40/val_top/* | head -2 | tr '\n' ' ')
OUT=/scratch/$USER/smoke_headonly_out
rm -rf "$OUT" /scratch/$USER/train_state/smoke_headonly_out
exec scripts/memory/train_state_ddp_job.sh "$OUT" \
    '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
    --head smallread_lingbot --dec-depth 8 --read-depth 8 --write-oneway --clip-store cpu \
    --init-from $G/scenes944_96f_b12x4_write8oneway_read8_tap23_lingbothead_LORAr16_h200/best_step5200.pt \
    --unfreeze-head --freeze-memory --val-clips $VA --val-every 2 --val-frames 96 --async-val \
    --updates 6 --patience 0 --frames-min 160 --frames 160 --frames-step 16 --max-frames 160 \
    --probe-stops 16 --recent-window 8 --tbptt 0 --head-bf16 --head-down 2 --loss-keep 0.5 \
    --batch "${B:-24}" --lr 1e-4 --warmup 300 --save-every 100000 --viz-every 100000 --wandb off --log-every 1
