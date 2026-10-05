#!/usr/bin/env bash
# BIGRUN continued on 4 H200: global batch kept at 24 (6/rank), so lr and the
# per-update objective match the 2-GPU run; warm-started from its best.pt (the
# only checkpoint it left), now with the LoRA head. Two chained 12h segments.
set -euo pipefail
G=/group/compact-3dmem/campaigns/spatial_memory
name=spatialmem_BIGRUN_944scenes_96f_b6x4gpu_write8oneway_read8_tap23_lingbothead_LORAr16_initfrom_b12x2step100_lr1.4e-4_h200x4_12hseg
dep=""
for seg in 1 2; do
    jid=$(sbatch --parsable ${dep:+--dependency=$dep} --export=ALL,KEEPALIVE=1 --job-name=${name}${seg} \
        --partition=batch --constraint="zone-sof1|zone-msp3" --nodes=1 --gpus=h200:4 \
        --cpus-per-task=32 --mem=900G --time=12:00:00 \
        --chdir=/home/baris_bakay/lingbot-map \
        --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/bigrun4_%j.out \
        scripts/memory/train_state_ddp_job.sh \
        sof1:$G/scenes944_96f_b6x4_write8oneway_read8_tap23_lingbothead_LORAr16 \
        '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/* /data/lingbot-tapcache-v5-tap23/train/*' \
        --head smallread_lingbot --dec-depth 8 --read-depth 8 --write-oneway --clip-store cpu \
        --lora-head --lora-rank 16 --lora-alpha 16 \
        --init-from $G/scenes944_96f_b12x2_write8oneway_read8_tap23_lingbothead/best.pt \
        --val-clips /data/lingbot-tapcache-v4-40/val_top/210f741378_c0 /data/lingbot-tapcache-v4-40/val_top/260db9cf5a_c0 /data/lingbot-tapcache-v4-40/val_top/2ab7bea148_c0 /data/lingbot-tapcache-v4-40/val_top/47eb87b5bb_c0 /data/lingbot-tapcache-v4-40/val_median/0a7cc12c0e_c0 /data/lingbot-tapcache-v4-40/val_median/1730c7d709_c0 /data/lingbot-tapcache-v4-40/val_median/1841a0b525_c0 /data/lingbot-tapcache-v4-40/val_median/4291be3b44_c0 /data/lingbot-tapcache-v5-tap23/val/0a5c013435_c0 /data/lingbot-tapcache-v5-tap23/val/16c9bd2e1e_c0 /data/lingbot-tapcache-v5-tap23/val/192ab15daf_c0 /data/lingbot-tapcache-v5-tap23/val/19cfd590f4_c0 /data/lingbot-tapcache-v5-tap23/val/1ada7a0617_c0 /data/lingbot-tapcache-v5-tap23/val/1ae9e5d2a6_c0 /data/lingbot-tapcache-v5-tap23/val/20ff72df6e_c0 /data/lingbot-tapcache-v5-tap23/val/270ada6f0d_c0 /data/lingbot-tapcache-v5-tap23/val/281ba69af1_c0 /data/lingbot-tapcache-v5-tap23/val/30966f4c6e_c0 /data/lingbot-tapcache-v5-tap23/val/3864514494_c0 /data/lingbot-tapcache-v5-tap23/val/3aa115e55e_c0 /data/lingbot-tapcache-v5-tap23/val/3ce6d36ab5_c0 /data/lingbot-tapcache-v5-tap23/val/3e7e4b07c4_c0 /data/lingbot-tapcache-v5-tap23/val/40b56bf310_c0 /data/lingbot-tapcache-v5-tap23/val/4517d988d8_c0 \
        --val-every 100 --updates 200000 --patience 15 --min-delta 0.002 --resume auto \
        --frames 96 --max-frames 160 --batch 6 --lr 1.4e-4 --warmup 300 --n-past 4 --probe-every 1 \
        --tbptt 8 --save-every 100 --viz-every 1000 --wandb online)
    echo "seg$seg $jid"
    dep=afterany:$jid
done
