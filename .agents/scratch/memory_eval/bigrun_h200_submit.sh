#!/usr/bin/env bash
# BIGRUN moved from 4x rtx6000 (b6/rank, global 24, lr 1.4e-4) to 4x H200 at
# b12/rank: the b12 H200 run peaked at 123 GB of 141. Global batch 48, so lr is
# sqrt-scaled to 2e-4 (AdamW). Continues the rtx run's optimizer and step via
# SEED_FROM under a new OUT, starting only after the rtx job has ended.
# Usage: FIRST_DEP=afterany:<rtx job> bigrun_h200_submit.sh
set -euo pipefail
NSEG=${NSEG:-4}; TIME=${TIME:-24:00:00}
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(tr '\n' ' ' < "$(dirname "$0")/val24.txt")
name=spatialmem_BIGRUN_944scenes_96f_b12x4gpu_write8oneway_read8_tap23_lingbothead_LORAr16_seededfrom_rtx6000b6x4_lr2e-4sqrtscaled_h200x4_24hseg
dep="${FIRST_DEP:-}"
for seg in $(seq 1 "$NSEG"); do
    jid=$(sbatch --parsable ${dep:+--dependency=$dep} \
        --export=ALL,KEEPALIVE=1,STAGE=1,SEED_FROM=sof1:$G/scenes944_96f_b3x8_write8oneway_read8_tap23_lingbothead_LORAr16_rtx6000 \
        --job-name=${name}${seg} \
        --partition=batch --constraint=zone-msp3 --nodes=1 --gpus=h200:4 \
        --cpus-per-task=32 --mem=900G --time=$TIME \
        --chdir=/home/baris_bakay/lingbot-map \
        --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/bigrun_h200_%j.out \
        scripts/memory/train_state_ddp_job.sh \
        sof1:$G/scenes944_96f_b12x4_write8oneway_read8_tap23_lingbothead_LORAr16_h200 \
        '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/* /data/lingbot-tapcache-v5-tap23/train/*' \
        --head smallread_lingbot --dec-depth 8 --read-depth 8 --write-oneway --clip-store cpu \
        --lora-head --lora-rank 16 --lora-alpha 16 \
        --init-from $G/scenes944_96f_b12x2_write8oneway_read8_tap23_lingbothead/best.pt \
        --val-clips $VAL \
        --val-every 100 --updates 200000 --patience 15 --min-delta 0.002 --resume auto \
        --frames 96 --max-frames 160 --batch 12 --lr 2e-4 --warmup 300 --n-past 4 --probe-every 1 \
        --tbptt 8 --save-every 100 --viz-every 1000 --wandb online)
    echo "seg$seg $jid"
    dep=afterany:$jid
done
