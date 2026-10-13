#!/usr/bin/env bash
# Write from lingbot taps 11+17 (channel-concat, 4096-d) instead of tap 23.
# Warm start from FRESHDPT_depth best (step 6500, val24 0.2720): everything loads
# except in_proj (2048 -> 4096 input), which starts fresh. Otherwise FRESHDPT's
# exact recipe (96 scenes, 96f, b4, lr 1e-4, tbptt 8, fresh lingbot-architecture
# DPT), so the curve compares with FRESHDPT_depth's own continuation past 6500.
# Why 11/17: lingbot's depth head reads only taps 4/11/17 (its tap-23 branch is
# dead) and leans on 11 then 17 (.agents/lingbot_tap_ablation.md).
# val24: the 16 tap-23-only v5 clips come from lingbot-tapcache-v5-val16-4tap.
set -euo pipefail
NSEG=${NSEG:-3}; TIME=${TIME:-12:00:00}
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(sed 's#/data/lingbot-tapcache-v5-tap23/val/#/data/lingbot-tapcache-v5-val16-4tap/val/#' \
      "$(dirname "$0")/val24.txt" | tr '\n' ' ')
arm=scenes96_96f_b4_write8oneway_read8_TAPS11-17_FRESHDPT_depth_initFRESHDPTs6500
name=spatialmem_TAPS11-17_write_from_lingbot_layers11and17_initFRESHDPTdepth_s6500_inprojreinit_scenes96_96f_b4_freshdpt_depth_h200_12hseg
dep="${FIRST_DEP:-}"
for seg in $(seq 1 "$NSEG"); do
    jid=$(sbatch --parsable ${dep:+--dependency=$dep} --export=ALL,KEEPALIVE=1 --job-name=${name}${seg} \
        --partition=batch --constraint="zone-sof1|zone-msp3" --gpus=h200:1 \
        --cpus-per-task=12 --mem=320G --time=$TIME \
        --chdir=/home/baris_bakay/lingbot-map \
        --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/taps1117_%j.out \
        scripts/memory/train_state_job.sh \
        sof1:$G/$arm \
        '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
        --head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head \
        --taps 11,17 \
        --init-from $G/scenes96_96f_b4_write8oneway_read8_tap23_FRESHDPT_depth_scratch/best.pt \
        --val-clips $VAL --val-every 100 --updates 5000 --patience 0 --resume auto \
        --frames 96 --max-frames 160 --batch 4 --lr 1e-4 --wd 0.05 --warmup 300 \
        --n-past 4 --probe-every 1 --tbptt 8 --clip-store cpu \
        --save-every 250 --viz-every 1000 --wandb online)
    echo "seg$seg $jid"
    dep=afterany:$jid
done
