#!/usr/bin/env bash
# Depth vs world-frame xyz, both FROM SCRATCH: CUT3R weights for write/read (as
# BIGRUN's original start), fresh adapters, and a random-init fully trainable
# lingbot-architecture DPT head (no lingbot weights, no LoRA). BIGRUN's memory
# architecture (8 one-way write, 8 read). Fixed 5000-step budget, no early stop,
# 24-scene val. Six 12 h segments (~40 s/step); a spare one exits on resume.
set -euo pipefail
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(tr '\n' ' ' < "$(dirname "$0")/val24.txt")
for spec in "FRESHDPT_depth:" "FRESHDPT_xyzworld:--pts-frame world"; do
    tag=${spec%%:*}; flag=${spec#*:}
    name=spatialmem_${tag}_scratch_scenes96_96f_b4_write8oneway_read8_tap23_h200_5000upd_12hseg
    dep=""
    for seg in 1 2 3 4 5 6; do
        # shellcheck disable=SC2086
        jid=$(sbatch --parsable ${dep:+--dependency=$dep} --export=ALL,KEEPALIVE=1 --job-name=${name}${seg} \
            --partition=batch --constraint="zone-sof1|zone-msp3" --gpus=h200:1 \
            --cpus-per-task=12 --mem=192G --time=12:00:00 \
            --chdir=/home/baris_bakay/lingbot-map \
            --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/fresh_${tag}_%j.out \
            scripts/memory/train_state_job.sh \
            sof1:$G/scenes96_96f_b4_write8oneway_read8_tap23_${tag}_scratch \
            '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
            --head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head $flag \
            --val-clips $VAL --val-every 100 --updates 5000 --patience 0 --resume auto \
            --frames 96 --max-frames 160 --batch 4 --lr 1e-4 --wd 0.05 --warmup 300 \
            --n-past 4 --probe-every 1 --tbptt 8 --clip-store cpu \
            --save-every 250 --viz-every 1000 --wandb online)
        echo "$tag seg$seg $jid"
        dep=afterany:$jid
    done
done
