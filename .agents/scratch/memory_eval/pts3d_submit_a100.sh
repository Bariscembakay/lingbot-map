#!/usr/bin/env bash
# A100-80G variant of pts3d_submit.sh: clip taps on host RAM (they are ~80 GB of the
# H200 run's 124.7 GB peak); KEEPALIVE because the clip preload reads as idle (job 829672).
# Usage: pts3d_submit.sh <smoke_jobid>
set -euo pipefail
SMOKE=$1
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(printf '/data/lingbot-tapcache-v4-40/%s ' val_top/210f741378_c0 val_top/260db9cf5a_c0 val_top/2ab7bea148_c0 val_top/47eb87b5bb_c0 val_median/0a7cc12c0e_c0 val_median/1730c7d709_c0 val_median/1841a0b525_c0 val_median/4291be3b44_c0)
for spec in "PTS3D_camframe:--pts-frame cam" "PTS3D_worldframe:--pts-frame world"; do
    tag=${spec%%:*}; flag=${spec#*:}
    name=spatialmem_LORAHEAD_r16_${tag}_scenes96_96f_b4_write4_read2_lingbothead_initfrom_frozenhead_A100_clipcpu_to4900_24hseg
    dep=$([ "$SMOKE" = none ] || echo afterok:$SMOKE)
    for seg in 1 2; do
        # shellcheck disable=SC2086
        jid=$(sbatch --parsable ${dep:+--dependency=$dep} --export=ALL,KEEPALIVE=1 --job-name=${name}${seg} \
            --partition=batch --constraint=zone-gcp-eu1 --gpus=a100-80g:1 \
            --cpus-per-task=8 --mem=192G --time=24:00:00 \
            --chdir=/home/baris_bakay/lingbot-map \
            --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/pts3d_${tag}_%j.out \
            scripts/memory/train_state_job.sh \
            $G/scenes96_96f_b4_write4_read2_lingbothead_LORAHEAD_r16_${tag} \
            '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
            --head smallread_lingbot --dec-depth 4 --read-depth 2 \
            --lora-head --lora-rank 16 --lora-alpha 16 $flag \
            --init-from $G/scenes96_96f_b4_write4_read2_lingbothead/last.pt \
            --val-clips $VAL \
            --val-every 100 --updates 4900 --patience 15 --min-delta 0.002 --resume auto \
            --frames 96 --max-frames 160 --batch 4 --lr 1e-4 --wd 0.05 --warmup 100 \
            --n-past 4 --probe-every 1 --tbptt 8 --save-every 250 --viz-every 1000 --wandb online --clip-store cpu)
        echo "$tag seg$seg $jid"
        dep=afterany:$jid
    done
done
