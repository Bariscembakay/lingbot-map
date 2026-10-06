#!/usr/bin/env bash
# 3D-output ablation vs LORAHEAD_r16 (0.1503): identical config, only the head output differs.
# Usage: pts3d_submit.sh <smoke_jobid> [tag ...]   (no tags = all arms)
set -euo pipefail
SMOKE=$1; shift; ONLY=" $* "
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(printf '/data/lingbot-tapcache-v4-40/%s ' val_top/210f741378_c0 val_top/260db9cf5a_c0 val_top/2ab7bea148_c0 val_top/47eb87b5bb_c0 val_median/0a7cc12c0e_c0 val_median/1730c7d709_c0 val_median/1841a0b525_c0 val_median/4291be3b44_c0)
for spec in "PTS3D_camframe_smallinit:--pts-frame cam" "PTS3D_worldframe_smallinit:--pts-frame world" "FRESHLASTCONV_depth:--fresh-last-conv"; do
    tag=${spec%%:*}; flag=${spec#*:}
    [ "$ONLY" = "  " ] || [[ "$ONLY" == *" $tag "* ]] || continue
    name=spatialmem_LORAHEAD_r16_${tag}_scenes96_96f_b4_write4_read2_lingbothead_initfrom_frozenhead_h200_to4900_12hseg
    # SMOKE: a job id (afterok), "none", or a full spec like afterany:<seg> to
    # extend a running arm; NSEG segments are chained after it.
    case "$SMOKE" in none) dep="" ;; after*) dep=$SMOKE ;; *) dep=afterok:$SMOKE ;; esac
    for seg in $(seq 1 "${NSEG:-2}"); do
        # shellcheck disable=SC2086
        jid=$(sbatch --parsable ${dep:+--dependency=$dep} --export=ALL,KEEPALIVE=1 --job-name=${name}${seg} \
            --partition=batch --constraint="zone-sof1|zone-msp3" --gpus=h200:1 \
            --cpus-per-task=12 --mem=128G --time=12:00:00 \
            --chdir=/home/baris_bakay/lingbot-map \
            --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/pts3d_${tag}_%j.out \
            scripts/memory/train_state_job.sh \
            sof1:$G/scenes96_96f_b4_write4_read2_lingbothead_LORAHEAD_r16_${tag} \
            '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
            --head smallread_lingbot --dec-depth 4 --read-depth 2 \
            --lora-head --lora-rank 16 --lora-alpha 16 $flag \
            --init-from $G/scenes96_96f_b4_write4_read2_lingbothead/last.pt \
            --val-clips $VAL \
            --val-every 100 --updates 4900 --patience 15 --min-delta 0.002 --resume auto \
            --frames 96 --max-frames 160 --batch 4 --lr 1e-4 --wd 0.05 --warmup 100 \
            --n-past 4 --probe-every 1 --tbptt 8 --save-every 250 --viz-every 1000 --wandb online)
        echo "$tag seg$seg $jid"
        dep=afterany:$jid
    done
done
