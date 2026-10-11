#!/usr/bin/env bash
# World-frame xyz FROM SCRATCH with the fast recipe, against FRESHDPT_xyzworld
# (same 96 scenes, same init: CUT3R write/read + random-init DPT head, same
# val24 at 96 frames). Recipe: lengths 64..160 (step 16), 16 probe stops x
# (1 recent + 1 old) queries = constant budget, full BPTT, bf16 + half-res
# head in training, --loss-keep 0.5, batch 14, async validation.
# lr: sqrt-scaled from FRESHDPT's 1e-4 @ batch 4 (the BIGRUN convention).
# A100 smoke: 4.4x faster per update than the FRESHDPT recipe at batch 4.
set -euo pipefail
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(tr '\n' ' ' < "$(dirname "$0")/val24.txt")
NSEG=${NSEG:-5}
tag=FASTRECIPE_xyzworld
name=spatialmem_${tag}_scratch_scenes96_N64to160_stops16k2_bf16halfdpt_losskeep05_fullbptt_b14_lr1.9e-4_asyncval_h200_12hseg
dep=""
for seg in $(seq 1 "$NSEG"); do
    jid=$(sbatch --parsable ${dep:+--dependency=$dep} --export=ALL,KEEPALIVE=1 --job-name=${name}${seg} \
        --partition=batch --constraint="zone-sof1|zone-msp3" --gpus=h200:1 \
        --cpus-per-task=12 --mem=192G --time=12:00:00 \
        --chdir=/home/baris_bakay/lingbot-map \
        --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/fast_${tag}_%j.out \
        scripts/memory/train_state_job.sh \
        sof1:$G/scenes96_N64-160_stops16k2_b14_write8oneway_read8_tap23_${tag}_scratch \
        '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
        --head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head --pts-frame world \
        --val-clips $VAL --val-every 100 --val-frames 96 --async-val --updates 10000 --patience 0 --resume auto \
        --frames-min 64 --frames 160 --frames-step 16 --max-frames 160 \
        --probe-stops 16 --recent-window 8 --tbptt 0 \
        --head-bf16 --head-down 2 --loss-keep 0.5 \
        --batch 14 --lr 1.9e-4 --wd 0.05 --warmup 300 --clip-store cpu \
        --save-every 100 --viz-every 100000 --wandb online)
    echo "seg$seg $jid"
    dep=afterany:$jid
done
