#!/usr/bin/env bash
# CUT3R-style curriculum on the LoRA r16 96-scene setup, one chain:
#   A: 4 frames -> B: 16 frames -> C: 4..64 frames (revisits, grad on last 16).
# Frames come from CUT3R's ScanNet++ sampler; every stage validates on the same
# fixed 96-frame streams, so valm_self compares directly with LORAHEAD_r16
# (0.1503). Stage A runs batch 16 (4-frame clips are cheap); B and C keep the
# baseline's per-window memory (batch 4, tbptt 8 -- CUT3R itself cuts every 4).
# Clips live in host RAM: on the GPU they took ~80 GB and stage A OOMed (1079269).
set -euo pipefail
G=/group/compact-3dmem/campaigns/spatial_memory
B0=scenes96_cut3rcurr_write4_read2_lingbothead_LORAr16
VAL=$(tr '\n' ' ' < "$(dirname "$0")/val24.txt")   # default val set since 2026-10-08 (BIGRUN's 24 scenes)
common="--head smallread_lingbot --dec-depth 4 --read-depth 2 --lora-head --lora-rank 16 --lora-alpha 16
  --sampler cut3r --max-frames 160 --val-frames 96 --tbptt 0 --n-past 4 --probe-every 1
  --lr 1e-4 --wd 0.05 --warmup 100 --val-clips $VAL --val-every 100 --patience 15 --min-delta 0.002
  --resume auto --save-every 250 --viz-every 1000 --wandb online --clip-store cpu"
submit() {  # tag dep init extra...
    local tag=$1 dep=$2 init=$3; shift 3
    # shellcheck disable=SC2086
    sbatch --parsable ${dep:+--dependency=afterok:$dep} --export=ALL,KEEPALIVE=1 \
        --job-name=spatialmem_CURRICULUM_cut3rsampler_${tag}_scenes96_write4_read2_lingbothead_LORAr16_h200 \
        --partition=batch --constraint="zone-sof1|zone-msp3" --gpus=h200:1 \
        --cpus-per-task=12 --mem=192G --time=12:00:00 \
        --chdir=/home/baris_bakay/lingbot-map \
        --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/curr_${tag}_%j.out \
        scripts/memory/train_state_job.sh sof1:$G/${B0}_${tag} \
        '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/*' \
        $common --init-from "$init" "$@"
}
a=$(submit stageA_4f "" $G/scenes96_96f_b4_write4_read2_lingbothead/last.pt --frames 4 --batch 16 --updates 3000)
b=$(submit stageB_16f "$a" $G/${B0}_stageA_4f/best.pt --frames 16 --batch 4 --tbptt 8 --updates 3000)
c=$(submit stageC_4to64f "$b" $G/${B0}_stageB_16f/best.pt --frames 64 --frames-min 4 --allow-repeat --grad-last 16 --batch 4 --tbptt 8 --updates 4900)
echo "A $a  B $b  C $c"
