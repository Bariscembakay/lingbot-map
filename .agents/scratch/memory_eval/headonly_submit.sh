#!/usr/bin/env bash
# Head-only run: BIGRUN s5200's write/read memory frozen (writes without
# autograd), its LoRA merged into the lingbot DPT, whole head trained with the
# fast recipe. Smoke 1100060: b24/rank at N=160 peaked at 37 GB, ~17 s/update.
# lr 1e-4: the head is already trained, so no sqrt scale-up for the global 96.
set -euo pipefail
NSEG=${NSEG:-3}; TIME=${TIME:-24:00:00}
G=/group/compact-3dmem/campaigns/spatial_memory
VAL=$(tr '\n' ' ' < "$(dirname "$0")/val24.txt")
arm=scenes944_N64-160_stops16k2_b24x4_HEADONLY_unfrozenDPT_frozenBIGRUNs5200memory_lr1e-4
name=spatialmem_HEADONLY_unfrozenDPT_LoRAmerged_frozenBIGRUNs5200memory_944scenes_N64to160_stops16k2_bf16halfdpt_losskeep05_fullbptt_b24x4gpu_lr1e-4_asyncval_h200x4_24hseg
dep="${FIRST_DEP:-}"
for seg in $(seq 1 "$NSEG"); do
    jid=$(sbatch --parsable ${dep:+--dependency=$dep} \
        --export=ALL,KEEPALIVE=1,STAGE=1 \
        --job-name=${name}${seg} \
        --partition=batch --constraint=zone-msp3 --nodes=1 --gpus=h200:4 \
        --cpus-per-task=32 --mem=900G --time=$TIME \
        --chdir=/home/baris_bakay/lingbot-map \
        --output=/home/baris_bakay/lingbot-map/.agents/scratch/memory_logs/headonly_%j.out \
        scripts/memory/train_state_ddp_job.sh \
        sof1:$G/$arm \
        '/data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/* /data/lingbot-tapcache-v5-tap23/train/*' \
        --head smallread_lingbot --dec-depth 8 --read-depth 8 --write-oneway --clip-store cpu \
        --init-from $G/scenes944_96f_b12x4_write8oneway_read8_tap23_lingbothead_LORAr16_h200/best_step5200.pt \
        --unfreeze-head --freeze-memory \
        --val-clips $VAL --val-every 100 --val-frames 96 --async-val \
        --updates 10000 --patience 0 --resume auto \
        --frames-min 64 --frames 160 --frames-step 16 --max-frames 160 \
        --probe-stops 16 --recent-window 8 --tbptt 0 --head-bf16 --head-down 2 --loss-keep 0.5 \
        --batch 24 --lr 1e-4 --warmup 300 --save-every 100 --viz-every 1000 --wandb online)
    echo "seg$seg $jid"
    dep=afterany:$jid
done
