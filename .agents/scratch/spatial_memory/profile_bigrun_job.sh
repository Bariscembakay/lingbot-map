#!/usr/bin/env bash
# Why did BIGRUN 895314 take ~205 s/update? Same architecture (write8 oneway,
# read8, smallread_lingbot, LoRA head), 12 clips, 1 H200, 4 updates per variant:
#   A b12 cpu-store (the big run's per-rank setting)   C b6 gpu-store
#   B b6  cpu-store                                    D b4 cpu-store
# B vs C isolates the CPU store; A/B/D give scaling in batch (near-full memory
# at b12 is the other suspect). No keep-alive: its matmul bursts would skew timing.
set -uo pipefail
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
# The env script turns on errexit; one variant OOMing must not kill the rest.
set +e
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
dataset pull lingbot-tapcache-v5-tap23 >/dev/null
ALL=(/data/lingbot-tapcache-v5-tap23/train/*)
CLIPS="${ALL[*]:0:12}"
OUT=/scratch/$USER/profile_bigrun_$SLURM_JOB_ID
mkdir -p "$OUT"

run() {  # name batch store
    echo "=== $1: batch $2, clip-store $3 ==="
    # shellcheck disable=SC2086
    "$PY" scripts/memory/train_state.py --clips $CLIPS --out "$OUT/$1" \
        --head smallread_lingbot --dec-depth 8 --read-depth 8 --write-oneway \
        --lora-head --clip-store "$3" --batch "$2" --frames 96 --max-frames 160 \
        --n-past 4 --probe-every 1 --tbptt 8 --lr 1.4e-4 --warmup 300 \
        --updates 4 --log-every 1 --val-every 100000 --save-every 100000 \
        --viz-every 100000 --resume off --wandb off 2>&1 \
        | grep -E "^\[ *[0-9]+\] loss|lora-head|Error|error|OOM|out of memory"
}
run A_b12_cpu 12 cpu
run B_b6_cpu 6 cpu
run C_b6_gpu 6 gpu
run D_b4_cpu 4 cpu
rm -rf "$OUT"
