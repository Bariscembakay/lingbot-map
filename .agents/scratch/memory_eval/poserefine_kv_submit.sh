#!/usr/bin/env bash
# v0 KV-cache refiner overnight ablation (2026-10-15). Common: lingbot special-token K/V at
# global layers 4/11/17/23 as memory, anchors 8, mixed ScanNet++944 / KITTI-360 (drives
# 0007+0009 held out) / VKITTI2, prefixes 32..2048 (curriculum from 1024), 4096 frames per
# batch, scale-free + distance-normalised translation, abs + local relative loss, lr 3e-5
# cosine 10k, best.pt by mean(ATE / lingbot ATE) over val24 + kitti360_val.
# Usage: poserefine_kv_submit.sh A|B|all [GPU=h200:1]
set -euo pipefail
cd "$HOME/lingbot-map/.agents/scratch/memory_eval"
GPU=${GPU:-h200:1}
C="--anchor-frames 8 --scale-free-loss --norm-trans --train-store cpu
   --train-dirs train+train_v5:0.45,kitti360_train:0.35,vkitti2:0.2
   --t-min 32 --t-max 2048 --t-step 32 --t-max-start 1024 --frames-per-batch 4096 --batch 16
   --max-frames-rope 4096 --steps 10000 --warmup 500 --val-every 500
   --eval-sets val24:320,kitti360_val:full,nrgbd:500,kitti:full --select val24,kitti360_val"
A="--variant tokres_frozencausal --no-head-rope"
B="--variant tokres_bidirhead"
P=poserefine_kvmem_mixed944k360vk2_t32-2048_anchor8
sub() {  # sub <arm suffix> <args...>
    echo "$1 $(sbatch --parsable --partition=batch --gpus=$GPU --time=06:00:00 \
        -J "poserefine_KVMEM_$1_mixed_scannet944_kitti360_vkitti2_t32-2048_anchor8_lr3e-5_10k_${GPU%%:*}" \
        poserefine_kv_train_job.sh "${P}_$1" $C "${@:2}")"
}
case ${1:-all} in
A|all)
sub A_frozencausal_rel   $A --memory kvcache --rel-w 1 --lr 3e-5
sub A_frozencausal_rel_poseemb $A --memory kvcache --rel-w 1 --lr 3e-5 --pose-emb
sub A_frozencausal_rel_specialsmem $A --memory specials --rel-w 1 --lr 3e-5
sub A_frozencausal_rel_nomem $A --memory none --rel-w 1 --lr 3e-5
sub A_frozencausal_rel_reg0.1 $A --memory kvcache --rel-w 1 --lr 3e-5 --reg-w 0.1
sub A_frozencausal_rel_kvadapter $A --memory kvcache --rel-w 1 --lr 3e-5 --kv-adapter
sub A_frozencausal_absonly $A --memory kvcache --rel-w 0 --lr 3e-5
sub A_frozencausal_rel_lr1e-4 $A --memory kvcache --rel-w 1 --lr 1e-4
sub A_frozencausal_rel_seed1 $A --memory kvcache --rel-w 1 --lr 3e-5 --seed 1
;;&
B|all)
sub B_bidirhead_rel      $B --memory kvcache --rel-w 1 --lr 3e-5
sub B_bidirhead_rel_poseemb $B --memory kvcache --rel-w 1 --lr 3e-5 --pose-emb
sub B_bidirhead_rel_seed1 $B --memory kvcache --rel-w 1 --lr 3e-5 --seed 1
;;
esac
