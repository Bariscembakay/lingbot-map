#!/usr/bin/env bash
# 944-scene pose refiner: first 8 frames fixed to lingbot (anchors), lr 2e-5 warmup+cosine,
# full schedule (best.pt still kept by val24 ATE).
set -euo pipefail
cd "$HOME/lingbot-map/.agents/scratch/memory_eval"
R="--memory specials --train-dirs train,train_v5 --anchor-frames 8 --steps 20000 --batch 16 --lr 2e-5 --train-store cpu"
A=poserefine_scenes944_t16-320_specials4tap_anchor8_b16_lr2e-5_cosine
sub() { sbatch --parsable --partition=rendering --qos=rendering --constraint=zone-sof1 --gpus=a6000:1 --mem=128G --time=12:00:00 -J "poserefine_$1_specials4tap_anchor8_scenes944_t16-320_b16_lr2e-5_cosine_a6000" \
        poserefine_train_job.sh "${A}_$1" $R --variant "${@:2}"; }
echo tokres_bidirhead_rope $(sub tokres_bidirhead_rope tokres_bidirhead)
echo pose_residual         $(sub pose_residual pose_residual)
echo tokres_frozencausal   $(sub tokres_frozencausal tokres_frozencausal --no-head-rope)
