#!/usr/bin/env bash
# build (CPU) -> smoke (all variants, 20 steps + step-0 evals) -> 4 trainings.
set -euo pipefail
cd "$HOME/lingbot-map/.agents/scratch/memory_eval"
mkdir -p /group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs
B=$(sbatch --parsable poserefine_build_job.sh)
S=$(sbatch --parsable --dependency=afterok:$B --time=01:00:00 \
    -J poserefine_SMOKE_allvariants_20steps_eval0_specials4tap_a100 poserefine_train_job.sh smoke)
R="--memory specials --steps 20000 --batch 16 --lr 2e-4"
A=poserefine_scenes96_t16-320_specials4tap_b16_lr2e-4
sub() { sbatch --parsable --dependency=afterok:$S -J "poserefine_$1_specials4tap_scenes96_t16-320_b16_lr2e-4_a100" \
        poserefine_train_job.sh "${A}_$1" $R --variant "${@:2}"; }
echo build $B smoke $S
echo tokres_bidirhead_rope   $(sub tokres_bidirhead_rope tokres_bidirhead)
echo direct_pose             $(sub direct_pose direct_pose)
echo pose_residual           $(sub pose_residual pose_residual)
echo tokres_frozencausal     $(sub tokres_frozencausal tokres_frozencausal --no-head-rope)
