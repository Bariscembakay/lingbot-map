#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-17
#SBATCH --output=/leonardo_scratch/large/userexternal/bbakay00/logs/%x_%A_%a.out
# eval_lingbot_rel.py on one benchmark scene per array task, from the BSS copies in $CINECA_SCRATCH/insait.
# SET=kitti_vbr (default, --array=0-17): KITTI 00-10, then the 7 VBR scenes. SET=oxford / oxford_s12
# (--array=0-9): the 10 Oxford Spires scenes, dense (~3840 frames) or every 12th frame (320).
# Summarise with eval_lingbot_rel.py --summary <out>.
# Usage: [SET=oxford] sbatch -J <name> --time=.. [--array=0-9] eval_lingbot_rel_leonardo_job.sh <out dir>
#        (--run DIR [--ckpt last.pt] | --lingbot)
set -euo pipefail
S=$CINECA_SCRATCH
P=$S/micromamba/envs/lingbot_map/bin/python
SET=${SET:-kitti_vbr}
case $SET in
    kitti_vbr)
        B=$S/insait/campaigns/lingbot_map/default
        SCENES=(kitti/00 kitti/01 kitti/02 kitti/03 kitti/04 kitti/05 kitti/06 kitti/07 kitti/08 kitti/09 kitti/10
                vbr/campus_train0 vbr/campus_train1 vbr/ciampino_train1 vbr/colosseo_train0 vbr/diag_train0
                vbr/pincio_train0 vbr/spagna_train0) ;;
    oxford|oxford_s12)
        # the benchmark's dense_s1 and sparse_s12 trees
        B=$S/insait/campaigns/lingbot_map/$([ $SET = oxford ] && echo dense_s1 || echo sparse_s12)
        SCENES=(oxford_spires/bodleian-library-02 oxford_spires/christ-church-02 oxford_spires/christ-church-03
                oxford_spires/christ-church-05 oxford_spires/keble-college-02 oxford_spires/keble-college-03
                oxford_spires/keble-college-04 oxford_spires/keble-college-05 oxford_spires/observatory-quarter-01
                oxford_spires/observatory-quarter-02) ;;
esac
OUT=$1; shift
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SC=${SCENES[$SLURM_ARRAY_TASK_ID]}
echo "[eval_lingbot_rel] $(hostname) $SC git $(git -C $S/lingbot-map-rel rev-parse --short HEAD)"
"$P" $S/lingbot-map-rel/scripts/relpose/eval_lingbot_rel.py --scenes $B/$SC --out "$OUT" "$@"
