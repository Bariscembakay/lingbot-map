#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-32%8
#SBATCH --output=/leonardo_scratch/large/userexternal/bbakay00/logs/%x_%A_%a.out
# KITTI 00-10 framing test: our benchmark's cover-fit centre crop to 504x280 vs ABot-Recon's width-fit + pad
# (kitti_pad_scenes.py, run first). Tasks 0-10 released lingbot default on the crop, 11-21 on the pad, 22-32
# ABot-Recon on the pad (ABot on the crop is the abot_bench baseline run). All scored the same way.
# Usage: sbatch -J <name> --time=.. kitti_framing_leonardo_job.sh <out dir>
set -euo pipefail
S=$CINECA_SCRATCH
OUT=$1
export TORCH_HOME=$S/torch_hub HF_HOME=$S/hf_cache HF_HUB_OFFLINE=1
SEQ=$(printf "%02d" $((SLURM_ARRAY_TASK_ID % 11)))
CROP=$S/insait/campaigns/lingbot_map/default/kitti/$SEQ
PAD=$S/data/kitti_pad504x280/$SEQ
R=$S/lingbot-map-rel/scripts/relpose
echo "[kitti_framing] $(hostname) task $SLURM_ARRAY_TASK_ID seq $SEQ git $(git -C $S/lingbot-map-rel rev-parse --short HEAD)"
case $((SLURM_ARRAY_TASK_ID / 11)) in
    0) $S/micromamba/envs/lingbot_map/bin/python $R/lingbot_default_bench.py --scenes $CROP --out $OUT/lingbot_default --tag _crop ;;
    1) $S/micromamba/envs/lingbot_map/bin/python $R/lingbot_default_bench.py --scenes $PAD --out $OUT/lingbot_default --tag _pad ;;
    2) $S/micromamba/envs/abot_recon/bin/python $R/abot_bench.py run --scenes $PAD --out $OUT/abot_recon_pad --abot-root $S/ABot-Recon
       $S/micromamba/envs/lingbot_map/bin/python $R/abot_bench.py score --scenes $PAD --out $OUT/abot_recon_pad ;;
esac
