#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# periodic_refine_bench.py on one cached benchmark scene per array task (scenes = files of $BENCH_CACHE/<bench>).
# Usage: sbatch --array=0-(N-1) ... periodic_refine_leo.sh BENCH RUN_NAME [SCENE]   (SCENE overrides the array)
set -euo pipefail
S=/leonardo_scratch/large/userexternal/bbakay00
cd $S/lingbot-map
export BENCH_ROOT=$S/insait/campaigns/lingbot_map BENCH_CACHE=$S/benchcache BENCH_OUT=$S/insait/campaigns/spatial_memory/refiner_bench
export PYTHONUNBUFFERED=1
scenes=($(ls $BENCH_CACHE/$1 | sed 's/\.npz$//'))
sc=${3:-${scenes[${SLURM_ARRAY_TASK_ID:-0}]}}
$S/envs/lingbot_map/bin/python scripts/memory/pose_refine/periodic_refine_bench.py --bench "$1" \
    --run $S/insait/campaigns/spatial_memory/$2 --scenes "$sc" ${EXTRA:-}
