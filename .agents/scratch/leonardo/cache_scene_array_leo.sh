#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# One benchmark scene's lingbot test cache per array task (eval_refiner_bench.py cache), on Leonardo.
# Usage: sbatch --array=0-(N-1) ... cache_scene_array_leo.sh BENCH TREE   (TREE = path under BENCH_ROOT, e.g. default/vbr)
set -euo pipefail
S=/leonardo_scratch/large/userexternal/bbakay00
cd $S/lingbot-map
export BENCH_ROOT=$S/insait/campaigns/lingbot_map BENCH_CACHE=$S/benchcache PYTHONUNBUFFERED=1
scenes=($(for d in $BENCH_ROOT/$2/*/gt; do basename $(dirname $d); done))
$S/envs/lingbot_map/bin/python scripts/memory/pose_refine/eval_refiner_bench.py cache --benches "$1" \
    --scenes "${scenes[$SLURM_ARRAY_TASK_ID]}" --backend sdpa
