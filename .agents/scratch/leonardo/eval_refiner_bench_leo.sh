#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# eval_refiner_bench.py on Leonardo (A100), against the scratch mirror of the INSAIT benchmark trees.
# Results go to scratch and are copied back to INSAIT /group afterwards (Leonardo = compute only).
# Usage: sbatch [--mem --time --output -J] eval_refiner_bench_leo.sh TASK BENCHES [SCENES]   (refiner run dirs in $RUNS)
set -euo pipefail
S=/leonardo_scratch/large/userexternal/bbakay00
cd $S/lingbot-map
export BENCH_ROOT=$S/insait/campaigns/lingbot_map BENCH_CACHE=$S/benchcache BENCH_OUT=$S/insait/campaigns/spatial_memory/refiner_bench
export PYTHONUNBUFFERED=1
$S/envs/lingbot_map/bin/python scripts/memory/pose_refine/eval_refiner_bench.py "$1" --benches "$2" ${3:+--scenes $3} \
    --runs "${RUNS:-}" --backend sdpa
