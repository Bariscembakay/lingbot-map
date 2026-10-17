#!/usr/bin/env bash
# CPU-only: score one (arm, dataset) of the pose-headroom oracle with the
# benchmark's own evaluate.py. Build the arms first with gtpose_oracle_build.py.
# Usage: sbatch --array=0-9 gtpose_oracle_eval.sh
set -euo pipefail
CFGS=(predpose_rerun_neural_rgbd gtpose_oracle_neural_rgbd predpose_rerun_seven_scenes gtpose_oracle_seven_scenes gtdepth_predpose_neural_rgbd gtdepth_predpose_seven_scenes gtpose_framescale_neural_rgbd gtpose_framescale_seven_scenes gtpose_gtintr_neural_rgbd gtpose_gtintr_seven_scenes)
CFG=${CFGS[${SLURM_ARRAY_TASK_ID:?}]}
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_bench_env.sh
cd benchmark
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/bench/bin/python" evaluate.py \
    --config "configs/gtpose_oracle/$CFG.yaml"
