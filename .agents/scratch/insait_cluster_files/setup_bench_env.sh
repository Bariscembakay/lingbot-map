#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `bench` micromamba env (framework
# side: prepare.py, evaluate.py, report.py, run.py). Source alongside
# setup_lingbot_map_env.sh at the top of any job.
set -euo pipefail

export PATH="$HOME/bin:$PATH"  # see setup_lingbot_map_env.sh

ENV_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/bench"

# See setup_lingbot_map_env.sh: multiple jobs can share this node's /scratch.
# Scoped to a subshell so the lock releases once setup finishes rather than
# being held for the sourcing shell's entire life (see setup_lingbot_map_env.sh
# for why that's a real bug, not a style preference).
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
mkdir -p "$MAMBA_ROOT"
(
    exec 201>"$MAMBA_ROOT/.setup_bench_env.lock"
    flock 201

    if [ -d "$ENV_PREFIX" ]; then
        echo "[setup_bench_env] bench env already exists at $ENV_PREFIX, skipping."
    else
        echo "[setup_bench_env] Building bench env on $(hostname) ..."

        micromamba create -n bench python=3.11 -y

        micromamba run -n bench pip install \
            numpy opencv-python-headless open3d evo matplotlib pyyaml tqdm scipy \
            imageio trimesh plyfile OpenEXR Imath Pillow onnxruntime-gpu==1.23.2

        echo "[setup_bench_env] Done."
    fi
)
