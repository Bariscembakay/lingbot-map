#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `bench` micromamba env
# (framework side: prepare.py, evaluate.py, report.py, and run.py, which
# itself dispatches to the `lingbot_map` env via `conda run`).
#
# Usage: source this at the top of any job before running prepare.py/
# run.py/evaluate.py, alongside setup_lingbot_map_env.sh.
set -euo pipefail

ENV_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/bench"

if [ -d "$ENV_PREFIX" ]; then
    echo "[setup_bench_env] bench env already exists at $ENV_PREFIX, skipping."
    exit 0
fi

echo "[setup_bench_env] Building bench env on $(hostname) ..."

micromamba create -n bench python=3.11 -y

micromamba run -n bench pip install \
    numpy opencv-python-headless open3d evo matplotlib pyyaml tqdm scipy \
    imageio trimesh plyfile OpenEXR Imath Pillow onnxruntime-gpu==1.23.2

echo "[setup_bench_env] Done."
