#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `lingbot_map` micromamba env.
#
# /scratch (and $MAMBA_ROOT_PREFIX under it) is per-node local disk, so any
# job that lands on a node other than the one we built the env on by hand
# (sof1-h200-0) needs to recreate it. Fast no-op if the env already exists
# on this node.
#
# Usage: source this at the top of any job before `micromamba run -n lingbot_map ...`.
set -euo pipefail

ENV_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map"

# Repo root is three levels up from this script (.agents/scratch/insait_cluster_files/).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [ -d "$ENV_PREFIX" ]; then
    echo "[setup_lingbot_map_env] lingbot_map env already exists at $ENV_PREFIX, skipping base install."
else
    echo "[setup_lingbot_map_env] Building lingbot_map env on $(hostname) ..."

    micromamba create -n lingbot_map python=3.10 -y

    micromamba run -n lingbot_map pip install torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128

    micromamba run -n lingbot_map pip install -e "${REPO_DIR}[vis]"

    micromamba run -n lingbot_map pip install --index-url https://pypi.org/simple flashinfer-python

    micromamba run -n lingbot_map pip install --index-url https://pypi.org/simple \
        kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

    micromamba run -n lingbot_map pip install \
        numpy opencv-python Pillow matplotlib open3d plyfile tqdm scipy evo pyyaml OpenEXR Imath

    echo "[setup_lingbot_map_env] Base install done."
fi

# Separate idempotent step: nvcc (CUDA compiler), needed to JIT-build
# preprocess/oxford.py's CUDA visibility extension. Not part of the base
# torch install (pip wheels ship only the CUDA runtime, not the compiler),
# and older envs built before this fix won't have it either -- so this
# checks/installs regardless of whether the base env above was just built.
if [ -x "$ENV_PREFIX/bin/nvcc" ]; then
    echo "[setup_lingbot_map_env] nvcc already present, skipping."
else
    echo "[setup_lingbot_map_env] Installing cuda-nvcc (12.8, matching the torch cu128 build) ..."
    micromamba install -n lingbot_map -c nvidia -c conda-forge cuda-nvcc=12.8 -y
fi

echo "[setup_lingbot_map_env] Done."
