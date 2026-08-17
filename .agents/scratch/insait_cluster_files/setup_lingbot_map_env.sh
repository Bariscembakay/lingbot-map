#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `lingbot_map` micromamba env.
# /scratch is per-node local disk, so source this at the top of any job
# before `micromamba run -n lingbot_map ...` -- fast no-op if already built
# on this node.
set -euo pipefail

# ~/bin holds the `conda`->micromamba shim run.py's subprocess dispatch
# needs, but isn't on PATH by default in job shells.
export PATH="$HOME/bin:$PATH"

ENV_PREFIX="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/lingbot_map"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Multiple jobs can land on the same node (Slurm packs several 1-GPU jobs
# onto one 8-GPU node) and race on this shared per-node env -- serialize
# with a lock so the second job waits instead of colliding mid-install.
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
mkdir -p "$MAMBA_ROOT"
exec 200>"$MAMBA_ROOT/.setup_lingbot_map_env.lock"
flock 200

if [ -d "$ENV_PREFIX" ]; then
    echo "[setup_lingbot_map_env] lingbot_map env already exists, skipping base install."
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
        numpy opencv-python Pillow matplotlib open3d plyfile tqdm scipy evo pyyaml OpenEXR Imath pye57

    echo "[setup_lingbot_map_env] Base install done."
fi

# conda's cuda-* packages use targets/x86_64-linux/{include,lib}/[stubs/]
# instead of the standard toolkit layout (lib64/, top-level include/) that
# FlashInfer's JIT build and gcc/nvcc expect -- symlink lib64 and add the
# include dir to CPATH rather than patching every consumer.
if [ ! -e "$ENV_PREFIX/lib64" ]; then
    ln -s "$ENV_PREFIX/targets/x86_64-linux/lib" "$ENV_PREFIX/lib64"
fi

# nvcc: not part of the base install above (pip torch wheels ship only the
# CUDA runtime, not the compiler); needed to JIT-build
# preprocess/oxford.py's CUDA visibility extension.
if [ -x "$ENV_PREFIX/bin/nvcc" ]; then
    echo "[setup_lingbot_map_env] nvcc already present, skipping."
else
    micromamba install -n lingbot_map -c nvidia -c conda-forge cuda-nvcc=12.8 -y
fi

# cuda-driver-dev: separate package from cuda-nvcc/cuda-cudart-dev, needed
# for cuda.h (extensions that #include it directly, not just cuda_runtime.h).
if [ -f "$ENV_PREFIX/targets/x86_64-linux/include/cuda.h" ]; then
    echo "[setup_lingbot_map_env] cuda-driver-dev already present, skipping."
else
    micromamba install -n lingbot_map -c nvidia -c conda-forge cuda-driver-dev=12.8 -y
fi

export CPATH="$ENV_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"

echo "[setup_lingbot_map_env] Done."
