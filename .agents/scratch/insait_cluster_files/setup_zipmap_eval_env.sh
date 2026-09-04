#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `zipmap_eval` micromamba env.
# Separate from `zipmap`: the eval harness (ZipMap_eval/, upstream branch
# `evaluation`) pins torch==2.5.1 in its own requirements.txt, and that file is
# the reproduction contract for ZipMap paper Table 3.
set -euo pipefail
export PATH="$HOME/bin:$PATH"
MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
ENV_PREFIX="$MAMBA_ROOT/envs/zipmap_eval"
mkdir -p "$MAMBA_ROOT"
(
    exec 200>"$MAMBA_ROOT/.setup_zipmap_eval_env.lock"
    flock 200
    if [ -d "$ENV_PREFIX" ]; then
        echo "[setup_zipmap_eval_env] env exists, skipping."
    else
        echo "[setup_zipmap_eval_env] Building on $(hostname) ..."
        micromamba create -n zipmap_eval python=3.11 -y
        micromamba run -n zipmap_eval pip install -r "$HOME/lingbot-map/ZipMap_eval/requirements.txt"
        # einx used by the ZipMap model code the harness embeds; not in their reqs
        micromamba run -n zipmap_eval pip install einx easydict
        echo "[setup_zipmap_eval_env] done."
    fi
    # The harness's embedded models/zipmap wrapper does absolute `zipmap.*`
    # imports, i.e. upstream ran with the main-repo package installed alongside.
    # --no-deps keeps this env's torch==2.5.1 pin (ZipMap pyproject wants 2.6).
    PY="$ENV_PREFIX/bin/python"
    # The embedded ttt3r/cut3r wrappers import beyond the harness requirements.
    if ! "$PY" -c "import transformers, roma" >/dev/null 2>&1; then
        echo "[setup_zipmap_eval_env] Installing ttt3r/cut3r extras ..."
        "$PY" -m pip install transformers roma accelerate h5py scikit-learn
    fi
    if ! "$PY" -c "import zipmap" >/dev/null 2>&1; then
        echo "[setup_zipmap_eval_env] Installing zipmap package (--no-deps) ..."
        "$PY" -m pip install -e "$HOME/lingbot-map/ZipMap" --no-deps
    fi
    # The harness's ttt3r pos_embed hard-raises without compiled curope (no
    # torch fallback allowed upstream). torch 2.5.1/cu124 predates both compile
    # breakers that block curope under torch 2.8 (AT_DISPATCH removal, CUDA
    # 12.8/glibc cospi clash), so build it as upstream intended.
    CUROPE_DIR="$HOME/lingbot-map/ZipMap_eval/models/ttt3r/croco/models/curope"
    if ! ls "$CUROPE_DIR"/curope*.so >/dev/null 2>&1; then
        echo "[setup_zipmap_eval_env] Compiling curope (torch 2.5.1 / cu124) ..."
        # host gcc on these nodes is >13, which CUDA 12.4 refuses; use a
        # conda gcc-13 as nvcc's host compiler.
        micromamba install -n zipmap_eval -c nvidia -c conda-forge cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-cccl=12.4 cuda-libraries-dev=12.4 "gxx_linux-64=13" "gcc_linux-64=13" -y
        HOSTXX="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-g++"
        ( cd "$CUROPE_DIR" && CUDA_HOME="$ENV_PREFIX" LIBRARY_PATH="$ENV_PREFIX/targets/x86_64-linux/lib:$ENV_PREFIX/lib" CC="$ENV_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$HOSTXX" NVCC_PREPEND_FLAGS="-ccbin $HOSTXX" TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0" "$PY" setup.py build_ext --inplace )
    fi
)
export PATH="$ENV_PREFIX/bin${PATH:+:$PATH}"
echo "[setup_zipmap_eval_env] Done."
