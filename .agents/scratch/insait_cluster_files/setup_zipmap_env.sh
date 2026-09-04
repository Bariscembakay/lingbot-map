#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `zipmap` micromamba env.
#
# Separate env because ZipMap pins torch==2.6.0 (cut3r/lingbot_map run 2.8),
# and its pyproject is the reproduction contract -- install it as written.
# /scratch is per-node local disk, so source this at the top of any job.
set -euo pipefail

export PATH="$HOME/bin:$PATH"

MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
ENV_PREFIX="$MAMBA_ROOT/envs/zipmap"
mkdir -p "$MAMBA_ROOT"

(
    exec 200>"$MAMBA_ROOT/.setup_zipmap_env.lock"
    flock 200

    if [ -d "$ENV_PREFIX" ]; then
        echo "[setup_zipmap_env] zipmap env already exists, skipping install."
    else
        echo "[setup_zipmap_env] Building zipmap env on $(hostname) ..."
        micromamba create -n zipmap python=3.11 -y
        # pyproject's torch==2.6.0 resolves to the default cu124 build on PyPI,
        # which covers both sm86 (a6000) and sm90 (H200).
        micromamba run -n zipmap pip install -e "$HOME/lingbot-map/ZipMap"
        echo "[setup_zipmap_env] Install done."
    fi
)

export PATH="$ENV_PREFIX/bin${PATH:+:$PATH}"
echo "[setup_zipmap_env] Done."
