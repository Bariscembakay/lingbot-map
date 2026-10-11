#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `chorus` micromamba env: Chorus package
# mode (inference only) from ~/Chorus, pinned to its tested cu126 track
# (torch 2.7.1, spconv-cu126, torch-scatter, flash-attn 2.8.3 -- all prebuilt
# wheels, nothing compiles).
#
# Separate env because Chorus pins torch 2.7.1 / python 3.10, unlike cut3r.
# /scratch is per-node local disk, so source this at the top of any job.
set -euo pipefail

export PATH="$HOME/bin:$PATH"

MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
ENV_PREFIX="$MAMBA_ROOT/envs/chorus"
CHORUS_REPO="${CHORUS_REPO:-$HOME/Chorus}"
mkdir -p "$MAMBA_ROOT"

# Subshell-scoped lock: see setup_cut3r_env.sh for why a top-level flock is a bug.
(
    exec 202>"$MAMBA_ROOT/.setup_chorus_env.lock"
    flock 202

    # Marker written only after the import check passes, so a half-finished
    # install (killed job, failed wheel) is rebuilt instead of silently reused.
    if [ -f "$ENV_PREFIX/.chorus_ok" ]; then
        echo "[setup_chorus_env] chorus env already exists, skipping install."
    else
        echo "[setup_chorus_env] Building chorus env on $(hostname) ..."
        rm -rf "$ENV_PREFIX"
        micromamba create -n chorus python=3.10 -y -c conda-forge
        PIP="$ENV_PREFIX/bin/pip"
        "$PIP" install -r "$CHORUS_REPO/scripts/package/requirements-cu126.txt"
        "$PIP" install -e "$CHORUS_REPO" --no-deps
        "$ENV_PREFIX/bin/python" -c "import torch, spconv, torch_scatter, flash_attn, chorus; \
print('[setup_chorus_env] torch', torch.__version__, 'chorus', chorus.__version__)"
        touch "$ENV_PREFIX/.chorus_ok"
        echo "[setup_chorus_env] Install done."
    fi
)
