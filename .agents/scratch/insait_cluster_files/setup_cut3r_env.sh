#!/usr/bin/env bash
# Idempotent per-node bootstrap for the `cut3r` micromamba env.
#
# Deliberately SEPARATE from `lingbot_map`: this env exists to reproduce CUT3R's
# published numbers, and a dependency clash there must not be able to break the
# env our own benchmarks run in.
#
# Upstream pins numpy==1.26.4. That pin is stale and is deliberately NOT honoured
# here: current open3d 0.19, opencv-python 5.x and pandas 3.x all require
# numpy>=2, so holding 1.26.4 breaks open3d's import outright. The mv_recon code
# path was checked for numpy-2-removed APIs and is clean -- the only `np.in1d`
# uses are in datasets/megadepth.py and viz.py, neither of which mv_recon
# imports. The reproduction itself is the test: if the numbers match Table 4,
# numpy 2 is fine.
#
# /scratch is per-node local disk, so source this at the top of any job.
set -euo pipefail

export PATH="$HOME/bin:$PATH"

MAMBA_ROOT="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}"
ENV_PREFIX="$MAMBA_ROOT/envs/cut3r"
mkdir -p "$MAMBA_ROOT"

# Scoped to a subshell so the lock is released once setup finishes. `exec 200>`
# + flock at top level never unlocks, so fd 200 would stay open for the whole
# life of the sourcing shell and every child it spawns -- turning one job into a
# mutex over every other job packed onto the same node.
(
    exec 200>"$MAMBA_ROOT/.setup_cut3r_env.lock"
    flock 200

    if [ -d "$ENV_PREFIX" ]; then
        echo "[setup_cut3r_env] cut3r env already exists, skipping install."
    else
        echo "[setup_cut3r_env] Building cut3r env on $(hostname) ..."
        micromamba create -n cut3r python=3.11 -y

        micromamba run -n cut3r pip install torch==2.8.0 torchvision==0.23.0 \
            --index-url https://download.pytorch.org/whl/cu128

        micromamba run -n cut3r pip install \
            roma einops trimesh "huggingface-hub[torch]>=0.22" hydra-core omegaconf \
            accelerate transformers scikit-learn h5py opencv-python scipy \
            matplotlib tqdm "pillow==10.3.0" tensorboard

        # Eval-only, called out in upstream's README rather than requirements.txt.
        # NOT evo: it is only used by eval/relpose, and it pulls pandas, which
        # complicates the dependency set for no benefit here. Add it if the
        # relpose benchmark is ever wanted.
        micromamba run -n cut3r pip install open3d

        echo "[setup_cut3r_env] Install done."
    fi
)

# No curope build. croco/models/pos_embed.py:121 falls back to a pure-torch
# RoPE2D when the CUDA extension is absent, which is correct but slower. That is
# fine for eval; revisit if this env is ever used for training throughput.

export CUDA_HOME="$ENV_PREFIX"
export PATH="$ENV_PREFIX/bin${PATH:+:$PATH}"

echo "[setup_cut3r_env] Done."
