# CUT3R — vendored into lingbot-map

This directory was a clone of the upstream CUT3R repository. Its `.git` has been
**removed on purpose** so the code is carried by lingbot-map's own history: the
cluster rule is that code reaches other zones by git and never by copy, and an
embedded repo is invisible to the parent's commits.

Everything upstream's `.git` was the only record of is written down here instead.

## Provenance

| | |
|---|---|
| upstream | https://github.com/CUT3R/CUT3R |
| commit | `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` |
| subject | "undo the changes of dynamic_replica.py" |
| author | ZHANG Yifei \<zhangyifei21a@gmail.com\> |
| date | 2025-08-27 |
| branch | `main` |
| vendored | 2026-08-26 |
| paper | Continuous 3D Perception Model with Persistent State, CVPR 2025 (arXiv 2501.12387) |
| licence | see `LICENSE` — CC BY-NC-SA 4.0 in parts, inherited from DUSt3R/CroCo |

To diff against upstream later:
`git clone https://github.com/CUT3R/CUT3R /tmp/cut3r-upstream && git -C /tmp/cut3r-upstream diff 8bc15dc --no-index -- . <this dir>`

## Local modifications

Anything we change here must be listed, or the reproduction stops being
attributable.

- **`eval/mv_recon/launch.py`** — added a `--datasets` argument selecting a
  subset of the hardcoded `datasets_all`. Marked `LOCAL PATCH` in the source.
  Needed because `datasets_all` always builds both 7-Scenes and NRGBD, and
  `SevenScenes` under `full_video=True` uses a hardcoded 13-sequence list rather
  than listing its root — so an absent 7-Scenes directory could not be skipped.
  **Selection only: the model, protocol and metric code are untouched.**

- **`src/dust3r/model.py`** (`load_model`) — added `weights_only=False` to
  `torch.load`. torch >= 2.6 flipped that default to `True`, and this checkpoint
  pickles an `omegaconf.DictConfig` in `args`, which the restricted unpickler
  rejects. Upstream predates the change. **Compatibility only.**

- **`src/croco/models/pos_embed.py`** — the pure-torch `RoPE2D` fallback now
  handles negative positions. It built its cos/sin table over `arange(seq_len)`
  and indexed it with `F.embedding`, which asserts on a negative index — while
  the CUDA kernel evaluates the angle analytically, so CUT3R's pose token at
  position `-1` is fine there. Offsetting the table origin to `pos_min`
  evaluates the same formula at the same integers, so it is numerically
  equivalent to the kernel rather than an approximation. **Only reachable when
  curope is absent**, which is always: the CUDA extension does not build against
  torch 2.8 (`AT_DISPATCH(tokens.type(), ...)` is the pre-2.4 API, and glibc
  2.41+ clashes with CUDA 12.8's `cospi`/`sinpi` declarations). Repairing it
  would mean editing the kernel, which removes the only reason to prefer it.
  Equivalence is verified numerically against an independent analytic
  implementation of `kernels.cu`'s arithmetic: **max abs error 0.000e+00**,
  including at `pos = -1`.

## What is deliberately not tracked

- `assets/` (41 MB of README gifs), `examples/` (47 MB of demo media) and
  `cut3r.pdf` (11 MB) are gitignored by the parent repo. None is needed to run
  `eval/` or to import `src/dust3r`; `demo.py` is the only consumer of
  `examples/`.
- `src/*.pth` and `data/` stay ignored by this directory's own `.gitignore`.
  Both are recreated per zone by
  `.agents/scratch/reproduction/run_cut3r_mv_recon.sh`, which is the portable
  way to do it — a committed symlink would point at a path that only resolves
  in the zone it was made in.

## Environment

Built by `.agents/scratch/insait_cluster_files/setup_cut3r_env.sh` into a
`cut3r` micromamba env, separate from `lingbot_map`. Upstream's
`numpy==1.26.4` pin is deliberately not honoured; the reasoning is in that
script's header. The CUDA RoPE extension (`src/croco/models/curope`) is **not**
compiled — `croco/models/pos_embed.py:121` falls back to a pure-torch RoPE2D.
