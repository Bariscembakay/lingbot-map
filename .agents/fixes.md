# Technical fixes & gotchas — lingbot-map

Local-only. Reusable technical knowledge, not current state (that's
`AGENTS.md`).

## Per-node env

`/scratch` is per-node. Source `.agents/scratch/insait_cluster_files/
setup_lingbot_map_env.sh` (+ `setup_bench_env.sh`) at the top of any job —
idempotent, no-op if already built on that node.

## CUDA extension build (nvcc + driver headers + CPATH + lib64)

`preprocess/oxford.py`'s CUDA extension and FlashInfer's JIT kernels both
silently fall back / fail to link unless the env has:

- `cuda-nvcc=12.8` (pip torch only ships the CUDA runtime, not the compiler).
- `cuda-driver-dev=12.8` (`cuda.h`, separate from `cuda-cudart-dev`).
- `$ENV_PREFIX/targets/x86_64-linux/include` on `CPATH` — conda's cuda-*
  packages don't use the standard toolkit layout gcc/nvcc expect.
- A `lib64 -> targets/x86_64-linux/lib` symlink — FlashInfer's build
  links with `-Llib64 -Llib64/stubs -lcuda -lcudart`, which doesn't exist
  under conda's layout either.

All in `setup_lingbot_map_env.sh`. Without the fix: NumPy fallback ran at
~35 sec/frame; with it, ~5.5 frames/sec on an a6000.

Build artifacts (`.ninja_*`, `*.o`, `*.so`) land directly in
`preprocess/points_visibility/` (source dir doubles as build dir) — safe
to delete, never the 3 tracked `.cpp`/`.cu` files.

## `exit 0` in a sourced script kills the whole job

A `source`d script's `exit` terminates the caller's shell, not just
itself. `setup_bench_env.sh` had this in its early-return path — silently
no-op'd an entire job whenever `bench` already existed on that node. Use
if/else, never a bare `exit` in anything meant to be sourced.

## `run.py` needs the `conda` shim on `PATH`

`run.py` spawns `conda run -n {env} ...` as a real subprocess — the
`.bashrc` shell-function `conda` doesn't exist there. A tracked
`~/bin/conda` shim (real executable) already exists cluster-wide, but
`~/bin` isn't on `PATH` by default in job shells. Both bootstrap scripts
now `export PATH="$HOME/bin:$PATH"`.

## Cross-zone GPU

`gcp-eu1-a100-80g-*` shares sof1's `/group`/`/home` (verified via ssh +
`ls`) — unlike msp3, which is genuinely separate. Usable as a drop-in
extra sof1 node.

A node showing free GPUs doesn't mean a job schedules instantly — check
`squeue -w <node> --start` for other users' backfill-scheduled chains or
`PLANNED` reservations.

## Git push auth

VS Code's `GIT_ASKPASS` doesn't reach Bash-tool-spawned shells. Use a
GitHub PAT via `git config --global credential.helper store`.

## Idle-GPU deallocation

See global `~/.claude/CLAUDE.md` — this cluster kills jobs at low GPU
utilization/VRAM. Mitigation: `.agents/scratch/insait_cluster_files/
gpu_keep_alive.py`, wired into `submit_sof1.sh`/`submit_msp3.sh` by
default (`GPU_KEEP_ALIVE_FRACTION`/`GPU_KEEP_ALIVE=0` to adjust).

## TAT (Tanks and Temples) Google Drive

Legacy (~2018) Drive file IDs need a `resourcekey` param `gdown` (any
version) doesn't support — its errors are misleading, but bypassing it
with raw `requests` and following the actual confirm-token flow reveals
the real cause: a genuine, sustained per-file Google Drive quota (not
about file popularity — hit both Barn and Meetingroom). No mirror found
(checked nerfbaselines, HF `kairunwen/InstantSplat`). GT metadata
(log/trans/json) downloaded fine via plain `gdown`; only `.ply` and image
zips hit the quota, and `.ply` isn't even needed (`points` eval disabled).
Direct links + a working retry implementation:
`.agents/scratch/reproduction/tat/retry_tat_images.py`.

## TAT image zips: inconsistent internal layout

Most scene zips extract to `{Scene}/{NNNNNN}.jpg`, but Caterpillar and
Truck are flat (`{NNNNNN}.jpg` at zip root, no subfolder) — extracting
all scenes into a shared parent dir at once causes filename collisions
between the flat ones. Extract each zip to its own `TAT/{scene}/` target
individually, don't batch-extract into one parent.

## VBR / DROID-W / TUM sources

- VBR: LoGeR's HF release (`Junyi42/vbr_processed`) already matches
  lingbot-map's expected content, just a directory rename.
- DROID-W: direct ETH Zurich host, matched the expected layout as-is.
- TUM: no documented "9 sequences" subset exists anywhere (paper doesn't
  use TUM at all) — downloaded the full public benchmark instead.

## Per-node env race condition (multiple jobs, one node)

Requesting 1 GPU per job lets Slurm pack several jobs onto the same 8-GPU
node — and since the micromamba env lives on that node's `/scratch`, two
jobs sourcing `setup_lingbot_map_env.sh`/`setup_bench_env.sh` at once race
on the same check-then-act steps (`lib64` symlink, package installs).
Killed a VBR and a TUM run with `ln: File exists`. Fixed by wrapping both
scripts' bodies in `flock` on a lockfile under `$MAMBA_ROOT_PREFIX`
(distinct fds/files per script since both are sourced together) — second
job waits instead of colliding.

## `seven_scenes.py` bugs (both hit during benchmark testing)

- `get_scenes()` raised `FileNotFoundError` if any `raw_data_root`
  subdirectory lacked a split file — crashed the whole run on a stray
  `eval_gt/` dir left by the unrelated `paper_reproduction` project sharing
  `datasets/7-scenes`. Fixed to `continue` (skip) instead of raising, same
  as Oxford Spires' scene-validation pattern.
- `load_frame_data()` looked for `frame-{id}.depth.proj.png`; the actual
  7-Scenes files (and the class's own docstring) are `.depth.png`. 100%
  frame-load failure across all 18 scenes until fixed — no `.proj` variant
  exists on disk.

## `prepare.py` also aborts the whole pipeline on any failed scene

Not just `run.py` (see below) — `prepare.py` itself exits nonzero whenever
`Total failed > 0`, so `run_benchmark.sh`'s `set -e` aborts before
`run.py`/`evaluate.py` ever run, discarding every scene that DID prepare
successfully. Hit this with TUM: 66/80 scenes prepared fine, but the 14
`_secret` scenes (see `tum.py` fix above) failing was enough to kill the
whole job before any inference ran. General lesson: any dataset loader
that can produce a "no GT" scene must filter it out at `get_scenes()` time
— logging-and-continuing per-scene isn't enough, since the phase-level
nonzero exit still nukes the rest of the pipeline.

## `lingbot_map_v1` — unconfigured method, only referenced by seven_scenes.yaml

`configs/methods/lingbot_map_v1.yaml` is an upstream leftover: placeholder
`_checkpoint: /path/to/lingbot-map.pt` (never filled in) and the same
`env: lingbot-map` (hyphen) vs. actual `lingbot_map` (underscore) env-name
bug fixed earlier for the main method config. We have no v1 checkpoint —
out of scope, same as KITTI/TartanAir. `configs/seven_scenes.yaml` was the
only config listing it under `methods:`; removed. Side effect worth
knowing: `run.py` exits nonzero on ANY failed combination, and
`run_benchmark.sh`'s `set -euo pipefail` then aborts before `evaluate.py`
ever runs — so one broken method config silently discarded a fully
completed `lingbot_map` run alongside it.

## Oxford Spires upstream data bugs

`get_tls_pcd_path()` only checked singular `merged-cloud-{1,5}cm.pcd` —
`observatory-quarter`'s HF files are `merged-clouds-{1cm-no-colour,5cm}.pcd`
(plural + suffix), same bug class as the christ-church-05 date issue.
`bodleian-library`'s TLS cloud is `.e57`, not `.pcd` (open3d has no `.e57`
support) — converted via `pye57` in
`.agents/scratch/reproduction/oxford/convert_bodleian_e57.py`.
