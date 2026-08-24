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

## Calling the env's python directly loses FlashInfer's nvcc (2026-08-23)

The cluster rule "never `micromamba run` inside a job" (it holds a CephFS lock
for the job's whole lifetime) means calling
`"$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"` instead. That skips env
*activation*, and activation was the only thing putting `$ENV_PREFIX/bin` on
`PATH`. FlashInfer finds its toolchain by looking for `nvcc` on `PATH` and then
falling back to `/usr/local/cuda`, which does not exist on these nodes, so every
FlashInfer arm dies per scene with:

```
RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist
```

`cuda-nvcc` *was* installed (see the section above) — it just was not reachable.
Cost 3 arms x 10 scenes before it was spotted. Note the job does not fail fast:
`run_worker.py` catches per scene, so it grinds through all ten and reports
`0/10 scenes succeeded` at the end.

`setup_lingbot_map_env.sh` now exports `CUDA_HOME="$ENV_PREFIX"` and prepends
`$ENV_PREFIX/bin` to `PATH` alongside the existing `CPATH` export, so the direct
interpreter call behaves like an activated env. SDPA arms are unaffected — they
JIT nothing — which is exactly how the cause was isolated: `_use_sdpa: true` ran
fine on the same node while every FlashInfer arm failed.

### Second, separate problem: FlashInfer's JIT is broken on this cluster

With nvcc reachable, the JIT fails outright:

```
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(79): error: exception specification
is incompatible with that of previous function "cospi"
```

glibc 2.41 (current cluster-wide) declares `cospi`/`sinpi`/`cospif`/`sinpif`
`noexcept(true)`; CUDA 12.8's `crt/math_functions.h` declares them without it. So
`ninja` fails and **every FlashInfer arm dies**, on every node, every time.

It masquerades as flaky for two reasons, and I misread it as a race before
checking properly. First, the JIT only fires on the first attention call, several
frames into scene 1 — so a doomed job prints "Running streaming inference" and a
progress bar before dying, and looks healthy for a minute. Second,
`run_worker.py` catches per scene, so the job grinds through all ten and reports
`0/10 scenes succeeded` at the very end rather than failing fast. The check that
settles it is whether `traj.txt` files actually appear: SDPA arms wrote them,
FlashInfer arms wrote none while logging 30 `Ninja build failed` per scene.

Two ways out:

- `_use_sdpa: true`. Works today, ~4.5 min/scene against FlashInfer's ~0.5 on an
  A100 — roughly 9x slower — and it is not the shipped default, so a sweep run
  this way is internally comparable but is not literally the shipped config.
- Install `cuda-nvcc` 12.9+ (guards those declarations) and keep FlashInfer. The
  faithful fix; not attempted here because the env is shared per node and other
  jobs were mid-run.

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
`Total failed > 0`, so the stage runner's `set -e` aborts before
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
the stage runner's `set -euo pipefail` then aborts before `evaluate.py`
ever runs — so one broken method config silently discarded a fully
completed `lingbot_map` run alongside it.

## Oxford Spires upstream data bugs

`get_tls_pcd_path()` only checked singular `merged-cloud-{1,5}cm.pcd` —
`observatory-quarter`'s HF files are `merged-clouds-{1cm-no-colour,5cm}.pcd`
(plural + suffix), same bug class as the christ-church-05 date issue.
`bodleian-library`'s TLS cloud is `.e57`, not `.pcd` (open3d has no `.e57`
support) — converted via `pye57` in
`.agents/scratch/reproduction/oxford/convert_bodleian_e57.py`.

## `setup_lingbot_map_env.sh`/`setup_bench_env.sh` flock held for the job's entire life, not just setup

The 2026-08-16 flock fix (see "Job submission" in `.agents/AGENTS.md`) used
`exec 200>"$LOCK"; flock 200` at the top level of a script that's
`source`d into the caller's shell. `flock` with no `-u` and no subshell
never releases — the fd (and the lock) stays open for the rest of that
shell's life, including everything that runs after sourcing, and is
inherited by every child process it forks. So instead of guarding only the
idempotent create-env check, one job holding the lock during setup keeps
holding it for its *entire runtime*, turning it into an accidental
per-node mutex.

Harmless for short jobs (the next job just waits a few minutes for the
first one's whole process to exit), but wedges everything behind a
long-running job. Found 2026-08-17: `oxford_long` (multi-hour dense
Oxford Spires run, Table 3) grabbed the lock on a shared `gcp-eu1` node at
setup and never let go; five unrelated 1-GPU viewer-cache precompute jobs
Slurm packed onto that same node all blocked on `flock 200` indefinitely
— `ps` showed them parked at the `submit_sof1.sh` line, no error, `sacct`
said RUNNING, nothing looked wrong until `lsof` on the lockfile showed the
oxford job's shell (and everything it had forked) still holding fd 200
hours later. Fixed by scoping `exec/flock` to a `( ... )` subshell around
just the create-env body — the lock now releases when the subshell exits,
before the caller's actual job runs. Lesson: a `source`d script's `flock`
must be subshell-scoped, or "guard a check" silently becomes "serialize
the whole node."

## `max_frame_num` silently caps *keyframes* at 1024; the error names tensor dims

`--max_frame_num` / `_max_frame_num` (default 1024) looks like a 3D-RoPE
table size, but it drives **two independent ceilings**, both counted in
**keyframes**, not input frames:

| limit | derived as | at the 1024 default |
| --- | --- | --- |
| RoPE frequency table | `WanRotaryPosEmbed(max_seq_len=max_frame_num)` | **1024 keyframes** — binds first |
| FlashInfer special page pool | `max_total_frames = max_frame_num + 100` | 3,237 keyframes |

Both count keyframes because `total_frames_processed` (which becomes RoPE's
`f_start`) only increments when `_skip_append` is false —
`aggregator/stream.py:526` — and non-keyframes are rolled back out of the
paged cache.

Past the RoPE ceiling, `freqs[0][slice(f_start, f_end)]` in
`layers/rope.py:399` runs off the end of a `[max_seq_len, ...]` tensor.
Python slicing doesn't raise — it silently returns an **empty** slice, the
temporal frequency component vanishes, and `pos`'s last dim drops 32 → 22.
It surfaces one call later as:

```
RuntimeError: The size of tensor a (32) must match the size of tensor b (22)
              at non-singleton dimension 3
```

Loud, so it can't corrupt results quietly — but the message names neither
`max_frame_num` nor `keyframe_interval`, which is what makes it expensive
to diagnose. Verified 2026-08-19 by calling `WanRotaryPosEmbed` directly at
`f_start` 1000/1023/1024/2000.

Raising `_max_frame_num` lifts both ceilings together; it also grows the
preallocated paged cache, which is a **fixed** allocation, not per-frame
growth: 109 pages x 24 blocks ~ **7.75 GB** at the default, ~9.5 GB at 4096
(+128 MB FlashInfer workspace). Sizing is `flashinfer_cache.py:132-138`.

## `auto` keyframe_interval pins keyframes at ~320 for ANY sequence length

`_keyframe_interval: auto` resolves to `ceil(N / auto_keyframe_threshold)`
with the threshold at 320 (= the paper's training length, §4.4: "trained on
sequences of up to 320 views"). So the *number* of cached keyframes is

```
keyframes ~ num_scale_frames + N / ceil(N/320) ~ num_scale_frames + 320
```

— **constant in N**, not proportional. 3,840 frames -> interval 12 -> ~328
keyframes; 25,000 frames -> interval 79 -> ~325. That's why the ceilings
above are never reached in practice, and it is doing more than bounding
memory: **it means a "dense" run never actually stresses the streaming
state.** Two runs at different frame counts get near-identical KV cache
occupancy, so any sparse-vs-dense comparison under `auto` is confounded
(see `reproduction.md`, Table 3).

Related paper<->code gap: the paper's keyframe selection is **flow-based**
(predict depth+pose, optical flow vs. the last keyframe, promote above a
threshold) and it states the mechanism is "shared by both inference modes."
In the released code `flow_threshold` exists **only** in
`models/gct_stream_window.py` (the windowed/VO model); `models/gct_stream.py`
has no flow logic at all, `demo.py` doesn't expose the flag in either mode,
and `benchmark/methods/lingbot_map.py` passes it only in the windowed
branch. So Direct/streaming mode has no adaptive keyframing available —
only a fixed interval.

Consequence worth remembering: the released streaming path offers no way to
run the paper's dense protocol. `auto` collapses to ~328 keyframes, a fixed
`1` crashes at keyframe 1024 unless `_max_frame_num` is raised too, and
flow-based selection isn't implemented there.
