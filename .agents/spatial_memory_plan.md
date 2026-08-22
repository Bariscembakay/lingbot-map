# Persistent summary memory — implementation plan

Architecture is in `spatial_memory_design.md`. This is the build order.

Rule for the whole plan: **nothing runs on a dev set until the validation suite
in Phase 4 is green.** The point of Phase 4 is that no GPU-hours are spent on a
harness that was never going to learn.

Revision 2026-08-22 for the read/write-between-aggregator-and-heads
architecture. Superseded: the standalone-memory build order, the 4-tap
partitioned state, D=2048, teacher-distillation as the primary loss, 512-frame
clips at stride 5.

## The two-stage structure, and why it survives

Injection happens **after** the aggregator, so the aggregator's output for frame
*i* does not depend on the state. It is still precomputable once.

```
STAGE 1 (once)   frozen lingbot-map, per clip
                   +- 4 tap tensors per frame      -> write input (tap 23 second
                   |                                  half) + frozen head inputs
                   +- depth + conf per frame       -> recall-loss target
                   +- pose_enc per frame           -> raymap queries for Loss 2
                   +- GT depth, GT pose, scale s   -> Loss 1 targets
                   +- revisit score per frame      -> can this clip show anything

STAGE 2 (many)   train write/read on cached tensors.
                 The aggregator never runs again; the heads do.
```

The frozen aggregator forward is ~7 TFLOPs/frame. In-loop it would run once per
clip pass -- hundreds of times per training run. Precomputed it runs once, ever.
The label is also frozen, so two arms are exactly comparable.

**Training must not import the aggregator.** If `scripts/memory/train.py` does,
something is wrong. It needs only the depth head, the camera head, and the cache.

## Layout

```
lingbot_map/memory/
    cache_format.py   ClipCache / ClipMeta -- the on-disk contract      [done]
    attention.py      CrossAttention, SelfAttention, Mlp, LayerScale    [done]
    write.py          WriteBlock, WriteTransformer                      [done]
    read.py           ReadBlock, ReadTransformer, zero-init gates        [done]
    raymap.py         pose_enc -> Plucker rays -> encoder               [done, Loss 2 only]
    schedule.py       WriteSchedule: disjoint / overlap + coverage audit [done]
    camera_bridge.py  teacher camera cache + per-step refined pose        [done]
    model.py          SummaryMemory.step / .read_at_camera              [done]
    losses.py         L_depth / L_abs_pose / L_rel_pose, recall, hindsight
    data.py           ClipReader, tap-23 split, query sampling             [done]
    baselines.py      frozen-state arm, no-read arm
scripts/memory/
    export_head.py       depth_head.pt + camera_head.pt                 [done, depth only]
    build_cache.py       stage 1                                        [done, needs GT + revisit]
    submit_build_cache.sh                                               [done]
    validate.py          the Phase 4 gate
    time_clip.py
    train.py
```

## Phase 1 — cache builder

- Decode `iphone/rgb.mkv` at **stride 20**, take 320-frame clips (a 168 s /
  10,110-frame recording gives ~505 frames at stride 20, so one full clip per
  scene; shorter recordings truncate and `meta.json` records the actual length).
  Stride 20 rather than 10 because the measured revisit score roughly doubles;
  see the design record.
- Resize with the benchmark's two steps: width 518, height floored to a multiple
  of 14, then `area_budget=255000` with `align=14`. LANCZOS, no crop. 4:3 ->
  518x378, 999 patches.
- Run the frozen model via `inference_streaming` with the **published config** --
  `num_scale_frames=8, kv_cache_sliding_window=64, keyframe_interval=1,
  enable_3d_rope=True, use_sdpa=False` -- so the teacher is the published model,
  not a variant.
- Per clip, contiguous, fp16:

  | file | shape | for |
  |---|---|---|
  | `taps.npy` | `[320, 4, 1005, 2048]` | write input + frozen head input |
  | `depth.npy` | `[320, 378, 518]` | recall target |
  | `conf.npy` | `[320, 378, 518]` | recall weighting |
  | `pose_enc.npy` | `[320, 9]` fp32 | Loss 2 raymap queries |
  | `gt_depth.npy` | `[320, 378, 518]` | **Loss 1 target** |
  | `gt_pose.npy` | `[320, 4, 4]` fp32 | **Loss 1 target**, c2w, canonical frame |
  | `meta.json` | | see below |

  ~17.6 MB/frame -> 5.6 GB per clip.

- **GT preparation** (`memory/gt.py`, done):
  - `iphone/depth.bin` is `uint32` length-prefixed **LZ4 blocks** of 256x192
    `uint16` millimetres -- established by inspection, not documentation. Reads a
    320-frame clip in 0.7 s with 0% invalid pixels. Resized nearest so invalid
    pixels never bleed. Switch to the laser mesh at `aligned_pose` before any
    number goes in a paper.
  - **The camera axis convention is detected, not assumed.** `detect_convention`
    scores the four candidates against the model's own trajectory using
    *long-baseline* relative rotations. Consecutive-frame rotations are
    near-identity in a slow room scan and separated the candidates by only ~5
    degrees, comparable to the teacher's own error; long baselines give a **40
    degree** margin. All four scores go in `meta.json` so a wrong pick is visible.
  - `s = mean ||x||_2` over the **anchor frames'** GT point cloud, the paper's §3.2
    rule. Divide GT depth and GT translations by it; store `s`. Measured
    s = 3.43 m on the first scene, after which GT depth percentiles
    (0.25 / 0.54 / 1.70) sit close to the model's own (0.30 / 0.67 / 1.58) --
    independent evidence the transform is right.
  - GT poses as **camera-to-world**, expressed relative to the anchor frame
    convention, translations scaled by `s`. This transformation is where mistakes
    hide; V1c below checks it.
- **Revisit score per frame**: fraction of this frame's visible surface last
  observed more than 72 frames ago, from the teacher's own depth and poses. This
  decides whether the clip can show a Loss-1 effect at all, and it is how the
  final stride gets chosen. Store per frame and as a histogram in `meta.json`.
- `meta.json` also carries: scene, clip index, stride, frame ids, H, W, P,
  patch_start_idx, tap layers, `s`, ray-origin magnitude percentiles, checkpoint
  sha256, git sha and dirty flag. A cache from a different checkpoint is a
  different dataset.
- **Storage: `/group/compact-3dmem/datasets/<name>`, never `/scratch`.** `/scratch`
  is `/dev/md40`, a per-node local ext4 disk, so a cache written by a build job on
  one node is invisible to a training job on another. `/group` is CephFS,
  zone-shared, 2.8 T free. Same reason sbatch `--output` goes under `/home`:
  Slurm opens the log on the compute node, and a `/scratch` path there does not
  exist (this cost job 731866, exit 127 in 1 s with no log).
  Then `dataset create` on sof1 and `dataset pull` + `dataset pin` on msp3.
- Whether a training job reads the cache straight off CephFS or stages its clips
  to local `/scratch` first is a **Phase 5 measurement**, not a guess. Phase 5
  predicted 0.3-1.2 GB/s sustained; if CephFS falls short, stage per job.

`export_head.py` exports **both** frozen heads: depth (62 keys, 32.7 M) and
camera (69 keys, **216.2 M**). 995 MB total, at
`/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt`. [done]

## Phase 2 — modules

Rework from the earlier 4-tap design:

- `SummaryMemory`: one state `[N, 1024]`, N=512. `state_init` is an
  `nn.Parameter`, reset at every clip start.
- `step(x_i, S_prev) -> (S_i, refined_x_i)`, computing both **from `S_prev`**.
- `WriteTransformer`: `L_w` pre-norm layers of
  `[cross-attn(Q=S, KV=x) -> self-attn(S) -> MLP]`.
- `ReadTransformer`: `L_r` pre-norm layers of
  `[cross-attn(Q=x, KV=S) -> MLP]`, **separately gated** so the state path's
  contribution can be measured rather than inferred.
- Zero-init `LayerScale` on the read residual, so step 0 is bit-identically
  published lingbot-map and every improvement is attributable.
- `raymap.py` stays as built but outputs 1024-d, and is used only by Loss 2.
- `WriteSchedule` selects `disjoint` (lag = window-1 = 63, exact partition with
  the cache) or `overlap` (lag 0, CUT3R's regime). Both implemented; disjoint is
  the default and the two are a first-class ablation.
- Flags wired from day one: `position_mode {none,pose,xyz}`,
  `write_residual_gate`, `frozen_state` (the control arm), and later `interleaved`
  (the CUT3R-style escalation).
- The read's residual gates are **always present and zero-initialised** -- that is
  what makes the untrained model bit-identical to published lingbot-map (V9).
  Cross-attention and MLP gate separately so the state path's contribution is
  measurable.

## Phase 3 — training loop

- Iterate a clip sequentially: at step *i*, `S_i, y_i = step(x_i, S_{i-1})`.
- **Unroll long, supervise sparsely.** 32-64 steps in the graph; evaluate the
  heads and Loss 1 every 4th step. The recurrence is ~435 GFLOPs/frame and the
  DPT head is ~1.5 TFLOPs, so this buys 4-8x longer write gradient at the same
  activation budget. This is the highest-leverage knob in the setup.
- Keep a rolling buffer of the last 64 **predicted** poses for `L_rel_pose`;
  detach those outside the BPTT window.
- Mirror the paper's **progressive view training**: start on short subsequences
  and grow the view count.
- AdamW, lr 1e-4, linear warmup + cosine (CUT3R and the paper agree), grad clip
  1.0, bf16 autocast.
- Loss weights `lambda_depth / abs / rel / trans`, `alpha`, `eps` are not in the
  paper -- start from VGGT's and sweep small.
- Build order for Loss 1: **depth term first**, then add the pose terms once
  depth works. The camera head's never-evicting causal cache becomes a second
  recurrence as soon as it sees refined tokens, and that is worth confronting
  separately from getting depth to move.
- Loss 2 (recall, hindsight) implemented but its experiments deferred. Planned
  arms: loss1 alone / loss1+hindsight / loss1+recall.

### wandb

`wandb login` on msp3 once (it has public internet), `WANDB_DIR` on `/scratch`,
never `/home`.

- project `lingbot-summary-memory`, run name = arm name, full config logged.
- per step: `loss/{total,depth,abs_pose,rel_pose}`, `lr`, `grad_norm/{write,read}`,
  `state/norm`, `gate/{read_cross,read_mlp}`, `nan_count`.
- per eval, the chart that is the experiment: **depth error bucketed by revisit
  score**, with the frozen-state arm and frozen lingbot-map on the same axes. An
  average over all frames will hide a real effect.
- images every few thousand steps: predicted vs GT depth and the error map.
- system: s/clip-pass, peak VRAM, dataloader wait fraction.
- **Resumable**: `wandb.init(resume="allow")` plus step-level checkpointing.
  `PreemptMode=REQUEUE` means a `batch` job gets bounced back into the queue by
  someone's `debug`/`rendering` job and restarted. V11 tests this.

## Phase 4 — validation suite (gate)

`scripts/memory/validate.py`, minutes on one a6000.

| id | check | catches |
|---|---|---|
| V1 | cached taps -> frozen DPT head reproduces cached `depth.npy` within fp16 tol | wrong tap order, wrong slicing, corrupt memmap |
| V1b | `sign(d)*log1p(|d|)` round-trips through `inv_log` back to `d` | the y-space loss built on a wrong inverse |
| V1c | GT poses, after the canonical transform, reproduce the teacher's poses to within its own error | the c2w / anchor-frame / scale transform, where mistakes hide |
| V1d | `camera_bridge.pose_at` fed the teacher's own token reproduces the teacher's pose exactly, and leaves the teacher cache unmutated | cache slicing, `frame_idx` (which flips `is_scale_frames`), accidental in-place writes. **already passing: 0.0 err** |
| V2 | perturb one patch token; depth changes at the expected pixel | transposed / column-major grid |
| V3 | refined tokens feed both heads and yield documented shapes | interface drift, P hardcoding |
| V4 | after one backward: every trainable param has finite non-zero grad; every frozen param has `grad is None` | detached graph, unfrozen backbone |
| V5 | zeroing the state changes the read output; the read at step *i* is unaffected by frame *i*'s write | that parallel read/write is wired as designed, not accidentally sequential |
| V5b | `coverage_report` in disjoint mode: zero gaps and zero overlaps over a full clip | the lag being off by one, which silently loses a frame from both memories |
| V6 | gradient does not reach writes older than the unroll window | detach in the wrong place |
| V7 | 320 untrained steps; log `norm(S)` per step | state drift or collapse -- must be known before training, not after |
| V8 | overfit one clip to ~0 loss | if it cannot memorise one sample the wiring is wrong |
| V9 | at init, with the zero-init gate, output is **bit-identical** to frozen lingbot-map | the attributability baseline |
| V10 | fixed seed reproduces bitwise | hidden nondeterminism |
| V11 | kill and resume mid-run; the loss curve continues without a discontinuity | non-resumable run under REQUEUE preemption |
| V4b | Loss-2 path: the raymap encoder receives gradient | the raymap branch silently disconnected |
| V4c | refined tap 23 reaches the camera head and back; the zero-init gate opens, then the branch goes live | passing the 1024-d refined half where the head needs the full 2048-d token, and mistaking zero-init gating for a dead path |
| V12 | report peak VRAM and s/clip-pass at target config | sizing the real run |

**Status: 12/12 passing** against a synthetic cache
(`make_synthetic_cache.py`) on CPU. V1 is provisional until it runs on a real
teacher cache: the synthetic depth is produced by the same frozen head, so V1
currently proves the memmap/slicing/dtype plumbing but not that the *builder*
stores what it claims.

Four real defects the gate caught before any GPU time:

1. **The depth head's activation is `exp`, not `inv_log`**, and `output_dim=2`,
   not 4. `DPTHead`'s defaults belong to the point heads. Wrong shape fails
   loudly; wrong activation would have failed silently. Fixed by routing all head
   construction through `memory/frozen.py`.
2. **The write recurrence diverged**: `||S||` 14.5 -> 40,206 (x2775) over 128
   untrained frames. Fixed with a LayerNorm on the state (now x1.000) and a
   matching `std=1.0` state init.
3. **The camera arm passed the 1024-d refined half** where the camera head needs
   the full 2048-d tap-23 token. Now both heads consume one `rebuild_tap23`.
4. **V2 was degenerate** -- `patch_h//3 == patch_w//4`, so the transposed
   candidate coincided with the row-major one and the check could not fail.

## Phase 5 — timing run

One clip, one GPU, measured not estimated. Current paper estimate: write ~258 +
read ~177 GFLOPs/frame plus a ~1.5 TFLOPs DPT head, so the head is ~77% of cost
and a 320-frame clip pass should be seconds. Also confirms whether the job is
I/O bound, which sets `/scratch` layout and worker count.

## Phase 6 — first dev run

- **24 train / 4 val scenes**, 3 clips of 320 frames each. ~473 GB cache at
  17.6 MB/frame; final sizing from the Phase 5 number, not this line.
- Arms in the first batch, because the controls are what make the result mean
  anything:

  | arm | |
  |---|---|
  | frozen lingbot-map | the baseline, exact at step 0 |
  | **frozen state** | same read transformer, state pinned at init and never written |
  | **arm A** `refine_taps=(0,1,2,3)`, depth + pose | the full claim |
  | **arm B** `refine_taps=(3,)`, pose only | tap 23 is inert for depth, so this is the clean trajectory arm |
  | full, N=512, disjoint, window 64 | story (a): same cache, better predictions |
  | full, N=512, disjoint, **window 16** | story (b): smaller cache, same quality -- and where the revisit score says the signal is |
  | full, N=1024, disjoint | separates capacity from mechanism |
  | full, N=512, overlap | does duplicating the cache help or waste capacity |

- Go / no-go: does depth error improve **over the frozen-state arm**, bucketed by
  revisit score. Beating only frozen lingbot-map proves nothing -- a plain adapter
  would do that.
- The measured revisit score says only 33-43% of frames at window 64 can show a
  Loss-1 effect at all, so **report bucketed, never averaged**, and run the
  window-16 arm alongside.
- Then: trajectory-memory on/off x state on/off; write layers; gate vs plain;
  N sweep.
- Then Loss 2, then `a` vs `c` (position_mode).

## Open items, in priority order

1. **Validate the pose pipeline on a benchmark dataset with trusted GT.** Run the
   repo's own `benchmark/` ATE path (evo, `correct_scale=True`) on e.g. TUM-RGBD or
   ETH3D and compare against the published numbers. This is the only way to
   separate "the model's pose is poor on this clip" from "our GT transform is
   wrong", and no trajectory claim can be made before it. **Contamination-free
   signal: use a benchmark, not our ScanNet++ cache.**
2. Build a clip starting mid-recording rather than at source frame 0 -- the
   opening seconds of a handheld scan make poor anchor frames, and all 8 anchors
   set the trajectory's reference.
3. Implement V1c (GT poses reproducing the teacher's within its own error). It was
   specced and never written, which is why (1) is still open.
4. Re-measure the revisit score and the depth-ratio trend on a 320-frame clip and
   across scenes; the current numbers are one 128-frame clip.

## Cluster notes

**Everything on msp3.** One-time `dataset pull ScanNetpp` + `dataset pin`
(1.79 TiB, ~6.6 h at the observed 75 MB/s; lands as a `/data` mount, does not
touch msp3's 300 G `/group`). Worth it because the cache will be rebuilt more
than once. `--constraint=zone-msp3 --gpus=h200:1`.

Validation runs anywhere; `hala` a6000 with `--partition=rendering
--qos=rendering` is cheapest, and both hala partitions **require** a matching
`--qos` or the job is rejected outright.

**Only committed, pushed code runs on msp3.** `lingbot_map/memory/` and
`scripts/memory/` must be pushed before any msp3 job; an agent shell needs a PAT
via `git config --global credential.helper store`.

**Frozen surface at training time:** `frozen_heads.pt` (995 MB, depth + camera)
at `$GROUP_ROOT/checkpoints/lingbot-map/`. No aggregator.

**Always pass `--cpus-per-task` and `--mem` explicitly** -- nothing is auto-sized
here and the default is `cpu=1, mem=2G`.

**Nothing durable on `/scratch`, and no sbatch `--output` there either.** It is a
per-node local disk. Logs go to `/home`, caches to `/group`, only genuine
per-job intermediates to `/scratch`.

**Run `gpu_keep_alive.py 0.05`** alongside training: small kernels plus heavy I/O
reads as idle to the reaper.

**Results reduce in-zone; the record lives on sof1.** `rsync -a` to the `sof1`
alias under `campaigns/summary-memory/<arm>/<benchmark>/<scene>` via
`campaign_dir()`. Arm names carry what makes numbers incomparable, e.g.
`a_N512_L2_320f_s10`. Smoke runs are deleted, not filed.
