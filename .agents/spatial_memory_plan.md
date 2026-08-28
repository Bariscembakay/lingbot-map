# Spatial memory — implementation plan

Companion to `spatial_memory_design.md` (design closed 2026-08-26). This file is
sequencing and status only; every architectural decision lives in the design
record and is not restated here.

**Revision 2026-08-26.** Rewritten for the CUT3R-fed-by-lingbot-map
architecture. The previous plan (Loss 1, read/write pair, four smoke arms) is
recoverable at `3d34cd5`.

## The two results

**E1** — CUT3R hallucinates when a past camera is queried by raymap alone.
**E2** — lingbot-map + CUT3R recalls it instead.

Everything below is either a prerequisite for measuring that, or the measurement.

## Phase table

| phase | what | blocks | runs where |
|---|---|---|---|
| **0** | unblock: checkpoints, env, data | 1 | login / CPU |
| **1** | reproduce CUT3R on NRGBD | trust in arm A | hala a6000 |
| **2** | implement our model + validators + one-scene overfit | 4 | sof1 |
| **3** | shared recall harness (drives arm A and arm C) | E1, E2 | hala / sof1 |
| **4** | small-dataset training; arm A recall eval; compare | — | sof1, msp3 at scale |

0 and 2 overlap: 0a/0b are waiting, 2 is work. 1 and 2 overlap. 3 is written
against arm A first, because arm A exists before our weights do.

---

## Phase 0 — unblock

- [ ] **0a. CUT3R checkpoints** -> `/group/compact-3dmem/checkpoints/CUT3R/`.
      `cut3r_512_dpt_4_64.pth` is the one we build on; `cut3r_224_linear_4.pth`
      is a backup. Record sha256 for both.
      Note: this gdown build has no `--fuzzy`; use the Python API with `id=`.
      Google Drive quota can bite -- HF mirror is the fallback.
- [ ] **0b. `cut3r` micromamba env**, separate from `lingbot_map` so it cannot
      break the working one. torch cu128, accelerate, open3d, einops,
      transformers. **Skip gsplat** (training-logging only). **No evo** -- only
      `eval/relpose` uses it, and it pulls pandas.
      **curope MUST be built** (needs `cuda-nvcc=12.8`). An earlier reading of
      this repo said the pure-torch RoPE2D fallback made it optional; that was
      wrong. The fallback is not merely slower, it is not equivalent: the CUDA
      kernel computes the angle analytically so a negative position is a
      negative angle, while the fallback indexes a table built over
      `arange(seq_len)` with `F.embedding`, which asserts on a negative index --
      and CUT3R gives its pose token position -1. Also note **upstream's
      `numpy==1.26.4` pin is stale** and deliberately not honoured; reasoning is
      in the setup script's header.
- [ ] **0c. NRGBD symlink**: `CUT3R/data/neural_rgbd -> /data/NRGBD`. The layout
      already matches CUT3R exactly -- `{scene}/images/img{N}.png`,
      `depth/depth{N}.png`, `poses.txt`, `focal.txt`. Nothing to preprocess.
- [ ] *(deferred)* **7-Scenes**. CUT3R wants `frame-{N}.depth.proj.png` and we
      have none. We do have the code -- `benchmark/datasets/seven_scenes.py:233`
      `_register_depth`, exposed by `benchmark/configs/datasets/seven_scenes_depthproj.yaml`.
      **Deliberately not on the critical path**: our reprojection uses estimated
      Kinect calibration (zinsmatt/7-Scenes-Calibration), not Spann3R's pipeline,
      so a mismatch there would not cleanly mean "we are driving CUT3R wrong".
      NRGBD alone gates that.

## Phase 1 — reproduce CUT3R

`eval/mv_recon`, `cut3r_512_dpt_4_64.pth`, protocol already encoded in
`launch.py` (`full_video=True`, `kf_every=500` for NRGBD -- the paper's
"2 to 4 frames per scene" sparse setting).

Targets, Table 4:

| | Acc mean/med | Comp mean/med | NC mean/med |
|---|---|---|---|
| **NRGBD** | 0.099 / 0.031 | 0.076 / 0.026 | 0.837 / 0.971 |
| 7-Scenes *(deferred)* | 0.126 / 0.047 | 0.154 / 0.031 | 0.727 / 0.834 |

Table 5, `--revisit 2 --freeze`:

| | Acc | Comp | NC |
|---|---|---|---|
| NRGBD, Ours Revisit | 0.094 | 0.076 | 0.844 |

**Gate: PASSED 2026-08-26** (job 750327, `cut3r_512_dpt_4_64.pth` sha256
`45f7e98a...`, rtx6000, 9/9 scenes, 33 s). Measured against Table 4:

| metric | ours | paper | delta |
|---|---|---|---|
| Acc | **0.0999** | 0.099 | **+0.9%** |
| Comp | **0.0781** | 0.076 | +2.7% |
| NC | **0.8339** | 0.837 | -0.4% |
| NC med | **0.9701** | 0.971 | -0.1% |
| Acc med | 0.0366 | 0.031 | +18.1% |
| Comp med | 0.0318 | 0.026 | +22.3% |

Four of six within 3%, including the headline Acc. **We are driving the model
correctly.**

### The median residual, and what it is not

The two medians sit ~20% high, reproducibly. Two explanations were tested and
neither is the cause, so the residual is recorded as **unexplained** rather than
argued away:

1. **Not run-to-run noise.** `launch.py` passes no `seed`, so `base.py:83` falls
   back to `torch.initial_seed()` and `_crop_resize_if_necessary` draws
   `rng.integers(2)` -- which looked like nondeterminism. Four repeat runs
   (750432-4) returned **identical numbers to four decimals, spread 0.0000**.
   The draw happens but does not change the outcome at this resolution.
2. **Not the depth variant, on its own.** NRGBD ships `depth/`,
   `depth_filtered/` and `depth_with_noise/`; the loader reads `depth/`, while
   reconstruction work often scores against the filtered maps (which mask 8.3%
   of pixels here and differ by 1.3 cm mean where both are valid). Tested
   (job 750447, `NRGBD_DEPTH=depth_filtered`): it helps four metrics and hurts
   two -- Acc goes +0.9% -> **-5.8%**, NC med -0.1% -> -2.5%, while Acc med
   +18.1% -> +7.9% and NC lands at -0.0%. No single variant matches on all six.

**Decision: both arms use raw `depth/`**, which is what the released loader
reads, and which matches on Acc and NC med. What matters for E1 vs E2 is that
arm A and arm C are scored against the *same* GT, not which variant is closer to
a published table.

Untested candidates, not worth GPU time now: a different NRGBD release (frame
counts, pose file), a paper revision predating the released code, or a different
median aggregation (pooled over points vs mean of per-scene medians -- the last
is not testable from the logs).

Nothing in Phase 3 or 4 was trustworthy until this passed.

Worth reproducing Table 5 as well, because **"Ours Revisit" is CUT3R's own recall
experiment** -- freeze the final state, re-process the same inputs. It is weaker
than our probe (it still hands the model the *image*), but it is the authors' own
evidence that the state accumulates scene knowledge, and it is a free reference
point on the same axis.

- [ ] 1a. NRGBD, online. Compare to Table 4.
- [ ] 1b. NRGBD, `--revisit 2 --freeze`. Compare to Table 5.
- [ ] 1c. Record both in the campaign record with the checkpoint sha256.

## Phase 2 — implement our model

- [ ] **2a. Cache v3 -> v4.** Add `gt_intrinsics.npy [N,3,3]` (post-resize, per
      frame); bump `FORMAT_VERSION`. GT pointmaps stay derived on the fly.
      Extend one existing clip rather than rebuilding.
- [ ] **2b. Model.** `in_proj` (2048->768 on tap 23), state + `s0` from the
      checkpoint, both decoder stacks from the checkpoint, raymap encoder (2
      blocks, scratch), mod token, two DPT heads at `patch_size=14` with the
      terminal `interpolate` to (H, W).
- [ ] **2c. Token-provider interface.** The training loop takes tokens from a
      provider, not from the aggregator directly, so **arm B** (CUT3R's own
      encoder, our objective) is later a config flag rather than a rewrite.
      Implemented now because it is tidier now; not run now.
- [ ] **2d. Validators**, in priority order:

  | | check |
  |---|---|
  | V1 | **probe gradient reaches the write at step *q* for *q* << *t*.** The load-bearing claim of the design. If this fails, nothing else matters. |
  | V2 | probe gradient reaches `s0` |
  | V3 | zeroing `s_t` changes the probe output -- no dead path |
  | V4 | head output shape `== (H, W)` after the terminal interpolate |
  | V5 | token counts and slices: 1005->999 write, 1000->999 probe |
  | V6 | **leak check**: the probe forward never touches the tap tensors |
  | V7 | raymap origins are c2w, checked **positionally** (magnitude checks are blind: `||t_w2c|| == ||t_c2w||`) |
  | V8 | peak VRAM and s/step at 160 frames x 4 probes, checkpointed |

- [ ] **2e. One-scene overfit + visualisation.** Purpose is implementation and
      gradient flow, not a result. Visualise probe pointmaps against GT per lag
      via `vis/glb_export.py` / `vis/viser_wrapper.py`.
      **The no-write control runs here too** -- see the design record: it is
      mandatory at every rung, and its failure signature is a lag sweep that is
      flat *and* good.

## Phase 3 — the shared recall harness

One implementation driving **both** CUT3R (`inference_step`) and ours, so E1 and
E2 are the same measurement rather than two similar ones.

### The query is not sequential, and that changes the harness

A probe needs only `s_t` and a raymap. So once the clip has been consumed, every
frame's camera can be queried **in one batched pass** -- no recurrence, batch
over *q*. Two consequences:

1. **Eval is cheap and embarrassingly parallel**, unlike training's per-frame
   loop. Batch dimension = number of probes.
2. **Unioning all probe pointmaps gives a whole-scene point cloud** in one shot,
   which is exactly the input `eval/mv_recon/utils.py` (`accuracy`,
   `completion`) already consumes. So arm A and arm C land on the paper's own
   axis with their own code.

### Two modes, same harness

| mode | what | gives |
|---|---|---|
| **final-state** | probe every frame's camera against `s_T` | the headline number, and the direct parallel to CUT3R's "Revisit" -- except we withhold the image |
| **online** | probe frame *q* against `s_t` for several *t > q* | the lag curve |

### Metrics

- **Primary: Acc / Comp / NC**, via their `Regr3D_t_ScaleShiftInv`, so arm A sits
  directly on Table 4's axis and we reuse their code.
- **Secondary: AbsRel and delta<1.25 on ray depth** -- readable, and standard.
- **Report both unaligned (metric) and per-sequence-scale.** CUT3R claims metric
  while our canonical normalisation is a different convention; conflating them
  would quietly decide the result.
- **Everything broken down by lag `t - q`.** That is the plot that carries the
  claim.

### Protocol details read out of `eval/mv_recon/launch.py` (must be matched)

- **GT points come from unprojected GT depth, not a mesh.** No NRGBD ground-truth
  mesh is needed.
- **Predicted points are aligned to GT by point-to-point ICP** before Acc / Comp /
  NC (`o3d.pipelines.registration.registration_icp`, identity init). The metric is
  therefore invariant to a rigid misalignment -- generous, and *not* what "metric
  scale" alone implies. Our recall harness must apply the same alignment, or arm A
  and arm C are not on the same axis.
- Normals for NC come from `estimate_normals()` on both clouds *after* that
  alignment.
- NRGBD intrinsics are **hardcoded** in the loader (`fx = fy = 554.2562584220408,
  cx = 320, cy = 240`); `focal.txt` is ignored.
- Frame subsampling is `img_idxs[:: min(kf_every, len(img_idxs) // 2)]`, so
  `kf_every=500` yields the paper's "2 to 4 frames per scene".

- [ ] 3a. Harness against arm A (exists before our weights do). This *is* E1.
- [ ] 3b. Same harness against arm C.

## Overfit result (2026-08-27) -- the setup works end to end

Job 753962, scene `285efbc7cf`, rtx6000, 200 updates in 46 min, **19.1 GB flat**.

| | start | end | ratio |
|---|---|---|---|
| loss | 3.6859 | **0.6501** | 5.7x |
| L21 self | 0.9325 | **0.1736** | 5.4x |
| L21 world | 1.0491 | **0.4788** | 2.2x |
| state norm | 641.5 | 639.1 | **stable -- no drift over 200 recurrent updates** |

The stable state norm is a real confirmation, not a formality: the design argued
that the per-frame `dec_norm_state` keeps `d s_t / d s_{t-1}` from drifting, and
this is that claim holding across a full run rather than a synthetic check.

**A premature call, recorded so it is not repeated.** At step 80 `L21 world` was
flat (1.05 -> 1.03) while `L21 self` had halved, and it was flagged as a likely
fault in `dpt_cross`'s adaLN modulation -- the one path with no pretrained
weights. It was not a fault. The world head simply lags the self head and ended
2.2x better. Do not diagnose a head from its first 80 steps.

### Overfit v2 (2026-08-27, job 754903) -- eliminate "too short" before touching the architecture

753962's curve never flattened (L21 self 0.43 -> 0.23 -> 0.16 over the last 100
steps), half the run was warmup (`--warmup 100` of 200), the first ~25 steps were
a clipped gradient explosion, and probe density was still the a6000-OOM setting
(`--probe-every 8 --n-past 1`) despite landing on a 96 GB card. None of that is
architectural, so no architecture change until this run plateaus:

- same scene/cache, `--updates 2000 --probe-every 4 --n-past 4` (design
  defaults), warmup now 5% of the run, viz + ckpt every 250 steps
- `gcp-eu1-rtx6000-txtb` (RTX PRO 6000 Blackwell, 96 GB -- checked, not 24/48),
  est ~22 s/step ~= 12 h, 16 h walltime
- wandb online (`--wandb`, project `spatial_memory`, graceful no-op on failure)
- job script now mirrors /scratch -> /group every 10 min: an EXIT trap does not
  run under SIGKILL after KillWait, so a walltime kill would have stranded
  everything on the node
- out: `/group/compact-3dmem/campaigns/spatial_memory/overfit_1scene_2k`

Known-minor, deliberately untouched (one variable at a time): every step clips
at |g|~150 vs threshold 1.0 (AdamW largely absorbs a uniform rescale); loss
regresses metric points with no avg_dis normalisation unlike CUT3R's Regr3D --
fallback axis if this run plateaus high.

### 32-scene stage 1 (2026-08-28, jobs 756322 write / 756323 no-write twin)

The single-scene overfit's no-write control matched the write arm at every
step -- the scene was in the weights, so the arms measuring write-side axes
went uninformative. This pair is the designed fix and the new decisive
experiment: 32 train scenes, batch 4 (scene-specific memorisation gradients
cancel across the batch; the read-the-state gradient is common and adds),
16-frame random windows anchored at their own start (short-stream curriculum,
CUT3R-style; stage 2+ warm-starts longer via --init-from), tbptt-8, msp3 H200s,
data via `dataset pull lingbot-tapcache-v4-40`.

**The verdict metric is `valm_*`/`valx_*`: recall on the 8 unseen val scenes,
write vs no-write.** Weights cannot memorise unseen scenes, so the gap is the
memory. Untrained reference valm_self = 1.003. Measured scaling at 16 frames:
peak = ~37 GB + 15.3 GB x batch (B=8 needs ~160 GB -- does not fit an H200;
B=4 = 98.6 GB), steady ~8-11 s/step at B=4, 4000 updates ~= 10 h.

### The 32-scene axis fleet (2026-08-28, evening)

Nine one-flag arms, one protocol (16-frame windows, batch 4, 4000 updates),
one verdict metric (`valm_self` on the 8 unseen scenes), 8 H200s at the
etiquette cap (state1536 gated `afterany` on the twin's slot):

| job | arm | axis |
|---|---|---|
| 756322 | write | reference |
| 756323 | no-write twin | state-usage floor |
| 757883 | fullbptt (`--tbptt 0`) | truncation cost, honest re-run |
| 757884 | probecur_off | lag-0 echo |
| 757885 | raymap_true | input convention |
| 757886 | freeze_s0 | trainable-constant channel |
| 757997 | raydepth (`--head raydepth`) | read head capacity: 0.8 M vs 59 M, one geometry on true rays; untrained val 0.32 vs 1.00 (exp(0)=1 is a plausible depth) |
| 757998 | tapsall (`--taps all`) | encoder taps; 105 GB CPU-resident, 66 MB/step H2D |
| 757952 | state1536 | state capacity x2; loaded prior tiled with 1e-4 noise |

Still held: small read DECODER (2-4 cross-attn blocks -- needs decoupling the
probe path from the interconnected write path) and per-window optimizer
stepping.

### Stage-1 verdict (2026-08-28): the write earns 43-47% on unseen scenes

756322/756323 both COMPLETED (13h20 / 10h08). Mean of the last five val
passes, 8 unseen scenes:

| | write | no-write | gap |
|---|---|---|---|
| valm_self | 0.169 | 0.316 | **47%** |
| valm_world | 0.192 | 0.338 | 43% |
| valx_self (96f) | 0.227 | 0.332 | 32% |
| valx_world | 0.274 | 0.360 | 24% |

First defensible claim of the project: recall on unseen scenes materially
beats the weights+prior ceiling, so the state is carrying scene content.
The extrapolation gap (valx < valm) says the advantage shrinks -- not
vanishes -- at 6x the trained stream length; stage 2 (longer windows,
--init-from) is the designed answer. Per-lag numbers from the 160-frame
unseen-scene viz are single-frame samples at 10x the trained horizon --
jagged (lag1 med 0.93 vs lag159 med 0.13) and not to be over-read; the
averaged val metrics are the evidence. Clouds:
`scenes32_16f_b4/ply_val_unseen_final` (browsable).

### Render naming convention (2026-08-28)

Render dirs are viser browser labels (`<dir> | <arm>`), so they encode
split, scene, checkpoint step and stream length:
`val_<scene>_step<K>_<N>frames` / `train_<scene>_step<K>_<N>frames`,
one render per (checkpoint, scene) -- no duplicates under second names.
The arm identity comes from the parent campaign dir.

### What the runs corrected in this document

- **`--probe-every` is not an optional axis.** The clip's graph is retained by
  construction, so cost scales with *probe passes*, not frames: 48, 96 and 160
  frames all hit the same 44 GB ceiling, because it is reached before the clip
  ends. At every-frame density a 160-frame clip needs ~144 GB and **does not fit
  an H200 either**. The "794 probes per clip" and "31-48 GB" figures elsewhere in
  the design are wrong.
- **The decoder runs in bf16.** Checkpointing stores block *inputs*; in fp32 that
  is ~65 MB per pass. CUT3R trains every stage with `amp=1`.
- **DPT output convs need a small non-zero init** (`std=1e-4`, zero bias).
  `depth_mode` exponentiates the raw output, so random init gave `|grad| = 5.5e12`
  on the first real step. Zeroing the weight instead is a **dead path** -- V1 and
  V3 caught it at exactly 0.0, the same trap the old design hit with `to_taps`.
- **No `gpu_keep_alive` in training jobs.** The loop is near-continuous GPU work,
  so the reaper is not a risk, and the 6.79 GiB it holds was itself an OOM cause.
- Ray-origin norms on `285efbc7cf` are **p50 0.90 / p100 1.95**, about 2x the
  0.48 / 0.97 recorded from an earlier scene. This feeds the raymap encoder's
  sinusoidal band.

## Phase 4 — train small, compare

- [ ] 4a. Train arm C on a few scenes. Full BPTT over 160 frames (320-frame clip
      subsampled by 2, effective stride 40), 4 probes/frame, gradient
      checkpointing, recall loss only.
- [ ] 4b. Arm A recall eval on the same scenes (Phase 3 harness). Runs in
      parallel -- no training needed.
- [ ] 4c. Compare, with the controls from the design record at this rung too.
- [ ] 4d. **Report as a curve over clip length**, with a point at <= 64 frames.
      Arm A has no KV cache and never trained past 64 views, so at 160 frames it
      is out of distribution; without the <= 64 point the comparison is open to
      exactly that objection.
- [ ] *(later)* arm B, all scenes, the tap-23-vs-four-taps probe.

---

## Operating conventions for this campaign

- **Zone**: sof1 for repro, overfit and small training; msp3 when training gets
  heavy. A rule of thumb, not a constraint -- be flexible.
- **Storage**: jobs run on `/scratch` always; artifacts ship to **sof1**
  `/group/compact-3dmem`. msp3's `/group` is not a consideration -- nothing is
  filed there.
- **Small evals** (Phase 1, Phase 3 against arm A) -> hala a6000,
  `--partition=rendering --qos=rendering`. Both hala partitions carry
  `AllowQos=<own name>`, so a matching `--qos` is mandatory.
- **Always pass `--cpus-per-task` and `--mem` explicitly.** There is no
  auto-sizing on this cluster; the default is `cpu=1, mem=2G`.
- Job logs -> `/group/compact-3dmem/campaigns/_joblogs`.
- Never `micromamba run` inside a job; call the env's interpreter directly.

## Deferred, recorded so it is not lost

7-Scenes reproduction · other CUT3R benchmarks (monodepth, video_depth, relpose)
· arm B execution · four-taps input (sweep axis (b)) · KV-window shrinking ·
current-frame **token-space** loss at low weight (A/B), which is *not* the lag-0
raymap probe -- that is on by default · `--probe-current {on,off}`: does the
lag-0 term teach an echo buffer rather than a map? · `--raymap-convention
{cut3r,true}` · probe density 4-per-frame vs 4-every-8th · unfreezing the
aggregator · 7-Scenes `.depth.proj.png`.
