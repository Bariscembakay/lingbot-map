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
