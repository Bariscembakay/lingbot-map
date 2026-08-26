# CUT3R evaluations — run log and findings

Everything we have run against the **published** CUT3R checkpoint. Kept separate
from `spatial_memory_design.md` (architecture) and `spatial_memory_plan.md`
(sequencing) because it is a measurement record: it should accumulate runs and
never be rewritten.

Arm A of the E1/E2 comparison is the published checkpoint, so this file is also
the provenance for arm A's numbers.

## What we are evaluating

| | |
|---|---|
| checkpoint | `/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth` |
| sha256 | `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103` |
| size / tensors | 3.17 GB, 1248 tensors under a `module.` prefix, 803.3 M params |
| upstream commit | `8bc15dc92a6d7fd92920b4ec81540d3dec7d3ecf` (see `CUT3R/UPSTREAM.md`) |
| env | `cut3r` micromamba, built by `setup_cut3r_env.sh` |
| runner | `.agents/scratch/reproduction/run_cut3r_mv_recon.sh` |
| artifacts | `/group/compact-3dmem/campaigns/spatial_memory/cut3r_repro/` |
| job logs | `/group/compact-3dmem/campaigns/_joblogs/cut3r_repro_*.out` |

## Benchmark: `eval/mv_recon` on NRGBD

Protocol comes entirely from upstream's `launch.py`: `full_video=True`,
`kf_every=500`, `resolution=(512, 384)`, `Regr3D_t_ScaleShiftInv`, point-to-point
ICP alignment before scoring, `estimate_normals` after, GT from unprojected GT
depth (**no mesh needed**). Frame selection is
`img_idxs[:: min(kf_every, len(img_idxs) // 2)]`, which yields **2-4 views per
scene** and matches the paper's stated sparse setting:

| scene | frames | views | | scene | frames | views |
|---|---|---|---|---|---|---|
| breakfast_room | 1167 | 3 | | morning_apartment | 920 | **2** |
| complete_kitchen | 1211 | 3 | | staircase | 1149 | 3 |
| green_room | 1442 | 3 | | thin_geometry | 395 | 3 |
| grey_white_room | 1493 | 3 | | whiteroom | 1676 | 4 |
| kitchen | 1517 | 4 | | | | |

## Results

Paper Table 4 (NRGBD): Acc 0.099 / 0.031, Comp 0.076 / 0.026, NC 0.837 / 0.971
(mean / median). Numbers below are from upstream's own `logs_all.txt`, not a
re-derivation.

| run | jobs | acc | comp | nc | acc_med | comp_med | nc_med |
|---|---|---|---|---|---|---|---|
| **paper** | — | 0.099 | 0.076 | 0.837 | 0.031 | 0.026 | 0.971 |
| raw `depth/` | 750327, 750432-4 | **0.100** | **0.078** | 0.834 | 0.037 | 0.032 | 0.970 |
| `depth_filtered/` | 750447 | 0.093 | 0.077 | **0.837** | 0.033 | 0.031 | 0.946 |

**Means and normal consistency reproduce. Medians do not** -- acc_med +18%,
comp_med +22% on raw depth. Not an exact reproduction.

### What has been ruled out

1. **Our aggregation.** Upstream's `logs_all.txt` is written by their own code
   (`sum/len` over per-scene values, `nc = (nc1+nc2)/2`) and agrees with our
   independent re-derivation to four decimals. The post-processing is theirs.
2. **Run-to-run noise.** `launch.py` passes no `seed`, so `base.py:83` falls back
   to `torch.initial_seed()` and `_crop_resize_if_necessary` draws
   `rng.integers(2)` -- which looked like nondeterminism. **Four repeat runs
   returned identical numbers, spread 0.0000.** At `(512, 384)` against NRGBD's
   4:3 input the crop is a pure resize, so the draw has no effect.
3. **The depth variant.** NRGBD ships `depth/`, `depth_filtered/` and
   `depth_with_noise/`; the loader reads `depth/`. Filtered masks 8.3% of pixels
   and differs by 1.3 cm mean where both are valid. It helps four metrics and
   hurts two (Acc +0.9% -> **-5.8%**, NC med -0.1% -> -2.5%). No variant matches
   all six.
4. **A corrupt scene.** `grey_white_room` carries the whole gap (acc_med 0.1027
   against 0.014-0.046 for the rest; drop it and acc_med is 0.0284, comp_med
   0.0229, both *below* the paper). But its data is structurally normal: 1493
   frames, 1493 poses, no NaN, mid-range trajectory extent and depth, 100% valid
   pixels.

### The remaining hypothesis

The same scene must be bad in *their* run too, or their **mean** Acc could not be
0.099 -- `grey_white_room` contributes ~30% of that mean. Yet it does not drag
their median column. That points at the median statistic, not the data:

```
mean-of-per-scene-medians   0.037 / 0.032   <- what logs_all.txt computes, what we get
median-over-scenes          0.029 / 0.025   <- paper reports 0.031 / 0.026
```

So the paper's median column was plausibly **not** produced by the released
aggregation script -- a pooled-over-points median, or a median across scenes,
would suppress one bad scene exactly as observed while leaving the mean column
alone. Consistent with all four observations; not proven, since testing a pooled
median needs per-point distances the logs do not retain.

One smaller possibility, noted for completeness: their `logs_all.txt` reports
`nc1: 0.837` and `nc2: 0.830` separately, and the paper's NC column is 0.837 --
**exactly nc1**. The paper may report NC1 rather than the mean the script
computes.

### Deviations from upstream's `run.sh`

| | ours | theirs | assessed |
|---|---|---|---|
| processes | 1 (also tested 2) | `accelerate launch --num_processes 8` | `split_between_processes` defaults to `apply_padding=False`, so each scene is evaluated once either way; measured with `NPROC=2` |
| GPU | rtx6000 (Turing, sm_75) / h200 | 8x A100 (sm_80) | **real numerical difference**: Turing has no TF32, so our fp32 convs are exact where an A100's `cudnn.allow_tf32=True` path is ~10 mantissa bits. Cuts the wrong way to explain our numbers. |
| torch / numpy | 2.8 / 2.x | early-2025 era | upstream's `numpy==1.26.4` pin is unsatisfiable against current open3d/opencv/pandas |
| source | 3 patches | — | `--datasets`, `weights_only=False`, RoPE negative positions (verified bit-identical). See `CUT3R/UPSTREAM.md`. |

## Verdict

**Not an exact reproduction.** Means and NC land on the paper; the two medians
are reproducibly ~20% high and localize to one scene, most consistently
explained by an aggregation convention that is not in the released code.

**Decision: both arms use raw `depth/`**, the released default. What E1 vs E2
requires is that arm A and arm C are scored through the *same* harness, GT,
alignment and aggregation -- an offset against a published table cancels in a
within-harness comparison. `NRGBD_DEPTH` stays selectable so this is revisitable.

## Not yet run

- **Table 5 "Ours Revisit"** (`--revisit 2 --freeze`), paper: 0.094 / 0.076 /
  0.844. This is CUT3R's own recall experiment -- freeze the final state and
  re-process the same inputs. Weaker than our probe (it still hands the model the
  image), but a free reference point on the same axis.
- 7-Scenes: needs `.depth.proj.png`, which we do not have. Deferred; our
  reprojection would use estimated Kinect calibration rather than Spann3R's, so
  a mismatch there would not cleanly mean anything.
- `monodepth`, `video_depth`, `relpose` (the last needs `evo`, deliberately not
  in the env).
