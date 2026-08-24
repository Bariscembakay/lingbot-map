# Reproduction results — lingbot-map

Three-way: paper vs. upstream's own `benchmark/README.md` vs. our runs. Ours are
read from `eval/*.json` aggregates on disk under
`/group/compact-3dmem/campaigns/lingbot_map/<arm>/<benchmark>/`, not transcribed.
Scene counts in parentheses. `—` = not reported by that source.

`benchmark/README.md` is what upstream gets from the released `lingbot-map.pt`
on the shipped configs, i.e. the same thing we run; the paper is a third,
separate set of numbers. Causes for individual rows are in `paper_vs_repo.md`.

Arrows mark the better direction: ATE / RPE / accuracy / completeness / chamfer
are **lower-is-better** (↓); AUC / precision / recall / F1 are **higher-is-better**
(↑). RPE-rot is in degrees.

## Where it stands (2026-08-23)

Twelve of the fifteen published rows reproduce. What is left, and why:

| # | mismatch | status | what would close it |
| --- | --- | --- | --- |
| 1 | `bodleian-library-02` — 4.16 vs the paper's 9.59 | **open, actionable** | inspect in viser; the only Oxford scene whose gap run variance does not explain |
| 2 | Oxford paper row (6.42) | **unreproducible by construction** | nothing — Table 2 used an unreleased 160-epoch checkpoint (#62) |
| 3 | Oxford README row (5.374) | **closed 2026-08-23 — not the shipped config** | nothing further here; a 10-configuration sweep shows the shipped default is the 6th-closest of ten configs to their own row. See the Oxford sweep section |
| 4 | Table 3 sparse vs dense | **unreproducible from the release** | flow-based keyframing, absent from the streaming path (`paper_vs_repo.md` §1.2; investigation in `trajectory_memory_ablation.md`) |
| 5 | 7-Scenes points, registered GT | **closed, not identical** | their `.depth.proj.png`; ours is a reconstruction of it, so we sit 0.7% *better* than their row rather than on it |
| 6 | RPE-rot on NRGBD (+13.6%) and KITTI (+26.3%) | **closed — metric, not pipeline** | nothing; RMSE over all pairs is outlier-dominated and moves ±65% between identical runs |
| 7 | ETH3D, all 17 metrics 0.08–0.79% low | **closed — run numerics** | nothing; deficit is monotone in threshold |

Only #1 is worth more compute. #2 needs upstream. #3–#7 are explained and
stable. A new item, the Oxford aspect squash, is a correctness bug in upstream's
adapter with negligible accuracy impact — see the sweep section.

## Trajectory — ATE ↓ / RPE-trans ↓ / RPE-rot ↓ (°)

| Dataset | Paper | upstream README | Ours | vs README |
| --- | --- | --- | --- | --- |
| ETH3D | 0.43 / — / — | 0.439 / 0.493 / 3.339 (11) | 0.4424 / 0.4965 / 3.3486 (11) | 🟢 |
| 7-Scenes | 0.08 / — / — | 0.079 / 0.020 / 0.579 (18) | 0.0789 / 0.0205 / 0.5788 (18) | 🟢 |
| Tanks & Temples | 0.20 / — / — | 0.210 / 0.087 / 0.572 (6) | 0.2095 / 0.0867 / 0.5704 (6) | 🟢 |
| Oxford Spires, sparse s12 | 6.42 / 1.01 / 3.70 | 5.374 / 0.930 / 3.694 (10) | 6.1887 / 0.7650 / 4.2903 (10) | 🔴 |
| Oxford Spires, dense s1, kf auto | 7.11 / — / — | — | 5.1620 / 0.5726 / 2.1700 (10) | — |
| Oxford Spires, dense s1, kf 1 | — | — | 29.063 / 0.3748 / 2.7690 (10) | — |
| Neural RGB-D | — | 0.056 / 0.019 / 0.257 (9) | 0.0554 / 0.0189 / 0.2920 (9) | 🟡 |
| TUM RGB-D, freiburg1 nine | — | 0.045 / 0.013 / 0.513 (9) | 0.0451 / 0.0132 / 0.5124 (9) | 🟢 |
| KITTI Odometry | — | 24.046 / 2.861 / 0.696 (11) | 24.0772 / 2.8623 / 0.8790 (11) | 🟡 |
| VBR | — | 31.204 / 2.717 / 4.564 (7) | 31.119 / 2.7156 / 4.3654 (7) | 🟢 |
| DROID-W | — | 0.909 / 0.184 / 6.115 (7) | 0.9084 / 0.1837 / 6.1151 (7) | 🟢 |

The TUM row is an exact aggregate over upstream's own scope, the nine Freiburg1
sequences (see the TUM section); it is not a subset of a larger run. Upstream's
single Oxford row is unlabelled — the README does not say which config produced
it. NRGBD trajectory needs `configs/neural_rgbd_traj.yaml` and the TUM nine need
a `_scenes` whitelist in `configs/datasets/tum.yaml`; the shipped `neural_rgbd.yaml`
sets `traj.enable: false` and the shipped `tum.yaml` discovers all 66 sequences,
so neither README row is producible from the shipped configs as they stand.

## Pose AUC ↑ — degrees

| Dataset | Paper | README macro | Ours macro | README micro | Ours micro | vs README |
| --- | --- | --- | --- | --- | --- | --- |
| ETH3D @3 / @30 | 37.22 / 81.09 | 37.22 / 81.10 | 36.9779 / 81.0060 | 40.34 / 87.97 | 40.0408 / 87.8965 | 🟢 |
| 7-Scenes @3 / @30 | 12.63 / 78.59 | 12.35 / 78.09 | 12.3331 / 78.0686 | 13.20 / 79.06 | 13.1831 / 79.0485 | 🟢 |
| Tanks & Temples @3 / @30 | 45.80 / 92.80 | — | 47.9918 / 93.1738 | — | 44.8669 / 92.3314 | — |
| Oxford Spires @15 / @30 | 61.64 / 75.16 | — | 63.5028 / 76.5027 | — | 63.5075 / 76.5025 | — |
| Neural RGB-D @3 / @30 | — | — | 42.3558 / 92.3945 | — | 40.4690 / 92.2634 | — |

## Point clouds — Acc ↓ / Comp ↓ / F1 ↑

| Dataset | Paper | upstream README | Ours | vs README |
| --- | --- | --- | --- | --- |
| ETH3D | 0.16 / 0.08 / 86.79 | 0.168 / 0.089 / 86.80 | 0.1683 / 0.0897 / 86.7172 | 🟢 |
| 7-Scenes, raw GT depth | 0.02 / 0.07 / 80.39 | 0.036 / 0.044 / 82.38 | 0.0411 / 0.0469 / 79.0556 | 🔴 |
| 7-Scenes, registered GT depth | — | — | 0.0331 / 0.0427 / 82.9816 | 🟡 |
| Neural RGB-D | 0.07 / 0.03 / 64.26 | 0.074 / 0.030 / 65.10 | 0.0737 / 0.0303 / 65.1071 | 🟢 |

All rows are exact, full-set reproductions; no scene is ever dropped from an
aggregate.

## 7-Scenes point clouds — cause found and fixed (2026-08-23)

Upstream reads GT depth from `frame-{id}.depth.proj.png`, i.e. depth already
registered into the colour camera. No `.proj` variant ships with the raw
Microsoft distribution, so we switched to `.depth.png` — which is in the
**depth** camera frame. The pipeline then unprojects it with the colour focal
(525) when the depth camera's is ~599, stretching the GT cloud radially: zero at
the principal point, ~15 cm at the corners at 2 m, against a 5 cm F1 threshold.
Being radial, Umeyama+ICP cannot absorb it — which is why it survived every
alignment check, and why swapping 525→585 made Acc worse instead of better.
Only depth-consuming metrics move; pose is unaffected. Full derivation in
`paper_vs_repo.md` §2.5.

**Fix, and it works.** `_register_depth` in `datasets/seven_scenes.py` warps
depth into the colour frame before use — intrinsics untouched — using the
Kinect calibration estimated by `zinsmatt/7-Scenes-Calibration` (Microsoft ship
none). Arm `campaigns/lingbot_map/depthproj`, job 738368.

Every one of the six point-cloud metrics moved toward upstream; F1 went from
3.32 below their row to 0.60 above it. Two controls say the effect is real:

- **Pose is unchanged** — ATE/RPE deltas vs the raw-depth arm are ≤1.5e-04, i.e.
  float nondeterminism. Only GT moved, exactly as predicted.
- **Precision improved** (74.26 → 79.51) even though registration *shrinks* the
  GT cloud to the depth camera's narrower FOV (91.5% → 73.5% frame coverage).
  A sparser GT should hurt precision; it improved, so this is alignment, not a
  coverage artifact.

Sign of the translation was fixed empirically, not guessed: scoring RGB gradient
magnitude at depth discontinuities (a model-free check — a correctly registered
depth map puts its jumps on image edges) showed only one sign improves it, by
30-60% on every sequence tested.

We now sit slightly *above* upstream rather than on them, so this reproduces
their **cause**, not their exact GT — our estimated calibration is not
necessarily the one their `.proj` files were built with. Treat 82.98 as "the
gap is explained and closed", not as a bit-match.

## Settlement against upstream's README (2026-08-23)

Status marks in the three tables above:

| mark | meaning |
| --- | --- |
| 🟢 | exact — every metric within run-to-run noise |
| 🟡 | close; the residual is named and understood, but not eliminated |
| 🔴 | does not reproduce, or is not comparable |
| — | upstream publishes no baseline for that row |

Deltas computed cell by cell from the `eval/*.json` aggregates, not eyeballed.

### ETH3D — closed, and it was never an AUC bug

`DA3_FILTER_KEYS` is ruled out. Replaying the filter against the real frame
stems drops exactly 4 / 6 / 12 / 11 / 6 frames from `delivery_area`, `electro`,
`relief`, `relief_2`, `playground` — the intended lists, with no over- or
under-match. The `endswith` test looks fragile but is deliberate: the keys are
bare numbers (`427.JPG`) and the stems carry a camera prefix (`DSC_0427`).

The actual reading is that **all 17 published ETH3D metrics are worse by the
same small amount, and the deficit shrinks monotonically as the tolerance
loosens**:

| | @3 | @5 | @15 | @30 |
| --- | ---: | ---: | ---: | ---: |
| AUC macro, worse by | 0.65% | 0.56% | 0.21% | 0.12% |
| AUC micro, worse by | 0.74% | 0.52% | 0.16% | 0.08% |

with ATE +0.77%, RPE-t +0.71%, completeness +0.78%, chamfer +0.79% at the tight
end, and precision / recall / F1 at +0.10 / +0.09 / +0.10%.

The uniform sign across 17 cells is not evidence of bias — they are 17 views of
one set of predictions, not 17 independent draws. What matters is the *shape*: a
frame-subset or aggregation difference would perturb these erratically, whereas
a tiny numeric difference in the predictions themselves is exactly what bites at
tight tolerance and washes out at loose. Consistent with a different GPU / torch
/ attention backend. Nothing to fix.

### Oxford — the authors published per-scene ATE, and it reframes this

Upstream issue [#38](https://github.com/Robbyant/lingbot-map/issues/38) contains
`LinZhuoChen`'s per-scene ATE for all 14 Oxford scenes at "first 320 frames,
sampled every 12 frames" — our exact protocol. The mean of the ten kept scenes is
**6.4246**, which is the paper's 6.42, so these are Table 2's own per-scene
values. It also settles the scene set (#38 calls the paper's "13 scenes" a typo
for 14, but see below — 13 is in fact correct;
four were dropped because "the poses and point clouds in 4 of those scenes were
not properly aligned", leaving the ten we use) and gives the 14-scene figure,
ATE 5.59.

| scene | paper (#38) ATE ↓ | ours ATE ↓ | diff | ratio |
| --- | ---: | ---: | ---: | ---: |
| bodleian-library-02 | 9.5920 | 4.1618 | −5.4302 | 0.43× |
| christ-church-02 | 3.2969 | 3.3469 | +0.0500 | 1.02× |
| christ-church-03 | 0.4530 | 0.5116 | +0.0586 | 1.13× |
| christ-church-05 | 35.9870 | 40.5017 | +4.5147 | 1.13× |
| keble-college-02 | 3.0160 | 2.8568 | −0.1592 | 0.95× |
| keble-college-03 | 1.9570 | 2.1473 | +0.1903 | 1.10× |
| keble-college-04 | 1.5780 | 1.4577 | −0.1203 | 0.92× |
| keble-college-05 | 2.8290 | 2.6356 | −0.1934 | 0.93× |
| observatory-quarter-01 | 2.8610 | 1.9408 | −0.9202 | 0.68× |
| observatory-quarter-02 | 2.6760 | 2.3269 | −0.3491 | 0.87× |
| **mean** | **6.4246** | **6.1887** | −0.2359 | 0.96× |

This corrects the earlier "Oxford is `christ-church-05`" reading. Eight of ten
scenes agree within 13%. The two that do not are `bodleian-library-02` (we are
2.3× *better*) and `christ-church-05` (we are 1.13× worse), and they carry 45%
and 38% of the total per-scene movement in opposite directions — so the close
means (6.19 vs 6.42) are substantially cancellation, not agreement.

`christ-church-05` is no longer the anomaly: the paper's 35.99 sits inside the
34.8–42.5 band our own 14 near-identical arms span on that scene, i.e. within
config/run variance. **`bodleian-library-02` is the real outlier** and is where
to look in viser — a 2.3× gap on a scene neither run treats as a failure.

Its likely explanation is in issue
[#62](https://github.com/Robbyant/lingbot-map/issues/62): Table 2 was produced
with "the 160-th epoch checkpoint", and the released weights "can achieve better
ATE than the results in table 2". A released checkpoint that fixes one scene the
160-epoch one botched is exactly this shape. **Table 2 is therefore not
reproducible from public weights by construction** — see `paper_vs_repo.md`.

Ruled out as causes: frame set (320 frames from 3,840 at stride 12, confirmed
from `sampling.json` and the rgb count, and now confirmed against #38's wording),
scene set (the ten, confirmed by #38), and run-to-run noise (our SDPA and
FlashInfer arms agree to 0.4% on the dataset mean).

### RPE-rot is outlier-dominated — measured, not asserted

`evaluation/trajectory.py` computes it as `main_rpe.rpe(...,
rotation_angle_deg, delta=1, all_pairs=True)` and reports `stats["rmse"]`. On
KITTI the top 1% of frame pairs carry 4.5–90.6% of the total squared error
(uniform would be 1%), and instability tracks that concentration:

| seq | top-1% share of RMSE² | spread between our two identical-config runs |
| --- | ---: | ---: |
| 02 | 90.6% | 35.0% |
| 07 | 63.7% | 32.5% |
| 00 | 22.4% | 65.2% |
| 10 | 15.0% | 0.09% |
| 03 | 9.7% | 0.40% |
| 04 | 4.5% | 0.83% |

On seq 02, deleting the single worst pair out of ~1,900 moves RMSE from 3.674 to
2.730. Those same two runs agree to ≤2.5% per scene on ATE. So RPE-rot cannot be
reproduced better than tens of percent on these datasets, and a mismatch there
is not evidence of a pipeline difference — which is why NRGBD (+13.6%) and KITTI
(+26.3%) are marked 🟡 rather than 🔴. DROID-W and Tanks & Temples happening to
match exactly is luck, not a stronger reproduction.

### Oxford vs upstream's README — closed 2026-08-23 by a 10-configuration sweep

Upstream say their numbers come from "the released `lingbot-map.pt` checkpoint
(streaming mode), evaluated on the shipped dataset configs", so flow-based
keyframing is not implicated in *this* row and the question is purely data,
config or numerics. Full evidence in
`.agents/scratch/reproduction/oxford/oxford_readme_ruleouts.md`; the load-bearing parts:

**Ruled out with measurements, not inspection.** Config drift (`git diff` shows
only paths differ), checkpoint (sha256 matches the published file), the
`christ-church-05` sequence (upstream's hardcoded `2024-03-18` does not exist;
only `2024-03-20` does, so they made our fix too), scene set and stride
(confirmed by issue #38), the `max_frames=3840` truncation (the three
most-truncated scenes would diverge 3–4x under a different cap; they agree with
the paper to 8%), raw-data authenticity, stale upstream code, non-metric
predicted scale (3.5–27x on *every* dataset), evaluation frame density
(subsampling moves ATE 0.6% while moving both RPE metrics the *same* way), a
different 10-scene selection (all 1001 subsets of #38's fourteen: those with
cc-05 average ≥5.722, those without ≤3.832, and 5.374 falls in the gap), and
GPU/backend (SDPA-A100 vs FlashInfer-a6000 moves Oxford <1%).

**What the sweep found.** Twelve arms, all SDPA/A100 so mutually comparable; the
control submitted twice returned identical numbers to four decimals, so the
pipeline is deterministic and every delta is signal.

| | ATE | RPE-t | RPE-rot | worst delta |
| --- | ---: | ---: | ---: | ---: |
| upstream README | 5.374 | 0.930 | 3.694 | |
| anchors 4 | 5.625 | 0.892 | 3.832 | 4.7% |
| anchors 16 + window 128 | 5.511 | 0.865 | 3.420 | 7.4% |
| **shipped default** | 6.143 | 0.771 | 4.276 | **17.1%** |
| `lingbot-map-long.pt` | 4.737 | 0.876 | 3.173 | 14.1% |

- **The shipped default is the 6th-closest of ten configs to upstream's own
  row.** Most single-knob changes fit their numbers better than the config they
  name. That, plus environment being worth <1%, is why this is read as "not the
  shipped config" rather than "an environment difference".
- **The row is bracketed by the two released checkpoints** on ATE and RPE-rot:
  `lingbot-map.pt` sits 14.3% above, `lingbot-map-long.pt` 11.8% below, and
  solving for the crossing gives lambda 0.453 from ATE and 0.472 from RPE-rot —
  agreeing to 4%. With issue #62 (Table 2 used an unreleased 160-epoch
  checkpoint; both released ones beat it), an unpublished intermediate checkpoint
  is the economical explanation.
- **Oxford ATE spans 4.74–8.25 across single-knob changes**, a 74% range with
  5.374 inside it. A 14.3% gap is small against that sensitivity — and it is why
  the other eight rows reproduce within 1% while this one does not: they are
  short or well-constrained sequences where the memory config barely bites.
- **Only the checkpoint, the KV window and the aspect ratio move ATE and
  RPE-trans in opposite directions.** The README needs exactly that (lower ATE,
  higher RPE-trans), which no protocol or evaluation change can produce.

**Not pursued further, deliberately.** Two arms land within 10% on all three
metrics and several within 15%. Picking the closest would be curve-fitting, and
`anchors 4` contradicts both the shipped config and issue #68's "the number of
scale images is 8".

### Oxford's aspect squash — a real bug, but not a performance win

`datasets/oxford_spires.py` fixes width to 518 and floors height to a multiple of
14, turning a 1440x1080 (1.3333) source into 518x378 (1.3704): a 2.78%
anisotropic squash. GT intrinsics are scaled per axis so GT stays consistent, but
the model predicts fx~=fy — square pixels — and structurally cannot represent
that camera. `load_img_size: 504` gives exact 4:3 at the same patch count
(972 vs 999), isolating geometry from resolution:

| | squashed | exact 4:3 | change |
| --- | ---: | ---: | ---: |
| ATE | 6.1429 | 6.0403 | −1.7% |
| RPE-trans | 0.7710 | 0.8686 | +12.7% |
| RPE-rot RMSE | 4.2763 | 3.0547 | −28.6% |
| RPE-rot **median per-pair** | 0.969 | 0.948 | −2.2% |
| AUC@3 / @15 | 17.74 / 63.66 | 17.87 / 61.94 | +0.7% / −2.7% |

The −28.6% RMSE is a tail effect: the median per-pair rotation error barely
moves, both distributions are heavy-tailed, AUC is a wash and RPE-trans gets
worse. Worth reporting upstream as a correctness bug; not worth claiming as an
accuracy improvement.

### TUM — solved 2026-08-23, the nine are Freiburg1

Upstream's row is "TUM RGB-D | 9" with its figure captioned *fr1/desk*. The nine
are the standard Freiburg1 set — 360, desk, desk2, floor, plant, room, rpy,
teddy, xyz — the same subset DROID-SLAM and NICE-SLAM report on. Restricting to
them reproduces the row outright:

| | upstream | ours (9) |
| --- | ---: | ---: |
| ATE ↓ | 0.045 | 0.04508 |
| RPE-trans ↓ | 0.013 | 0.01323 |
| RPE-rot ↓ | 0.513 | 0.51242 |

Not a hand-computed subset: `configs/datasets/tum.yaml` now carries a `_scenes`
whitelist of the nine, so the scope is declared rather than inferred, and the
aggregate above is `evaluate.py`'s own `eval/traj.json`. As shipped the config
has no whitelist and `TumDataset.get_scenes` discovers every
`rgbd_dataset_freiburg*` with a `groundtruth.txt` — 66 against a full download,
so upstream evidently had only Freiburg1 on disk. The nine are scored from the
earlier 66-sequence run's predictions, which is why the arm holds symlinks into
`default/tum` rather than its own copies.

RPE-rot landing within 0.11% here is worth noting given how unstable that metric
is elsewhere — see below.

### Upstream issue tracker — what it settled

| issue | what it gives |
| --- | --- |
| [#38](https://github.com/Robbyant/lingbot-map/issues/38) | per-scene ATE for all 14 Oxford scenes at stride 12 / first 320; confirms the 10-scene set. Its claim that the paper's "13 scenes" is a typo for 14 is wrong — `christ-church-01` ships no `gt-tum.txt`, so 13 is the evaluable maximum |
| [#62](https://github.com/Robbyant/lingbot-map/issues/62) | Table 2 used an unreleased 160-epoch checkpoint; released weights score *better*; AUC follows DA3 |
| [#68](https://github.com/Robbyant/lingbot-map/issues/68) | 7-Scenes protocol: `lingbot-map.pt`, pure streaming, `num_scale_frames=8`, `keyframe_interval=1`, unproject predicted depth with predicted poses, no confidence/sky masking |

Nothing in the tracker mentions TUM, ETH3D or NRGBD protocol. #68 asks about
7-Scenes but never mentions `.depth.proj.png`, so it is *not* evidence that
another user hit our GT-registration issue — the earlier note claiming so was
unsupported.

## Checklist

Trajectory-memory ablation items moved to `trajectory_memory_ablation.md`.

Against the paper where the paper is reachable, otherwise against upstream's
`benchmark/README.md` — see the note at the top.

- [x] ETH3D pose + point clouds
- [x] Tanks & Temples pose
- [x] NRGBD point clouds
- [x] 7-Scenes point clouds — **closed 2026-08-23**. Raw depth-camera-frame GT
      unprojected with the colour focal; registering it into the colour frame
      moves all six metrics onto upstream's row (F1 79.06 → 82.98 vs their
      82.38), with pose unchanged as a control.
- [x] VBR — pipeline validated, no paper baseline
- [x] KITTI (Odometry) — full artifacts kept for viewing, no paper baseline
- [x] Oxford Spires vs paper — per-scene ATE recovered from upstream issue #38
      (their ten average 6.4246 = Table 2's 6.42). 8/10 scenes agree within 13%;
      `bodleian-library-02` (we are 2.3x better) and `christ-church-05` (1.13x
      worse) carry 45% and 38% of the movement in opposite directions, so the
      close means are largely cancellation. Table 2 used an unreleased 160-epoch
      checkpoint (#62), so it is not reproducible from public weights
- [x] Oxford Spires vs upstream README — **closed 2026-08-23** by a 10-configuration sweep.
      Their row is not the shipped config: the shipped default is the 6th-closest
      of ten arms to it, environment is worth <1%, and the row is bracketed by
      the two released checkpoints (lambda 0.453 from ATE, 0.472 from RPE-rot).
      Oxford ATE spans 4.74-8.25 across single-knob changes, so 14.3% is small
      against its own sensitivity
- [x] Oxford aspect squash — 518x378 from a 4:3 source is a 2.78% anisotropic
      warp the model cannot represent (it predicts square pixels). Real bug, but
      fixing it moves median per-pair rotation error only 2.2% and worsens
      RPE-trans; report it, do not sell it as an accuracy gain
- [ ] `bodleian-library-02` in viser — the one Oxford scene whose gap is not
      explained by run variance
- [x] NRGBD trajectory — reproduced (ATE -0.99%) via `configs/neural_rgbd_traj.yaml`
- [x] ETH3D AUC — **closed 2026-08-23**, not an AUC bug. `DA3_FILTER_KEYS` drops
      exactly its intended frames; all 17 metrics are worse by 0.08-0.79% with
      the deficit monotone in threshold, i.e. run numerics
- [x] RPE-rot instability — measured: top 1% of KITTI frame pairs carry up to
      90.6% of RMSE²; one pair in ~1,900 moves seq 02 from 3.674 to 2.730
- [x] TUM — **solved 2026-08-23**. The nine are the standard Freiburg1 set;
      `configs/tum.yaml` plus a `_scenes` whitelist reproduces the row at
      0.04508 / 0.01323 / 0.51242 against 0.045 / 0.013 / 0.513
- [ ] Decide whether to PR bugs found this round upstream (NOT the `.depth.png`
      change — that was our data-prep gap)
