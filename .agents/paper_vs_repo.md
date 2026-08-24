# Paper ↔ repo differences — lingbot-map

Systematic comparison of `lingbot-map_paper.pdf` (30 pp.) against the released
code at `upstream/main` (`c95c33c`), plus a three-way reconciliation of the
paper's tables, upstream's own `benchmark/README.md` numbers, and our runs.

Written 2026-08-22. Line references are to `upstream/main` unless noted; our
local commits don't touch any of the files cited in Part 1.

**Three sources, three different sets of numbers.** The most consequential
finding is not any single mismatch but that *upstream's own published benchmark
results do not reproduce their paper's tables either* — see Part 2.

**Updated 2026-08-23 from the upstream issue tracker.** Issue
[#62](https://github.com/Robbyant/lingbot-map/issues/62) establishes that
**Table 2 was produced with an unreleased 160-epoch checkpoint**, not the
released `lingbot-map.pt`, and that the released weights "can achieve better ATE
than the results in table 2". Table 2 is therefore not reproducible from public
artifacts by construction — see §1.8. Issue
[#38](https://github.com/Robbyant/lingbot-map/issues/38) supplies Table 2's
per-scene ATE, and [#68](https://github.com/Robbyant/lingbot-map/issues/68) the
7-Scenes protocol; both are used in `reproduction.md`.

---

## Part 1 — Code differs from the method as described

### 1.1 Training is entirely unreleased

§3.3 (composite depth + absolute-pose + relative-pose loss), §4.1–4.2 (AdamW,
2 × 160K iterations, progressive 24→320 view curriculum, Ulysses context
parallelism at dim 16, TorchTitan + Magi Attention, FSDP + bf16), §4.3 (29-dataset
corpus, nearby sampler, foldback video sampler) and the whole appendix (Blender
render settings, gaming-capture protocol, MatrixCity Alg. 1/2) have no
counterpart in the repo. `grep -rE 'AdamW|Ulysses|FSDP|torchtitan|fully_shard'`
hits only vendored DINOv2 and a registration helper.

The repo is inference + evaluation only. Every training claim in §3.3–§4.3 is
unverifiable from it.

### 1.8 Table 2 used a checkpoint that was never released

Confirmed by the authors in issue #62: "The results in table 2 using the 160-th
epoch checkpoint of lingbot-map.pt, both lingbot-map.pt and lingbot-map-long.pt
can achieve better ATE than the results in table 2."

Two consequences. First, the Oxford Spires row cannot be reproduced from the
release at any configuration — the weights that produced it are not public, and
the public ones are *better*, so a faithful reproduction should overshoot the
paper rather than match it. Ours does (6.19 vs 6.42), as does upstream's own
README (5.374). Second, it is inconsistent with issue #68, where the same
maintainer states the 7-Scenes numbers in Tables 4/5 *do* use `lingbot-map.pt`.
Different tables in the same paper therefore rest on different checkpoints, and
the paper says so nowhere.

The per-scene data in #38 shows where it bites: our `bodleian-library-02` scores
4.16 against the paper's 9.59, which is the shape of a released checkpoint
fixing a scene the 160-epoch one handled badly.

### 1.2 Flow-based keyframe selection is missing from Direct mode

§4.4 defines adaptive keyframing — predict depth+pose, compute optical flow
against the last keyframe, promote when it exceeds a threshold — and states
*"This mechanism is shared by both inference modes."*

`flow_threshold` exists only in [gct_stream_window.py](../lingbot_map/models/gct_stream_window.py)
(the VO/windowed model). [gct_stream.py](../lingbot_map/models/gct_stream.py) —
Direct mode, the paper's default for every table — has no flow logic at all,
[demo.py](../demo.py) never exposes the flag, and
[benchmark/methods/lingbot_map.py:205](../benchmark/methods/lingbot_map.py#L205)
passes it only down the windowed branch. Direct mode gets a **fixed** interval.

Consequence: the released streaming path cannot run the paper's dense protocol
at all. `auto` pins the cache at ~328 keyframes regardless of N; a forced `1`
leaves the trained frame range; flow-based selection isn't implemented. See
`reproduction.md` for the measured ΔATE of −1.03 vs +22.87 either side of that.

### 1.3 VO-mode window fusion is not a Sim(3) fit

§4.4: *"we compute a Sim(3) alignment between the overlapping regions of
consecutive windows, recovering the relative scale, rotation, and translation."*

[gct_stream_window.py:757](../lingbot_map/models/gct_stream_window.py#L757)
`_pairwise_alignment` instead takes **one** anchor frame — the last keyframe
paired in the overlap — for R and t, and derives scale from the **median depth
ratio** over the paired keyframes. There is no least-squares fit over the
overlapping trajectory. A single-frame relative pose plus a robust scale
estimate is a different (and drift-wise weaker) estimator than a Sim(3) fit.

### 1.4 Video RoPE is applied to everything, not just the trajectory memory

§3.2: *"we incorporate video temporal positional encodings [71] into the
retained tokens"* — i.e. the 6 context tokens surviving eviction.

[attention.py:263-268](../lingbot_map/layers/attention.py#L263-L268): with
`enable_3d_rope=True` (the default everywhere) the 3D Wan RoPE **replaces** the
2D spatial RoPE in every cross-frame block and is applied to all tokens of every
frame — the 6 specials *and* all ~1000 image patches. The §6.4 "V. RoPE"
ablation row therefore describes a much narrower intervention than the flag
actually performs.

### 1.5 Dead code

`lingbot_map/models/gct_stream_window_v2.py` — 1,349 lines, tracked, imported by
nothing. Both [demo.py:134](../demo.py#L134) and
[benchmark/methods/lingbot_map.py:117](../benchmark/methods/lingbot_map.py#L117)
import `gct_stream_window`.

### 1.6 Configuration constants

| | Paper | Released code |
|---|---|---|
| keyframe interval | **m = 1** (§4.4, "Default Inference Configuration") | `_keyframe_interval: auto` → `ceil(N/320)` ([lingbot_map.yaml:15](../benchmark/configs/methods/lingbot_map.yaml#L15)). Oxford dense (3,840 f) → **m = 12** |
| anchor frames | **n = 3** (complexity analysis), **n = 2** (Fig. 3) | **8** — [demo.py:367](../demo.py#L367), [stream.py:48](../lingbot_map/aggregator/stream.py#L48), method yaml |
| image tokens/frame | **M ≈ 500** | 518×378 at patch 14 = 37×27 = **999** |
| resolution | **518×378** (abstract, §3.4, §4.4) | profiler defaults to **504×378** ([gct_profile.py:229-230](../gct_profile.py#L229-L230)) while its own `--compile` help says "518×378"; the benchmark uses neither (below) |
| window k, dtype, 6 context tokens, DINOv2 ViT-L/14, depth 24 | — | all match |

The M discrepancy is load-bearing for two headline claims. At the paper's own
stated resolution the per-frame growth reduction is `(999+6)/6 ≈ 167×`, not the
claimed **80×** (abstract, §3.2, §7), and the complexity figures "∼5×10⁶ vs
∼7×10⁴ tokens" are both computed from M = 500.

**Benchmark resolution is a third value again.** `_area_budget: 255000` with
align 14 ([resize.py:79-100](../benchmark/benchmark/geometry/resize.py#L79-L100))
downscales to ≤255k px, so a 4:3 source lands at ~574×434 — ~30% more pixels
than the 518×378 (195,804 px) the paper says all experiments use. The 518-crop
path in `load_and_preprocess_images` is used by `demo.py`, not by the benchmark.

### 1.7 Evaluation protocol

- **ETH3D voxel size.** §5.2 specifies "F1 threshold of 0.25 with a voxel size
  of 0.039 m". The threshold matches ([eth3d.py:83](../benchmark/datasets/eth3d.py#L83))
  but ETH3D does **no** voxel downsampling — 7-Scenes and NRGBD do (4.0/512),
  ETH3D doesn't.
- **ETH3D scene count.** `DA3_FILTER_SCENES = ["meadow", "terrace"]` drops 2 of
  13 → 11 scenes, though §5.1 lists "terraces" among the covered scene types.
- **7-Scenes count.** Paper says 7 scenes; the shipped config evaluates the
  **18** test sequences those 7 scenes contain.
- **Oxford ICP threshold** 0.5 ([oxford.yaml](../benchmark/configs/datasets/oxford.yaml))
  vs the paper's 0.1 — inert, Oxford reconstruction is not a paper table.
- **FPS is never measured.** Table 3's FPS column has no counterpart in the
  benchmark harness; only `gct_profile.py` measures throughput, on synthetic
  frames, and it is not wired in.

Matching correctly: Oxford stride 12 and the 3,840-frame cap
([preprocess/oxford.py:763](../preprocess/oxford.py#L763)), the exact 10 Oxford
scenes, stride 5 for 7-Scenes and NRGBD, the 6 T&T scenes, Umeyama + ICP@0.1,
F1@0.05, voxel 4.0/512, Sim(3)-aligned ATE, 6 context tokens
(1 camera + 4 register + 1 anchor), k = 64, bf16, DINOv2 ViT-L/14, 24 alternating
blocks, FlashInfer paged KV cache, anchor frames never evicted, append-only
trajectory memory.

---

## Part 2 — Numbers: paper vs upstream README vs our runs

Ours = on-disk aggregates under `/group/compact-3dmem/campaigns/lingbot_map/`,
read directly from `eval/*.json` (not transcribed). `#sc` = scenes evaluated.

### 2.1 Trajectory — ATE / RPE-trans / RPE-rot

| Dataset | Paper | upstream README | ours |
|---|---|---|---|
| ETH3D | 0.43 / – / – | 0.439 / 0.493 / 3.339 (11) | **0.4424 / 0.4965 / 3.3486** (11) |
| 7-Scenes | 0.08 / – / – | 0.079 / 0.020 / 0.579 (18) | **0.0789 / 0.0205 / 0.5788** (18) |
| Tanks & Temples | 0.20 / – / – | 0.210 / 0.087 / 0.572 (6) | **0.2095 / 0.0867 / 0.5704** (6) |
| Oxford sparse (s12) | 6.42 / 1.01 / 3.70 | 5.374 / 0.930 / 3.694 (10) | **6.1887 / 0.7650 / 4.2903** (10) |
| Oxford dense (s1, auto) | 7.11 / – / – | — | **5.162 / 0.5726 / 2.170** (10) |
| Neural RGB-D | — | 0.056 / 0.019 / 0.257 (9) | not produced — shipped config disables `traj` |
| TUM RGB-D | — | 0.045 / 0.013 / 0.513 (**9**) | 0.1303 / 0.0273 / 0.9189 (**66**) |
| KITTI | — | 24.046 / 2.861 / **0.696** (11) | 24.0772 / 2.8623 / **0.8790** (11) |
| VBR | — | 31.204 / 2.717 / **4.564** (7) | 31.119 / 2.7156 / **4.3654** (7) |
| DROID-W | — | 0.909 / 0.184 / 6.115 (7) | **0.9084 / 0.1837 / 6.1151** (7) |

### 2.2 Pose AUC (degrees)

| Dataset | Paper | README macro | ours macro | README micro | ours micro |
|---|---|---|---|---|---|
| ETH3D @3 / @30 | 37.22 / 81.09 | 37.22 / 81.10 | 36.98 / 81.01 | 40.34 / 87.97 | 40.04 / 87.90 |
| 7-Scenes @3 / @30 | **12.63** / **78.59** | 12.35 / 78.09 | 12.33 / 78.07 | 13.20 / 79.06 | 13.18 / 79.05 |
| T&T @3 / @30 | 45.80 / 92.80 | not published | 47.99 / 93.17 | not published | 44.87 / 92.33 |
| Oxford @15 / @30 | 61.64 / 75.16 | not published | 63.50 / 76.50 | not published | 63.51 / 76.50 |

### 2.3 Reconstruction — Acc / Comp / F1

| Dataset | Paper | upstream README | ours |
|---|---|---|---|
| ETH3D | 0.16 / 0.08 / 86.79 | 0.168 / 0.089 / 86.80 | 0.1683 / 0.0897 / 86.72 |
| 7-Scenes | 0.02 / 0.07 / 80.39 | 0.036 / 0.044 / 82.38 | 0.0411 / 0.0469 / 79.06 |
| Neural RGB-D | 0.07 / 0.03 / 64.26 | 0.074 / 0.030 / 65.10 | 0.0737 / 0.0303 / 65.11 |

### 2.4 Reading of the three-way comparison

**Our pipeline reproduces upstream's released code faithfully.** On DROID-W,
7-Scenes trajectory, T&T trajectory, and NRGBD reconstruction we match their
README to 3–4 significant figures; ETH3D (traj, AUC, points) is within ~0.3%.
That is the strongest available evidence that the harness, configs and
checkpoint wiring are correct.

**Upstream's README does not reproduce upstream's paper.** Independently of
anything we did:

| claim | paper | their own README |
|---|---|---|
| Oxford Spires ATE | 6.42 | 5.374 |
| 7-Scenes Acc / Comp / F1 | 0.02 / 0.07 / 80.39 | 0.036 / 0.044 / 82.38 |
| NRGBD F1 | 64.26 | 65.10 |
| 7-Scenes AUC@3 | 12.63 | 12.35 macro, 13.20 micro — neither |

So three of the paper's reported rows are not obtainable from the released
checkpoint on the shipped configs *by the authors*. Our failure to hit 6.42 on
Oxford and 80.39 on 7-Scenes was never a reproduction defect on our side.

**Four residual gaps between us and their README, with causes:**

1. **7-Scenes reconstruction (0.0411/0.0469/79.06 vs 0.036/0.044/82.38).**
   Root cause identified — see §2.5. Not a like-for-like comparison.
2. **Oxford Spires trajectory (6.1887 vs 5.374).** Unexplained. Their 5.374
   matches neither of our two shipped Oxford configs (sparse 6.1887, dense-auto
   5.162), and no two of the three sources agree on all three metrics: our ATE
   is close to the paper's (6.19 vs 6.42) while their RPE-rot is close to the
   paper's (3.694 vs 3.70) and ours is not (4.29). Suggests a config the repo
   doesn't ship. Open.
3. **TUM (9 vs 66 scenes).** Different subsets, not comparable. No documented
   9-sequence subset exists anywhere in the repo or paper (`fixes.md`).
4. **KITTI and VBR RPE-rot** (0.879 vs 0.696; 4.365 vs 4.564) while ATE and
   RPE-trans match to 3 decimals on both. A rotation-only divergence at matching
   translation smells like a GT-convention or frame-pairing difference, not a
   model difference. Neither dataset is in the paper. Low priority.

Also: their README publishes an **NRGBD trajectory row** (ATE 0.056) that the
shipped `neural_rgbd.yaml` cannot produce — it sets `traj.enable: false` and
`auc.enable: false`. Another number generated from an unshipped config.

### 2.5 The 7-Scenes reconstruction gap — resolved (data prep, not code)

This closes the long-standing open item in `AGENTS.md` ("Acc/Comp off ~1.5-2×,
unresolved").

Upstream's [seven_scenes.py](../benchmark/datasets/seven_scenes.py) reads GT
depth from `frame-{id}.depth.proj.png`. We changed it to `frame-{id}.depth.png`
(`fixes.md`, "seven_scenes.py bugs") because no `.proj` variant exists on our
disk and every frame failed to load.

That was not a bug in their code. `benchmark/README.md` says to prepare
7-Scenes *"following the data preparation in Pi3"*; `.depth.proj.png` is the
**projected** depth that preprocessing emits — raw 7-Scenes depth is captured in
the depth-camera frame and is not registered to the RGB camera. Our
`/group/compact-3dmem/datasets/7-scenes/` holds the raw Microsoft distribution
(`.color.png` / `.depth.png` / `.pose.txt` only), so the "fix" silently
substituted unregistered depth for registered depth as ground truth.

Evidence this is the cause:

- The defect is confined to metrics that consume GT depth. Pose and AUC, which
  don't, match their README to 3–4 s.f.
- It is confined to the one dataset whose GT loader we modified. ETH3D and
  NRGBD adapters are untouched and both match their README almost exactly.
- Direction is consistent with a misregistered GT cloud: our Acc is ~2× the
  paper's and our Comp ~1.5× lower, i.e. the two clouds are offset rather than
  uniformly noisier.
- Upstream issue #68 asks about reproducing 7-Scenes, but never mentions
  `.depth.proj.png`, so it is not evidence of another user hitting this. It does
  confirm the protocol: `lingbot-map.pt`, pure streaming, `num_scale_frames=8`,
  `keyframe_interval=1`, depth unprojected with predicted poses, no masking.

**Resolved 2026-08-23.** Rather than regenerate Pi3's `.depth.proj.png`, we warp
the raw depth into the colour frame ourselves (`_register_depth`). The prediction
held: Acc/Comp moved to 0.0331/0.0427 against their README's 0.036/0.044, not
toward the paper's 0.02/0.07, and F1 went 79.06 → 82.98 versus their 82.38. Pose
was unchanged as a control. Details in `reproduction.md`.

The `get_scenes()` change in the same file (skip instead of raise on a stray
`eval_gt/`) is unrelated and should stay.

### 2.6 Bookkeeping

The 7-Scenes reconstruction figures originally in `reproduction.md` were a
scene-subset aggregate, not the full 18-sequence result. All 18 scenes were
evaluated in one pass (2026-08-16), so nothing was overwritten. Corrected
2026-08-23 to the exact full-set values used above; subset aggregates are not
recorded anywhere in these notes.

---

## Part 3 — Paper-internal contradictions

Listed because several of them are why a faithful reproduction can look like a
mismatch.

- **§6.2 prose vs Table 4.** "our ATE of 0.22 is nearly 4× lower than Wint3R
  (0.86)" — Table 4 says **0.43** (= 2×). Code, their README and our run all
  give 0.43–0.44. The prose is wrong.
- **§6.3 prose vs Table 5.** "ETH3D … F1 of 98.98 … accuracy (0.09 vs 0.28) and
  completeness (0.03 vs 0.21)" — Table 5 says **0.16 / 0.08 / 86.79**, which is
  what everyone reproduces. The prose is wrong.
- **AUC thresholds.** §5.2 defines 3° and 30°; Table 2 reports AUC@**15** and
  @30, a threshold §5.2 never introduces.
- **ETH3D F1 threshold.** §5.1 says d = 0.1 m; §5.2 says 0.25. Code uses 0.25.
- **Keyframe selection.** §3.4 says "select a key frame every m frames"; §4.4
  says adaptive flow-based. The code implements §3.4 for Direct and §4.4 for VO.
- **Benchmark count.** Intro and the contributions list name 4 datasets; §5.1
  defines 5 (adds NRGBD), and Fig. 1's radar plots 5.
- **Training data.** Table 1 has 31 rows; §4.3 says "29 datasets" and "we draw
  from all 29" for Stage 1, but only 26 rows carry a Stage-1 ratio. **Replica
  [62]** is named in the Stage-1 text and absent from Table 1.
- **T&T runner-up.** §6.3 calls Stream3R (0.76) the ATE runner-up; TTT3R is 0.66
  in the same table, making the ratio 3.3× not 3.8×.
- **Sequence length.** Abstract and README claim ~20 FPS "over long sequences
  exceeding 10,000 frames"; §4.4 says Direct mode is stable to ~3,000, beyond
  which VO mode — which resets state per window and adds alignment drift — is
  required.
- **Duplicate reference.** [78] and [79] are the same CUT3R paper.

---

## Part 4 — What changes for our open items

| item | status after this pass |
|---|---|
| 7-Scenes reconstruction gap | **Fixed** (§2.5): raw vs Pi3-projected GT depth; registering it lands all six metrics on their README row. |
| Table 2 not reproduced | **Closed** (§1.8): it used an unreleased 160-epoch checkpoint, so it is unreachable from public weights by construction. Upstream's own README row is separately unreachable — not the shipped config (10-configuration sweep, `scratch/reproduction/oxford/oxford_readme_ruleouts.md`). |
| Table 3 not reproduced | Unchanged, and now better explained: §1.2 means the paper's dense protocol (flow-based keyframing) is not implementable from the released streaming code at any fixed interval. |
| Whether to PR upstream | The `.depth.proj.png` change should **not** be PR'd — it was our data-prep gap, not their bug. The `get_scenes()`, `tum.py`, `prepare.py`, flock and Oxford TLS-naming fixes still stand. |
| New | Their README's NRGBD trajectory and TUM rows are not producible from the shipped configs (`traj.enable: false`; no Freiburg1 whitelist), and the Oxford row is not the shipped config either — all worth an upstream issue. |

## Method / reproducibility of this comparison

- Paper text extracted with `pypdf` to
  `/tmp/.../scratchpad/paper.txt`; all §/table references are to that text.
- Code compared against `upstream/main` = `c95c33c`, which is also our
  merge-base, so nothing in Part 1 is an artifact of our local commits. Our
  local changes to files cited: none in `lingbot_map/models/`, `demo.py`,
  `gct_profile.py`, `resize.py`, `eth3d.py`, `preprocess/oxford.py`.
- Our numbers read from `eval/*.json` aggregates under
  `/group/compact-3dmem/campaigns/lingbot_map/{default,sparse_s12,dense_s1}/`.
- Not verified: the ~20 FPS / ~10.5 FPS throughput claims (no harness),
  anything in Part 1.1 (no code exists), and the exact contents of Pi3's
  7-Scenes preprocessing (inferred from the filename and `benchmark/README.md`).
