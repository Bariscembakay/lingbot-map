# Oxford README row — what has been ruled out (2026-08-23)

> Campaign data retired 2026-08-24. Aggregates and trajectories are in
> `/group/compact-3dmem/archive/retired/oxford_scope_and_gt_2026-08`; the
> per-frame artifacts and the preprocessed 13-scene / v0.1.0 roots were deleted,
> and `datasets/oxford_spires` holds only the paper's 10 sequences again.

Target: upstream `benchmark/README.md` reports 5.374 / 0.930 / 3.694 over 10 scenes.
Ours (`sparse_s12`, a6000): 6.1887 / 0.7650 / 4.2903.

The README states the numbers come from "the released `lingbot-map.pt` checkpoint
(streaming mode), evaluated on the shipped dataset configs", so flow-based
keyframing — which the paper uses and the repo lacks — is not implicated here.
That makes the row a pure data/config/numerics question.

## Ruled out, with the evidence

| # | candidate | verdict |
| --- | --- | --- |
| 1 | config drift | `git diff upstream/main` on `configs/methods/lingbot_map.yaml` and `configs/datasets/oxford.yaml`: only `raw_data_root`, `_checkpoint` and the `env` name differ. Every knob is upstream's. |
| 2 | wrong checkpoint | `sha256(ckpt/lingbot-map.pt)` = `ee665103…cd72`, bit-identical to the file published at `robbyant/lingbot-map`. |
| 3 | wrong `christ-church-05` sequence | Upstream's `PROCESS_SCENE` names `2024-03-18-christ-church-05`, which does not exist in the release; only `2024-03-20` does. Our fix is forced, and upstream must have made the same one, so it is the same physical sequence. |
| 4 | scene set | The ten are confirmed by upstream issue #38. Note its "13 scenes is a typo for 14" is wrong: `christ-church-01` ships no `gt-tum.txt` (verified against the HF file listing), only COLMAP/SLAM estimates, so 13 is the evaluable maximum and the paper's original wording was right. |
| 5 | frame set / stride | 320 frames at stride 12 (319 for `christ-church-02`, whose pose association drops 19). Matches #38's "first 320 frames, sampled every 12 frames". |
| 6 | `--max_frames` truncation | `preprocess/oxford.py` defaults to 3840, which truncates the raw sequences to 28–97% coverage depending on length. If upstream had used a different cap, the three most-truncated scenes (28–33%) would diverge 3–4×; instead they agree with the paper's per-scene ATE to within 8%, and corr(coverage, ours/paper) = −0.10. |
| 7 | raw data authenticity | Fetched by `snapshot_download` from `ori-drs/oxford_spires_dataset`, which verifies content hashes. |
| 8 | stale upstream code | Only 11 commits ever touched `benchmark/`. `datasets/oxford_spires.py` and `evaluation/trajectory.py` have not changed since the initial commit, and the results table was added in the same commit as the last method-wrapper change. The published numbers correspond to the code we run. |
| 9 | different RPE convention | Our RPE-trans matches upstream on **every** other dataset to ≤2.5% (ETH3D, 7-Scenes, NRGBD, KITTI, VBR, DROID-W, TAT, TUM-fr1). The metric is identical; only Oxford's is off, and in the *opposite* direction (ours is 18% better). |

| 10 | non-metric predicted scale | The Umeyama scale is ~15x on Oxford, but it is 3.5-27x on every other dataset too (ETH3D 6.1, KITTI 27.0, NRGBD 3.5, TAT 7.8). The model is non-metric everywhere; Oxford is unremarkable. |
| 11 | different evaluation frame density | Subsampling our own predictions to effective strides 12/24/36/48/72 moves ATE by only +0.6% (6.1887 -> 6.2284) while RPE-t rises 0.765 -> 2.186 and RPE-rot 4.290 -> 7.602. Upstream needs RPE-t **up** and RPE-rot **down** at once, which no density change produces, and their lower ATE cannot come from density at all. |
| 12 | a different 10-scene selection | Upstream's row says "10" but never names them. Enumerating all 1001 ten-subsets of #38's fourteen: every subset containing `christ-church-05` averages **>= 5.722**, every subset without it **<= 3.832**, and 5.374 falls in the gap. No scene selection produces it. (The same arithmetic validates #38: its fourteen average 5.5925 against their stated 5.59, and its ten average 6.4246 against Table 2's 6.42.) |

| 13 | GPU / attention backend | The control arm — shipped config, but SDPA on an A100 instead of FlashInfer on an a6000 — gives 6.1429 / 0.7710 / 4.2763 against the baseline's 6.1887 / 0.7650 / 4.2903: −0.74% / +0.78% / −0.33%. Changing *both* backend and GPU moves Oxford by under 1%, so environment is not where a 15% gap lives. |

## The finding: the README row is bracketed by the two released checkpoints

All arms below are SDPA on an A100 (FlashInfer's JIT is broken cluster-wide — see
`fixes.md`), so they are mutually comparable. The `ctrl` and `sdpa` arms are the
same config submitted twice and returned **identical** numbers to four decimals,
so the pipeline is deterministic here and every difference below is signal.

| | `lingbot-map.pt` | upstream README | `lingbot-map-long.pt` | bracketed |
| --- | ---: | ---: | ---: | :-- |
| ATE | 6.1429 | 5.374 | 4.7373 | yes |
| RPE-trans | 0.7710 | 0.930 | 0.8760 | **no — above both** |
| RPE-rot | 4.2763 | 3.694 | 3.1734 | yes |

The released `lingbot-map.pt` lands 14.3% *above* the README row and
`lingbot-map-long.pt` 11.8% *below* it. Solving for where between the two the
README would sit gives lambda = 0.453 from ATE and 0.472 from RPE-rot — agreeing
to 4%, consistent with one intermediate checkpoint rather than a coincidence.

**This is also the only lever found that moves ATE and RPE-trans in opposite
directions.** Going `lingbot-map.pt` -> `-long.pt`, ATE falls 6.14 -> 4.74 while
RPE-trans *rises* 0.771 -> 0.876. Every config knob and every evaluation-protocol
change moves them together, which is exactly why nothing else could reproduce the
README's combination of a lower ATE with a higher RPE-trans.

Read with upstream issue #62 — Table 2 used "the 160-th epoch checkpoint" and
both released checkpoints "achieve better ATE than the results in table 2" — the
economical reading is that **the README's Oxford row was produced by a checkpoint
that is neither released file**, most likely an intermediate training state. That
would make the row unreproducible from public artifacts, the same way Table 2 is.

Residual: RPE-trans at 0.930 is above both checkpoints (0.771 and 0.876), so an
interpolated checkpoint predicts ~0.83 and still misses by ~12%. Better than the
29% the uniform-factor model missed by, but not explained.

## Scene set and GT revision — measured, and neither explains the row

Suggested by Baris 2026-08-23. Two things were tested together: whether the row
is a wider-scene aggregate, and whether the dataset's ground truth changed under
us.

**The dataset does have two GT states.** HF `main` and tag `v0.1.0` differ for
five sequences; `christ-church-01` exists only at `v0.1.0` and
`christ-church-05` only on `main`, so **no single published revision covers all
fourteen** — the authors evaluated a state that no longer exists as one revision.
Our local copy matches `main` exactly (git blob OIDs), so nothing was mis-
downloaded.

**But the revisions differ only in coverage, not in pose values.** On every
timestamp present in both, positions are bit-identical (max diff 0.00e+00 m);
`v0.1.0` merely extends further at each end, and the extra poses fall outside the
first 3,840 images. Preprocessed poses come out byte-identical and all four
overlapping scenes score the same under either revision. `main` was trimmed, not
corrected — so **the GT revision does not explain `bodleian-library-02`**, which
had looked like a promising lead.

**All 128 scene-set x GT-revision combinations were enumerated** from the
per-scene `eval/traj.json` files, scoring each against the published triple and
requiring all three metrics to match:

| | ATE | RPE-t | RPE-rot |
|---|---:|---:|---:|
| upstream README | 5.374 | 0.930 | 3.694 |
| 13 scenes, main GT | **5.3966** (+0.42%) | 0.7308 (−21.4%) | 3.4626 (−6.3%) |
| 10 scenes, shipped | 6.1429 (+14.3%) | 0.7710 (−17.1%) | 4.2763 (+15.8%) |

**Zero combinations match all three within 1%.** The 13-scene ATE lands within
0.42% — better than any config arm — but RPE-trans stays ~21% low, and across
the entire search *every* candidate sits at 0.73–0.90 against their 0.930.

That is the most durable fact here: ATE can be moved onto their row by scene
selection, but nothing tested — scene set, GT revision, config knobs, evaluation
density, backend, GPU — lifts RPE-trans. Only the checkpoint moves ATE down and
RPE-trans up together.

Incidental result worth keeping: with its `v0.1.0` GT, `christ-church-01` scores
**3.7427** against the paper's 3.6793 — 1.7% apart, the closest per-scene
agreement in the whole set.

## The sweep, and what it settles

Twelve arms, all SDPA on A100 (FlashInfer's JIT is broken cluster-wide, see
`fixes.md`), so they are mutually comparable. `ctrl` and `sdpa` were the same
config submitted twice and returned **identical numbers to four decimals** — the
pipeline is deterministic here, so every difference below is signal, not noise.
Arm aggregates were read back from the pipeline's own `eval/traj.json`, so every
number here is the evaluator's, not a re-derivation.

| arm | ATE | RPE-t | RPE-rot | dATE | dRPEt | dRPEr | worst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **upstream README** | **5.374** | **0.930** | **3.694** | | | | |
| anchors 4 | 5.625 | 0.892 | 3.832 | +4.7% | −4.1% | +3.7% | **4.7%** |
| anchors 16 + window 128 | 5.511 | 0.865 | 3.420 | +2.6% | −7.0% | −7.4% | 7.4% |
| window 128 | 6.029 | 0.842 | 3.771 | +12.2% | −9.5% | +2.1% | 12.2% |
| anchors 16 | 5.642 | 0.831 | 4.177 | +5.0% | −10.7% | +13.1% | 13.1% |
| `lingbot-map-long.pt` | 4.737 | 0.876 | 3.173 | −11.8% | −5.8% | −14.1% | 14.1% |
| **shipped default** | 6.143 | 0.771 | 4.276 | +14.3% | −17.1% | +15.8% | 17.1% |
| 504x378 exact 4:3 | 6.040 | 0.869 | 3.055 | +12.4% | −6.6% | −17.3% | 17.3% |
| fp32 (no autocast) | 6.094 | 0.778 | 4.653 | +13.4% | −16.3% | +26.0% | 26.0% |
| window 32 | 8.249 | 0.898 | 5.098 | +53.5% | −3.4% | +38.0% | 53.5% |
| 3D RoPE off | 8.704 | 1.075 | 6.564 | +62.0% | +15.6% | +77.7% | 77.7% |

(`window 256` OOMed: SDPA has no paged KV cache, so a 256-frame resident window
exceeds 80 GB. fp32 lands within 1% of the bf16 default on ATE and RPE-trans, so
numerical precision is not a lever either.)

The arm configs were deleted after the sweep; each was a one-line delta from the
shipped `configs/methods/lingbot_map.yaml`, all with `_use_sdpa: true` on top:

| arm | delta from shipped |
| --- | --- |
| ctrl / sdpa | none (SDPA only) |
| long | `_checkpoint: lingbot-map-long.pt` |
| noamp | `_use_amp: false` |
| norope | `_enable_3d_rope: false` |
| scale4 | `_num_scale_frames: 4`, `_kv_cache_scale_frames: 4` |
| scale16 | `_num_scale_frames: 16`, `_kv_cache_scale_frames: 16` |
| win32 / win128 | `_kv_cache_sliding_window: 32` / `128` |
| combo | `_num_scale_frames: 16`, `_kv_cache_scale_frames: 16`, `_kv_cache_sliding_window: 128` |

`configs/methods/lingbot_map_sdpa.yaml` is kept as the shipped config plus SDPA,
since FlashInfer is unusable while the glibc 2.41 / CUDA 12.8 clash stands.

Four things this settles.

**1. The shipped default is the 6th-closest of ten to upstream's own row.** Most
single-knob perturbations fit their published numbers *better* than the config
they say produced them. Combined with environment being worth <1% and data,
protocol and scene selection all ruled out, the economical reading is that the
README row was not produced by the shipped config as we run it.

**2. Oxford ATE spans 4.74–8.25 across single-parameter changes** — a 74% range,
with the README's 5.374 comfortably inside. A 14.3% gap is small against that
sensitivity. This also explains why the other eight datasets reproduce within 1%:
they are short indoor or well-constrained sequences where the memory
configuration barely bites, while Oxford's 200–780 m outdoor traversals lean on
it heavily.

**3. Only three levers move ATE and RPE-trans in opposite directions** — the
checkpoint, the KV window, and the aspect ratio. Every other knob, and every
evaluation-protocol change, moves them together. The README needs a *lower* ATE
with a *higher* RPE-trans, which is why no amount of protocol reasoning could
reach it and why the uniform-quality-factor model missed RPE-trans by 29%.

**4. The sweep cannot identify their configuration, and should not try.** Two
arms land within 10% on all three metrics and several more within 15%. With nine
arms against a three-number target, picking the closest (`anchors 4`, 4.7%) would
be curve-fitting — and it contradicts both the shipped config and issue #68,
where the maintainer states the number of scale frames is 8. The defensible
conclusions are the bracketing result and the sensitivity range, not a winner.

## Oxford is the only row that misses

Same checkpoint, same pipeline, same evaluator, every published ATE row:

| dataset | README | ours | delta |
| --- | ---: | ---: | ---: |
| ETH3D | 0.439 | 0.4424 | +0.77% |
| 7-Scenes | 0.079 | 0.0789 | −0.13% |
| TUM (freiburg1 nine) | 0.045 | 0.04508 | +0.18% |
| Neural RGB-D | 0.056 | 0.0554 | −1.07% |
| KITTI | 24.046 | 24.0772 | +0.13% |
| VBR | 31.204 | 31.1190 | −0.27% |
| DROID-W | 0.909 | 0.9084 | −0.07% |
| Tanks & Temples | 0.210 | 0.2095 | −0.24% |
| **Oxford Spires** | **5.374** | **6.1887** | **+15.16%** |

Eight of nine land within 1.07% (median 0.24%); Oxford misses by 14x the worst of
the rest. Whatever differs is Oxford-specific, not a global environment or
checkpoint effect — which also argues *against* the plain "they used a different
checkpoint" reading, since a different checkpoint would move every row.

## The shape of what is left

A single uniform quality factor `k` applied to our numbers fits two of the three
metrics almost exactly:

| | ours x k (k = 0.8647) | upstream README | miss |
| --- | ---: | ---: | ---: |
| ATE | 5.3513 | 5.3740 | −0.4% |
| RPE-rot | 3.7098 | 3.6940 | +0.4% |
| RPE-trans | 0.6615 | 0.9300 | **−28.9%** |

k from ATE and RPE-rot independently is 0.8684 and 0.8610 — they agree to 0.8%,
which is unlikely to be coincidence. So upstream's Oxford predictions are
uniformly ~13.5% better than ours globally and in rotation, while being ~22%
*worse* in local translation. That is not the signature of a straightforwardly
better run; it is a different trade-off between local smoothness and global
accuracy.

## The aspect squash is real but costs little — tested, not assumed

`datasets/oxford_spires.py` fixes the width to `load_img_size` (518) and floors
the height to a multiple of 14, giving 518x378 = 1.3704 from a 1440x1080 = 1.3333
source: a **2.78% anisotropic squash**. GT intrinsics are scaled per axis
(fx 251.147, fy 244.480) so the GT stays self-consistent, but the model predicts
fx~=fy — square pixels — so it structurally cannot represent that camera.
Upstream runs the same code, so this is not what separates us from them.

`load_img_size: 504` gives 504x378, exactly 4:3, at essentially the same patch
count (972 vs 999). Same backend, same GPU, same config — only geometry differs:

| | 518x378 squashed | 504x378 exact 4:3 | change |
| --- | ---: | ---: | ---: |
| ATE | 6.1429 | 6.0403 | −1.7% |
| RPE-trans | 0.7710 | 0.8686 | **+12.7%** |
| RPE-rot (RMSE) | 4.2763 | 3.0547 | −28.6% |
| RPE-rot **median per-pair** | 0.969 | 0.948 | **−2.2%** |
| AUC@3 / @15 | 17.74 / 63.66 | 17.87 / 61.94 | +0.7% / −2.7% |
| Racc@3 | 53.6 | 53.4 | −0.2 pt |

**The eye-catching −28.6% on RPE-rot is a tail effect, not an accuracy gain.**
The median per-frame-pair rotation error barely moves, both distributions are
heavy-tailed (top 1% of pairs carry 13.5% and 17.7% of the squared error), AUC is
a wash, and RPE-trans gets materially worse. So the squash is a genuine
correctness bug worth reporting upstream, but fixing it does not buy accuracy —
do not sell it as a performance improvement.

## Structure of the error, for the viser session

ATE tracks path length: corr = +0.87 (ours) and +0.94 (the paper's per-scene).

| scene | path | our ATE | drift | paper ATE | drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| christ-church-05 | 780.9 m | 40.502 | 5.19% | 35.987 | 4.61% |
| bodleian-library-02 | 539.9 m | 4.162 | 0.77% | 9.592 | 1.78% |
| observatory-quarter-01 | 266.4 m | 1.941 | 0.73% | 2.861 | 1.07% |
| (median of all ten) | | | 1.05% | | |

`christ-church-05` is the longest sequence (3× the median) *and* the worst relative
drift; its error is spread evenly across the path (deciles 23–57 m), so it is a
global drift failure, not one excursion. `bodleian-library-02` is the opposite: a
clean track at 0.77% drift where the paper's checkpoint managed only 1.78%, which
is what issue #62's "released weights beat Table 2" would predict.
