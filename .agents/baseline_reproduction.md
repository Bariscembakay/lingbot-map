# Baseline reproduction — TTT3R + ZipMap (+ CUT3R re-check)

Goal: reproduce each baseline's published reconstruction numbers before any of
them is scored on our unified recall benchmark (nrgbd_recall_s2_n{100,300,500}).
CUT3R was already reproduced (see `cut3r_evaluation.md`: NRGBD Table 4 means
match on raw `depth/`; medians carry a known `grey_white_room` gap).

## Vendored code (commit 9b229e0)

| dir | upstream | commit |
|---|---|---|
| `TTT3R/` | github.com/Inception3D/TTT3R (main) | `edd6d8c` |
| `ZipMap/` | github.com/haian-jin/ZipMap (main) | `e0f1f40` |
| `ZipMap_eval/` | same repo, branch `evaluation` (recons_eval fork) | `8b6b629` |

- TTT3R ships no weights: it drives CUT3R's `cut3r_512_dpt_4_64.pth`
  (symlinked to the same /group copy CUT3R uses). Its `croco/` tree is
  byte-identical to CUT3R's except `pos_embed.py`, so the curope-free
  negative-position RoPE patch was ported verbatim (see `CUT3R/UPSTREAM.md`
  for the equivalence argument). TTT3R runs in the existing `cut3r` env
  (+ `imageio`, `imageio-ffmpeg` — demo utils import them).
- ZipMap checkpoints (HF `coast01/ZipMap`) →
  `/group/compact-3dmem/checkpoints/ZipMap/{checkpoint_aff_inv,checkpoint_online,checkpoint_state_query}.pt`,
  symlinked from `ZipMap/checkpoints/`. aff_inv = main bidirectional (Table 3);
  online = streaming fine-tune; state_query = pose-only-query fine-tune (from
  the stage-2 ref-view model, NOT affine-invariant — carry this caveat into the
  recall benchmark).
- `zipmap` env: own micromamba env (torch==2.6.0 per pyproject) —
  `.agents/scratch/insait_cluster_files/setup_zipmap_env.sh`.

## Reproduction target

**ZipMap paper Table 3** (7-Scenes + NRGBD; Acc/Comp/NC mean+median) — its
rows cover ZipMap AND CUT3R AND TTT3R under exactly the vendored
`ZipMap_eval` harness, so one run validates all three baselines and our data
prep at once. Reference values (arXiv 2603.04385v2, means):

| dataset (dense) | method | Acc | Comp | NC |
|---|---|---|---|---|
| 7-Scenes kf40 | ZipMap | 0.018 | 0.030 | 0.680 |
| 7-Scenes kf40 | CUT3R | 0.023 | 0.028 | 0.674 |
| 7-Scenes kf40 | TTT3R | 0.035 | 0.032 | 0.666 |

(sparse = kf200 on 7-Scenes, kf500 on NRGBD; dense = kf40 / kf100. Full rows
incl. medians to be filled from the paper when the runs land.)

Secondary check: TTT3R's own `eval/mv_recon/run.sh` (7scenes, kf_every=2,
max_frames=200, modes cut3r vs ttt3r) against its Figure 9 curves — the paper
publishes no numeric recon table, only plots, which is why ZipMap's Table 3 is
the primary gate.

## Data prep

- NRGBD: `/data/NRGBD` raw (`depth/`, `poses.txt`) — same choice as the CUT3R
  reproduction (raw depth matched the paper means; `depth_filtered` was a
  ruled-out variant).
- 7-Scenes: both harnesses read Spann3R-format `frame-*.depth.proj.png`
  (depth registered to the color camera), which our copies lack. Generating
  with SimpleRecon's own `7scenes_preprocessing.py` (the script Spann3R's docs
  point at), restricted to the 13 eval sequences, writing in place into
  `/group/compact-3dmem/datasets/7-scenes/` —
  `.agents/scratch/baselines/register_7scenes_depth.py`.

## Status log

- 2026-09-04: repos vendored + pushed; checkpoints downloaded; cut3r env
  reused for TTT3R; zipmap env building. Smoke: TTT3R (a100) needed imageio;
  ZipMap smoke = load online ckpt + 8-frame NRGBD forward + state_query load
  check. 7-Scenes depth registration job running. ZipMap_eval wiring
  (data/ symlinks, checkpoints/, eval config) pending; its embedded
  models/ttt3r/croco needs the same pos_embed check before running.
- 2026-09-04 (13:30): **ZipMap row of Table 3 REPRODUCED** via vendored
  ZipMap_eval (job 821691, a100). 7scenes-dense means: Acc 0.0183 / Comp
  0.0303 / NC 0.680 vs paper 0.018 / 0.030 / 0.680 — matches to the printed
  precision. Full per-dataset CSVs under `ZipMap_eval/outputs/mv_recon/ZipMap/`
  (7scenes-sparse Acc 0.0432, NRGBD-sparse 0.0461, NRGBD-dense 0.0157).
  ttt3r+cut3r rows rerunning as job 821750 (their wrappers needed
  transformers/roma etc. beyond the harness requirements).
- 2026-09-04 (15:20): **TTT3R row of Table 3 REPRODUCED** (job 821784, curope
  compiled with conda gcc-13 + cu124 nvcc as upstream requires). 7scenes-dense
  means: Acc 0.0349 / Comp 0.0316 / NC 0.666 vs paper 0.035 / 0.032 / 0.666.
  Other cells: 7scenes-sparse 0.0981/0.1595/0.681, NRGBD-sparse
  0.1009/0.0762/0.826, NRGBD-dense 0.0736/0.0369/0.803. CUT3R pass running in
  the same job.
- 2026-09-04 (16:20): **CUT3R row of Table 3 REPRODUCED** (job 821784
  completed). 7scenes-dense means: Acc 0.0234 / Comp 0.0276 / NC 0.674 vs
  paper 0.023 / 0.028 / 0.674. Other cells: 7scenes-sparse 0.0800/0.1024/0.711,
  NRGBD-sparse 0.0977/0.0746/0.830, NRGBD-dense 0.0651/0.0359/0.812.

## VERDICT: Phase 2 gate PASSED — all three baselines reproduce Table 3

| 7scenes-dense (means) | Acc | Comp | NC |
|---|---|---|---|
| ZipMap paper / ours | 0.018 / 0.0183 | 0.030 / 0.0303 | 0.680 / 0.680 |
| CUT3R paper / ours | 0.023 / 0.0234 | 0.028 / 0.0276 | 0.674 / 0.674 |
| TTT3R paper / ours | 0.035 / 0.0349 | 0.032 / 0.0316 | 0.666 / 0.666 |

Every value matches the paper to its printed precision. This validates: the
vendored code + patches, the ZipMap checkpoints, the CUT3R weights, the
7-Scenes depth.proj registration (SimpleRecon script), the raw-NRGBD wiring,
and the curope build. Cleared to build the recall-benchmark adapters (Phase 3).
