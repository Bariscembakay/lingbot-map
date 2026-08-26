# Working notes — lingbot-map

Local-only (gitignored). Current state — commit history covers what
changed and why. Technical gotchas live in `fixes.md`.

## Goal

Reproduce the paper's benchmark tables via `benchmark/`. Table 2 = Oxford
Spires "sparse" = `benchmark/configs/oxford.yaml` (stride 12, not
`oxford_long.yaml`). Paper's real benchmarks (via arXiv HTML, not this
repo's README): Oxford Spires, ETH3D, 7-Scenes, Tanks and Temples, NRGBD
(main tables) + TartanAir/TartanGround (ablation only, out of scope — no
adapter exists, user doesn't need it).

Pipeline: `preprocess/<dataset>.py` (raw → BSS layout, Oxford only so far)
→ `prepare.py` → `run.py` (dispatches to method env) → `evaluate.py` →
`report.py`. Envs: `lingbot_map` (method) and `bench` (framework) —
`run.py` shells from `bench` to `lingbot_map` via `conda run`.

## Reproduction status

All numbers, per-benchmark settlement and open threads live in
`reproduction.md`. Summary as of 2026-08-23: every published upstream row
reproduces within ~1% except Oxford Spires, and both Oxford rows are settled as
unreachable — Table 2 used an unreleased 160-epoch checkpoint (issue #62) and the
README row is not the shipped config (10-configuration sweep). Bugs found and fixed getting
here are in `fixes.md`.

## Cluster facts specific to this project

- Envs `lingbot_map`/`bench`: micromamba, per-node `/scratch` — bootstrap
  scripts + why in `fixes.md`. Built on sof1 and msp3.
- Checkpoint: `ckpt/lingbot-map.pt` (gitignored symlink) →
  `/group/compact-3dmem/checkpoints/lingbot-map/lingbot-map.pt`, synced to
  msp3's own path too.
- Git: `origin` = fork (`Bariscembakay/lingbot-map`), `upstream` =
  `Robbyant/lingbot-map`. Push auth: PAT via `credential.helper store`
  (see `fixes.md`). msp3 has its own clone.
- Retired campaigns keep metrics only, under `archive/retired/<name>/`.
- `/group/compact-3dmem`: run outputs → `campaigns/lingbot_map/<arm>/<benchmark>/`
  (arm above benchmark since 2026-08-20); raw downloads → `datasets/<name>/`;
  preprocessed → `datasets/<name>_processed/`.
- `campaigns/paper_reproduction/` is a different project's — don't touch.

## Dataset status

| Dataset | Raw data | Ready to run | Registered |
|---|---|---|---|
| Oxford Spires | ✅ `datasets/oxford_spires` (~196G, 14 scenes — the paper's 10 plus the 4 it excluded) | ✅ processed | ✅ raw + processed |
| VBR | ✅ `datasets/vbr` (113G, 7 scenes) | ✅ | ✅ |
| DROID-W | ✅ `datasets/DROID-W` (8G, 7 scenes) | ✅ | ✅ |
| TUM RGB-D | ✅ `datasets/TUM-RGBD` (73G, 80 seqs) | ✅ | ✅ |
| 7-Scenes | ✅ `datasets/7-scenes` | ✅ | already was |
| NRGBD | ✅ `datasets/nrgbd` | ✅ | already was |
| ETH3D | ✅ `datasets/eth3d`, Pi3-preprocessed in place | ✅ | ✅ re-registered |
| TAT | ✅ full (GT + images, `datasets/TAT`) | ✅ | ✅ |
| KITTI (Odometry) | ✅ `datasets/kitti_odometry` (63G, seq 00-21, GT poses 00-10) | ✅ no preprocessing needed | ✅ |
| TartanAir/TartanGround | — | — | out of scope |

All config yamls wired to real paths, committed.

## Benchmark pipeline testing round (2026-08-16)

Ran prepare→run→evaluate for all data-ready datasets on sof1, plus one
msp3 workflow validation run (NRGBD — bit-identical to sof1's own run,
confirming cross-zone reproducibility). Found + fixed 4 real bugs along
the way (flock race condition, `seven_scenes.py` eval_gt + depth filename,
`lingbot_map_v1` unconfigured method, `tum.py` `_secret` sequences +
prepare.py's any-failure-aborts-everything behavior) — see `fixes.md`.

Per-benchmark results and status: `reproduction.md`.

**7-Scenes reconstruction: solved 2026-08-23** — the cause was ground truth, not
the model: raw `.depth.png` is in the depth-camera frame but the pipeline
unprojects it with the colour focal. `_register_depth` in
`benchmark/datasets/seven_scenes.py` (opt-in, default off) warps it into the
colour frame. Full explanation in `reproduction.md`.

**This change must NOT be PR'd** — it is our data-prep gap, not upstream's bug.
Upstream reads `.depth.proj.png`, which their Pi3-based prep emits and the raw
Microsoft distribution does not ship.

## Open items

- Notes layout: `reproduction.md` = reproducing upstream (status, tables,
  per-benchmark settlement). `trajectory_memory_ablation.md` = our own context-token
  experiments. `paper_vs_repo.md` = full paper↔repo comparison (code, configs,
  three-way number reconciliation, paper-internal contradictions).
  `.agents/scratch/reproduction/oxford/oxford_readme_ruleouts.md` = the Oxford sweep
  evidence. `spatial_memory_design.md` = the CUT3R-fed-by-lingbot-map architecture
  (decisions, closed). `spatial_memory_plan.md` = its implementation sequencing.
  `cut3r_evaluation.md` = every run against the published CUT3R checkpoint.
- Oxford vs the **paper** — the authors' per-scene ATE is in upstream issue #38
  (their ten average 6.4246 = Table 2's 6.42), and issue #62 says Table 2 used an
  unreleased 160-epoch checkpoint, so Table 2 is unreproducible from public
  weights. 8/10 scenes agree within 13%; `bodleian-library-02` (we are 2.3x
  better) and `christ-church-05` (1.13x worse) move in opposite directions, so
  the close means are largely cancellation. **Open: inspect
  `bodleian-library-02` in viser** — the only scene whose gap is not explained by
  run variance.
- Oxford vs the **README** — **closed 2026-08-23** by a 10-configuration sweep
  (`.agents/scratch/reproduction/oxford/oxford_readme_ruleouts.md`). Their row is not
  the shipped config: the shipped default is the 6th-closest of ten arms to it,
  GPU+backend is worth <1%, and data/protocol/scene-selection are all ruled out
  by measurement. The row is bracketed by the two released checkpoints, and
  Oxford ATE spans 4.74-8.25 across single-knob changes, so the 14.3% gap is
  small against the benchmark's own sensitivity. Not pursued further: several
  configs land within 10%, so naming one would be curve-fitting.
- Oxford **aspect squash** (new, reportable): `datasets/oxford_spires.py` warps
  1440x1080 to 518x378, a 2.78% anisotropic squash the model cannot represent
  since it predicts square pixels. `load_img_size: 504` fixes it at the same
  patch count. Real correctness bug, but negligible accuracy effect — median
  per-pair rotation error moves 2.2% and RPE-trans worsens.
- TUM — **solved 2026-08-23.** Upstream's nine are the standard Freiburg1 set
  (360, desk, desk2, floor, plant, room, rpy, teddy, xyz), given away by the
  README's fr1/desk figure. `configs/datasets/tum.yaml` now carries a `_scenes`
  whitelist of the nine and `configs/tum.yaml` points at the `tum_fr1` arm, which
  reproduces their row: 0.04508 / 0.01323 / 0.51242 against 0.045 / 0.013 / 0.513.
  Without the whitelist the adapter finds all 66 sequences in a full download.
- ETH3D — **closed 2026-08-23.** Not an AUC bug: `DA3_FILTER_KEYS` drops exactly
  its intended frames, and all 17 published metrics are worse by 0.08-0.79% with
  the deficit monotone in threshold (@3 worst, @30 best) — the signature of run
  numerics, not a frame-subset or aggregation difference.
- Whether to PR the Oxford Spires upstream bugs + the 4 bugs found this round
  back to `Robbyant/lingbot-map`: undecided. The strongest candidates are the
  aspect squash and the non-monotonic anchor-count response; the `.depth.png`
  change is not one of them.
- msp3 workflow validated end-to-end (dataset pull → compute → rsync back
  to sof1) via the NRGBD run above.
- **CUT3R's raymap direction channel — open, and we deliberately match it.**
  `get_ray_map` (`src/dust3r/datasets/base/base_multiview_dataset.py:13`, and
  identically in `viser_utils.py:453`) builds the direction as

      rd = inv(K) @ [u, v, 1]                    # a direction, camera frame
      rd = (c2w @ vstack([rd, ones]))[:3]        # <- the `ones` make it a POINT,
      rd = rd / |rd|                             #    so c2w applies R *and* t

  i.e. `normalize(R·d_cam + t)` where a ray direction is `normalize(R·d_cam)`.
  Verified numerically: exact when `t = 0` (which is camera 0, since the raymap
  is relative to it), **10.57 deg off at `t = (1.5, -0.3, 2.0)`**. Our camera
  origin norms are p50 0.48 / p100 0.97 canonical with `|d_cam| ~ 1`, so we sit
  in that regime.

  Not asserted to be a bug. `inv(K) @ [u,v,1]` has z exactly 1, so
  `c2w @ [d_cam; 1]` is the world **point** at camera-depth 1 along the ray, and
  origin-plus-a-point-on-the-ray is a valid encoding -- what makes it odd is
  normalising a point afterwards, which discards the distance. The paper (§3.2)
  says "encoding the origin and direction of rays at each pixel" and the variable
  is named `rd`, so code and paper disagree; a stray homogeneous `1` is the most
  likely explanation, but the code alone cannot settle intent.

  Recoverability: **as a field, not per pixel.** `p = λn` for unknown λ, and
  `normalize(λn - t)` depends on λ; λ is pinned only because `d_cam.z == 1`,
  which needs `R`, which is shared across the image. So it is learnable, by a
  weaker route than a true direction.

  **Decision: match their construction.** It is the distribution the released
  raymap encoder was trained on, so arm A *must* use it -- feeding correct rays
  would be off-distribution input and would make E1 understate CUT3R, flattering
  us. Arm C matches it too, to keep A/B/C on one footing and to keep the
  inherited 25 M `enc_blocks_ray_map` weights in-distribution.
  `--raymap-convention {cut3r,true}` stays a one-flag sweep axis. Settling intent
  properly means asking upstream.
