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

## Table 2 (Oxford Spires, sparse) — reproduced

Ran 2026-08-15 on sof1 (hala, a6000). Results:
`/group/compact-3dmem/campaigns/lingbot_map/oxford_spires/sparse_s12/
oxford/eval/{traj,auc_macro,auc_micro}.json`, HTML report in
`.../sparse_s12/report/`.

| Metric | Paper | Ours |
|---|---|---|
| AUC@15 | 61.64 | 63.50 |
| AUC@30 | 75.16 | 76.50 |
| ATE | 6.42 | 6.19 |
| RPE-trans | 1.01 | 0.765 |
| RPE-Rot | 3.70 | 4.29 |

Ran on sof1 only — msp3 env/checkpoint are also set up (generic runner at
`.agents/scratch/insait_cluster_files/submit_msp3.sh`) but unused, sof1
succeeded first. msp3's checkout has local, uncommitted config overrides
(`raw_data_root: /data/oxford_spires_processed`, `workspace: /scratch/...`)
— expect it to show dirty.

Bugs found + fixed getting here (committed, details in `fixes.md`):
3 upstream Oxford Spires TLS-cloud naming bugs, a `setup_bench_env.sh`
sourcing bug, two separate CUDA build gaps (`cuda-driver-dev`+`CPATH`;
FlashInfer's `lib64` symlink), and `run.py`'s `conda` shim needing `PATH`.

## Cluster facts specific to this project

- Envs `lingbot_map`/`bench`: micromamba, per-node `/scratch` — bootstrap
  scripts + why in `fixes.md`. Built on sof1 and msp3.
- Checkpoint: `ckpt/lingbot-map.pt` (gitignored symlink) →
  `/group/compact-3dmem/checkpoints/lingbot-map/lingbot-map.pt`, synced to
  msp3's own path too.
- Git: `origin` = fork (`Bariscembakay/lingbot-map`), `upstream` =
  `Robbyant/lingbot-map`. Push auth: PAT via `credential.helper store`
  (see `fixes.md`). msp3 has its own clone.
- `/group/compact-3dmem`: run outputs → `campaigns/lingbot_map/<benchmark>/
  <arm>/`; raw downloads → `datasets/<name>/`; preprocessed → 
  `datasets/<name>_processed/`.
- `campaigns/paper_reproduction/` is a different project's — don't touch.

## Dataset status

| Dataset | Raw data | Ready to run | Registered |
|---|---|---|---|
| Oxford Spires | ✅ `datasets/oxford_spires` (161G, 10 scenes) | ✅ processed, Table 2 done | ✅ raw + processed |
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

Paper comparison (Table 2/4/5), where finished:

| Dataset | Pose (AUC/ATE) | Reconstruction (Acc/Comp/F1) |
|---|---|---|
| Oxford Spires | ✅ matches | — (not evaluated) |
| ETH3D | ✅ matches closely | ✅ matches closely |
| TAT | ✅ matches closely | — (points disabled) |
| NRGBD | — (not evaluated) | ✅ matches closely (sof1 + msp3 bit-identical) |
| 7-Scenes | ✅ matches closely | ❌ Acc/Comp off ~1.5-2x, unresolved — see below |
| DROID-W | n/a — not one of the paper's benchmarks, no baseline to compare |
| VBR / TUM | pipeline run, still in progress as of this writing |

**7-Scenes reconstruction: could not reproduce, deferred.** Pose/AUC
matches the paper closely, but points.json (Acc/Comp/F1) doesn't, even
after ruling out: sampling/eval config (identical to NRGBD's, which
matched), frame correspondence, resolution, camera intrinsics (tried the
empirically-"corrected" 585 vs hardcoded 525 — made Acc *worse*, reverted).
Closest lead: the model's pred/GT depth scale ratio drifts slightly more
across a 7-Scenes sequence (~7%) than NRGBD's (~4.5%), which Umeyama's
single global scale can't fully correct — real but not enough to explain
the whole gap on its own. `stairs` sequences are also genuine severe
outliers (F1 ~57 vs 73-86 elsewhere) independent of this. Revisit later.

## Open items

- 7-Scenes reconstruction gap (above) — revisit with a fresh angle.
- Whether to PR the Oxford Spires upstream bugs + the 4 bugs found this
  round back to `Robbyant/lingbot-map`: undecided. Note issue #68 on that
  repo shows another user hit the same 7-Scenes symptom, unresolved there too.
- msp3 workflow validated end-to-end (dataset pull → compute → rsync back
  to sof1) via the NRGBD run above.
