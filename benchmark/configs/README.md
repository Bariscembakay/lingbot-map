# Configs — what is upstream's and what is ours

Upstream's files are unmodified apart from `raw_data_root` / `_checkpoint` /
`workspace` paths. Everything below that is ours exists because a shipped config
could not produce a number we needed.

Base configs may sit in a campaign subdirectory; `datasets/` and `methods/` are
found by walking up to the configs root. Config **names are filename stems** and must be unique across subdirectories —
`core/config.py` discovers `datasets/**/*.yaml` and `methods/**/*.yaml`
recursively and raises on a duplicate stem. Subdirectories are for grouping only;
the BSS workspace keys method output directories by that same stem, so moving a
config is safe but **renaming one orphans its results on `/group`**.

## Ours, and why

| config | why it exists | results |
|---|---|---|
| `neural_rgbd_traj.yaml` | shipped `neural_rgbd.yaml` sets `traj.enable: false`, so it cannot produce upstream's own NRGBD trajectory row | `campaigns/lingbot_map/default/neural_rgbd` |
| `tum.yaml` + `datasets/tum.yaml` `_scenes` | upstream's row is the nine Freiburg1 sequences; without the whitelist the adapter finds all 66 | `campaigns/lingbot_map/tum_fr1` |
| `seven_scenes_depthproj.yaml` + `datasets/seven_scenes_depthproj.yaml` | GT depth registered into the colour frame (upstream reads a `.depth.proj.png` the raw distribution does not ship) | `campaigns/lingbot_map/depthproj` |
| `oxford_aspect.yaml` + `datasets/oxford_ar504.yaml`, `oxford_ar1008.yaml` | tests the adapter's 2.78% anisotropic squash at exact 4:3 | retired 2026-08-24; metrics in `archive/retired/oxford_scope_and_gt_2026-08` |
| `methods/lingbot_map_sdpa.yaml` | shipped config with SDPA; FlashInfer's JIT is unusable on glibc 2.41 + CUDA 12.8 (`.agents/fixes.md`) | — |
| `ctx_ablation/` + `methods/ctx_ablation/` | the trajectory-memory context-token ablation (8 base configs, 16 arms) | `campaigns/lingbot_map/context_token_ablation` |

The Oxford README-row sweep's arms (`lm_sw_*`) were deleted once it concluded;
each was a one-line delta from `methods/lingbot_map.yaml`, tabulated in
`.agents/scratch/reproduction/oxford_readme_ruleouts.md`.

The scene-scope and GT-revision configs (`oxford13`, `oxford_v010gt`) were removed
2026-08-24 with their data: the investigation concluded, and the aggregates live in
`/group/compact-3dmem/archive/retired/oxford_scope_and_gt_2026-08` with the findings
in `.agents/scratch/reproduction/oxford_readme_ruleouts.md`. `datasets/oxford_spires`
now holds only the paper's 10 sequences again.
