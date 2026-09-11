# nrgbd_recall_s2 — the recall benchmark

Measures the *recall* ability of streaming 3D reconstruction systems: after
ingesting a video stream, the model is queried with **camera poses only** (no
images) for every past frame and must decode the geometry it saw there.
Equivalently: standard streaming-reconstruction evaluation where the second
pass hands the model raymaps instead of images.

## Protocol (frozen; manifests/ is the source of truth)

- **Scenes**: the 7 NRGBD scenes with >=1000 raw frames (whiteroom, kitchen,
  grey_white_room, green_room, complete_kitchen, breakfast_room, staircase).
  Every tier uses the identical 7-scene set — no per-tier scene dropping.
- **Ingestion**: first 1000 raw frames, stride 2 → ingest indices 0,2,…,998.
  Tiers n100 / n300 / n500 are *prefixes* of that one list.
- **Queries**: after ingestion, EVERY ingested frame's camera, images withheld.
- **GT**: raw `depth/` (16-bit mm, valid 1mm–10m — the variant every prior
  CUT3R/ASVGGT/lingbot eval reads) + `poses.txt`.
  **`poses.txt` is OpenGL c2w** — runners multiply by `diag(1,-1,-1,1)`
  (proven: cross-frame depth reprojection 1.9mm with flip vs 137mm raw).

## Query-pose semantics (the one subtle decision)

A raymap query is only meaningful in *some* frame. Ours is posed (GT frame
native); CUT3R/TTT3R/ZipMap reconstruct in their own frame-0-anchored,
up-to-scale worlds. Protocols:

- **selfpose (primary)**: query the pose the model itself estimated for frame
  t — pure memory, uncontaminated by localisation error.
- **gtpose (secondary, CUT3R-family only)**: GT poses relative to frame 0.

Fused-cloud metrics use **one Sim(3) per (scene, tier)** — Umeyama on camera
centres — applied uniformly to every method (near-identity for posed/metric
systems; supplies scale for ZipMap). Per-view depth metrics are view-local and
need no alignment (scale only).

## Metrics per (method, scene, tier)

- fused cloud vs GT cloud: Acc / Comp (bidirectional NN dist), Chamfer,
  means + medians;
- per-view depth: AbsRel, delta<1.25 (per-scene scale);
- **recall-vs-lag curve**: per-query accuracy binned by frame age — the
  headline forgetting plot.

## Baselines and caveats

| method | ingestion | query mechanism | caveat |
|---|---|---|---|
| CUT3R | sequential | native raymap revisit (state read-only) | ~metric |
| TTT3R | sequential (TTT state rule) | same as CUT3R (same weights) | — |
| ZipMap | **bidirectional** | state-query ckpt: 9-ch Plücker rays → render() | its state-query ckpt is a stage-2 fine-tune with no streaming variant; recorded in every metrics.json |
| ours | sequential (posed) | GT raymap (native) | posed system — declared as its own column |

All three baselines were reproduced against their papers before being scored
here (`.agents/baseline_reproduction.md`): mv_recon 72/72 numbers, relpose all
cells, videodepth 5/6 (ZipMap-KITTI checkpoint caveat).

## Outputs

`/group/compact-3dmem/campaigns/spatial_memory/recallbench/<method>/<scene>_n<tier>/`:
`metrics.json`, `pred_cloud_rgb.ply` (coloured by the query frame's image),
`pred_cloud_err.ply` (error heat), `gt_cloud_rgb.ply`. Viser walker sees them.

## How a cell is produced (two phases, since 2026-09-11)

Ingest is ~3 min of what used to be a ~10 h cell; the rest is GT unprojection
and KD-tree work that needs no GPU. Holding an H200 through that made the cells
*unschedulable* on a full cluster, so the runners split:

1. **GPU** — `run_cut3r_family.py --update-rule {cut3r,ttt3r} --dump-dir D`
   (ours: `run_ours.py --dump-dir D`). Ingest + query only, ~5 min for all
   three tiers of a scene. Writes `D/<method>/<scene>_n<tier>.npz`
   (0.24 GB at n100, 1.18 GB at n500).
2. **CPU, no GPU requested** — `run_recall_score.py --method M --scene S
   --tier T --dump-dir D --out OUT` (ours adds `--posed`). Metrics + clouds.
   These schedule instantly while every GPU is busy.

Verified bit-identical to the old single-process path on the same GPU
(cut3r/breakfast_room/n100: acc 0.0938, comp 0.0642, absrel 0.0969 both ways).
`run_zipmap.py` is still single-phase. Every runner resumes per tier.

## Provenance: one GPU type per table

**Cells do not reproduce across GPU type or code version.** Over 10 overlapping
cut3r cells (a100 originals vs h200 re-runs): selfpose mean |delta| 14.1%
(max 65%), gtpose mean |delta| 20.3% (max 169%). Both modes move -- an earlier
note here claimed gtpose was stable to 1e-4, which was generalised from one
cell and is wrong.

Two causes, easily confused. (a) **Stale code**: `breakfast_room_n500` was
written 2 min before the Sim(3) commit and 8 h before the OpenGL-flip fix, so
its +169% gtpose delta is a bug. A file's mtime does not certify it -- a job
that started before a fix keeps the old code in memory. (b) **Genuine
sensitivity**: selfpose fits a Sim(3) to the model's own predicted camera
centres, worst where the reconstruction is already poor (green_room, acc > 1 m,
swung 65%) and the fit is ill-conditioned. Ours fits nothing and is exempt.

Consequence: cut3r and ttt3r were fully regenerated on h200 under one code
version rather than completing a part-a100 table. The superseded a100 cells
are kept at `campaigns/spatial_memory/recallbench_superseded_a100/`. ZipMap's
cells are a100 and are **not** directly comparable on selfpose to the
regenerated rows; its column is footnoted.

## Viewing

`python view_cell.py --method M --scene S --tier T [--show pred gt err]`
serves the full-resolution clouds in viser. Files on disk are never
decimated; `--stride` only thins what reaches the browser.

Future work: VBR / HM3D long-loop extension.
