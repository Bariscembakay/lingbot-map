# CUT3R official recall probe — conventions (scene 210f741378, 16 frames)

Produced by `run_cut3r_recall.py` → `cut3r_recall_210f741378.npz` (now in `/group/compact-3dmem/campaigns/spatial_memory/_recall_dumps/`).
Model: official `cut3r_512_dpt_4_64.pth` (512-res DPT head, ViT patch 16),
loaded with `ARCroco3DStereo.from_pretrained`.

## Which CUT3R code ran

The **vendored, patched copy at `lingbot-map/CUT3R`** (same upstream commit
`8bc15dc` as the fresh clone at `~/CUT3R`), because the fresh clone cannot run
on the cut3r env's torch 2.8: it needs `weights_only=False` in `load_model`
and the RoPE2D negative-position fix (pose token sits at position −1; the
pure-torch fallback's `F.embedding` asserts on it, and the curope CUDA
extension does not compile against torch 2.8). Both patches are additive,
numerically equivalent to upstream's CUDA path, and documented in
`lingbot-map/CUT3R/UPSTREAM.md`. **Neither CUT3R tree was modified.**
No extra pip installs were needed; the stock `cut3r` env sufficed.

## Raymap-query API (the "virtual camera / revisit" ability)

Native, first-class API — no hacks needed:

- Every view dict carries `img_mask`, `ray_mask`, `update`, `reset` flags
  (see `demo.py:prepare_input` and `viser_utils.py:_connect_client`, which
  does exactly this for its interactive virtual camera).
- A raymap-only query view is: `img = NaN [1,3,H,W]`, `ray_map = [1,H,W,6]`
  (channel-LAST), `true_shape=[[H,W]]`, `img_mask=False`, `ray_mask=True`,
  `update=False`, `reset=False`, `camera_pose=I` (placeholder, unused).
- Query call: `dust3r.inference.inference_step(view, state_args, model,
  device)` with `state_args = state_args_list[-1]` from
  `inference(views, model, device)` (the ingest pass).
- **State cannot be contaminated by queries, twice over:** inside the model
  `update_mask = img_mask & update` (`model.py` ~line 800) so a raymap-only
  view never writes state; and `inference_step` discards the returned new
  state anyway. Verified at runtime: all five state tensors
  (`state_feat, state_pos, init_state_feat, mem, init_mem`) are **bitwise
  identical** before and after the 16 queries (asserted in the script).
  No snapshot/restore was therefore needed — all 16 queries read the same
  final state.
- Ingest was the demo's own streaming path: 16 image views in order,
  `img_mask=True, ray_mask=False, update=True`, single `inference()` call
  (internally sequential-recurrent; autocast disabled, fp32).

## Raymap formula (CUT3R's own, copied verbatim)

From `src/dust3r/datasets/base/base_multiview_dataset.py:get_ray_map`
(identical in `viser_utils.py`):

```
c2w  = inv(c2w_first) @ c2w_i          # pose relative to camera 0
ro   = c2w[:3,3]                       # broadcast per pixel, channels 0:3
rd   = normalize( (c2w @ [K^-1 @ (u,v,1); 1])[:3] )   # channels 3:6
```

- Pixel grid is `arange(w) x arange(h)`, `indexing="xy"`, **no +0.5**.
- **Quirk (do not "fix"):** the unit-depth pixel point is transformed as a
  homogeneous POINT, so the translation contaminates the "direction":
  `rd = normalize(R @ d_cam + t)`, not `normalize(R @ d_cam)`. This matches
  what the released raymap encoder was trained on. It agrees with
  `lingbot_map/memory/probe_data.py:build_raymap(convention="cut3r")` exactly.
- Frame: everything is relative to **camera 0 of the ingested clip** (CUT3R's
  world frame). `gt_c2w.npy` is already relative-to-first, so `c2w_first = I`.

## Scale of the query poses — METRIC, not canonical

`gt_c2w.npy` in the cache has translations in canonical units (divided by
`gt_scale = 2.4621531782876516` from `meta.json`). CUT3R is trained on metric
raymaps (its datasets gate raymap inputs on `is_metric`), and its internal
map built from the 16 RGB frames is in its own ~metric scale — so the query
translations were **multiplied back by `gt_scale` to metres**
(`query_c2w_used` in the npz is what was actually fed, metric, relative to
camera 0, OpenCV c2w). The scorer must NOT assume the npz predictions are in
canonical GT units: CUT3R's output scale is its own (~metric); use median/
similarity alignment as planned.

## Frames, resolution, intrinsics

- Video frames `range(0, 640, 40)` = cache indices `range(0, 32, 2)` (cache
  stride 20). Decoded with ffmpeg at native 1920x1440 (rawvideo, no jpeg
  round-trip), then preprocessed exactly like `load_images(size=512)`:
  long-edge lanczos resize 1920x1440 → 512x384, centre crop to multiples of
  16 → **H=384, W=512** (crop is a no-op at this aspect ratio). `hw=[384,512]`.
- Intrinsics: cached `gt_intrinsics.npy` (per-frame, for the 378x518 cache
  grid) rescaled axis-wise to native then through the same resize+crop math
  → `intrinsics_used` [16,3,3]. Same K used for each frame's query raymap.

## What is in the npz (all float32 unless noted)

| key | shape | meaning |
|---|---|---|
| `pred_pts_self` | [16,384,512,3] | query-view pointmap in the **query camera's own frame** (`pts3d_in_self_view` head) |
| `pred_pts_world` | [16,384,512,3] | query-view pointmap in **CUT3R's world frame = camera 0 of the clip**, CUT3R's own scale (`pts3d_in_other_view` head = its DPT-cross output, i.e. the model's own world-frame prediction, not our geotrf) |
| `pred_conf` | [16,384,512] | `conf_self` (confidence of `pred_pts_self`) — matches the `pred_conf` semantics of `ours_*.npz` |
| `pred_conf_other` | [16,384,512] | `conf` (confidence of `pred_pts_world`) |
| `pred_rgb` | [16,384,512,3] | model's hallucinated image at the query view, range [-1,1] |
| `pred_query_c2w` | [16,4,4] | **model's own predicted pose** for each raymap query (pose head `absT_quaR` decoded via `pose_encoding_to_camera`), c2w in CUT3R's world frame. Available — not identity. NOTE: for the DPT head the world pointmap is an independent `dpt_cross` branch, NOT `geotrf(pose, self)` — measured `\|pred_pts_world − geotrf(pred_query_c2w, pred_pts_self)\|` mean 0.027, so the three outputs are consistent but not bitwise redundant |
| `ingest_depth` | [16,384,512] | z of `pts3d_in_self_view` predicted at ingest time (lag-0 reference, CUT3R's scale) |
| `ingest_pts_world` | [16,384,512,3] | ingest-time world-frame pointmaps |
| `ingest_conf` | [16,384,512] | ingest-time `conf_self` |
| `ingest_c2w` | [16,4,4] | model's predicted ingest trajectory (for scale alignment against GT if wanted) |
| `query_c2w_used` | [16,4,4] | GT poses actually fed to the raymaps: relative to camera 0, **metric**, OpenCV c2w |
| `intrinsics_used` | [16,3,3] | K at 384x512 used for the raymaps |
| `gt_scale` | scalar f64 | 2.4621531782876516 (canonical→metric factor; divide predictions' translations/depths by it to compare in canonical units) |
| `hw` | [2] int | [384, 512] |
| `video_frame_ids` | [16] int | 0..600 step 40 |
| `cache_indices` | [16] int | 0..30 step 2 |

## Caveats for the scorer

- `pred_pts_world` lives in the frame of **camera 0** (first ingested frame),
  in **CUT3R's own predicted scale** (~metric but not exact); the query pose
  fed in was metric GT, but the model treats the raymap as a soft conditioning
  signal — its output pose `pred_query_c2w` is its own estimate and can differ
  from `query_c2w_used` in both pose and scale.
- `pred_pts_self` is the cleanest thing to score depth/pointmap recall on
  (frame-invariant up to the model's scale); scale-align before comparing.
- GT depth in the cache is canonical (metres ÷ gt_scale); CUT3R predictions
  are not — one global scale alignment is expected and fair.
- Resolution differs from the cache grid (384x512 vs 378x518); resize
  predictions to the GT grid (the existing `score_recall.py` already does).

## Run record + sanity checks

Slurm job 762701 on `hala` (a6000, rendering+qos, zone-sof1), 2026-08-28.
Wall: 45.2 s total — ingest of 16 frames 2.1 s, all 16 raymap queries 2.4 s,
rest model load + video decode. npz is 177 MB compressed.

- No NaNs in any pred/ingest array.
- Per-view median z of the query pointmaps tracks the lag-0 ingest depths
  (e.g. 2.09/2.02, 5.09/4.55, 3.76/3.83 m) — the state is genuinely being
  read out at the queried cameras.
- The model's own predicted query poses match the fed metric GT poses to a
  few percent in translation norm across all 16 views (e.g. fed 1.156 m vs
  pred 1.148 m at view 15) — confirming metric raymap translations were the
  right scale and CUT3R's internal frame is ~metric, anchored at camera 0.
