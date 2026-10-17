# Frame-anchor diagnostic — does CUT3R degrade with distance from its frame-0 origin?

Motivation: a proposed moving (egocentric / re-anchored) output frame instead of
CUT3R's fixed frame 0. Before any training, test on the frozen checkpoint
whether frame-0 outputs degrade with **distance from the origin** (fixable by a
moving frame) or with **stream index** (state saturation, which a moving frame
does not fix). Measurement record: append runs, don't rewrite.

## Setup

| | |
|---|---|
| checkpoint | published `cut3r_512_dpt_4_64.pth` (provenance in `cut3r_evaluation.md`) |
| code | `.agents/scratch/frame_anchor/` — `cut3r_anchor_dump.py` (GPU), `analyze_frame_anchor.py` (CPU), `cut3r_anchor_job.sh`, `check_stream_equivalence.py` |
| stream loop | one frame per step through the model's own `_forward_decoder_step`; matches batched `model(views)` to fp32 reordering noise (median rel. diff ≤ 5e-4, p99 ≤ 3e-3 over 12 frames, strict fp32) |
| numerics | **strict fp32**: `croco.py` turns TF32 matmul on at import; off here so rounding cannot pose as a distance effect. With TF32 on, streamed vs batched p99 reaches 6.6e-2 |
| GPU | RTX PRO 6000 Blackwell (`gcp-eu1-rtx6000-kzdd`), all runs; jobs 1113467-71 |
| order | palindrome `0..N-1..0`; first N positions are exactly the forward run (causal) |
| resolution | CUT3R's `load_images(size=512)`; GT depth nearest-resized + cropped identically, K transformed with dust3r's pixel-centre convention |
| artifacts | `/group/compact-3dmem/campaigns/frame_anchor/cut3r_pub_palindrome_s1/<bench>/<scene>/` (`frames.npz`, `points_s8.npz`, `meta.json`); analysis in `_analysis/` (`summary.md`, `summary.json`, `headline.png`, `plots/`) — 3.9 G |

Scope — every scene of each set:

| benchmark | scenes | frames | GT |
|---|---|---|---|
| `nrgbd` | all 9 | 10,970 | depth + pose; ≤ 5.5 m from frame 0 |
| `kitti` (odometry) | 00-10 | 23,201 | pose only; up to 2.1 km |
| `kitti_depthval` | all 13 val drives | 3,556 | LiDAR depth only; K from the devkit 1216×352 selection intrinsics + crop offset (verified against odometry P2 on two dates) |

Per-frame metrics: `self_shape` / `world_shape` = residual of a per-frame Sim(3)
fit of X_self / X_world to GT camera-frame points, median, ÷ median GT depth
(non-rigid error only); `ws_ratio` = world/self; `sw_res` = residual of Sim(3)
X_self→X_world (GT-free); RPE@10 from the predicted frame-0 poses.

## Results (2026-10-16)

**1. X_world degrades with distance from the origin; X_self does not.**
KITTI depth-val, medians per predicted-distance bin:

| pred. dist (m) | X_world | X_self | ratio fwd | ratio return |
|---|---|---|---|---|
| 0-10 | 0.099 | 0.098 | 1.01 | 1.03 |
| 25-50 | 0.100 | 0.086 | 1.17 | 1.41 |
| 50-100 | 0.204 | 0.122 | 1.69 | 2.04 |
| 100-200 | 0.299 | 0.093 | 2.82 | 4.82 |
| 200-400 | 0.542 | 0.105 | 5.10 | 6.82 |

NRGBD over 0-5 m: ratio 1.08 → 1.45, X_self flat (0.016-0.021).

Within-scene partial Spearman over the full palindrome, ρ(·, dist | index) / ρ(·, index | dist):

| | dist given index | index given dist |
|---|---|---|
| KITTI depth-val `ws_ratio` | 0.69 | 0.24 |
| KITTI odometry `sw_res` | 0.80 | 0.08 |
| NRGBD `ws_ratio` | 0.075 | 0.088 |

The forward pass alone is mostly rank-collinear on KITTI (guarded to nan above 0.95); the return trip is what decouples the two.

**2. The pose head cannot place the camera more than 298 m from frame 0.**
Max predicted camera distance over all 53,490 KITTI stream positions is
**297.94 m** (raw pre-`expm1` norm 5.700); 0 positions above 298 m; the
histogram piles up below it. GT reaches 2,130 m (seq 01). Per-sequence max
predicted 148-298 m. The pose head is a linear MLP on a LayerNorm-ed token, so
its range is bounded; 5.70 is consistent with that bound being hit, not derived.

| GT dist from frame 0 (m) | 0-25 | 25-50 | 50-100 | 100-200 | 200-400 | 400-800 | 800-1600 | >1600 |
|---|---|---|---|---|---|---|---|---|
| pred / GT distance | 2.12 | 1.12 | 1.02 | 0.77 | 0.27 | 0.16 | 0.12 | 0.054 |

| GT dist from frame 0 (m) | 0-50 | 50-100 | 100-200 | 200-400 | 400-800 | 800-1600 | >1600 |
|---|---|---|---|---|---|---|---|
| 10-frame step length pred / GT (GT step > 1 m) | 0.98 | 0.95 | 0.96 | 0.89 | 0.88 | 0.57 | 0.49 |

Step lengths stay near right to 800 m; the trajectory stays inside the ceiling
mostly by heading drift curling it up (RPE@10 rot ≈ 4°).

**3. Pose rotation degrades with stream index, not distance.**
NRGBD (bounded distance), RPE@10 rotation by stream position: 0.32° (0-50) →
0.42 → 0.72 → 2.13 (200-400) → 3.01 (400-800) → 2.34 (800-1600). Partial
Spearman: dist|index 0.079, index|dist 0.37.

Palindrome, same image, first 10% of each sequence (forward = earliest and
nearest the origin; return = the last positions of the stream):

| | return / forward | pred. dist fwd → ret |
|---|---|---|
| NRGBD RPE@10 rot | 6.09 | 0.74 → 1.88 m |
| NRGBD X_world shape | 2.03 | |
| NRGBD X_self shape | 1.02 | |
| KITTI odometry RPE@10 rot | 1.48 | 73 → 74 m |
| KITTI depth-val X_world shape | 1.47 | 14 → 35 m |
| KITTI depth-val X_self shape | 0.97 | |

### Reading

- Origin distance is real and large for the frame-0 **pointmap** (≈5× at a few
  hundred metres) and is a hard **range ceiling** for the frame-0 **pose**. Both
  are what a moving / re-anchored output frame removes.
- Rotation / local pose degrades with stream index at fixed distance (NRGBD,
  palindrome) — a state effect a moving frame does not address.
- X_self (egocentric) is unaffected by either, everywhere.

## Hardening result 2: repeated straight segments (2026-10-16)

Two GT-straight segments (chord = arc), each streamed 4× back to back, with and
without a state reset after the last frame of each pass. Arms
`cut3r_pub_repeat4_s1` / `cut3r_pub_repeat4_reset_s1`, job 1113659, analysis in
`campaigns/frame_anchor/_analysis_repeat4/` (`summary.md`, `repeat4.png`).

| segment | length | speed |
|---|---|---|
| KITTI 01, f214-484 (highway) | 717 m, 271 frames | 2.66 m/frame |
| KITTI 06, f731-1100 (urban) | 442 m, 370 frames | 1.27 m/frame |

**No pass of 16 exceeds 298 m.** Max is 297.7 m (06 pass 1, raw norm 5.699).

Pass 1, predicted distance from the segment start at a given GT distance (m):

| GT | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 442 | 600 | 717 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 06 | 51 | 103 | 148 | 180 | 220 | 263 | 295 | 283 | 261 | | |
| 01 | 42 | 80 | 101 | 118 | 133 | 143 | 144 | 151 | 158 | 129 | 116 |

01 plateaus near 150 m, well below the cap, so the plateau level is
sequence-dependent; 298 m is only the maximum ever reached.

Without reset, predicted displacement within each pass (GT 442 / 717 m every pass):

| pass | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 06 | 261 | 92 | 37 | 27 |
| 01 | 116 | 28 | 39 | 48 |

The model neither continues forward nor re-traces pass 1 (median 61-103 m from
pass 1's positions); it hovers at 75-200 m.

**A reset starts a new coordinate frame.** The first frame after each reset is
predicted at |t| = 0.01 m, rotation 0.03-0.06°, matching training, where the
single-view datasets reset after every view with `camera_pose = eye(4)`. Passes
2-4 are bitwise identical to each other; they differ from pass 1 by up to
31.6 m (01) / 18.3 m (06) because pass 1's first frame uses the learned
`pose_token` and a post-reset frame takes its pose token from the pose
retriever on the initial memory (`model.py:765-768`). The reset predicts no
transform from the old frame to the new one.

## Ceiling or drift? (2026-10-16)

`plot_trajectories.py` → `campaigns/frame_anchor/_analysis_trajectories/`
(`straight_segments.png`, `kitti_full_sequences.png`). Heading = predicted vs GT
optical axis in frame-0 x-z; forward speed = predicted motion along the
camera's own optical axis ÷ GT forward motion, 10-frame windows.

KITTI 00-10 forward passes, by stream position:

| stream position | 0-50 | 50-100 | 100-200 | 200-400 | 400-800 | 800-1600 | 1600-3200 | >3200 |
|---|---|---|---|---|---|---|---|---|
| forward speed ratio | 1.01 | 0.92 | 0.81 | 0.64 | 0.19 | 0.22 | 0.26 | 0.20 |
| frac. windows backward | 0.00 | 0.00 | 0.04 | 0.16 | 0.41 | 0.37 | 0.32 | 0.36 |
| local heading drift (deg / 10 frames) | 0.61 | 1.79 | 3.48 | 3.52 | 3.20 | 3.22 | 3.91 | 3.60 |

By predicted distance the speed ratio is 0.23-0.45 in every bin below 280 m,
then **-0.21 with 59% of windows backward at 280-300 m**.

- **What keeps most KITTI trajectories inside ~150-200 m is index-driven
  crumpling**: forward progress collapses after ~400 frames and the path
  tangles back and forth. This is the state effect of result 3, not the cap.
  The pred/GT distance ratios in result 2's table (0.05 at > 1.6 km) mostly
  measure this.
- **The 298 m wall is real but rarely reached**: in KITTI 04 (271 frames) and
  the 06 straight segment, heading stays within 13° and forward speed ≈ 1.0
  until ~200 m, then falls to ≤ 0 as the prediction meets 298 m, with the
  camera still looking down the road.
- 01 highway segment: speed drops to 0.3-0.5 by GT 100 m with heading still
  aligned (< 5° until GT 200 m); the 90° heading drift comes after the stall.

## Reset-and-chain (2026-10-17)

`--order chunked --chunk K --overlap 1`: reset the state after every K frames;
chunk j+1 starts on chunk j's last frame o, so o has a pose in both chunk
frames and chunk j+1 → chunk j is `T_j(o) @ inv(T_j+1(o))`. Rigid links, no
scale fit (CUT3R is metric). Scored by `analyze_chunked.py` →
`campaigns/frame_anchor/_analysis_resetchain/` (`summary.md`,
`kitti_trajectories.png` = continuous / K=25 / K=50, `kitti_trajectories_allK.png`).
Arms `cut3r_pub_chunk{K}_ov1_s1`, jobs 1113714-23. Continuous = forward pass of
`cut3r_pub_palindrome_s1`. Checks: chunk 0 is bitwise equal to the continuous
run's first K frames in every run; chaining recovers a synthetic trajectory
to 1e-13.

KITTI odometry 00-10, mean over sequences (t_err / r_err = KITTI devkit,
100-800 m sub-paths, no alignment):

| run | t_err % | r_err deg/100 m | ATE Sim(3) m | ATE SE(3) m | RPE@10 rot deg | RPE@10 trans m | fwd speed | max dist m |
|---|---|---|---|---|---|---|---|---|
| continuous | 70.2 | 24.7 | 210 | 218 | 3.65 | 9.50 | 0.44 | 240 |
| reset every 25 | 12.1 | 5.20 | 42.9 | 59.1 | 0.93 | 1.13 | 1.00 | 636 |
| reset every 50 | 13.2 | 4.10 | 47.8 | 70.2 | 1.00 | 1.50 | 1.01 | 622 |
| reset every 100 | 24.4 | 8.52 | 86.3 | 131 | 1.45 | 2.39 | 1.05 | 716 |
| reset every 200 | 46.3 | 16.2 | 127 | 237 | 2.16 | 4.64 | 1.00 | 824 |

Every sequence improves at K = 25 and 50 (per-sequence t_err in `summary.md`).

NRGBD, all 9:

| run | ATE Sim(3) m | RPE@10 rot deg | X_world shape | X_self shape | X_self abs-rel |
|---|---|---|---|---|---|
| continuous | 1.18 | 2.17 | 0.0313 | 0.0204 | 0.0286 |
| reset every 50 | 0.394 | 0.435 | 0.0184 | 0.0185 | 0.0264 |
| reset every 100 | 0.432 | 0.516 | 0.0194 | 0.0191 | 0.0269 |

KITTI depth-val, all 13:

| run | X_world shape | X_self shape | world/self | X_self abs-rel |
|---|---|---|---|---|
| continuous | 0.218 | 0.111 | 1.87 | 0.0849 |
| reset every 25 | 0.124 | 0.120 | 1.02 | 0.0887 |
| reset every 50 | 0.125 | 0.116 | 1.05 | 0.0874 |
| reset every 100 | 0.133 | 0.112 | 1.13 | 0.0849 |
| reset every 200 | 0.156 | 0.110 | 1.34 | 0.0846 |

- A reset changes two things at once — the origin moves and the state is
  fresh — so this cannot attribute the gain between them. K = 200 chunks
  mostly stay under the wall yet sit at 46% t_err with forward speed ≈ 1, so on
  trajectories the within-chunk rotation degradation (result 3) dominates.
- Losing memory does not cost accuracy even in NRGBD rooms with revisits.
- No loop closure; error accumulates over N/K rigid links.

## Caveats

- The return trip is also a revisit (CUT3R Table 5: revisiting can help), so
  return ≤ forward is not by itself evidence against an index effect.
- Distance is the *predicted* distance (the quantity the heads output); on
  KITTI it is capped (result 2), so bins above ~150 m are thin and mix drift.
- NRGBD and KITTI do not collapse onto one curve in distance ÷ scene depth
  (at 1-2 depths: NRGBD ratio 1.42, KITTI 1.06).
- One checkpoint, one GPU type, strict fp32. Not run: TUM / 7-Scenes / Oxford.
