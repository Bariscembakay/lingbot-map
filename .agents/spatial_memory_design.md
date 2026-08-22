# Persistent summary memory — design record

Status: design settled, implementation in progress. Every entry is revisitable;
"decided" means "the current default we build against".

Revision 2026-08-22: the architecture moved from a standalone queryable memory
to a **read/write pair inserted between the aggregator and the heads**. Earlier
revisions (D=2048, 4 partitioned states, raymap-only reads, teacher-distillation
loss) are superseded throughout.

## Purpose

The summary state is the **compressed version of the evicted tokens**, so it is
complementary to what lingbot-map already holds in its KV cache. Two regimes are
implemented and interchangeable (`memory/schedule.py`):

```
disjoint (default) -- state and cache tile the sequence exactly

  frame:  0 ....... 7 | 8 .............. i-64 | i-63 ......... i
          └ anchors ─┘  └── in the summary ──┘  └── in cache ──┘
          └──────── in cache ────────┘

overlap (CUT3R's regime) -- the state absorbs every frame as processed

  frame:  0 ........................................ i-1 | i
          └──────────── in the summary ──────────────────┘
          └ anchors ─┘                  └─── in cache ───────┘
```

**The disjoint lag is derived, not chosen.** Attention sees the cache *after*
eviction (`layers/attention.py`: append :700, evict :703, attend :729), so at
step `i` the model sees `{0..7} u {i-63..i}`. The read consumes `S_{i-1}`, not
`S_i`. Requiring the two to tile:

    (i-1) - lag == i - window   =>   lag == window - 1 == 63

Verified numerically over 320 steps: zero gaps, zero overlaps, first write at
step 71, 249 writes per clip. Off by one in either direction leaves a frame in
neither memory or in both. (An earlier revision of this document said `i-73`;
that was wrong.)

Anchor frames 0-7 are never evicted, so in disjoint mode they never enter the
state -- correctly, since they stay permanently in the cache.

Two things follow, and they are the two losses:

1. Because it holds what the cache dropped, it should **improve the current
   reconstruction** — depth and pose for frame *i*.
2. Because it is a compression of the past, it should be a **memory of the
   past** — queryable at an old camera, either faithfully (recall) or better
   than the model managed at the time (hindsight).

## Architecture

```
frame ─> DINOv2 ─> Aggregator (24 frame + 24 global blocks)
                        │
                        │  x_i = last-layer tokens [P, 1024]
                        │       (the 1024-d stream after block group 23,
                        │        NOT the 2048-d tap)
              ┌─────────┴─────────┐
              │                   │
      ╔═══════▼═══════╗   ╔═══════▼════════╗
      ║ WRITE         ║   ║ READ           ║
      ║  Q  = S_{i-1} ║   ║  Q  = x_i      ║
      ║  KV = x_i     ║   ║  KV = S_{i-1}  ║
      ╚═══════╤═══════╝   ╚═══════╤════════╝
              │                   │
            S_i             refined x_i
     (for the next step)          │
                                  ▼
                    tap_23 = cat(frame_inter_23, refined x_i)
                    taps 4, 11, 17 unchanged
                                  ▼
                      Depth head + Camera head
```

A recurrent cell: `S_i = WRITE(x_i, S_{i-1})` and `y_i = READ(x_i, S_{i-1})`.
**Write and read run in parallel and both consume the OLD state.** Consequences:

- The read provably cannot see frame *i* through the state. Temporal leakage is
  impossible by construction, not by a min-gap heuristic.
- The write's only gradient is through **future** reads, since `S_i` is first
  consumed at step *i+1*. This is the central optimization difficulty.

## Decided

| | |
|---|---|
| state | `[N, 1024]`, **N = 512** default, sweep {256, 512, 1024} |
| write input | last-layer tokens, **unrefined** (a write fed refined tokens would feed itself) |
| write schedule | **`disjoint`** default (lag 63), **`overlap`** (lag 0) selectable via `write_mode` |
| write positional input | nothing beyond the tokens (arm **a**); `position_mode` flag carries "pose"/"xyz" for later |
| read output | refines the last layer only |
| tap 23 | `cat(frame_inter_23, refined)`; taps 4/11/17 pass through untouched |
| residual gate | zero-init, so step 0 is bit-identically published lingbot-map |
| clip length | **320** frames (the paper's own maximum training length) |
| stride | **20** — see "Why stride matters" |
| loss 1 | lingbot-map's own composite loss vs GT |
| loss 2 | recall and hindsight both implemented; experiments deferred |
| frozen | DINOv2, aggregator, both heads |
| trainable | write, read, raymap encoder, state init |

## BLOCKER: tap 23 has zero influence on the depth head

Measured, not inferred. `d(depth)/d(tap23) == 0` exactly, by both autograd and
finite differences, while taps 4/11/17 all have large gradients.

Mechanism, in the **published** frozen head:

```
layer_4_rn = scratch.layer4_rn(tap23_projected)     # 100% NEGATIVE, |.| ~ 470
out        = resConfUnit2(layer_4_rn)
             |
             +- out = ReLU(inplace=True)(x)   <-- mutates x to zeros in place
             +- out = conv2(ReLU(conv1(out)))
             +- return skip_add.add(out, x)   <-- x is now the ZEROED tensor
```

`ResidualConvUnit.activation` is `ReLU(inplace=True)`, so the residual `+ x` adds
the *mutated* tensor rather than the original. Normally that only loses the
negative half of the skip; here `layer4_rn`'s output is **100% negative on every
frame measured** (max -58, min -1000, fraction > 0 = 0.0000% across frames
0/8/32/64/100/127), so the first ReLU annihilates it entirely and `refinenet4`
emits a constant. For contrast `layer3_rn` (tap 2) is 11.4% positive and behaves
normally.

So the deepest DPT branch is a constant, and the head effectively runs on three
taps. This is a property of the published checkpoint, not of anything we added.

**Consequence for this design: the depth path is void as built.** Refining the
last-layer tokens cannot change depth. It *can* still change pose, because the
camera head reads `aggregated_tokens_list[-1][:, :, 0]` directly with no DPT
fusion in between -- verified working (V4c). The earlier "band limitation" note
below was wrong: the state's influence does not enter at the coarsest scale, it
does not enter at all.

**Resolved: run both arms.** `refine_taps` selects which taps the read touches;
one shared state, one reader per refined tap.

| arm | `refine_taps` | losses | trainable |
|---|---|---|---|
| **A** | `(0,1,2,3)` | depth + abs-pose + rel-pose | 136 M |
| **B** | `(3,)` | pose only (depth auto-disabled) | 61 M |

Arm A's depth path is verified live: V8 overfits a real frame 0.5575 -> 0.066
with `refine_taps=(0,1,2)`, where tap 23 alone could not move the loss at all.
The "an intermediate the aggregator already consumed" objection stands but is
weaker than it looked: it concerned the *aggregator's* internal consistency, and
the frozen head has no such expectation.

Rejected: making the head's ReLU non-inplace. It would fix the residual, but the
head's weights were trained with this behaviour, so the result would no longer be
published lingbot-map.

**V13 guards this permanently**: it asserts exactly which taps influence depth
and fails if that ever changes, so `refine_taps` can never silently optimise a
constant again.

## Why only the last layer

Taps 4/11/17 are intermediate activations that the aggregator itself consumed.
Refining tap 11 would leave blocks 12-23 having run on the unrefined version, so
the head would receive four grids that never coexisted in any real forward pass.
Refining only the last layer avoids that: nothing downstream inside the
aggregator reads it.

The price is a **band limitation**. In the DPT head the deepest tap goes through
`Conv2d(k3, s2)` and is the lowest-resolution branch, entering `scratch_forward`
first. So the state's influence enters coarse and propagates up through the
fusion blocks; fine detail still comes from the untouched shallow taps.

Refining tap 23 also changes the **camera head's** input, which is what gives the
pose terms of Loss 1 any gradient at all.

## Why stride matters more than clip length

The state can only contribute what the cache does not already hold. The cache is
72 frames:

### Measured revisit score -- fraction of visible surface last seen > window frames ago

Three ScanNet++ scenes, 320-frame clips, from GT depth and poses:

| window | stride 10 | stride 20 | stride 30 |
|---|---|---|---|
| **64** (published) | 10-35% of frames over 10% stale | **33-43%** | 27-41% |
| **16** | 52-56% | **56-67%** | 54-66% |

Two conclusions, both material:

1. **Stride 20 beats stride 10 substantially and stride 30 adds nothing** (and
   truncates clips on shorter recordings). Hence stride 20.
2. **The window matters more than the stride.** Dropping it from 64 to 16 roughly
   doubles the fraction of frames where the state has anything to contribute, at
   every stride.

That second point reframes the experiment. Two legitimate stories:

- **(a) same window, better predictions** -- window 64 + memory beats window 64
  alone. Directly useful, but only 33-43% of frames can show it.
- **(b) smaller window, same quality** -- window 16 + memory approximates window 64
  alone. That is a *memory-efficiency* claim: the KV cache is ~76 MB/frame, so
  replacing 48 frames of it with a 512x1024 state is a ~700x compression. It is
  also the regime where the signal demonstrably exists.

So **window is an arm, not a constant**, and the money plot becomes quality vs KV
cache size with and without the memory. If the state lets the window shrink 4x at
equal quality, that is the paper.

## Loss 1 -- lingbot-map's own composite loss (paper §3.3)

```
L = lambda_depth * L_depth + lambda_abs * L_abs_pose + lambda_rel * L_rel_pose
```

| term | form |
|---|---|
| `L_depth` (VGGT) | `sum_i  || Sigma_i . (Dhat_i - D_i) ||  +  || Sigma_i . (grad Dhat_i - grad D_i) ||  -  alpha log Sigma_i` |
| `L_abs_pose` (VGGT) | `sum_i || Phat_i - P_i ||_eps`  (Huber) |
| `L_rel_pose` (pi^3) | `1/(k(k-1)) sum_{i != j} [ L_rot(i,j) + lambda_trans L_trans(i,j) ]` over all pairs in the k=64 window |

- `Sigma_i` is the predicted uncertainty map -- the head's `conf` channel with
  `expp1` activation, `1 + exp(x)`, which is DUSt3R's parameterisation.
- The depth term supervises **the gradient of depth as well as depth**.
- `L_rot` is geodesic, `L_trans` is l1.
- **Poses are supervised camera-to-world, not world-to-camera** -- the paper is
  explicit: in w2c "rotation and translation are inherently coupled, making
  translation estimation highly sensitive to rotation errors, particularly in
  long sequences."
- The paper gives no values for `lambda_depth`, `lambda_abs`, `lambda_rel`,
  `lambda_trans`, `alpha`, `eps`. Start from VGGT's and treat as a small sweep.
- The paper also used **progressive view training** (short subsequences first,
  growing view count). Worth mirroring.

**Loss 1 cannot be self-distilled.** With the frozen model's own prediction as
the target, the gradient at the zero-init gate is exactly zero -- the model
already matches itself. That is what makes GT mandatory, and it is the single
biggest change from the earlier design.

## Loss 2 -- querying a past camera

No current tokens exist for a past camera, so the query is a **Plücker raymap**
through a lightweight encoder, into the **same read transformer**, with **no
state update**. CUT3R does exactly this and their raymap encoder is 2 blocks.

| variant | target | measures | ceiling |
|---|---|---|---|
| recall | lingbot-map's own causal prediction at time *i* | faithfulness | the teacher |
| hindsight | **GT depth** at frame *i* | is the past understanding now *better* | none |

Hindsight is the more interesting claim: the state has seen frames *after* *i*,
so its answer at *i*'s camera can beat what the model could have said at time
*i*. Experiment order (deferred): loss1 alone, loss1+hindsight, loss1+recall.

The raymap query starts far off the token manifold that the read was trained on,
so a warmup that fits `raymap(pose_i) -> x_i` is worth having. The target is
unreachable (tokens depend on appearance), so it converges to the mean token
field for that camera -- which is exactly the useful thing: it lands the query in
the right region.

## Controls that are not optional

**The read is queried by `x_i`, which is frame *i*'s own content.** So Loss 1 can
improve with the state contributing nothing at all: a 25 M-parameter residual
transformer on the last-layer tokens can learn to correct systematic bias in the
frozen head and look exactly like success.

| arm | isolates |
|---|---|
| full | read + write, state updated normally |
| **frozen state** | identical read, state pinned at its learned init, never written |
| no read | frozen lingbot-map (the zero-init-gate baseline) |

Gate the read's cross-attention separately from its MLP so the state path's
contribution is measurable rather than inferred.

**lingbot-map already has a competing long-term memory.** The paper's trajectory
memory keeps 6 tokens per evicted frame, forever -- ~1,488 tokens for a 320-frame
clip, nearly 3x a 512-slot state, and growing linearly. So the ablation grid is
{trajectory memory on/off} x {state on/off}, and a fixed-size state *replacing* a
linearly-growing one is the quotable result. It also sharpens what the state is
for: the trajectory memory already keeps each evicted frame's camera, register
and scale tokens, and throws away the 999 **patch** tokens. The evicted patch
geometry is precisely what the state uniquely holds.

## Why the write ends in a LayerNorm

Each pre-norm block adds `f(LN(s))`, whose magnitude is independent of `||s||`, so
those residuals accumulate over a clip. Measured with no training at all on a
128-frame clip: **`||S||` went from 14.5 to 40,206 (x2775)**. With a LayerNorm on
the state it is flat -- `724.1..724.1` across 105 writes (x1.000), where
`sqrt(512*1024) = 724` is the normaliser's fixed point.

The state init is therefore `std=1.0`, not a small init: the first write would
otherwise move the state onto a different scale from every later step, and the
read would see two regimes.

This was found by V7 before any GPU time, which is what that check is for.

## Sizing

Parameters at D=1024, 2 layers each (N barely matters):

| | params |
|---|---|
| write transformer | 33.6 M |
| read transformer | 25.2 M |
| state init (N=512) | 0.5 M |
| raymap encoder | 1.2 M |
| **total trainable** | **~61 M** |

Against a frozen 1.3 B backbone -- a genuine adapter. (The earlier D=2048,
4-state design was 953 M.)

Compute per frame, forward+backward: write ~258 GFLOPs, read ~177 GFLOPs at
N=1024 (less at 512), and the **DPT head ~1.5 TFLOPs -- 77% of training cost**.
That asymmetry is the lever behind "unroll long, supervise sparsely": put 32-64
write/read steps in the graph but evaluate the heads every 4th step, for 4-8x
longer write gradient at the same activation budget.

## Interface constraints (measured, not assumed)

- Tap entries are `cat([frame_intermediates, global_intermediates], -1)`
  (`aggregator/base.py:603`), 2048-d; the 1024-d halves are the token stream
  before and after the group's global block. `embed_dim=1024`, `num_heads=16`.
- The DPT head slices at `patch_start_idx=6` (`heads/dpt_head.py:206`) then
  permutes to `[2048, patch_h, patch_w]`, so tokens must be **row-major** with
  `P_patch = (H/14)(W/14)`. Never hardcode the count.
- `DPTHead.norm` is `nn.LayerNorm(2048)` applied to head input
  (`heads/dpt_head.py:66`, used at `:217`), so the refined tokens' scale and mean
  are absorbed downstream.
- **Depth activation is `exp`, not the `DPTHead` default `inv_log`.**
  `GCTBase._build_depth_head` (`models/gct_base.py:106`) passes
  `output_dim=2, activation="exp"`; `inv_log` belongs to the *point* heads. So
  depth is strictly positive and the pre-activation inverse is `log(depth)`.
  An earlier revision of this document claimed depth could be negative and that
  `log(depth)` was undefined -- wrong. Build the frozen heads only through
  `memory/frozen.py`, since `DPTHead`'s own defaults load the wrong shape and
  fail loudly, while the wrong *activation* would fail silently.
- `conf_activation="expp1"`: `conf = 1 + exp(x)`, >= 1 and unbounded. Normalise
  before using as a weight.
- Preprocessing follows the **benchmark**, not `load_fn.py`: resize width to 518,
  floor height to a multiple of 14, then the `area_budget=255000` cap with
  `align=14` (`benchmark/datasets/general.py:146`,
  `benchmark/benchmark/geometry/resize.py:79`). For 4:3 the cap is a no-op ->
  **518x378 -> 37x27 = 999 patches**, +6 specials = 1005 tokens, matching the
  paper's stated 518x378. It only bites near-square aspect ratios.
- The whole stack is resolution-agnostic (DINOv2 `interpolate_pos_encoding`,
  runtime RoPE grid, DPT head deriving `patch_h/patch_w` from `images.shape`), and
  the state has no P dimension -- so write and read resolutions are independent.
- `pred_normalization` defaults False and must stay off, or cached labels stop
  matching a per-frame decode.
- The model's output scale is **canonical**: paper §3.2 normalises GT by
  `s = mean ||x||_2` over the anchor-frame point cloud. GT depth and translations
  must be divided by the same `s`, computed from GT on the anchor frames. Do not
  add per-clip rescaling on top.

## Depth has no long-horizon headroom -- measured

`median(predicted_depth / GT_depth)` against frame index, 128 frames at stride 20:

| | |
|---|---|
| slope | +9.2e-05/frame = **+1.16% over the clip** |
| first 16 vs last 16 | 1.0071 -> 1.0224 (**+1.5%**) |
| median abs relative error | 2.72% |
| after removing per-frame scale | **1.81%**, and it *falls* 1.89% -> 1.01% |

Flat. Scale is stable to ~1.5% and the residual error is already ~2%, at the
level of iPhone LiDAR's own noise. **Depth is therefore a consistency metric, not
a headline claim** -- which is also why arm B (pose only) is worth running even
though arm A restores the depth path.

## Trajectory error: NOT yet measurable -- pipeline unvalidated

Raw numbers looked dramatic (RPE-rot 3.65 -> 7.60 deg, ATE RMSE 111% of
trajectory spread) but they are **not trustworthy and not drift**:

- error is already 3.65 deg in the first third against a **median inter-frame
  rotation of 5.61 deg** -- wrong from frame 0, not accumulating;
- GT and predicted motion statistics match almost exactly (median 5.61 vs 5.60,
  p90 11.05 vs 11.03, max 16.09 vs 16.23), while the per-frame rotations are
  nearly uncorrelated. Right magnitudes with wrong correspondence.

Ruled out: all 8 sign-flip conventions left and right, w2c vs c2w (best 4.91 deg,
still 87% of the motion); temporal offset between RGB and pose json (+-60 frames,
no minimum); a fixed change-of-basis (best is 1.5 deg from identity and reduces
nothing). Frame-index misalignment is ruled out too, since depth agrees to 2.7%
and is indexed by the same ids.

So either the model's pose is genuinely poor on this clip, or the GT pose
transform is still wrong. **Do not quote trajectory numbers until this is settled
on a benchmark dataset with trusted GT** -- see the plan's open items. One
concrete suspect: this clip starts at source frame 0, and the opening seconds of
a handheld scan make poor anchor frames.

## Measured on the first real cache build (scene 00777c41d4, 96 frames, stride 10)

| | value | consequence |
|---|---|---|
| inference throughput | 6.8-9.5 frames/s on one H200, degrading as the cache fills | a 320-frame clip is ~90 s; the whole dev set ~2 h |
| cache size | **17.2 MB/frame** | matches the estimate exactly |
| `tap_absmax` | **451** | fp16 storage is lossless with a 145x margin on fp16's 65504 range (fp16 has 10 mantissa bits vs bfloat16's 8, so the only risk was range) |
| camera origin norm | p50 **0.48**, p100 **0.97** | **sub-unit.** An earlier revision guessed "a corridor reaches ~20 canonical units" -- wrong for a room scan. The anchor normalisation keeps origins O(1), so the raymap sinusoidal band must target roughly [1e-3, 1], not [0, 20]. Re-measure on a large outdoor scene before assuming it generalises. |
| depth | p1 0.30, p50 0.67, p99 1.58 | O(1) in canonical units, consistent with the above |
| conf | p1 **1.7**, p50 **14.8**, p99 **30.6** | an 18x spread, so `conf` is unusable as a raw loss weight -- normalise per frame and clip |

Harmless log noise to expect: DINOv2's `pretrained_path` is empty so its own
weight load fails, but the full lingbot-map checkpoint then loads with
`missing=0 unexpected=0` and supplies `patch_embed` itself.

## The camera bridge -- how the pose terms get a gradient without a second recurrence

The camera head consumes `aggregated_tokens_list[-1][:, :, 0]` -- token 0 of tap
23, exactly what the read refines -- and carries its own causal KV cache. If that
cache saw refined tokens it would become a second recurrence interleaved with the
state's, with its own BPTT path.

`memory/camera_bridge.py` avoids that: **the cache holds the teacher's values for
every frame < i, and only frame i's token is refined.** Gradient reaches the read
through frame *i* alone and nothing recurs.

Two properties of the frozen head make this exact and nearly free:

- its cache **never evicts** -- the eviction guard is `shape[3] > 1` and the
  camera head has one token per frame, so it is always False. A single streaming
  pass therefore leaves a complete, in-order cache that can be sliced per step.
- appends **rebind** the dict entry via `torch.cat` (`layers/attention.py:285`)
  rather than mutating in place, so per-step views into the teacher cache are safe
  to reuse.

Verified: feeding the teacher's own token back through the bridge reproduces the
teacher's pose with **0.0 max abs error**; gradient reaches the token and no
frozen parameter; the teacher cache is unchanged after a step. Cost is ~1.6
GFLOPs/step against the DPT head's ~1.5 TFLOPs, so the pose terms are close to
free.

**Honest limitation.** This is a *first-order* approximation: frames < i use
teacher tokens, so it does not model the compounding effect of refining every
frame's pose along the trajectory. The fully-refined cache -- a genuine second
recurrence -- stays future work. It also means `L_rel_pose` over the window
compares one refined pose against 63 teacher poses, which is the right thing for
this arm but is not the deployed regime.

Cache rebuild needs no extra disk: the teacher camera tokens are already
`taps[:, 3, 0, :]`. The cache is 4 iterations x 4 trunk blocks x {k,v} of
`[B, 16, L, 1, 128]`, ~67 MB for a 320-frame clip, rebuilt once per clip pass.

## Open

## Reference points from CUT3R (arXiv 2501.12387)

Closest prior work; the read/write shape is deliberately similar.

| | CUT3R | here |
|---|---|---|
| state | 768 x 768 | 512 x 1024 |
| interaction | two interconnected decoders, cross-attention **every block**, bidirectional | single-shot parallel read/write on the old state |
| other memory | **none** -- the state is the only memory | a 72-frame KV cache plus the trajectory memory |
| past-view query | raymap -> 2-block encoder -> shared decoder, no state update | same |
| training length | <= 64 views | 320-frame clips |
| stage 4 | freeze encoder, train decoders **and heads** | heads stay frozen (late unfreeze is an ablation) |
| optimizer | AdamW 1e-4, linear warmup + cosine | same |

CUT3R's interleaved read/write is more expressive. If the parallel version
underperforms, "make it interleaved" is a known-good escalation rather than a
guess. CUT3R having no KV cache is also why it can train on 64 views while we
cannot: at 64 frames our eviction threshold of 72 is never reached and the state
would receive nothing.
