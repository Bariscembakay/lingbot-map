# Spatial memory — design record

Status: design settled for the first experiment ladder; implementation not
started. Every entry is revisitable; "decided" means "the current default we
build against". Where a number is quoted it was measured, and the measurement is
named.

**Revision 2026-08-25 — the architecture is now CUT3R fed by lingbot-map.**
DINOv2 + the aggregator (frozen, heads removed, KV cache kept) replace CUT3R's
ViT-L encoder. The aggregator's tokens drive CUT3R's two interconnected decoders
and its persistent state; new heads decode the readout. The one departure from
CUT3R is the objective: raymap queries address **past cameras that were actually
seen**, never novel views. The goal is faithful recall, not plausible completion.

Superseded and deleted from this document (recoverable at `fe7f781`): the
read/write-pair-between-aggregator-and-heads architecture, Loss 1 (lingbot's
composite depth+pose loss on the current frame), the zero-init read gate, the
token-space recall loss, the 8-stream vocabulary, the disjoint/overlap schedule,
and the camera bridge. Each existed to keep a *frozen lingbot head* in the graph;
none has a reason to exist now.

---

## The claim

A fixed-size state, written once per frame from a strong geometric encoder, can
be **read back at an old camera pose and asked what was there** — and it answers
faithfully, where CUT3R hallucinates.

That is the whole project. Two properties follow, and they are not the same
thing:

1. **Write it in.** At the moment frame *q* is processed, its content must enter
   the state.
2. **Don't overwrite it.** At every subsequent frame, the write must not destroy
   it.

A recall probe supervises **both at once**, which is why this objective works
where our earlier ones did not. See "Why full BPTT is a requirement".

---

## Architecture

```
  ┌─ FROZEN (lingbot-map, no heads, cached to disk) ─────────────────────────────┐
  │                                                                              │
  │  I_t ─► DINOv2 patch embed (518 wide, patch 14)                              │
  │            │                                                                 │
  │            ▼                                                                 │
  │      Aggregator: 24 × [frame-attn ; global-attn]   ◄──► KV cache             │
  │      embed_dim 1024, 16 heads                           8 scale/anchor frames│
  │            │                                            + 64 sliding window  │
  │            ▼  taps at layers 4, 11, 17, 23, each cat[frame,global] = 2048-d   │
  │      token layout [cam, reg×4, scale, patch×999]   patch_start_idx = 6        │
  └────────────┬─────────────────────────────────────────────────────────────────┘
               │   tap 23 only (default)  →  [B, 1, 1005, 2048]
               ▼
        ╔══ TRAINED ════════════════════════════════════════════════════════════╗
        ║   in_proj : Linear(2048 → 768)                                        ║
        ║            │                                                          ║
        ║      F_t (768-d, 1005 tokens)               s_{t-1}  (768 × 768)      ║
        ║            │                                    │                     ║
        ║   ┌────────┴────────────────────────────────────┴─────────────────┐   ║
        ║   │  interconnected decoders, 12 blocks; both read layer ℓ−1      │   ║
        ║   │     f_state^ℓ = blk_state^ℓ(x=state, y=img)   ← WRITE          │   ║
        ║   │     f_img^ℓ   = blk_img^ℓ  (x=img,   y=state) ← READ           │   ║
        ║   └────────┬───────────────────────────────────────┬──────────────┘   ║
        ║            │                                       │                  ║
        ║      F'_t  (discarded — see below)          s_t = dec_norm_state(·)   ║
        ╚═══════════════════════════════════════════════════════════════════════╝

  ── the probe: this is where the entire training signal comes from ─────────────

   for each frame t, sample 4 past cameras q < t (uniform)

   raymap R_q  (6-ch: origin, unit direction; from GT intrinsics + GT pose)
        │
        ▼
   [ probe_pose_token ; Encoder_r(R_q) ]      Encoder_r = 2 blocks, from scratch
        │
        ▼
   SAME decoders, SAME weights, s_t as memory   ── READ ONLY, state not updated ──
        │
        ▼
   dpt_self  → X_self(q), C_self       supervised against frame q's GT
   dpt_cross → X_world(q), C_world     (pose-modulated by the probe pose token)
```

**Why the probe is a clean measurement of the state, and nothing else.** The
raymap path never touches the aggregator. Its only route to frame *q*'s content
is `s_t`. Not DINOv2, not the KV cache, not the current image. Whatever the probe
answers correctly, the state was holding.

**`F'_t` is computed and discarded.** With no current-frame loss the image-side
readout is unused — but it cannot be skipped, because `blk_state` cross-attends
to `f_img` at every layer, so the write depends on it. What we *do* skip is the
heads on write passes, which is where the compute is.

---

## The objective

**Recall only.** No current-frame prediction, no pose loss, no RGB loss.

```
L = sum_t sum_{q in Q(t)}  [ conf-weighted regression on X_self(q)
                           + conf-weighted regression on X_world(q) ]
```

with CUT3R's confidence form, `c * ||xhat/shat - x/s|| - alpha * log c`, and
**alpha = 0.2** — the DUSt3R -> MASt3R -> CUT3R lineage constant, unchanged.

| | |
|---|---|
| probes per frame | **4** past, uniform over `q in [0, t-1]`, **plus 1 at the current camera `q = t`** (default on -- see axis) |
| at small *t* | `min(4, t)` distinct *q*; no probe at `t = 0` |
| probes per 160-frame clip | 634 past + 160 current = **794** |
| targets | **both** `X_self` (camera *q*'s own frame) and `X_world` (camera 0's frame) |

The raymap gives ray *direction* but not *distance*, so the probe is not
degenerate: the pose is handed over, the geometry is not.

### The lag-0 probe is not the current-frame loss we dropped

Read the mechanism, not the name. The dropped loss queried the state with the
frame's **image tokens**, which the aggregator's KV cache already covers -- so
the cheapest way to reduce it was to route around the state, which is the
suppression we measured. The lag-0 probe queries with a **raymap only**: no
image, no aggregator, so the sole path to frame *t*'s content is `s_t`. Same
clean-probe property as every other query.

It must read `s_t`, **after** frame *t* is written. Reading `s_{t-1}` would make
it prediction of an unseen view -- CUT3R's regime, and the thing we depart from.

What it buys: the lag>0 probes credit the write at step *q* through `t - q`
recurrence steps, so they supervise "write it in" and "don't overwrite it" only
jointly. Lag 0 credits the write through **one** step and isolates the first.

It also fills a real hole -- with past-only probes the loss at `t = 0` is
*empty* (there is no `q < 0`) and `t = 1` admits a single query, so the opening
of every clip is thinly supervised. The narrower claim is the correct one:
`write(frame 0)` was never *uncredited*, since later probes sampling `q = 0`
reach it through the recurrence; what was missing is a loss term at step 0.

Pose is GT, as for every probe -- the camera head is dropped, so there is no
predicted pose to use.

### Why the current-frame loss is dropped

We are not trying to beat frozen lingbot-map on the current frame and probably
cannot. Beyond that, the current-frame loss is actively suspect as a training
signal for the write: consecutive frames overlap heavily and the aggregator's KV
cache already covers the last 64, so the cheapest way to reduce it is to route
around the state entirely. That is not speculation — it is the suppression
measured in the earlier smoke arms, where both the write and the read
independently learned to attenuate the state path.

**Consequence, accepted deliberately:** the trained model answers raymap queries
only. It cannot produce a current-frame pointmap, and cannot be placed on
standard depth benchmarks. The evaluation protocol *is* the recall protocol.

Keeping it at low weight remains a live A/B — its one real merit is that it is
the only thing that ever shows the heads real image tokens.

---

## Why full BPTT is a requirement, not a convenience

This is the most important consequence of the objective, and it is where CUT3R's
training regime is structurally unable to follow.

A probe at time *t* of camera *q* reads `s_t`. Unrolling:

```
s_t = W(s_{t-1}, F_t) = W(W(s_{t-2}, F_{t-1}), F_t) = ...
                                      ... = W(... W(s_{q-1}, F_q) ...)
                                                    ^
                                    the write that ingested frame q
```

For the loss to teach "put frame *q* into the state", its gradient must traverse
**t - q recurrence steps**. With 4 uniform probes per frame the expected path
length is ~t/2 — on a 160-frame clip, tens of steps routinely.

CUT3R's `loss_of_one_batch_tbptt` does the opposite on both counts:

- `state_feat = state_feat.detach()` at every chunk boundary, `chunk_size=4`;
- every chunk except the last four runs under `torch.no_grad()`.

So a CUT3R write is never credited more than 4 frames past the read that used it,
and on a 64-view sequence frames 0–47 receive no gradient at all. Even given our
objective, that regime could not train recall.

**Therefore: no detach anywhere inside a clip.** Gradient checkpointing on the
decoder blocks (CUT3R already supports it) buys the memory back.

And the payoff: one loss supervises both halves of retention. The gradient at
step *q* teaches **write it in**; the gradient at steps *q+1 ... t* teaches
**don't overwrite it**. Our earlier designs needed schedule tricks to reach the
second property and never reached it cleanly.

---

## Decided

| | choice | why |
|---|---|---|
| encoder | lingbot-map DINOv2 + aggregator, **frozen**, heads removed, KV cache kept | the point of the project; frozen => cacheable |
| decoder input `F_t` | **tap 23 only**, `Linear(2048->768)` | see the sweep axis below |
| special tokens | all six pass through (camera, 4 registers, scale) | the scale token carries metric scale |
| state | CUT3R's, 768 tokens × 768-d, `state_pe='2d'` | inherit unchanged |
| state prior `s0` | CUT3R's learned initial state, **trainable throughout** | it gets trained in the end regardless; freezing it for the small-N rungs would only postpone the question. Verify at step 1 that gradient actually reaches it, then leave it alone -- the no-write control is what catches memorisation, not the freeze. |
| decoder init | **load CUT3R `cut3r_512_dpt_4_64.pth` decoders**, random-init heads | the head's `act_postprocess` assumes patch 16; ours is patch 14 |
| raymap encoder init | **load `enc_blocks_ray_map` + `enc_norm_ray_map` (25.2 M pretrained)**; random-init `patch_embed_ray_map` only | corrected 2026-08-26 after inspecting the checkpoint -- the design said "from scratch", which was wrong. The two blocks are patch-agnostic; only the 6-channel patch embed is patch-16-shaped (6*16*16 vs 6*14*14). |
| heads | CUT3R `dpt_self` + `dpt_cross`, trained from scratch | escapes the frozen-head constraints entirely |
| head patch size | `patch_size=14` to the head constructor, **plus one terminal `interpolate` to (H, W)** | croco's DPT upsamples by a hardcoded 16x; 14 is unreachable with powers of two. See below. |
| head conditioning | `dpt_self` unconditioned; `dpt_cross`'s **deepest tap only** modulated by the mod token | CUT3R's own arrangement, unchanged |
| mod token | `proj(mean(Encoder_r(R_q)))`, prepended to the query tokens, refined by the decoder, taken as `dec[-1][:, 0]` | CUT3R's structure minus `LocalMemory`; derived from the raymap alone, so leak-free |
| camera head | **dropped for this milestone** | under a probe-only objective the raymap already encodes the pose — the task is degenerate |
| `pose_retriever` (LocalMemory) | **dropped** | its job is ego-motion context across frames; the aggregator's KV cache does it far better |
| read gate | **none** | zero-init existed for attributability against frozen lingbot heads; with Loss 1 gone there is no baseline to stay attributable to, and CUT3R runs its state path live from step 0 |
| clip | 320-frame cached clip **subsampled by 2 -> 160 frames at effective stride 40** | wide baseline, more of the room per frame |
| BPTT | **full, over the whole clip**, gradient checkpointing on | required by the objective, above |
| grad clip | 1.0 | CUT3R's |
| raymap convention | `inv(c2w_0) @ c2w_q`, 6 channels, **built exactly as CUT3R's `get_ray_map`** -- direction channel included, which is `normalize(R*d_cam + t)` rather than `normalize(R*d_cam)` | it is the distribution the released raymap encoder was trained on. Arm A must use it or E1 understates CUT3R; arm C matches so A/B/C stay on one footing and the inherited 25 M raymap-encoder weights stay in-distribution. Open question and the numbers are in `AGENTS.md`; `--raymap-convention {cut3r,true}` is a sweep axis. |
| RoPE for non-patch tokens | **all of them at the same position, (-1, -1)**; patches at their grid coordinates | CUT3R's choice, so the loaded decoder weights stay in distribution. lingbot's `zeros` would alias every special onto patch (0,0). Distinct positions per special (-1,-2,...) were rejected: they put the loaded weights at RoPE phases never seen in training, and the six are already separated by content. |
| probe token block | **`[mod_token ; raymap patches x999]`** -- no register or scale analogues | registers exist to park global information and the state already is that place; scale is carried by the raymap's ray origins, which are in world coordinates and canonical units (p50 0.48). The write is expected to encode scale into the state. |

### Vocabulary

Two different things nearly share a name; keep them apart.

| term | what it is |
|---|---|
| **state prior**, `s0` | CUT3R's `register_tokens` — `nn.Embedding(768, 1024)` -> `decoder_embed_state`, one 768×768 tensor shared by all scenes. Badly named in their code: nothing per-frame, nothing register-like. |
| **register tokens** | lingbot's `register_token`, `[1, 2, 4, 1024]` — 4 DINOv2-style tokens **per frame** at indices 1–4 of every tap |

Taps are named by **aggregator layer** — tap 4, 11, 17, 23 — never by index 0–3.

---

## Sweep axes

### Which aggregator taps feed the decoders — default (a), (b) to be tried

| | `F_t` | input projection |
|---|---|---|
| **(a) default** | tap 23 only | `Linear(2048 -> 768)` |
| **(b)** | all four, concatenated on channels | `Linear(8192 -> 768)` |

Token-axis concatenation (4×1005 tokens) was considered and rejected: 4× decoder
FLOPs and activations for the same information.

This is **not** the DPT head's tap question. The head reads
`dec[0], dec[6], dec[9], dec[12]` — four taps manufactured inside the 12-block
decoder stack — under either choice. The two questions touch only at tap 0, which
`_decoder` seeds *before* `decoder_embed`. **Seed it with the projected 768-d
tokens**, so tap 0 is uniform with the rest and the head gets no state-free path
at full aggregator width.

Why (a) is the default:

- **CUT3R is the existence proof.** Its decoder receives exactly one stream — the
  ViT-L's last layer — and builds the DPT hierarchy internally over 12 blocks.
  Multi-scale in DPT is about *where fusion happens*, not a claim that the last
  layer lacks the information.
- **The cache shrinks 4×**: 4.3 MB/frame against 17.2. A 320-frame clip goes
  5.5 GB -> 1.4 GB; 276 ScanNet++ scenes 1.5 TB -> 380 GB; and I/O per update
  drops 4×, which is what caps batch size.
- Tap 23 is the **deepest** tap, therefore the most global and least spatially
  resolved. A room-scale summary is exactly the global end.

The reservation that keeps (b) alive:

- Tap 23 is demonstrably excellent for **pose** — the only tap lingbot's camera
  head reads, and that head reproduces the paper.
- There is **no evidence it suffices for depth**, and one piece against:
  `d(depth)/d(tap23) == 0` exactly in the published checkpoint, so lingbot's own
  DPT head takes its depth from taps 4/11/17. That is an artifact of the trained
  head's weights (`layer4_rn` output all-negative, annihilated by
  `ReLU(inplace)` through the residual), not proof the tap is depth-free — but
  (a) bets the depth ceiling on an untested assumption, and a coarse-but-blurry
  state would be indistinguishable from a failing memory.

Settle it by measurement, at no extra cost: the existing four-tap cache already
contains tap 23. Train a small DPT probe head on tap 23 alone and compare depth
error against the same head on all four taps. (a) and (b) differ only in the
input projection, so switching is one line **as long as the cache carries four
taps** — keep four taps through the one-scene run, and decide before building the
276-scene cache, which is the only point where the storage difference costs
anything.

### Probe at the current camera (lag 0) -- default ON

`--probe-current {on,off}`. One extra probe at `q = t` against `s_t`, alongside
the four sampled past cameras.

**What the axis tests is echo.** Frame *t* has just been written, so lag 0 is the
easiest query the model will ever face: it could be satisfied by a recency buffer
that copies the frame into scratch state and reads it straight back, learning
nothing about compression or retention. Two things bound the risk -- an easy term
drives its own loss toward zero and stops contributing gradient, and the lag>0
probes demand retention regardless. But state capacity spent on a buffer is
capacity not spent on the map, so measure it rather than assume.

Cost: 954 decoder passes per clip against 794, about 20% more.

### Deferred to later axes

- **KV window shrinking.** Reducing `kv_cache_sliding_window` from 64 makes the
  state load-bearing for current-frame prediction too, but runs the aggregator off
  its trained operating point and requires a cache rebuild. Not now.
- Current-frame loss at low weight (A/B).
- Probe density: 4-per-frame vs 4-every-8th-frame, once the mechanism is proven.
- Unfreezing the aggregator.

---

## Experiment ladder

The aim is two results, in order.

**E1 — CUT3R hallucinates on old frames.** Query the published CUT3R checkpoint
at the raymap of a past camera and show the answer is a plausible completion, not
what was there.

E1 needs **no training**. CUT3R's `inference_step` already takes a view with
`ray_mask` set, encodes the raymap, reads the state, runs the heads, and returns
without touching the state. So E1 is an evaluation of a released checkpoint, and
it debugs our probe harness against a model known to work **before any of our own
weights exist**. Run it first.

Its outcome is close to predictable once you look at how CUT3R trains raymaps.
When frame *v* is replaced by its raymap, the query sits at **position *v* in the
sequence** — the state holds frames 0...*v*-1 and frame *v* was never written
(`img_mask=False` => no state update). CUT3R is trained to predict a view it has
**never seen**, from its camera alone. It is trained to *complete* and **never
once trained to recall**; nothing in its objective rewards preserving a frame it
did write.

**E2 — lingbot-map + CUT3R recalls.** Same protocol, our model.

### The arms, because E1 vs E2 alone proves the wrong thing

"Ours is better at recall" has three candidate causes: the recall objective, the
stronger geometric encoder, and the architecture that follows from it. The cheap
explanation a reader reaches for first is the objective.

| arm | encoder | objective |
|---|---|---|
| **A** CUT3R published | DUSt3R ViT-L | theirs (completion) |
| **B** CUT3R finetuned | DUSt3R ViT-L, frozen | **ours (recall)** |
| **C** ours | lingbot aggregator, frozen | ours (recall) |

A->B isolates the objective. **B->C isolates the encoder, which is the actual
thesis.**

**Both arms must use the same head.** Arm B runs at patch 16 and arm C at patch
14; if B used croco's head and C used lingbot's, B->C would differ in encoder
*and* head and the comparison would be confounded again. croco's head plus the
terminal resize is one implementation for both -- in arm B the resize is exactly
a no-op, since 16*N == H by construction.

B is not a new project — encoder frozen, decoders + heads trained, same
data, same loss, same cost as C. Treat it as required.

### CUT3R is out of distribution past 64 frames

It has no KV cache and its longest training was 64 views, so at 160 frames arm A
runs well past where it was ever trained. Report the probe error **as a curve over
clip length**, with an explicit point at <= 64 frames where A is in distribution.
If we win there too the claim is safe; if we only win past 64, the honest
statement is "the advantage appears where CUT3R's context ends" — still a real
result, arguably the more interesting one.

### Scale ladder for arm C

1. **One scene** — implementation and gradient flow only. Not a result.
2. **A few scenes** — first honest signal.
3. **All scenes.**

### Controls, which are not optional

With one training scene, `s0` is ~590 k parameters *identical for every scene*, so
gradient descent will happily bake that scene into it — and the probe then
succeeds with the update path completely dead. "The state holds the room" and
"the state prior was trained to be this room" produce the same loss curve.

| control | what it answers |
|---|---|
| **no-write**: `s_t == s0`, everything else identical | is the scene in the *writes* or in `s0`'s weights? **This is the decisive one.** |
| **zero-state**: `s_t = 0` at read time | is the read path used at all? (removes `s0` too, so it cannot separate the two hypotheses) |
| **lag sweep**: probe error vs `t - q` | does retention decay, and where? |

`s0` stays **trainable at every rung**. Freezing it would remove one memorisation
route, but not the one that matters: with a single scene the *decoder weights*
can encode the scene just as easily -- the write can learn "whatever you see,
emit state X". So the freeze buys interpretation, not proof, and the proof has to
come from the controls either way.

Hence **no-write is mandatory at every rung**, not only the first. Its signature
when something has gone wrong is a lag sweep that is flat *and* good: a genuine
memory decays with `t - q`, a memorised scene does not.

The one-scene milestone is structurally incapable of settling this on its own.
The real fix is more than one scene, since `s0` cannot be all of them at once.

---

## Sizing and cost

Token geometry, measured: preprocessing follows the **benchmark**, not
`load_fn.py` — resize width to 518, floor height to a multiple of 14, then the
`area_budget=255000` cap with `align=14`. For 4:3 the cap is a no-op =>
**518×378 -> 37×27 = 999 patches**, +6 specials = **1005 tokens**. The raymap is
at the same resolution => 999 patches + 1 probe pose token = **1000 tokens**.

Decoder passes per 160-frame clip: 160 writes + 794 probes = **954**.

| | |
|---|---|
| activations per decoder block per pass, bf16 | ~38 MB `((1005+768) tokens × ~14×768 floats)` |
| × 12 blocks | ~**0.46 GB per pass** |
| 954 passes, no checkpointing | ~439 GB — **impossible** |
| 954 passes, **gradient checkpointing** | ~**31–48 GB** — fits an H200 at B=1 |

Heads run on probes only (794), never on write passes. In the earlier
architecture the DPT head was ~77% of training cost, so this matters.

These are estimates from token counts. **Measure peak VRAM on the first run
before fixing the clip length.**

---

## Interface constraints (measured, not assumed)

- Tap entries are `cat([frame_intermediates, global_intermediates], -1)`
  (`aggregator/base.py:603`), 2048-d; the 1024-d halves are the token stream
  before and after the group's global block. `embed_dim=1024`, `num_heads=16`.
- Token layout is `[camera, register×4, scale, patch...]`, `patch_start_idx = 6`
  (`aggregator/stream.py:203`). Patch tokens are **row-major** with
  `P = (H/14)(W/14)`; never hardcode the count.
- Taps are collected at `selected_idx=[4, 11, 17, 23]`
  (`models/gct_stream_window.py:318`).
- KV cache defaults: `kv_cache_sliding_window=64`, `kv_cache_scale_frames=8`
  (`models/gct_stream_window.py:160`).
- The stack is resolution-agnostic (DINOv2 `interpolate_pos_encoding`, runtime
  RoPE grid, DPT head deriving `patch_h/patch_w` from `images.shape`), and the
  state has no `P` dimension — so write and probe resolutions are independent.
- Output scale is **canonical**: paper section 3.2 normalises GT by
  `s = mean ||x||_2` over the anchor-frame point cloud. GT depth and translations
  must be divided by the same `s`, computed from GT on the anchor frames. Do not
  add per-clip rescaling on top.
- `pred_normalization` defaults False and must stay off, or cached labels stop
  matching a per-frame decode.
- Poses are **camera-to-world**. The repo contradicts itself in one dead function
  (`models/gct_base.py:268`, `_unproject_depth_to_world`, never called);
  `benchmark/methods/lingbot_map.py:225` and `models/gct_stream_window.py:82-93`
  both use c2w, and the benchmark reproduces the paper.

### CUT3R side

- `_decoder` seeds its tap list **before** `decoder_embed`, hence
  `dim_tokens = [enc_dim, dec_dim, dec_dim, dec_dim]`. We override tap 0 to the
  projected width (above).
- `DPTPts3dPose.forward` splits `pose_token = x[-1][:, 0]`, `token = x[-1][:, 1:]`.
  The strip count must equal the number of leading non-patch tokens: **6** on the
  write path (`patch_start_idx`), **1** on the probe path. A wrong *count* fails
  loudly -- `rearrange` is given `nh` and `nw` explicitly, so it checks
  `n_tokens == nh*nw` and raises. Only a wrong *offset* with the right count
  would shear the image silently.
- **croco's DPT upsamples by a hardcoded 16x** and has no terminal resize. With
  the 27x37 token grid: `act_postprocess` builds the pyramid at 4x/2x/1x/(1/2)x,
  the four fusion blocks each `interpolate(scale_factor=2)`, and `head` adds one
  more -- 216x296 (8N) then 432x592 (16N). Target is 378x518. Every factor is a
  conv stride and therefore a power of two, so **14 = 2*7 is unreachable**;
  the original DPT paper also resizes at the end. Fix: keep the head's final x2
  and add `interpolate` down from 16N to (H, W) -- decimating real detail rather
  than inventing it. (Dropping the final x2 and upsampling 1.75x from 8N is
  literally lingbot's head, and accepts a blurrier ceiling; we don't have to.)
  Cost is ~28 GFLOP per head call, so ~0.6 s per optimizer step across 794
  probes and both heads including backward -- not a constraint. The earlier
  "DPT head is 77% of training cost" figure was lingbot's 2048-d head.
- The state is LayerNormed every frame before propagating (`dec_norm_state`
  applied to `final_output[-1][0]`, then `new_state_feat = new_state_feat[-1]`).
  Its scale is re-fixed each step and cannot drift over the clip.
- `dec_blocks_state` and `dec_blocks` are **separate parameters initialised as
  copies** of each other when a checkpoint lacks the former.
- Decoder blocks are pre-norm with residuals, so `d s_t / d s_{t-1}` is
  identity-plus-corrections rather than a raw weight matrix. Combined with the
  per-frame LayerNorm and `clip_grad=1.0`, a vanilla-RNN-style explosion is not
  expected. **Measured 2026-08-26** (`scripts/memory/validate_state.py`, tiny
  config: `dec_depth=2`, 64 state tokens, random init, rollout T=32, probe at
  t=32). Gradient into each frame's tap, by lag:

  | lag | 32 | 24 | 16 | 8 | 4 |
  |---|---|---|---|---|---|
  | max abs grad | 1.58e-6 | 1.69e-7 | 1.41e-7 | 1.15e-7 | 1.60e-7 |

  **Flat — no decay with lag**, and every step non-zero. Step 0 is the outlier
  because its write acts on `s0` directly: the first write perturbs an empty
  prior, later ones perturb an already-populated state. Indicative for 160
  frames, not conclusive -- the real model is 12 blocks and 768 state tokens and
  will be trained. Still log the state norm and per-frame gradient norm on the
  first real run.

---

## Measured facts carried forward

From the first real cache build (scene `00777c41d4`, 96 frames, stride 10):

| | value | consequence |
|---|---|---|
| inference throughput | 6.8–9.5 frames/s on one H200, degrading as the cache fills | a 320-frame clip ~90 s |
| cache size | **17.2 MB/frame** (four taps) | tap 23 alone => 4.3 MB/frame |
| `tap_absmax` | **451** | fp16 storage lossless with a 145× margin on fp16's 65504 range |
| camera origin norm | p50 **0.48**, p100 **0.97** | **sub-unit** — the raymap sinusoidal band must target roughly [1e-3, 1], not [0, 20]. Re-measure on a large outdoor scene before assuming this generalises. |
| depth | p1 0.30, p50 0.67, p99 1.58 | O(1) in canonical units |
| conf | p1 1.7, p50 **14.8**, p99 30.6 | an 18× spread — unusable as a raw loss weight |

Harmless log noise: DINOv2's `pretrained_path` is empty so its own weight load
fails, but the full lingbot-map checkpoint then loads with
`missing=0 unexpected=0` and supplies `patch_embed` itself.

### Revisit score — fraction of visible surface last seen > window frames ago

Three ScanNet++ scenes, 320-frame clips, from GT depth and poses:

| window | stride 10 | stride 20 | stride 30 |
|---|---|---|---|
| **64** (published) | 10–35% of frames over 10% stale | **33–43%** | 27–41% |
| **16** | 52–56% | **56–67%** | 54–66% |

Stride 20 beats stride 10 substantially and stride 30 adds nothing while
truncating shorter recordings — hence the cache is built at stride 20. The window
matters more than the stride, which is what makes window-shrinking a future axis
rather than a detail.

### Contamination

ScanNet++ is in lingbot-map's training data, so a ScanNet++ split is held out from
*us* but not from the *encoder*. Final evaluation must leave it: **NRGBD**
(primary — room-scale, real revisits, per-frame GT depth at `depth/depth{N}.png`)
and Oxford Spires sparse (`depth/000000.npy`, likely near-zero revisit since it is
a traverse).

---

## Resolved while designing

1. **The probe's mod token.** `dpt_cross` needs one conditioning vector for its
   adaLN (`ConditionModulationBlock` -> `ModLN(dim, dim)`), and `dpt_self` needs
   none. CUT3R sources it from `LocalMemory`; `inference_step` shows the raymap
   case explicitly -- `global_img_feat_i = mean(feat_i)` then
   `pose_retriever.inquire(...)`, prepended, refined, and read back as
   `dec[-1][:, 0:1]`. So the token never *supplied* pose information the raymap
   lacked; it is only the vector adaLN requires, and CUT3R derives it from the
   query itself. We keep that structure and drop `LocalMemory`:
   `proj(mean(Encoder_r(R_q)))`. Mean-pooling a raymap cannot reveal anything
   about frame *q*'s image, so it is leak-free by construction.
   - Note for interpretation, not a design change: for a raymap query the pose is
     *given*, so `X_world = c2w_q * X_self` exactly, and a rigid transform
     preserves distances -- the two losses are the same up to scale
     normalisation. CUT3R's "redundancy simplifies training" argument rests on
     pose being *unknown* for image inputs and is much weaker here. Both heads
     are kept anyway; their disagreement is a free consistency diagnostic.

2. **The DPT token slice.** Not a problem -- an offset. Strip
   `patch_start_idx` (6) on the write path, 1 on the probe path, and the
   remaining 999 tokens fold into 27x37 exactly. Our layouts are *more* uniform
   than CUT3R's, which juggles three different ones across its four taps.

3. **The head's upsample factor.** Real, and fixed by one terminal
   `interpolate`. Details under "CUT3R side" above. Patch 14 is a fixed
   constraint of the frozen encoder, not a parameter -- DINOv2 ViT-L/**14** is
   what lingbot is built on. The grid divides exactly (518/14 = 37,
   378/14 = 27); only the head's internal factor mismatched.

## Checkpoint contents (measured 2026-08-26)

`cut3r_512_dpt_4_64.pth`, sha256 `45f7e98a0a64dbeb54901ae2b878cd8cd125f20a4497316483f0bd6f109f8103`,
3.17 GB, 1248 tensors under a `module.` prefix, **803.3 M parameters**. Loading
it requires `omegaconf` -- `args` pickles an omegaconf config.

| module | params | for us |
|---|---|---|
| `enc_blocks` (ViT-L) | 302.3 M | **discarded** -- replaced by lingbot's aggregator |
| `dec_blocks` | 113.4 M | **loaded** |
| `dec_blocks_state` | 113.4 M | **loaded** (present in this checkpoint, not synthesised) |
| `pose_retriever` | **152.3 M** | **dropped** |
| `downstream_head` | 91.8 M | discarded -- ours is patch 14 and random-init |
| `enc_blocks_ray_map` | 25.2 M | **loaded** |
| `register_tokens.weight` | (768, 1024) | **loaded** -- this is `s0` |
| `decoder_embed_state.weight` | (768, 1024) | **loaded** |
| `pose_token` | (1, 1, 768) | not applicable -- our mod token is derived, not a bare parameter |

Two things this measurement changed:

1. **The raymap encoder is not from scratch.** 25.2 M of pretrained raymap
   encoding transfers; only the 6-channel patch embed is patch-16-shaped and
   must be random-init.
2. **`pose_retriever` is 19% of CUT3R's parameters**, not the minor module it
   reads as in their code. Dropping it -- replacing a 152 M learned ego-motion
   memory with the aggregator's KV cache, which we get for free -- makes our
   model materially smaller than CUT3R rather than marginally so.

Inherited pretrained weight totals **~254 M**; trainable total is **~321 M**
(decoders 226.8 + raymap blocks 25.2 + heads ~65 + projections/`s0` ~4).

---

## Cache format changes required (v3 -> v4)

The existing cache stores `taps, depth, conf, pose_enc, gt_depth, gt_c2w,
revisit`. The probe needs one thing it does not have.

**Add `gt_intrinsics.npy`, `[N, 3, 3]`, post-resize, per frame.** Bump
`FORMAT_VERSION` 3 -> 4. Cost is 36 bytes per frame. Everything else the probe
needs is then derivable:

```
R_q        = rays through K_q, posed by inv(c2w_0) @ c2w_q
X_self(q)  = unproject(gt_depth[q], K_q)
X_world(q) = c2w_q @ X_self(q)
```

Two rules that go with it:

- **GT intrinsics, not the teacher's.** `pose_enc` is `absT_quaR_FoV`, so
  intrinsics *are* recoverable via `pose_encoding_to_extri_intri` -- but those
  are the teacher's *predictions*. The raymap defines the question being asked
  and must be consistent with the GT pointmap it is scored against.
- **Derive GT pointmaps on the fly; never store them.** At 378x518x3 in fp16
  that is ~1.2 MB/frame, which would roughly double a tap-23-only cache
  (4.3 MB/frame) to save an unprojection that costs nothing.

Consistency note: `gt_depth` and `gt_c2w` translations are already divided by
the canonical `gt_scale` in `meta`, and `K` is in pixels, so `X_self` and
`X_world` land in canonical units without further rescaling. Do not add a
per-clip rescale on top.

---

*Design closed 2026-08-26. No open items. Next artifact is the implementation
plan.*
