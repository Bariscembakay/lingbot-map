# Inference pipeline — lingbot-map

Debugging reference. Pseudocode, real shapes, real line numbers.

Numbers below are for:

```
demo.py --model_path ckpt/lingbot-map.pt --image_folder example/university \
        --first_k 24 --num_scale_frames 4 --keyframe_interval 2 --use_sdpa
```

Source images are 518×294, so the patch grid is 37×21. Change the resolution and
every `783` / `777` below moves; nothing else does.

---

## 0. Constants for this run

| symbol | value | origin |
| --- | --- | --- |
| `H, W` | 294, 518 | `load_and_preprocess_images(mode="crop", image_size=518)` |
| patches | 21×37 = **777** | `H//14 × W//14` |
| `num_special_tokens` | **6** | camera(1) + register(4) + scale(1), `stream.py:177` |
| `P` (tokens/frame) | **783** | 6 + 777 |
| `patch_start_idx` | **6** | `stream.py:176` |
| `C` (embed_dim) | 1024 | |
| aggregator heads / head_dim | 16 / 64 | |
| `depth` | 24 | 24 frame blocks + 24 global blocks |
| `aa_block_size` / `aa_block_num` | 1 / 24 | `base.py:97` — **the inner loops run once** |
| `selected_idx` | `[4, 11, 17, 23]` | `gct_stream.py:299` |
| camera head dim / heads / head_dim | 2048 / 16 / 128 | 2048 = 2×C, see §5 |
| camera trunk depth / iterations | 4 / 4 | |
| `pose_enc` | 9 = T(3) + quat(4) + FoV(2) | |

Anchor frames = 4, sliding window = 64 (default), keyframe interval = 2.

---

## 1. Entry point

```
demo.main()
  images = load_images(...)                 # [24, 3, 294, 518], float in [0,1]
  model  = load_model(args, device)         # picks the class by --mode:
                                            #   streaming → models/gct_stream.GCTStream
                                            #   windowed  → models/gct_stream_window.GCTStream
                                            # (both named GCTStream; check type(model).__module__)
  model.aggregator.to(bfloat16)             # heads stay fp32, demo.py:461

  with autocast(bf16):
      preds = model.inference_streaming(images, num_scale_frames=4,
                                        keyframe_interval=2)
  postprocess(preds)                        # pose_enc → extrinsic/intrinsic
  PointCloudViewer(...).run()               # BLOCKS. stop at demo.py:581 instead.
```

---

## 2. Class wiring — control bounces between files

Neither subclass defines `forward`; it lives once in the abstract base and calls
back down through `@abstractmethod` hooks.

```
GCTBase (models/gct_base.py:25)  ── abstract, owns forward() at :287
   ├── GCTStream (models/gct_stream.py:96)          streaming
   └── GCTStream (models/gct_stream_window.py:115)  windowed

AggregatorBase (aggregator/base.py:60)  ── abstract, owns forward() at :539
   └── AggregatorStream (aggregator/stream.py:23)
```

```
gct_stream.py:445   self.forward(...)                    # NOT via nn.Module.__call__
 └─ gct_base.py:287     GCTBase.forward
      ├─ :316   _normalize_input               (base)
      ├─ :318   _aggregate_features            ABSTRACT → gct_stream.py:277
      │            └─ self.aggregator(...)     # via __call__, hooks DO fire
      │                 └─ base.py:539  AggregatorBase.forward
      │                      ├─ :566  _embed_images                     (base)
      │                      │     └─ :385 _prepare_special_tokens  ABSTRACT → stream.py:297
      │                      ├─ :583  _process_frame_attention          (base)
      │                      └─ :587  _process_global_attention     ABSTRACT → stream.py:370
      ├─ :327   _predict_camera                (base, drives CameraCausalHead)
      ├─ :337   _predict_depth                 (base, drives DPTHead)
      └─ :342   _predict_points                → {}  (no head in this ckpt)
```

**Construction order gotcha.** `GCTStream.__init__` sets ~20 attributes *before*
`super().__init__()` (`gct_stream.py:180-202`) because `GCTBase.__init__` calls
`self._build_aggregator()`, which reads them. A new param set after `super()` →
`AttributeError` at construction.

---

## 3. Phase 1 — anchor frames (`gct_stream.py:431-458`)

```
clean_kv_cache()                              # ONE reset for the whole sequence
                                              # (windowed resets PER WINDOW instead)

scale_images = images[:, :4]                  # [1, 4, 3, 294, 518]
scale_output = forward(scale_images,
                       num_frame_for_scale=4,
                       num_frame_per_block=4)

# returns, all with S=4:
#   pose_enc       [1, 4, 9]
#   pose_enc_list  list of 4 × [1, 4, 9]      ← DISCARDED by the caller
#   depth          [1, 4, 294, 518, 1]
#   depth_conf     [1, 4, 294, 518]
#   images         [1, 4, 3, 294, 518]        ← DISCARDED
# absent: world_points / world_points_conf (point_head not built, no ckpt weights)

all_pose_enc = [scale_output["pose_enc"]]     # ONE element with S=4
```

Because the cache is empty and `S=4`, attention here is **fully bidirectional
over 4×783 = 3,132 tokens**. This is the only place information flows backward
in time. It fixes the global scale and the world coordinate frame; nothing
downstream ever revises it.

---

## 4. Phase 2 — streaming (`gct_stream.py:470-497`)

```
for i in 4 .. 23:
    frame = images[:, i:i+1]                          # [1, 1, 3, 294, 518]
    is_keyframe = (i - 4) % keyframe_interval == 0

    if not is_keyframe: _set_skip_append(True)        # gct_stream.py:322
    out = forward(frame, num_frame_for_scale=4,       # keeps anchor-token logic right
                         num_frame_per_block=1)
    if not is_keyframe: _set_skip_append(False)

    all_pose_enc.append(out["pose_enc"])              # [1, 1, 9]

predictions["pose_enc"] = cat(all_pose_enc, dim=1)    # [1, 24, 9]
```

**`len(all_pose_enc) == 21`, not 24** — one fat chunk (S=4) plus 20 thin ones (S=1).
The `cat(..., dim=1)` is what reassembles the sequence.

### What actually switches bidirectional → causal

Not `num_frame_per_block`. It is **`images.shape[1]`**, which becomes `S_global`
→ `num_frames`, and that alone drives both backends' branch:

```
block.py:245       is_streaming   = kv_cache is not None and num_frames <= 1
attention.py:473   is_multi_frame = num_frames > 1
```

**Neither phase uses a causal mask.** `attention.py:677` even builds an explicit
all-ones one. Causality is purely a property of *what is in the cache*.

---

## 5. Inside `forward()` — the 72-block stack

Per frame, three stacks of 24 blocks each run:

```
AggregatorBase.forward (base.py:539)

  tokens = _embed_images(images)
      patch_tokens = patch_embed(images)          # DINOv2 ViT-L, 24 blocks
                                                  # [S, 777, 1024]
      special      = _prepare_special_tokens(...) # [S, 6, 1024]
      tokens       = cat([special, patch_tokens]) # [S, 783, 1024]
                                                  # layout: [cam, r0..r3, scale, p0..p776]

  for step in 0..23:                              # aa_block_num = 24
      tokens = frame_blocks[step](tokens)         # per-frame, NO cache, 2D RoPE
      tokens = global_blocks[step](tokens, kv_cache=...)   # CROSS-frame, cached

      if step in [4, 11, 17, 23]:
          out.append(cat([frame_out, global_out], dim=-1))  # [B, S, 783, 2048]

  return out, patch_start_idx=6                   # 4 tensors
```

That channel-axis concat is the origin of `dim_in = 2*embed_dim = 2048` in every
head. Heads see frame-local and cross-frame features side by side, never merged.

### Which attention class each stack uses

`layers/__init__.py:5` aliases `Attention as MemEffAttention`, so DINOv2 uses the
same class as `frame_blocks`.

| stack | blocks | class | QK-norm | RoPE | causal |
| --- | --- | --- | --- | --- | --- |
| `patch_embed.blocks` (DINOv2) | 24 | `Attention` | **no** (Identity) | no (learned `pos_embed`) | no |
| `frame_blocks` | 24 | `Attention` | yes | 2D | no |
| `global_blocks` | 24 | `FlashInferAttention` / `SDPAAttention` | yes | 2D or 3D | **yes** |
| `camera_head.trunk` | 4 | `CausalAttention` | **no** | **no** | yes |

Verified against the checkpoint: only `frame_blocks` and `global_blocks` carry
`q_norm`/`k_norm` tensors. A breakpoint in `Attention.forward` (`attention.py:65`)
fires **48× per frame**; condition on `self.rope is None` for DINOv2 only.

---

## 6. KV cache — SDPA backend (`--use_sdpa`)

One flat dict on the aggregator, shared by all 24 global blocks
(`stream.py:186-194`):

```
self.kv_cache = {
    "k_0" .. "k_23"                  [1, 16, F, 783, 64]   anchor + window frames
    "v_0" .. "v_23"                  same
    "k_0_special" .. "k_23_special"  [1, 16, E, 6,   64]   trajectory memory
    "v_0_special" .. "v_23_special"  same
    "_skip_append": bool                                   non-keyframe flag
}
```

`F` clamps at `scale + window`. `E` grows **without bound** — one entry per
evicted frame. `frame_blocks` have no cache at all.

```
SDPAAttention.forward (attention.py:595)

  q,k,v = qkv(x).reshape(...)                    # [1, 16, N, 64]
  q,k   = q_norm(q), k_norm(k)                   # :603 — BEFORE RoPE (order required)
  q,k   = rope(q), rope(k)                       # baked into K before caching

  # split N tokens into frames for the 5-D cache layout
  if cache non-empty:                            # :636
      num_frame_per_block = k.shape[2] // cache["k_i"].shape[3]
  k_reshaped = k.view(1, 16, num_frame_per_block, N//num_frame_per_block, 64)

  if not skip_append:                            # KEYFRAME
      cache["k_i"] = k_reshaped  or  cat(cache["k_i"], k_reshaped, dim=2)
      _apply_kv_cache_eviction(...)              # :687
      k_full = cache["k_i"].clone()              # full copy, every block, every frame
  else:                                          # NON-KEYFRAME
      k_full = cat(cache["k_i"], k_reshaped)     # attend, do NOT persist

  k_full = k_full.reshape(1, 16, F*783, 64)
  if cache["k_i_special"] is not None:           # :668
      k_full = cat([specials.reshape(1,16,E*6,64), k_full], dim=2)

  x = scaled_dot_product_attention(q, k_full, v_full, attn_mask=ones)
```

Visible context at frame N: `E*6 + F*783` keys. The `.clone()` at `:655` is why
SDPA is memory-heavy and slow — fine for 24 frames, unusable at 3,840.

---

## 7. KV cache — FlashInfer backend (default)

`self.kv_cache` stays `{}`; everything lives in `self.kv_cache_manager`
(`layers/flashinfer_cache.py`). Two page streams per block:

```
kv_caches[block]  tensor [max_num_pages, 2, page_size, 16, 64]   # dim1: 0=K, 1=V
page_size          = patches_per_frame = 777      (exact, fa3=False → no padding)
max_total_frames   = max_frame_num + 100 = 1124   (sizes the special pool)

# Pool sizing, flashinfer_cache.py:132-136.  NOTE: the `max_num_frames` arg
# passed from stream.py:214 is DEAD -- the manager recomputes this itself.
max_patch_pages    = scale + window + 16 = 84     # +16 = free-list headroom
max_special_pages  = ceil(1124*6/777) + 16 = 25   # +16 again
# 16 is a round safety constant, not a derived bound: append_frame allocates
# before evict_frames runs (needs +1), and _defer_eviction suppresses eviction
# during the non-keyframe rollback.  Cost: one page = 72.8 MiB across 24 blocks,
# so the two headrooms burn ~2.3 GiB that is allocated and never written.
# Whole pool at this config: 109 pages = 7.76 GiB.

scale_patch_pages[block]         deque, ≤ scale_frames   NEVER recycled
live_window_patch_pages[block]   deque, ≤ sliding_window recycled FIFO
all_special_pages[block]         list, append-only       NEVER recycled
free_patch_pages[block]          free list
```

```
append_frame(block, k, v)                       # :202
    sp_k    = k[:6]                             # specials → special stream
    patch_k = k[6:]                             # 777 patches → one whole page
    _write_patch_page(...)                      # :474  routes:
        if len(scale_patch_pages) < scale_frames: → scale_patch_pages
        else:                                     → live_window_patch_pages
    _write_special_tokens(...)                  # :514  append-only, 6 tokens
    frame_count[block] += 1

evict_frames(block, ...)                        # :229, while-loop at :252
    while len(live_window_patch_pages) > sliding_window:
        free_patch_pages.append(live_window_patch_pages.popleft())
    # scale pages and special pages are unreachable here

build_visible_page_table(block)                 # :438  ← BEST single breakpoint
    return scale_pages + window_pages + special_pages   # strict order matters:
                                                        # only the last page is partial

compute_attention(block, q)                     # :349
    if block == 0: prefill_wrapper.plan(...)    # once per frame, reused by 23 blocks
                                                # causal=False, pos_encoding_mode="NONE"
    return prefill_wrapper.run(q, kv_caches[block])
```

Non-keyframe path (`block.py:256`): `_defer_eviction=True` → `append_frame` →
`compute_attention` → `rollback_last_frame` (`:268`, undoes all three sub-ops).

---

## 8. Eviction arithmetic

| backend | trigger | steady state |
| --- | --- | --- |
| SDPA | `cached > sliding_window + scale_frames` (`attention.py:694`) | rebuild `[:scale] ⧺ [-window:]` |
| FlashInfer | `len(live_window) > sliding_window` (`flashinfer_cache.py:252`) | `scale + window` pages |

Both fire on the **same frame**. On eviction the frame's 777 patch tokens are
dropped and only its **6 special tokens** survive, in `k_i_special` /
`all_special_pages`. That demotion is the trajectory memory.

`demo.py:145` wires `kv_cache_scale_frames = args.num_scale_frames`, so
`--num_scale_frames` controls both the phase-1 block size and the permanent
cache quota. There is no separate `--kv_cache_scale_frames` flag.

**First eviction fires at frame index `i` where `i + 1 > scale + window`:**

| config | threshold | first eviction | resident |
| --- | --- | --- | --- |
| `scale 4, window 64` (default) | 68 | `i = 68` — **never reached in a 24- or 40-frame run** | 68 |
| `scale 2, window 4` (debug) | 6 | `i = 6` | 6 |
| `scale 4, window 64`, `--first_k 140`, `interval 2` | 68 | `i = 132` | 68 |

To see eviction at all: `--first_k 40 --num_scale_frames 2 --kv_cache_sliding_window 4`.
Then `k_0.shape[2]` pins at 6 and `k_0_special.shape[2]` climbs to 34 by `i=39`.

---

## 9. Camera head (`heads/camera_head.py:157` — `CameraCausalHead`)

> `CameraHead` (`:23`) and `CameraDecoder` (`:390`) are **never instantiated**.
> Their bodies look nearly identical. Breakpoint the right one.

```
_predict_camera (gct_base.py:165)
    tokens → float()                              # bf16 → fp32 boundary
    with autocast(enabled=False):                 # head runs entirely in fp32

pose_tokens = aggregated_tokens_list[-1][:, :, 0] # token 0 = camera token
                                                  # [1, S, 2048]  — 777 patches discarded

if kv_cache is None:                              # :287
    kv_cache = [ {k_0..k_3, v_0..v_3} for _ in range(4) ]   # ONE CACHE PER ITERATION

pred = None
for it in 0..3:                                   # :341
    cond   = embed_pose(empty_pose_tokens if pred is None else pred.detach())
    shift, scale, gate = poseLN_modulation(cond).chunk(3)
    x = gate * modulate(adaln_norm(pose_tokens), shift, scale) + pose_tokens
    for b in 0..3:
        x = trunk[b](x, kv_cache=kv_cache[it], global_idx=b)
    pred = (pred or 0) + pose_branch(trunk_norm(x))
    out.append(activate_pose(pred))               # linear / linear / relu
frame_idx += S                                    # :375  must track total_frames_processed

return out[-1]                                    # only the last iteration is used
```

`pose_tokens` never changes across iterations — it is fixed image evidence.
Only the *conditioning* changes. AdaLN-Zero, DiT-style.

**This cache never evicts.** `_apply_kv_cache_eviction_causal` guards on
`cache["k_i"].shape[3] > 1`, and that dim is exactly `1` here (one camera token
per frame). So every frame's camera token is retained forever, at all
4 blocks × 4 iterations ≈ **256 KB/frame fp32** → ~1 GB at 3,840 frames,
~6 GB at 25,000. `--camera_num_iterations 1` shrinks it 4×.

---

## 10. Decode

```
DPTHead.forward (heads/dpt_head.py:115)
    for layer in [0,1,2,3]:                       # the 4 selected aggregator outputs
        x = tokens[layer][:, :, 6:]               # drop specials → [B,S,777,2048]
        x = reshape to [B*S, 2048, 21, 37] → project → resize
    fuse → interpolate → output_conv2
    activate_head(out, "exp", "expp1")            # depth=exp, conf=1+exp
    → depth [B,S,294,518,1], depth_conf [B,S,294,518]

pose_encoding_to_extri_intri (utils/pose_enc.py:72)
    T    = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]      → quat_to_mat  (NOT normalized in the head)
    fov  = pose_enc[..., 7:9]
    fy = (H/2) / tan(fov_h/2);  fx = (W/2) / tan(fov_w/2)
    cx, cy = W/2, H/2                             # principal point ASSUMED centred
```

`conf = 1 + exp(·)`, hence `--conf_threshold 1.5` as a sane default.

The viewer unprojects depth+pose itself (`vis/point_cloud_viewer.py:167`) because
`world_points` is absent.

---

## 11. Parameters that look live but are not

Threaded through the call stack and never read:

| param | where it dies |
| --- | --- |
| `mask` (`GCTBase.forward`) | never referenced; `gct_base.py:330` passes `mask=ordered_video` instead |
| `query_points`, `enable_track` | no track head exists anywhere; `_normalize_input` returns it and it is dropped |
| `point_masks`, `gather_outputs` | accepted by all `_predict_*`, ignored in every body |
| `sliding_window_size` | `_process_causal_stream` never reads it; real windowing is `kv_cache_sliding_window` |
| `num_frame_for_scale` | signature-only in both aggregator attention classes; only affects `_prepare_special_tokens` |
| `num_frame_per_block` | **unused by FlashInfer**; live only on SDPA (cache reshape) |
| `is_scale_frames`, `attend_to_scale_frames`, `num_random_frames` | signature-only in `CausalAttention` |
| `causal_inference` | reaches only `_predict_camera`; allocates the camera cache. Makes nothing causal. |

`**kwargs` on `GCTBase.forward` swallows typos silently.

Checkpoint contains **only** `aggregator` (1211) + `camera_head` (69) +
`depth_head` (62) = 1342 keys. No point head, no track head.

---

## 12. Paper ↔ code terminology

| paper (§3.2, §4.4) | code |
| --- | --- |
| anchor context / **anchor token** | `scale_token`, `num_scale_frames`, `scale_patch_pages` |
| local pose-reference window | `kv_cache_sliding_window`, `live_window_patch_pages` |
| trajectory memory | the append-only 6-token special stream |
| **context tokens** | the 6 special tokens, i.e. `[cam, r0..r3, scale]` |
| **Direct Output mode** | `--mode streaming` |
| **Visual Odometry (VO) mode** | `--mode windowed` |
| Sim(3) window alignment | `_pairwise_alignment` → `(s, R, t)` |

§3.2 verbatim, and the reason "context tokens" is ambiguous in conversation:
"for frames that fall outside both the anchor set and the active sliding window,
we retain only the camera, anchor, and register tokens (i.e. **6 context tokens**
per frame) while discarding the memory-intensive image tokens". So the paper's
"context tokens" are the 6 specials -- NOT the image/patch tokens, which it calls
image tokens. §3.3 names them camera token / register tokens (4) / **anchor
token**; `scale_token` in code is the anchor token.

Paper: trained on ≤ 320 views → source of `_DEFAULT_AUTO_KEYFRAME_THRESHOLD = 320`.
Direct mode stable to ~10× that (~3,000 frames) → source of the README's
">3000 frames use windowed" guidance.

**Gap:** the paper says flow-based keyframe selection is "shared by both inference
modes". In the released code `flow_threshold` exists **only** in
`gct_stream_window.py`; `gct_stream.py` has none, and `demo.py` never exposes it.
Streaming uses a fixed `keyframe_interval` instead.
