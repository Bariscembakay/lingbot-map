# Trajectory-memory ablation — lingbot-map

Our own experiments on the paper's 6 context tokens, split out of
`reproduction.md` 2026-08-23: reproducing upstream and probing the mechanism are
different questions. Reproduction status lives there; this file is the mechanism.

Arrows mark the better direction: ATE / RPE are lower-is-better (↓); AUC / F1 are
higher-is-better (↑). RPE-rot is in degrees.

## Table 3 — sparse vs. dense, why the trend is inverted

### Leading hypothesis (2026-08-19): the two rows aren't a controlled comparison

`_keyframe_interval: auto` resolves to `ceil(N/320)`, so it lands on
**different values either side of the 320 threshold** — and the two rows we
compare sit either side of it:

| row | frames | `auto` → interval | keyframes cached |
| --- | --- | --- | --- |
| sparse (`oxford.yaml`, stride 12) | 320 | **1** | 320 — every frame |
| dense (`oxford_long.yaml`, stride 1) | 3,840 | **12** | 8 + 320 = **328** |

Same `raw_data_root`, so both end up caching ~320 frames at ~12-frame
spacing over the same physical trajectory. **The dense run's KV cache is
essentially the sparse run's cache.** The only real difference is that the
dense run additionally emits predictions for the 11-in-12 non-keyframes,
which attend to that same memory and are then scored against GT.

If that's right, we didn't measure "does the method degrade over 3,840
frames" — we measured "does denser output sampling help at fixed memory,"
and ATE improving is the expected answer (12x more poses, smoothly sampled,
one global Sim(3) alignment). Which would explain −1.03 where the paper got
+0.69.

See `fixes.md` for why `auto` keeps the keyframe count *constant* in N, and
for the paper↔code gap: the paper's dense protocol uses **flow-based**
keyframe selection, which is not implemented in the streaming path at all
(only in the windowed/VO model).

**Decisive test** — rerun `oxford_long` with:

```yaml
_keyframe_interval: 1
_max_frame_num: 4096      # required, or it dies at keyframe 1024 (see fixes.md)
```

3,840 real keyframes, genuine long-sequence cache, ~9.5 GB preallocation
(fine on an H200; patch pages stay bounded by the 64-frame window).
Prediction: ΔATE flips positive, near the paper's +0.69. If it stays
negative, the cause is elsewhere (eval protocol or scene selection) and
Table 3 needs a different investigation.

#### Decisive test RESULT (2026-08-21) — hypothesis confirmed, but it overshoots

Ran exactly as specified above (`_keyframe_interval: 1`, `_max_frame_num: 4096`),
10 scenes x 3,840 real keyframes, as the control arm of the context-token
ablation. Same scenes, same `raw_data_root`, only the interval differs:

| dense protocol | keyframes cached | ATE ↓ |
| --- | --- | --- |
| `auto` -> 12 (the Table 3 row above) | ~328 | **5.16** |
| forced `1` (genuine long-sequence cache) | 3,840 | **29.06** |

**ΔATE vs sparse flips from -1.03 to +22.87.** So the hypothesis is right in
direction: the original dense row never stressed the streaming state, and that
is why ours improved where the paper degraded.

But +22.87 overshoots the paper's +0.69 by ~33x, so the paper's dense protocol
is **neither** of these. That is consistent with the documented paper<->code gap:
the paper uses flow-based keyframe selection, which lands on some intermediate
keyframe count and is not implemented in the streaming path at all. Table 3
cannot be reproduced from the released streaming code at any fixed interval —
`auto` undershoots the stress, `1` massively overshoots it.

Mechanism for the overshoot: at interval 1 the 3D RoPE frame index runs to 3,840
while the model was trained on <=320 views, so positions 320-3,840 are out of
distribution. `_auto_keyframe_threshold = 320` exists precisely to prevent this.

**Consequence worth carrying:** in the regime this model actually works in, the
trajectory memory can never exceed ~2.6% of visible context — `auto` caps
keyframes at ~320, hence ~248 evicted frames x 6 tokens against 72 resident
frames x 783. Raising the memory's share requires either shrinking the resident
window (in-distribution, see the ablation's w8/w4 blocks) or leaving the trained
frame range (out-of-distribution, as here).

Note this does **not** explain Table 2 — at 320 frames / interval 1 we're
in-distribution and matching the paper's training length, so keyframing
isn't the suspect for the RPE gap there. Keep the two separate.

## Trajectory-memory context-token ablation (2026-08-20)

Question: which of the paper's 6 "context tokens" does the trajectory memory
actually need? §3.2 keeps camera + anchor + 4 registers for every frame evicted
from the anchor set and the sliding window; these arms vary WHICH survive.
Anchor and window frames keep all 783 of their tokens in every arm.

Results: `campaigns/lingbot_map/context_token_ablation/` — 13 method configs,
20 arm-runs (all 13 on Oxford across 3 protocols, 7 on NRGBD). Provenance (base
commit + working-tree diff + config snapshot) under its `_provenance/`.

**Only the SDPA backend can run this.** `FlashInferKVCacheManager.evict_frames`
accepts `cross_frame_special` / `include_scale_frames` / `camera_only` and then
ignores all three, so a FlashInfer run silently returns baseline numbers. The
aggregator now raises instead. Hence the `lingbot_map` arm below: same 6 tokens,
FlashInfer, purely to size the backend gap (-0.07 AUC@15 — negligible next to
the effect being measured).

### Oxford Spires sparse s12, 10 scenes — pose

| arm | kept per evicted frame | n | AUC@15 ↑ | ATE ↓ | RPE-rot ↓ |
|---|---|---|---|---|---|
| `traj_regonly` | register | 4 | 63.81 | 6.076 | 4.574 |
| `traj_noscale` | camera+register | 5 | 63.77 | 6.115 | 4.299 |
| `traj_nocam` | register+scale | 5 | 63.70 | 6.132 | 4.321 |
| `traj6` (control) | camera+register+scale | 6 | 63.66 | 6.143 | 4.276 |
| `lingbot_map` (FlashInfer) | camera+register+scale | 6 | 63.58 | 6.167 | 4.311 |
| `traj_noreg` | camera+scale | 2 | 60.81 | 6.861 | 4.782 |
| `traj_camonly` | camera | 1 | 60.79 | 6.761 | 4.871 |

**The registers carry the memory; the camera and anchor tokens contribute
nothing measurable.** Keep the 4 registers and you are within +0.15 AUC@15 of
control no matter what else is dropped; drop them and you lose ~2.85 AUC@15 and
~0.7 m ATE regardless of what else is kept. It is not token count: 4-token
`regonly` matches control while 2-token `noreg` and 1-token `camonly` are
equally bad, and 2 -> 1 costs nothing further. `regonly` beats control on 8/10
scenes, so 6 -> 4 tokens (-33% memory) looks free here — though the deltas among
register-keeping arms are within noise at n=10, and `regonly` does pay RPE-rot
+0.30 while ATE/AUC improve.

`noreg` degrades on 8/10 scenes, worst on `bodleian-library-02` (62.6 -> 46.3).
`christ-church-05` sits at AUC@15 ~2.7 in every arm — a scene the method fails
on regardless, unrelated to the ablation.

Control sanity: `traj6` at 63.66 / 6.14 reproduces the `sparse_s12` run
(63.50 / 6.19) and still beats the paper's 61.64 / 6.42.

### Validation

`.agents/scratch/ctx_ablation/validate_trajmem.py` (invariants) and
`ctx_ablation/analyze_ctx_ablation.py` (divergence). Key facts established:

- The two stores behave differently: `k_i_special` is `cat`-ed on every eviction
  and grows **without bound** (one entry per evicted frame, never trimmed or
  recycled); `k_i` is **rebuilt** as `[:scale] ⧺ [-window:]` and stays pinned at
  scale+window frames. `memory + resident == frames processed` every frame, so
  no frame is double-counted or lost. Non-keyframes persist to neither store.
- Resident tokens/frame stays 783 under **every** arm — the window and anchor
  tokens are provably untouched.
- All 50 (10 scenes x 5 ablations) Oxford comparisons are **bit-identical to
  control through frame 71 and first diverge at exactly frame 72** = anchor 8 +
  window 64. Divergence earlier would mean the change leaked into the anchor or
  window; never diverging would mean the flag never reached the model.
- The 6-token spec reproduces the original contiguous slice byte for byte, so
  the control arm is unchanged from upstream.

### The floor: no trajectory memory at all (2026-08-22)

`kv_cache_cross_frame_special: false` — an evicted frame retains 0 of its 6 tokens.
Added because the NRGBD nulls were indistinguishable from a failed ablation without
it, and because nothing else puts an absolute scale on the per-token deltas.

| benchmark | with memory | memory OFF | mechanism worth |
|---|---|---|---|
| Oxford sparse (AUC@15 ↑) | 63.66 | 60.58 | **3.08** |
| Neural RGB-D (F1 ↑) | 65.108 | 65.161 | **~0** |

**Oxford: the registers are 93% of the whole mechanism** (2.85 of 3.08). Arms form
two clusters, not a gradient — registers present (6/5/5/4 tokens) 63.66-63.81
spread 0.15; registers absent (2/1/0) 60.58-60.81 spread 0.23. Inside the
register-less cluster the metric ordering contradicts itself (AUC says fewer is
worse, ATE says fewer is better), which is the signature of noise: once the
registers are gone, what else you keep is irrelevant. `none` worse on 9/10 scenes.

**NRGBD: the mechanism is inert.** All 8 arms within 0.113 F1 and memory-off is the
highest. Confirmed independently of the eval protocol by trajectory deviation:
Oxford `none` mean 1.073 vs NRGBD `none` mean 0.0094 — **114x**. The ablation is
still correctly graded on NRGBD (`none` diverges 2.5x more than `noreg`); every step
is just tiny. So every NRGBD null in this work is a property of revisiting indoor
trajectories, not a failed ablation.

Verification that the NRGBD arms really were ablated (the doubt that prompted this):
every arm logs its own composition at load, and divergence starts at exactly frame
72 on 8/9 scenes — the 9th being `whiteroom`, which has 336 frames, crosses the 320
threshold so `auto` gives interval 2, putting the 72nd *keyframe* at frame
8 + (72-8)*2 = **136**. Observed 136. The mechanism is behaving exactly as designed.

### Memory-share stress test (does the finding survive when the memory matters?)

At the default anchor 8 / window 64 the memory is only 2.6% of visible context.
Shrinking the resident store raises that share on the same data, far cheaper than
a 3,840-frame run — and anchor 2 / window 4 lands at 28.6%, exactly matching
`oxford_long` at 3,840 frames with `keyframe_interval 1`.

ΔAUC@15 against each block's OWN control (blocks are different protocols; never
compare absolute numbers across them):

| resident store | memory share | `regonly` (4 reg) | `noreg` (no reg) | block control AUC@15 ↑ |
|---|---|---|---|---|
| anchor 8 / window 64 | 2.6% | +0.15 | -2.85 | 63.66 |
| anchor 8 / window 8 | 12.7% | -0.29 | -4.02 | 30.87 |
| anchor 2 / window 4 | 28.6% | +0.13 | -2.20 | 24.19 |

**The finding is protocol-independent**: at every memory share tested, the 4
register tokens alone reproduce the full 6-token control (|Δ| ≤ 0.29) and dropping
them costs 2.2-4.0 AUC@15. So it is not an artifact of the memory being a
negligible slice of context.

The penalty magnitude is NOT monotonic in memory share (-2.85 → -4.02 → -2.20).
Most likely a floor effect rather than anything about the memory: shrinking the
window wrecks the model outright (control 63.66 → 30.87 → 24.19 AUC@15, ATE
6.14 → 18.90 → 22.77), leaving less to lose. Treat within-block direction as the
result and cross-block magnitude as uninterpretable.

### NRGBD reconstruction — no effect, and not because the metric is blunt

Complete arm set, 9 scenes, all evaluated (`context_token_ablation/neural_rgbd/`):

| arm | kept per evicted frame | n | Acc ↓ | Comp ↓ | Chamfer ↓ | F1 ↑ |
|---|---|---|---|---|---|---|
| `traj_none` | — (memory off) | 0 | 0.0734 | 0.0302 | 0.0518 | 65.1615 |
| `traj_nocam` | register+scale | 5 | 0.0736 | 0.0304 | 0.0520 | 65.1161 |
| `traj6` (control) | camera+register+scale | 6 | 0.0737 | 0.0303 | 0.0520 | 65.1084 |
| `lingbot_map` (FlashInfer) | camera+register+scale | 6 | 0.0736 | 0.0303 | 0.0520 | 65.0993 |
| `traj_noreg` | camera+scale | 2 | 0.0736 | 0.0303 | 0.0520 | 65.1036 |
| `traj_regonly` | register | 4 | 0.0736 | 0.0303 | 0.0520 | 65.0842 |
| `traj_camonly` | camera | 1 | 0.0736 | 0.0303 | 0.0520 | 65.0793 |
| `traj_noscale` | camera+register | 5 | 0.0737 | 0.0304 | 0.0520 | 65.0484 |

Total spread 0.113 F1 across every arm from 6 tokens down to none — smaller than
the SDPA/FlashInfer backend gap, and `noreg` (which costs 2.85 AUC@15 on Oxford)
is indistinguishable from control. Acc/Comp/Chamfer are flat to 4 decimals.

The poses genuinely barely move: max |delta| vs control is **7.9e-03** across all
9 scenes, against **3.06** for `noreg` on Oxford. Three orders of magnitude, at
comparable sequence lengths (184-336 frames) and eviction counts (112-264).

Likely driver is trajectory shape, not volume through the memory: NRGBD is
room-scale indoor capture that keeps revisiting the same space, so the 64-frame
window still overlaps nearly everything relevant and the distant past adds
nothing; Oxford traverses large outdoor sites and rarely returns, so evicted
frames hold information available nowhere else. Supporting detail: `whiteroom`
has the most evicted frames of any scene (264) and the *smallest* deviation
(1.5e-03).

### Long-span test (2026-08-21) — finding holds at 3,840 keyframes

Minimal arm set at stride 1, `keyframe_interval: 1`, `max_frame_num: 4096`, so
the trajectory memory spans 3,768 evicted frames instead of 248.

| arm | kept per evicted frame | n | ATE ↓ | dATE | RPE-t ↓ | RPE-rot ↓ | worse on |
|---|---|---|---|---|---|---|---|
| `long_traj6` (control) | camera+register+scale | 6 | 29.063 | — | 0.3748 | 2.769 | — |
| `long_regonly` | register | 4 | 28.931 | -0.132 | 0.3846 | 2.706 | 6/10 |
| `long_noreg` | camera+scale | 2 | 31.877 | **+2.814** | 0.2795 | **3.680** | **9/10** |

Registers necessary and sufficient, same as every other regime. The **relative**
penalty is near-identical across a 12x change in memory span:

| regime | keyframes | memory span | noreg ATE penalty |
|---|---|---|---|
| sparse s12 | 320 | 248 frames | +0.719 on 6.143 = **+11.7%** |
| long s1 | 3,840 | 3,768 frames | +2.814 on 29.063 = **+9.7%** |

Frame-72 property holds on 6 of 10 long scenes. The other four
(keble-college-04/05, observatory-quarter-01/02) diverge at frame **0** because
they ran on H200 while everything else ran on A100 — cross-hardware float
nondeterminism, not ablation leakage. Measured on identical inputs it is ~1% of
the ablation signal at frame 3839 (7.4e-03 vs 7.8e-01), and the subgroup means
agree (+2.745 A100 vs +2.917 H200), so the result stands. Details and the
platform-pair comparison are in the campaign's `RESULTS.md`.

**Read this with the caveat that the control is itself broken here** (ATE 29 m on
~100 m paths — see the Table 3 decisive-test result above for why interval 1
leaves the trained frame range). The floor effect that risked swamping the
necessity signal did not materialise — noreg is worse on 9/10 scenes — but the
trustworthy in-distribution evidence remains the sparse arms plus the w8/w4
memory-share blocks. `noreg` RPE-trans is *better* (0.2795 vs 0.3748) while ATE
and RPE-rot are worse: it produces a locally smoother path that drifts more
globally.

### Open

None — the ablation is complete. Both former caveats closed 2026-08-23:

- All 8 NRGBD arms have finished, including the two that were in flight; the
  reading is unchanged (table above).
- Temporal span was tested directly by the long-span run at 3,768 evicted
  frames, not just inferred from the memory-share blocks.
