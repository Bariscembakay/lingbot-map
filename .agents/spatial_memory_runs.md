s# Spatial-memory training runs — full register

Every training job of the CUT3R-state recall architecture (write → 768-token
state → read), in submission order. "best val" = lowest valm_self
(unweighted L21, canonical units) over the run's history; single-scene
overfits have no meaningful val, so their column is best TRAIN l21_self,
marked (t). All runs share the recall
objective (conf-weighted L21 on raymap probes of past cameras) unless noted.

## Era 0 — first overfits (1 scene, 16-frame windows, DPT head, full read)

| run | what it was | status | best val |
|---|---|---|---|
| `train_20scene_rtx` | earliest multi-scene training attempt, pre-fleet protocol | retired | 0.175 (t) |
| `overfit_1scene_rtx` / `overfit_1scene_2k` | first single-scene overfits (2000 upd, rtx) | done | 0.140 (t, 200 upd) |
| `overfit_1scene_2k_h200` | same, default settings, moved to H200 | done | 0.023 (t) |
| `overfit_1scene_tbptt8` | truncated-BPTT-8 axis | done | 0.070 (t, 340 upd) |
| `overfit_1scene_nowrite` | no-write control (state never written) | done | 0.021 (t) |
| `overfit_1scene_probecur_off` | echo (lag-0) probe disabled | done | 0.023 (t) |
| `overfit_1scene_raymap_true` | clean-ray query convention | done | 0.020 (t) |
| `overfit_1scene_taps_all` | all 4 aggregator taps as write input | done | 0.021 (t) |

## Era 1 — 32-scene 16-frame fleet (batch 4, tbptt-8, 4000 upd)

| run | axis it tested | status | best val |
|---|---|---|---|
| `scenes32_16f_b4` | reference (12-layer interconnected write+read, DPT head) | done | 0.151 |
| `scenes32_16f_b4_nowrite` | no-write twin (memory floor) | done | 0.286 |
| `scenes32_16f_b4_fullbptt` | full BPTT vs tbptt-8 | done | 0.158 |
| `scenes32_16f_b4_probecur_off` | echo probe off | done | 0.177 |
| `scenes32_16f_b4_raymap_true` | clean-ray convention (12-layer read) | cancelled early | 0.252 @2k |
| `scenes32_16f_b4_freeze_s0` | frozen initial state | done | 0.157 |
| `scenes32_16f_b4_state1536` | 2x state tokens | cancelled early | 0.221 @2k |
| `scenes32_16f_b4_tapsall` | all 4 taps (CPU-resident) | done | 0.178 |
| `scenes32_16f_b4_raydepth` | 0.8M linear ray-depth head | done | 0.177 |
| `scenes32_16f_b4_lingbot_frozen` | frozen lingbot DPT head + 4 linear adapters | done | 0.145 |
| `scenes32_16f_b4_smallread` | decoupled 2-block read + ray-depth head | done | 0.166 |
| `scenes32_16f_b4_write4layers` | writer 12→4 interconnected pairs (smallread read) | done | 0.165 |
| `scenes32_16f_b8n8_write4layers` | 4-layer writer, freed memory → batch 8 + 8 queries | done | 0.145 |
| `scenes32_16f_b4_write4layers_raymaptrue` | clean rays × light decoupled read | done | 0.171 |
| `scenes32_16f_b4_write4layers_oneway` | write interconnection removed (state-stack only) | done | 0.158 |

## Era 1b — CUT3R architecture controls (same data, loss, protocol; 16f)

| run | what it was | status | best val |
|---|---|---|---|
| `scenes32_16f_b4_CUT3R_CONTROL` | full CUT3R fine-tune, frozen ViT encoder (cached tokens) | done | 0.099 @0 (zero-shot; trained-best 0.142) |
| `scenes32_16f_b4_CUT3R_CONTROL_frozenhead` | + frozen heads | done | 0.099 @0 (trained-best 0.122) |
| `scenes32_16f_b4_CUT3R_CONTROL_randomdec_frozenhead` | + decoder stack random-init | done | 0.225 |

## Era 2 — 16f fixed-window smallread overfits (1 scene)

| run | what it was | status | best val |
|---|---|---|---|
| `overfit1scene_16f_smallread` | write arm, probe every frame, full BPTT | stopped early | 0.054 (t, 110 upd) |
| `overfit1scene_16f_smallread_nowrite` | no-write parity twin | stopped early | 0.013 (t) |

## Era 3 — 96-frame single-scene overfits

| run | what it was | status | best val |
|---|---|---|---|
| `overfit1scene_96f_smallread` | 96-frame fixed-window write arm; later continued to 10k upd, then continued again with LR-plateau decay ("exact overfit" push) | running (LR-decay leg) | 0.011 (t) |
| `overfit1scene_96f_smallread_nowrite` | no-write parity twin (96 frames) | done | 0.011 (t) |

## Era 4 — 96-frame 32-scene arms (random windows, early-stop on val plateau)

| run | architecture | status | best val |
|---|---|---|---|
| `scenes32_96f_b4_write4_read2_lingbothead` | 4-layer write, 2-block read, frozen lingbot head | done (early-stop) | 0.207 |
| `scenes32_96f_b4_write12_read2_lingbothead` | 12-layer write, 2-block read, frozen lingbot head | done (early-stop) | 0.207 |
| `scenes32_96f_b4_write12_read12_lingbothead` | 12-layer interconnected write+read, frozen lingbot head | done (timeout + resumed to early-stop) | 0.216 |
| `scenes32_96f_b4_write4oneway_read2_lingbothead` | one-way 4-layer write (no interconnection) | running | 0.259 so far |
| `scenes32_96f_b4_CUT3R_CONTROL_randomdec_frozenhead` | CUT3R arch, random decoder, frozen encoder+head, 96f | done (cancelled at 14/15 patience) | 0.261 |

## Era 5 — continue-runs: unfreeze the lingbot DPT trunk from each plateau

| run | continues from | status | best val |
|---|---|---|---|
| `scenes32_96f_b4_write4_read2_lingbothead_unfrozenhead` | write4 plateau | done (early-stop) | 0.183 |
| `scenes32_96f_b4_write12_read2_lingbothead_unfrozenhead` | write12+2read plateau | done (early-stop) | 0.190 |
| `scenes32_96f_b4_write12_read12_lingbothead_unfrozenhead` | write12+12read plateau | done (early-stop) | 0.198 |

## Era 6 — data scaling (96 scenes = 32 curated + 64 extension)

| run | what it was | status | best val |
|---|---|---|---|
| `scenes96_96f_b4_write4_read2_lingbothead` | 3x training scenes, same protocol/val as Era 4 | done (early-stop) | 0.167 |
| `scenes96_96f_b4_write4_read2_lingbothead_unfrozenhead` | + unfrozen head from its plateau (compounding test) | done (early-stop) | **0.151** |
| `scenes96_96f_b4_write4_read4_lingbothead` | A/B #1: 4-layer read (capacity symmetry with writer) | done (early-stop) | **0.158** |
| `scenes96_96f_b4_write4_read2_tapsall_lingbothead` | A/B #2: all 4 taps to writer (8192-d in_proj) | done (early-stop) | 0.175 |
