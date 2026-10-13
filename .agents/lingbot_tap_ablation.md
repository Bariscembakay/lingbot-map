# Which lingbot taps the depth and camera heads actually use

2026-10-13. Script `scripts/memory/lingbot_tap_ablation.py` (job script
`.agents/scratch/memory_eval/tap_ablation_job.sh`, job 1100331, hala A6000, 27 min).
Results: `/group/compact-3dmem/campaigns/spatial_memory/lingbot_tap_ablation/results_f96.json`.

## Setup

- Released `ckpt/lingbot-map.pt`, heads run on the cached aggregator taps of all 40
  4-tap clips (`lingbot-tapcache-v4-40`: 32 train + 8 val), first 96 cached frames
  per clip, streamed exactly as `build_cache.py` did (8 scale frames, then one by one).
  The aggregator is replaced by a stub that hands out the cached tokens.
- Baseline reproduces the cache's own lingbot outputs: depth rel. diff <= 2.6e-4,
  pose encoding max diff <= 4.3e-4.
- **Mean-ablation:** an input is replaced by its per-channel mean over the clip's
  frames (and patches). "drop k" ablates tap k; "only k" ablates the other three.
  A tap is `cat([frame_attn, global_attn])` (2048-d); each 1024-d half is also
  ablated alone.
- Depth: AbsRel and delta<1.25 against GT after per-frame median scaling.
  Pose: ATE RMSE after a Sim(3) fit of camera centres (metres), and mean rotation
  error between consecutive frames (RPE rot).
- These are **direct effects at the head inputs**. Taps 4/11/17 also shape layer 23
  (and so pose) inside the aggregator; that total effect needs an intervention in
  the aggregator and was not measured.

## What each head reads (from the code)

- Depth (DPT): patch tokens of taps 4, 11, 17, 23 (`intermediate_layer_idx=[0,1,2,3]`).
- Camera head: only token 0 (camera token) of tap 23, causal across frames.

## Results (mean over 40 clips)

| ablation | AbsRel | delta1 | ATE m | RPE rot deg |
|---|---|---|---|---|
| none | 0.044 | 0.969 | 0.044 | 0.67 |
| drop 4 | 0.049 | 0.967 | | |
| drop 4 frame / global | 0.046 / 0.046 | 0.968 / 0.969 | | |
| drop 11 | 0.188 | 0.704 | | |
| drop 11 frame / global | 0.100 / 0.103 | 0.900 / 0.893 | | |
| drop 17 | 0.109 | 0.874 | | |
| drop 17 frame / global | 0.075 / 0.065 | 0.946 / 0.959 | | |
| drop 23 (patch tokens) | 0.044 | 0.969 | 0.044 | 0.67 |
| only 4 | 0.282 | 0.563 | | |
| only 11 | 0.120 | 0.851 | | |
| only 17 | 0.202 | 0.680 | | |
| only 23 | 0.294 | 0.547 | | |
| drop camera token (tap 23) | | | 0.895 | 12.9 |
| drop camera token, frame half | | | 0.333 | 12.0 |
| drop camera token, global half | | | 0.093 | 2.3 |

## Findings

1. **The DPT's tap-23 branch is dead in the released checkpoint.** `scratch.layer4_rn`
   outputs are all <= about -58 on every clip/frame checked, and `refinenet4`'s
   ResidualConvUnit starts with `ReLU(inplace=True)`, so the branch emits a constant.
   Ablating tap 23 changes depth bit-for-bit nothing; "only 23" equals ablating every
   useful input. Depth depends on taps 4/11/17 only.
2. **Depth leans on tap 11, then 17; tap 4 adds little.** Tap 11 alone keeps AbsRel at
   0.120; removing it costs the most. Both halves matter about equally for tap 11.
3. **Pose depends on the camera token's frame-attention half.** Removing it costs
   almost as much rotation accuracy as removing the whole token; the global half
   matters much less.
4. **Consequence for the memory models.** `LingbotFrozenHead` feeds the read's 4th slot
   into the same dead branch, so that slot gets zero gradient under LoRA or full
   unfreezing (BIGRUN, the head-only run). Only `--fresh-head` escapes it. Memory
   meant to help depth has to act on the tap 11/17 paths; memory meant to help pose
   has to reach layer 23's camera token.

Follow-up launched: write from taps 11+17 instead of 23 (FRESHDPT depth init).
