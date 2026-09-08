# BIGRUN: 944-scene ScanNet++ training (submitted 2026-09-08, job 853887)

The data-scaling run: how far does recall go with all usable ScanNet++?

## Config (user-specified)
- write 8 layers **one-way** + read 8 layers, tap-23 only, frozen lingbot head
- 4x H200, one node, batch 12/rank (**global 48**), lr 2e-4 (sqrt-scaled from
  1e-4@4), warmup 300, 48h wall, patience 15 / min_delta 0.002, resume auto
- out: campaigns/spatial_memory/scenes944_96f_b12x4_write8oneway_read8_tap23_lingbothead

Evidence behind the oneway choice: the 32-scene oneway arm overtook its
interconnected twin (0.1849@6200 and falling vs 0.207 plateau) —
interconnection = faster early training, NOT higher capability.

## Data
- 1018 ScanNet++ scenes -> 968 buildable (50 missing iphone inputs).
- Split: **944 train / 24 val** (old 8 val scenes kept for comparability with
  every previous run + 16 new drawn from the revisit-screened pool).
  Lists: .agents/scratch/spatial_memory/bigrun_{train_scenes,val24}.txt
- Caches: `lingbot-tapcache-v5-tap23` = 864 new SLIM caches (tap 23 only,
  ~1.7GB/scene, 1.4TB total; full 4-tap would have been ~5.4TB > free space).
  The 96 v4 scenes are reused as-is (trainer resolves the tap-23 index from
  each cache's own meta.tap_layers).
- Adaptive stride: clips are 320 frames; stride = min(20, fits) — 209 scenes
  shorter than 6400 frames get stride 5-19; meta.json records the stride used.
- **Second ScanNet++ depth format discovered**: a scene subset stores
  depth.bin as raw-deflate float32 METRES (vs lz4 uint16 mm). gt.py detects
  the codec per file. Zero scenes lost to "corruption".

## Trainer changes (commit e888a4e and neighbours)
- Multi-GPU: torchrun + **manual grad all-reduce** (no DDP wrapper — tbptt
  runs several backward() per update and blocks use grad checkpointing, both
  fight DDP's hooks). Same-seed init, per-rank rng + clip shards (944/4=236
  clips/rank), rank-0-only val/ckpt/wandb/viz, early-stop broadcast.
- `--clip-store cpu`: taps+GT in host RAM (~450GB total), moved per access;
  944 clips cannot live on GPU. msp3/sof1 H200 nodes have 1.9TB RAM.
- Calibration (2-GPU smoke 841856, PASSED): 43.5GB @ batch 4/rank ->
  batch 12/rank ≈ 107GB, ~35GB margin; ~30s/step at batch 4.
- Gotchas fixed by smokes: SLURM_GPUS_ON_NODE unreliable (count GPUs via
  nvidia-smi -L); dump_viz indexed CPU-stored poses with a CUDA tensor;
  KEEPALIVE=1 guard for the long idle-looking load phase.

## Where results land
history.json / best.pt / last.pt under the campaign dir (mirrored every 10
min from /scratch); wandb run scenes944_96f_b12x4_write8oneway_read8_tap23_lingbothead.
