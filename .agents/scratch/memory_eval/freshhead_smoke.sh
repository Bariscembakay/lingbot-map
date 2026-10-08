#!/usr/bin/env bash
# Fresh-head duo smoke: BIGRUN architecture, random-init DPT for depth and for
# world xyz; train, resume, then load the checkpoint the way run_ours.py does.
set -euo pipefail
cd /home/baris_bakay/lingbot-map
export PYTHONPATH=/home/baris_bakay/lingbot-map
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
TR=(/data/lingbot-tapcache-v4-40/train/*)
R=/scratch/$USER/freshhead_smoke_$SLURM_JOB_ID
for arm in "" "--pts-frame world"; do
    out=$R/arm$(echo "$arm" | tr -d ' -')
    common=(--clips "${TR[0]}" "${TR[1]}" --val-clips /data/lingbot-tapcache-v4-40/val_top/210f741378_c0
        --head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head $arm
        --frames 8 --max-frames 16 --batch 2 --tbptt 8 --n-past 4 --probe-every 1 --clip-store cpu
        --val-every 2 --save-every 2 --viz-every 1000 --log-every 1 --wandb off --patience 0
        --lr 1e-4 --warmup 1 --resume auto --out "$out")
    echo "=== arm: ${arm:-depth}"
    "$PY" scripts/memory/train_state.py "${common[@]}" --updates 4 2>&1 | grep -E "fresh-head|lora|model\]|^\[ +[0-9]+\]|^\[val\]|Error|error|Traceback" | sed 's/valx.*//'
    "$PY" scripts/memory/train_state.py "${common[@]}" --updates 6 2>&1 | grep -E "resume|^\[ +[0-9]+\]|Error|Traceback"
    "$PY" - "$out/last.pt" <<'PYEOF'
import sys, torch
from lingbot_map.memory.cut3r_state import StateMemory
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); t = ck["args"]
m = StateMemory(patch_size=14, tap_dim=2048, state_tokens=int(t.get("state_tokens", 768)),
                dec_depth=int(t["dec_depth"]), read_depth=int(t["read_depth"]),
                write_oneway=bool(t["write_oneway"]), head_type=t["head"], grad_ckpt=False)
if t.get("fresh_head"): m.head.fresh_head(t.get("pts_frame"))
miss, unexp = m.load_state_dict(ck["model"], strict=False)
lb = torch.load("/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt", map_location="cpu", weights_only=False)["depth_head"]
k = "scratch.output_conv2.0.weight"
same = torch.equal(ck["model"]["head.dpt." + k], lb[k])
print(f"[eval-load] missing {len(miss)} unexpected {len(unexp)} | head differs from lingbot weights: {not same} | trainable head params in ckpt args: fresh_head={t.get('fresh_head')} lora={t.get('lora_head')}")
PYEOF
done
rm -rf "$R"
