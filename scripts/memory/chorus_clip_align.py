"""Per-clip map from clip GT units to the ScanNet++ world frame (metres).

The cache stores GT relative to the clip's first frame and divided by a canonical
scale s (lingbot_map/memory/gt.py: c2w_clip = inv(C0) @ c2w_raw, translations
and depth / s). A clip-frame point p therefore sits at C0 @ (s * p) in ScanNet++
world metres, where the Chorus voxel features live (checked: median 2.5-3.2 cm
to the nearest 5 cm voxel centre on three clips).

Writes {clip_dir_name: {"scene", "scale", "C0" (4x4)}} so training does not
need the ScanNet++ raw data at all.
Usage: python chorus_clip_align.py <out.json> <clip_dir> [clip_dir ...]
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lingbot_map.memory.cache_format import ClipMeta  # noqa: E402
from lingbot_map.memory.gt import read_iphone_meta  # noqa: E402

out = {}
for d in sys.argv[2:]:
    m = ClipMeta.load(Path(d))
    c2w, _ = read_iphone_meta(f"/data/ScanNetpp/data/{m.scene}/iphone/pose_intrinsic_imu.json", m.frame_ids[:1])
    out[Path(d).name] = {"scene": m.scene, "scale": float(m.gt_scale), "C0": np.asarray(c2w[0], float).tolist()}
Path(sys.argv[1]).write_text(json.dumps(out))
print(f"{len(out)} clips -> {sys.argv[1]}")
