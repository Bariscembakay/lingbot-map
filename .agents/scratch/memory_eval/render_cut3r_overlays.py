"""Bake pred+GT overlay entries for the CUT3R zero-shot recall96 renders,
from the already-written PLYs (no npz reload needed)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from scripts.memory.render_state import write_ply  # noqa: E402

def read_ply(p):
    # Mirrors render_state.write_ply exactly (binary LE, xyz f4 + rgb u1).
    raw = Path(p).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    n = int([l for l in raw[:end].split(b"\n") if l.startswith(b"element vertex")][0].split()[-1])
    rec = np.frombuffer(raw[end:], dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                          ("r","u1"),("g","u1"),("b","u1")], count=n)
    pts = np.stack([rec["x"], rec["y"], rec["z"]], -1)
    col = np.stack([rec["r"], rec["g"], rec["b"]], -1)
    return pts, col

D = Path("/group/compact-3dmem/campaigns/spatial_memory/CUT3R_zeroshot_512dpt/recall96_val8_allqueries")
for gt in sorted(D.glob("val_*-gt.ply")):
    tag = gt.name[:-len("-gt.ply")]
    pp, pc = read_ply(D / tag / "pred_cloud.ply")
    gp, gc = read_ply(gt)
    (D / f"{tag}_overlay").mkdir(exist_ok=True)
    write_ply(D / f"{tag}_overlay" / "pred_cloud.ply",
              np.concatenate([pp, gp]), np.concatenate([pc, gc]))
    print(f"[overlay] {tag}")
print("done")
