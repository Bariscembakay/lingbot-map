"""Is upcast.npy enough? Per voxel, compare projector(mean upcast) with the
stored mean of per-Gaussian projector outputs (lang.npy / dino.npy).

The projectors are non-linear (LayerNorm -> Linear -> GELU -> Linear -> L2), so
averaging before vs after them differs; this measures by how much. If cosine is
~1, lang/dino can be dropped and regenerated from upcast.
Usage: python chorus_upcast_check.py <feature_root> [--sample N] [--json out.json] <scene> [scene ...]
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

CKPT = "/group/compact-3dmem/checkpoints/Chorus/lang-dino-enc-pretrain-scan-ppv2-mp3d-mcmc.pth"


def load_chorus_projectors(ckpt=CKPT):
    """{'lang': 1232->1152, 'dino': 1232->1024}, without the L2 normalisation.

    Built from the checkpoint directly: Chorus's own loader insists on CUDA.
    """
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    out = {}
    for name, d in (("lang", 1152), ("dino", 1024)):
        p = f"module.projectors.{name}."
        m = nn.Sequential(nn.LayerNorm(1232), nn.Linear(1232, 4928), nn.GELU(), nn.Linear(4928, d))
        m.load_state_dict({"0.weight": sd[p + "input_norm.weight"], "0.bias": sd[p + "input_norm.bias"],
                           "1.weight": sd[p + "mlp_head.0.weight"], "1.bias": sd[p + "mlp_head.0.bias"],
                           "3.weight": sd[p + "mlp_head.2.weight"], "3.bias": sd[p + "mlp_head.2.bias"]})
        out[name] = m.eval()
    return out


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("scenes", nargs="+")
    ap.add_argument("--sample", type=int, default=0, help="random voxels per scene (0 = all)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    proj = load_chorus_projectors()
    rng = np.random.default_rng(0)
    res = {}
    for scene in a.scenes:
        d = root / scene
        up_mm = np.load(d / "upcast.npy", mmap_mode="r")
        idx = (np.sort(rng.choice(len(up_mm), a.sample, replace=False)) if 0 < a.sample < len(up_mm)
               else np.arange(len(up_mm)))
        up = torch.from_numpy(np.asarray(up_mm[idx], np.float32))
        cnt = np.load(d / "count.npy")[idx]
        line, res[scene] = [scene, f"V={len(up_mm)} n={len(idx)}"], {}
        for name in ("lang", "dino"):
            with torch.no_grad():
                z = torch.cat([nn.functional.normalize(proj[name](up[i:i + 32768]), dim=1)
                               for i in range(0, len(up), 32768)])
            ref = np.asarray(np.load(d / f"{name}.npy", mmap_mode="r")[idx], np.float32)
            ref = nn.functional.normalize(torch.from_numpy(ref), dim=1)
            c = (z * ref).sum(1).numpy()
            res[scene][name] = dict(mean=float(c.mean()), p1=float(np.percentile(c, 1)), min=float(c.min()),
                                    wmean=float(np.average(c, weights=cnt)))
            line.append(f"{name} cos mean {c.mean():.4f} p1 {np.percentile(c, 1):.4f} min {c.min():.4f} "
                        f"(Gaussian-weighted {np.average(c, weights=cnt):.4f})")
        print(" | ".join(line), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
