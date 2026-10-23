#!/usr/bin/env python3
"""Pack one lingbot-cache1048 dataset into what the bidirectional refiner reads: the tap-23 special
tokens of every frame in one fp16 memmap, and the small per-frame/per-clip arrays in one index.

  <out>/<dataset>/spec23.npy   [N, 6, 2048] fp16, clips concatenated in index order
  <out>/<dataset>/index.npz    name, seq, offset, length, hw, gt_scale, stride (per clip);
                               pose_enc [N, 9], gt_c2w [N, 4, 4], K [N, 3, 3] (per frame)

The cache npz members are stored uncompressed, so specials are memory-mapped inside the zip and only
the tap-23 slice of each frame is read (a quarter of the bytes).
  <out>/<dataset>/desc.npy     [N, 8448] fp16 MegaLoc descriptors (with --desc)
Caches written with --spec23 (tap 23 only) are read the same way.
Usage: build_tap23_pack.py --dataset tartanair-v2 --out /group/.../_staging/lingbot-tap23pack-1048
       [--cache-name lingbot-cachefull --cache-root /group/compact-3dmem/datasets/_staging --desc]
"""
from __future__ import annotations

import argparse
import ast
import time
import zipfile
from pathlib import Path

import numpy as np

TAP23 = 3


def npz_member_memmap(path, key):
    """np.memmap of an uncompressed .npy member of an .npz"""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(f"{key}.npy")
        assert info.compress_type == 0, f"{path}:{key} is compressed"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        local = f.read(30)
        n_name, n_extra = int.from_bytes(local[26:28], "little"), int.from_bytes(local[28:30], "little")
        start = info.header_offset + 30 + n_name + n_extra
        f.seek(start)
        magic = f.read(8)
        assert magic[:6] == b"\x93NUMPY", path
        hlen = int.from_bytes(f.read(2 if magic[6] == 1 else 4), "little")
        hdr = ast.literal_eval(f.read(hlen).decode("latin1"))
        data = f.tell()
    assert not hdr["fortran_order"]
    return np.memmap(path, dtype=np.dtype(hdr["descr"]), mode="r", offset=data, shape=tuple(hdr["shape"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache-root", type=Path, default=Path("/data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-name", default="lingbot-cache1048")
    ap.add_argument("--desc", action="store_true", help="also pack the per-frame MegaLoc descriptors")
    args = ap.parse_args()
    files = sorted(f for f in (args.cache_root / f"{args.cache_name}-{args.dataset}").glob("*.npz") if ".tmp" not in f.name)
    lengths = [int(f.stem.rsplit("_n", 1)[1]) for f in files]
    N = sum(lengths)
    out = args.out / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    spec = np.lib.format.open_memmap(out / "spec23.tmp.npy", mode="w+", dtype=np.float16, shape=(N, 6, 2048))
    desc = np.lib.format.open_memmap(out / "desc.tmp.npy", mode="w+", dtype=np.float16, shape=(N, 8448)) if args.desc else None
    pose, gt, K = np.zeros((N, 9), np.float32), np.zeros((N, 4, 4), np.float32), np.zeros((N, 3, 3), np.float32)
    meta = {k: [] for k in ("name", "seq", "offset", "length", "hw", "gt_scale", "stride")}
    o, t0 = 0, time.time()
    for i, (f, n) in enumerate(zip(files, lengths)):
        names = zipfile.ZipFile(f).namelist()
        if "spec23.npy" in names:
            sp = npz_member_memmap(f, "spec23")
            assert sp.shape[0] == n, (f, sp.shape, n)
            spec[o:o + n] = sp
        else:
            sp = npz_member_memmap(f, "specials")
            assert sp.shape[0] == n, (f, sp.shape, n)
            spec[o:o + n] = sp[:, TAP23]
        if desc is not None:
            desc[o:o + n] = npz_member_memmap(f, "megaloc")
        z = np.load(f)
        pose[o:o + n], gt[o:o + n], K[o:o + n] = z["pose_enc"], z["gt_c2w"], z["K"]
        for k, v in (("name", f.stem), ("seq", str(z["seq"])), ("offset", o), ("length", n), ("hw", z["hw"]),
                     ("gt_scale", float(z["gt_scale"])), ("stride", int(z["stride"]))):
            meta[k].append(v)
        o += n
        if i % 100 == 0:
            print(f"[pack] {args.dataset} {i + 1}/{len(files)} clips, {o} frames, {time.time() - t0:.0f} s", flush=True)
    spec.flush()
    del spec
    if desc is not None:
        desc.flush()
        del desc
        (out / "desc.tmp.npy").rename(out / "desc.npy")
    np.savez(out / "index.npz", pose_enc=pose, gt_c2w=gt, K=K, dataset=args.dataset,
             **{k: np.array(v) for k, v in meta.items()})
    (out / "spec23.tmp.npy").rename(out / "spec23.npy")
    print(f"[pack] {args.dataset}: {len(files)} clips, {N} frames, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
