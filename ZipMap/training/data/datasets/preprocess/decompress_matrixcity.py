#!/usr/bin/env python3
import os
import re
import sys
import shlex
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse

# !!! 
# After running this, go to ROOT, and run the following command to copy all transforms.json files:
# run "find ./ -name "transforms.json" -exec cp   --parents {} OUT_ROOT  \;" after decompressing
# find ./ -name "transforms.json" -exec cp   --parents {} /home/storage-bucket/haian/zipmap_data/processed_matrixcity/matrixcity_raw \; 

# --------- CONFIG ---------
# Point this to .../MatrixCity/big_city/aerial/train
ROOT = Path("/home/haian/codebase/zipmap_code/raw_data/MatrixCity")
OUT_ROOT = Path("/home/storage-bucket/haian/zipmap_data/processed_matrixcity/matrixcity_raw")
# How many parallel workers to use:
NUM_PROCS = max(1, min(cpu_count(), 64))  # cap at 64 by default
# --------------------------

SPLIT_RE = re.compile(r"^(?P<base>.+\.tar)(?P<part>\d{2,})$")  # matches .tar00, .tar01, ...

def find_archives(root: Path, out_root: Path = None, recursive: bool = False, layout: str = "archive-dir"):
    """
    Return a list of 'jobs'. Each job is:
      {
        'name': 'big_high_block_1' or 'subdir/big_high_block_1' when recursive,
        'parts': [Path(...tar00), Path(...tar01), ...] OR [Path(...tar)] (single),
        'dest': Path(.../subdir[/big_high_block_1] depending on layout),
        'marker': '.extracted_ok' or unique marker when sharing same dest
      }
    layout:
      - 'archive-dir' (default): extract into OUT_ROOT/<rel_dir>/<archive_base>/
      - 'preserve':     extract into OUT_ROOT/<rel_dir>/ (keep original folder structure without adding archive_base)
    """
    if out_root is None:
        out_root = root

    by_base = {}  # base name (full path without .tar?? suffix) -> list of part Paths
    singles = []  # regular .tar files

    iterator = root.rglob("*.tar*") if recursive else root.glob("*.tar*")
    for p in iterator:
        if p.is_dir():
            continue
        m = SPLIT_RE.match(p.name)
        if m:
            # Use the file's own directory for the base to avoid collisions across subdirs
            base = (p.parent / m.group("base")).as_posix()
            by_base.setdefault(base, []).append(p)
        elif p.suffix == ".tar":
            singles.append(p)

    jobs = []

    def _rel_under_root(path: Path) -> Path:
        try:
            rel = path.relative_to(root)
            return Path("") if str(rel) == "." else rel
        except Exception:
            # Fallback to a sanitized absolute segment to avoid collisions
            sanitized = path.as_posix().lstrip("/").replace("/", "__")
            return Path(sanitized)

    def _safe_marker(rel_dir: Path, base_name: str) -> str:
        # Produce a marker unique per-archive when extracting multiple into same dest
        tag = (rel_dir / base_name).as_posix() if str(rel_dir) else base_name
        tag = tag.replace("/", "__")
        return f".extracted_ok__{tag}"

    # Handle split archives
    for base, parts in by_base.items():
        parts_sorted = sorted(parts, key=lambda x: int(SPLIT_RE.match(x.name).group("part")))
        base_path = Path(base)
        block_name = base_path.stem  # remove ".tar" -> e.g. big_high_block_1
        rel_dir = _rel_under_root(base_path.parent)
        if layout == "preserve":
            dest = out_root / rel_dir if str(rel_dir) else out_root
            marker = _safe_marker(rel_dir, block_name)
            name = str((rel_dir / block_name).as_posix()) if str(rel_dir) else block_name
        else:  # archive-dir
            dest = out_root / rel_dir / block_name if str(rel_dir) else out_root / block_name
            marker = ".extracted_ok"
            name = str((rel_dir / block_name).as_posix()) if str(rel_dir) else block_name
        jobs.append({"name": name, "parts": parts_sorted, "dest": dest, "marker": marker})

    # Handle singles
    for t in singles:
        block_name = t.stem
        rel_dir = _rel_under_root(t.parent)
        if layout == "preserve":
            dest = out_root / rel_dir if str(rel_dir) else out_root
            marker = _safe_marker(rel_dir, block_name)
            name = str((rel_dir / block_name).as_posix()) if str(rel_dir) else block_name
        else:
            dest = out_root / rel_dir / block_name if str(rel_dir) else out_root / block_name
            marker = ".extracted_ok"
            name = str((rel_dir / block_name).as_posix()) if str(rel_dir) else block_name
        jobs.append({"name": name, "parts": [t], "dest": dest, "marker": marker})

    return jobs

def already_extracted(dest: Path, marker_name: str = ".extracted_ok") -> bool:
    return (dest / marker_name).exists()

def mark_done(dest: Path, marker_name: str = ".extracted_ok"):
    (dest / marker_name).write_text("ok\n")

def extract_with_tar(parts, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)

    if len(parts) == 1 and parts[0].suffix == ".tar":
        # single tar: tar -xf file.tar -C dest
        cmd = f"tar -xf {shlex.quote(parts[0].as_posix())} -C {shlex.quote(dest.as_posix())}"
    else:
        # split parts: cat part* | tar -xf - -C dest
        # Build a deterministic list (already sorted)
        cat_list = " ".join(shlex.quote(p.as_posix()) for p in parts)
        cmd = f"bash -lc 'cat {cat_list} | tar -xf - -C {shlex.quote(dest.as_posix())}'"

    # Run
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise RuntimeError(f"Extraction failed (code {res.returncode}) for {dest}")

def worker(job):
    name, parts, dest = job["name"], job["parts"], job["dest"]
    marker = job.get("marker", ".extracted_ok")
    force = job.get("force", False)
    if not force and already_extracted(dest, marker):
        return f"[SKIP] {name} (already extracted)"
    try:
        extract_with_tar(parts, dest)
        mark_done(dest, marker)
        return f"[OK]   {name}"
    except Exception as e:
        return f"[FAIL] {name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Extract MatrixCity archives, supporting split .tar parts.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Directory to scan for .tar and split parts (.tar00, .tar01, ...)")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT, help="Directory to extract into. Defaults to OUT_ROOT in script.")
    parser.add_argument("--recursive", action="store_true", help="Search for archives recursively under --root")
    parser.add_argument("--layout", choices=["archive-dir", "preserve"], default="archive-dir", help="Destination layout: 'archive-dir' creates OUT_ROOT/<rel_dir>/<archive_base>/; 'preserve' keeps OUT_ROOT/<rel_dir>/ only")
    parser.add_argument("--procs", type=int, default=NUM_PROCS, help="Number of parallel workers")
    parser.add_argument("--force", action="store_true", help="Re-extract even if marker exists")
    args = parser.parse_args()

    scan_root: Path = args.root
    out_root: Path = args.out_root or scan_root

    if not scan_root.exists():
        print(f"Root not found: {scan_root}", file=sys.stderr)
        sys.exit(1)

    jobs = find_archives(scan_root, out_root, recursive=args.recursive, layout=args.layout)
    if not jobs:
        hint = " Try --recursive if your archives are in subfolders." if not args.recursive else ""
        print(f"No .tar or split .tar?? files found under {scan_root}.{hint}")
        return

    # propagate flags to workers
    for j in jobs:
        j["force"] = args.force

    print(f"Discovered {len(jobs)} archives under {scan_root}\nDestination: {out_root}\nLayout: {args.layout}")
    for j in jobs:
        parts_desc = f"{len(j['parts'])} part(s)"
        print(f"  - {j['name']:>20}  -> {j['dest'].as_posix()}/  ({parts_desc})")

    with Pool(processes=args.procs) as pool:
        for msg in pool.imap_unordered(worker, jobs):
            print(msg)

if __name__ == "__main__":
    main()

"""
Notes & Tips
------------
1) Change ROOT at the top to your path, e.g.:
   ROOT = Path("/share/.../MatrixCity/big_city/aerial/train")

2) You can control the extraction destination with --out-root (defaults to OUT_ROOT in this script), e.g. extract under /fast_scratch while scanning --root for archives.

3) If your archives are nested in subfolders, pass --recursive to search below --root.

4) The script creates a '.extracted_ok' file inside each extracted folder. 
   Delete that marker if you want to re-extract.

5) If your environment lacks 'tar' (rare on Linux/HPC):
   - You can swap 'extract_with_tar' for a pure-Python version using the 'tarfile'
     module and a streaming reader for split parts. 'tarfile' is slower.

6) Safety: The script won’t delete archives. If you want to clean up after success,
   add os.remove(...) where appropriate.

7) You can raise --procs for faster I/O if your disk can handle it.

8) To avoid false 'already extracted' skips when different archives share the same filename in different subfolders, this script mirrors the source subdirectory structure under the output root. That is, big_high_block_1.tar inside sub/ will extract to OUT_ROOT/sub/big_high_block_1/.

- To avoid false 'already extracted' skips when multiple archives extract into the same folder, the script now supports two layouts:
  * archive-dir (default): OUT_ROOT/<rel_dir>/<archive_base>/ with a single .extracted_ok per dest
  * preserve: OUT_ROOT/<rel_dir>/ keeping original folder structure; a unique marker .extracted_ok__<rel_dir__archive_base> is used per archive
- Use --recursive to discover archives in subfolders.
- Use --force to re-extract even if a marker exists.
"""
