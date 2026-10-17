#!/usr/bin/env python3
"""/group/compact-3dmem tidy agreed 2026-10-11. Dry run unless --apply.

  python tidy_20261011.py            # print every delete / move / relink
  python tidy_20261011.py --apply    # do it, appending to tidy_20261011.log

Moves are renames inside one cephfs (instant, reversible by moving back).
Deletes are not reversible. Refuses to run while any of your jobs is RUNNING
with a work dir or args under a path it touches.
"""
import os
import shutil
import subprocess
import sys
import time

G = "/group/compact-3dmem"
C = f"{G}/campaigns"
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tidy_20261011.log")

DELETE = [
    # 12 single-scene overfit/debug training runs
    *(f"{C}/spatial_memory/{d}" for d in (
        "overfit1scene_16f_smallread", "overfit1scene_16f_smallread_nowrite",
        "overfit1scene_96f_smallread", "overfit1scene_96f_smallread_nowrite",
        "overfit_1scene_2k", "overfit_1scene_2k_h200", "overfit_1scene_nowrite",
        "overfit_1scene_probecur_off", "overfit_1scene_raymap_true",
        "overfit_1scene_rtx", "overfit_1scene_taps_all", "overfit_1scene_tbptt8")),
    # per-lag debug clouds
    f"{C}/spatial_memory/scenes32_16f_b4/val_step4000_16frames/_lags",
    f"{C}/spatial_memory/scenes32_16f_b4/val_step4000_160frames/_lags",
    # smoke / sanity / render scratch
    f"{C}/lingbot_map/msp3_test",
    f"{C}/lingbot_map/_sanity",
    f"{C}/lingbot_map/_render",
    # empty
    f"{C}/summary-memory",
    f"{G}/datasets/TAT_images_incoming",
]

# old -> new. BSS workspaces keep their internal <benchmark>/<scene>/<method>
# order; only the workspace root moves into a campaign named after the question.
MOVES = [
    *((f"{C}/lingbot_map/{a}", f"{C}/reproduction/{a}") for a in (
        "default", "dense_s1", "sparse_s12", "depthproj", "tum_fr1",
        "vo_useproj_s1", "direct_useproj_s1")),
    (f"{C}/lingbot_map/context_token_ablation", f"{C}/context_token_ablation/default"),
    *((f"{C}/lingbot_map/{a}", f"{C}/pose_oracle/{a}") for a in (
        "gtpose_oracle", "predpose_rerun", "gtdepth_predpose",
        "gtpose_framescale", "gtpose_gtintr")),
    # cross-zone comparability check: metrics only (24 MB)
    (f"{C}/lingbot_map/msp3_run", f"{G}/archive/validation/lingbot_msp3_zone_check"),
    # derived viewer cache + alias farm: "_" keeps it out of every catalog walk
    (f"{C}/reproduction/viewer_workspace", f"{C}/reproduction/_viewer_workspace"),
]
# The pose-oracle arms and the viewer farm hold ABSOLUTE symlinks into
# lingbot_map/default etc., which a rename leaves dangling.
RELINK_ROOTS = [f"{C}/reproduction", f"{C}/pose_oracle", f"{C}/context_token_ablation"]
RMDIR_AFTER = [f"{C}/lingbot_map"]


def log(msg, apply):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {'' if apply else '[dry] '}{msg}"
    print(line)
    if apply:
        with open(LOG, "a") as f:
            f.write(line + "\n")


def rbytes(p):
    try:
        return int(os.getxattr(p, "ceph.dir.rbytes")) if os.path.isdir(p) else os.path.getsize(p)
    except OSError:
        return 0


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024 or u == "T":
            return f"{n:.1f}{u}"
        n /= 1024


def running_job_conflicts():
    out = subprocess.run(["squeue", "--me", "-h", "-t", "R", "-o", "%i %Z %o"],
                         capture_output=True, text=True).stdout
    touched = [p for p in DELETE] + [o for o, _ in MOVES]
    return [l for l in out.splitlines() if any(t in l for t in touched)]


def main():
    apply = "--apply" in sys.argv
    if apply and (bad := running_job_conflicts()):
        sys.exit("refusing, running jobs touch these paths:\n" + "\n".join(bad))

    freed = 0
    for p in DELETE:
        if not os.path.lexists(p):
            log(f"SKIP missing {p}", apply)
            continue
        b = rbytes(p)
        freed += b
        log(f"DELETE {human(b):>7} {p}", apply)
        if apply:
            shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)

    prefix_map = []
    for old, new in MOVES:
        if not os.path.lexists(old):
            log(f"SKIP missing {old}", apply)
            continue
        if os.path.lexists(new):
            sys.exit(f"target exists, aborting before touching it: {new}")
        log(f"MOVE {old} -> {new}", apply)
        prefix_map.append((old, new))
        if apply:
            os.makedirs(os.path.dirname(new), exist_ok=True)
            os.rename(old, new)

    # Longest prefix first, so lingbot_map/default_x can never match lingbot_map/default.
    prefix_map.sort(key=lambda kv: -len(kv[0]))
    n_relink = n_dangling = 0
    for root in RELINK_ROOTS:
        # Dry run walks the pre-move location of anything that will land here.
        walk_roots = [root] if apply else [o for o, n in prefix_map if n.startswith(root + "/")] + [root]
        for wr in walk_roots:
            if not os.path.isdir(wr):
                continue
            for dp, dns, fns in os.walk(wr):
                for name in dns + fns:
                    p = os.path.join(dp, name)
                    if not os.path.islink(p):
                        continue
                    t = os.readlink(p)
                    for old, new in prefix_map:
                        if t == old or t.startswith(old + "/"):
                            nt = new + t[len(old):]
                            n_relink += 1
                            if apply:
                                os.remove(p)
                                os.symlink(nt, p)
                                if not os.path.exists(p):
                                    n_dangling += 1
                                    log(f"DANGLING after relink {p} -> {nt}", apply)
                            elif n_relink <= 5:
                                log(f"RELINK {p}: {t} -> {nt}", apply)
                            break
    log(f"RELINK total {n_relink} symlinks ({n_dangling} dangling)", apply)

    for d in RMDIR_AFTER:
        if apply and os.path.isdir(d):
            left = os.listdir(d)
            if left:
                log(f"KEEP {d}, still holds {left}", apply)
            else:
                os.rmdir(d)
                log(f"RMDIR {d}", apply)
        elif not apply:
            log(f"RMDIR {d} (if empty)", apply)

    log(f"freed {human(freed)}", apply)


if __name__ == "__main__":
    main()
