#!/usr/bin/env python3
"""Stream a train_pose_refine.py log into wandb (for runs started without wandb).

Parses "[  step] loss ... t ... |g| ..." and "[val step] {...}" lines and logs them at
their steps into the run named after the arm. The run id is derived from the name, so
re-running resumes the same wandb run instead of duplicating it. With --follow, keeps
tailing until the Slurm job leaves the queue.
Usage: wandb_sync_log.py <log> <arm name> [--job JOBID --follow]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import re
import subprocess
import time
from pathlib import Path

import wandb

TRAIN = re.compile(r"^\[\s*(\d+)\] loss ([\d.]+) t (\d+) \|g\| ([\d.]+)")
VAL = re.compile(r"^\[val (\d+)\] (\{.*\})$")


def alive(job):
    return bool(subprocess.run(["squeue", "-h", "-j", str(job)], capture_output=True, text=True).stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("name")
    ap.add_argument("--job")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--project", default="spatial_memory")
    args = ap.parse_args()
    rid = "pr-" + hashlib.sha1(args.name.encode()).hexdigest()[:12]
    run = wandb.init(project=args.project, name=args.name, id=rid, resume="allow")
    last = run.summary.get("_step", -1) if run.resumed else -1
    done = 0
    while True:
        lines = args.log.read_text(errors="ignore").splitlines()
        for line in lines[done:]:
            m = TRAIN.match(line)
            if m and int(m[1]) > last:
                last = int(m[1])
                run.log({"loss": float(m[2]), "t": int(m[3]), "grad_norm": float(m[4])}, step=last)
                continue
            m = VAL.match(line)
            if m:
                row = ast.literal_eval(m[2])
                step = int(m[1])
                vals = {k: v for k, v in row.items() if k != "step"}
                if step >= last:
                    last = step
                    run.log(vals, step=step)
                else:  # val lines print after the train line of the same step
                    run.log(vals, step=last)
        done = len(lines)
        if not (args.follow and args.job and alive(args.job)):
            break
        time.sleep(60)
    run.finish()


if __name__ == "__main__":
    main()
