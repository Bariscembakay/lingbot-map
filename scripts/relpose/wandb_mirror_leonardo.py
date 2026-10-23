#!/usr/bin/env python3
"""Mirror Leonardo runs of train_lingbot_rel.py into wandb from INSAIT, where the wandb key stays (Leonardo's
compute nodes are offline). Each arm's curves are rebuilt from its job logs (training losses, every 10 steps) and
its history.json (validation: only ATE and RPE-t go to wandb), and only the steps the wandb run does not have yet
are appended, so it is safe to rerun at any time. A step that ends in a validation is held back until that
validation is in history.json (or training has gone past it), because wandb steps can only increase.
The trainer's own offline wandb dirs on Leonardo are not synced: the old runs there log every validation metric.
Usage: wandb_mirror_leonardo.py [--every SECONDS]   (pulls with rsync over `ssh leonardo`; 0 = one round)
"""
import argparse
import getpass
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import wandb

REMOTE = "/leonardo_scratch/large/userexternal/bbakay00"
LOCAL = Path(f"/scratch/{getpass.getuser()}/wandb_mirror_leonardo")
LOG = re.compile(r"lingbotrel_B_train_(.+)_a100x\d+_leonardo_(\d+)\.out$")
TRAIN = re.compile(r"^\[\s*(\d+)\] pose (\S+) (.*?)\s*\|g\| (\S+) ")
SSH = ["-e", "ssh -o BatchMode=yes -o ConnectTimeout=30"]


def pull():
    (LOCAL / "logs").mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-am", *SSH, "--include=*/", "--include=history.json", "--include=config.json",
                    "--exclude=*", f"leonardo:{REMOTE}/runs/lingbot_rel/backbone/", f"{LOCAL}/runs/"],
                   check=True, capture_output=True)
    subprocess.run(["rsync", "-a", *SSH, "--include=lingbotrel_B_train_*", "--exclude=*",
                    f"leonardo:{REMOTE}/logs/", f"{LOCAL}/logs/"], check=True, capture_output=True)


def rows(arm, val_every):
    train = {}
    logs = sorted((int(m[2]), p) for p in (LOCAL / "logs").iterdir() if (m := LOG.match(p.name)) and m[1] == arm)
    for _, p in logs:
        for line in p.read_text(errors="replace").splitlines():
            if m := TRAIN.match(line):
                kv = m[3].split()
                train[int(m[1])] = {"train/pose": float(m[2]), "train/grad_norm": float(m[4]),
                                    **{f"train/{k}": float(v) for k, v in zip(kv[::2], kv[1::2])}}
                # a resumed job redoes the steps after its checkpoint, so the latest job's values win
    val = {}
    hp = LOCAL / "runs" / arm / "history.json"
    for r in json.loads(hp.read_text()) if hp.exists() else []:
        d = {}
        for metric in ("ate", "rpe_t"):
            v = {k.rsplit("/", 1)[0]: x for k, x in r.items() if k.endswith(f"/{metric}")}
            d |= {f"val_{metric}/{ds}": x for ds, x in v.items()} | {f"val_{metric}/mean": sum(v.values()) / len(v)}
        val[r["step"]] = d
    last_train = max(train, default=-1)
    out = []
    for s in sorted(set(train) | set(val)):
        if s % val_every == 0 and s not in val and s >= last_train:
            break
        out.append((s, train.get(s, {}) | val.get(s, {})))
    return out


def mirror(api, arm):
    cfg = json.loads((LOCAL / "runs" / arm / "config.json").read_text())
    rs = rows(arm, int(cfg.get("val_every", 500)))
    rid = "lbB" + hashlib.md5(arm.encode()).hexdigest()[:8]
    try:
        start = api.run(f"{api.default_entity}/lingbot-rel/{rid}").lastHistoryStep + 1
    except wandb.errors.CommError:
        start = 0
    new = [(s, d) for s, d in rs if s >= start]
    if not new:
        return 0
    run = wandb.init(project="lingbot-rel", group="backbone", name=arm, id=rid, resume="allow", config=cfg,
                     dir=LOCAL, settings=wandb.Settings(silent=True))
    for s, d in new:
        run.log(d, step=s)
    run.finish()
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=900)
    args = ap.parse_args()
    while True:
        try:
            pull()
            api = wandb.Api()
            arms = sorted(p.parent.name for p in (LOCAL / "runs").glob("*/config.json"))
            print(f"[wandb_mirror] {time.strftime('%H:%M', time.gmtime())} "
                  + " ".join(f"{a[:24]}+{mirror(api, a)}" for a in arms if not a.startswith("_")), flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[wandb_mirror] {time.strftime('%H:%M', time.gmtime())} pull failed (Leonardo certificate?): "
                  f"{e.stderr.decode()[-200:].strip()}", flush=True)
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == "__main__":
    main()
