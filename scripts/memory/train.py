#!/usr/bin/env python3
"""Stage 2: train the summary memory on the precomputed teacher cache.

The aggregator never runs here -- only the two frozen heads. If this file ends up
importing the aggregator, something has gone wrong.

Structure per clip:

    for i in 0..L-1:
        S_i, refined_i = memory.step(x_i, S_{i-1}, write_tokens=x_{schedule(i)})
        every `head_every` steps: decode with the frozen heads and take a loss

Long unroll, sparse supervision: the recurrence is ~0.4 TFLOPs/frame and the DPT
head ~1.5, so putting many steps in the graph while decoding rarely buys a much
longer write gradient at the same activation budget. The write's *only* gradient
path is through future reads, so unroll length is the knob that matters most.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory import camera_bridge as cb                  # noqa: E402
from lingbot_map.memory import frozen                               # noqa: E402
from lingbot_map.memory import losses as L                          # noqa: E402
from lingbot_map.memory.data import ClipReader, find_clips, query_frame  # noqa: E402
from lingbot_map.memory.model import SummaryMemory                  # noqa: E402
from lingbot_map.memory.schedule import DISJOINT                    # noqa: E402


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="root holding one dir per clip")
    ap.add_argument("--val-cache", default=None)
    ap.add_argument("--heads", default="/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="dev")
    # architecture
    ap.add_argument("--num-slots", type=int, default=512)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--write-layers", type=int, default=2)
    ap.add_argument("--read-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--write-mode", default=DISJOINT, choices=["disjoint", "overlap"])
    ap.add_argument("--scale-frames", type=int, default=8)
    ap.add_argument("--sliding-window", type=int, default=64)
    ap.add_argument("--frozen-state", action="store_true",
                    help="control arm: identical read, state never written")
    ap.add_argument("--refine-taps", default="0,1,2,3",
                    help="which taps the read refines. '0,1,2,3' = arm A (depth+pose); "
                         "'3' = arm B (pose only -- the depth head's tap-23 branch is "
                         "inert in the published checkpoint)")
    ap.add_argument("--share-read", action="store_true",
                    help="one reader shared across refined taps instead of one each")
    # objective
    ap.add_argument("--with-camera", action="store_true",
                    help="add abs/rel pose terms via the teacher camera bridge")
    ap.add_argument("--force-untrusted-pose", action="store_true",
                    help="train pose terms even when the cache marks its GT poses "
                         "untrusted. Only for deliberate experiments")
    ap.add_argument("--no-depth-loss", action="store_true",
                    help="pose-only. Implied when tap 23 is the only refined tap, "
                         "since d(depth)/d(tap23) == 0 in the published head")
    ap.add_argument("--y-space-depth", action="store_true")
    ap.add_argument("--w-depth", type=float, default=1.0)
    ap.add_argument("--w-abs", type=float, default=5.0)
    ap.add_argument("--w-rel", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    # schedule
    ap.add_argument("--unroll", type=int, default=32, help="truncated BPTT length")
    ap.add_argument("--head-every", type=int, default=4)
    ap.add_argument("--max-updates", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--wandb-project", default="lingbot-summary-memory")
    ap.add_argument("--no-wandb", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = build_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    clips = find_clips(args.cache)
    if not clips:
        raise SystemExit(f"no clips under {args.cache}")
    val_clips = find_clips(args.val_cache) if args.val_cache else []
    print(f"{len(clips)} train clips, {len(val_clips)} val clips")

    depth_head, camera_head = frozen.load_frozen(args.heads, device, args.with_camera)
    refine_taps = tuple(int(t) for t in args.refine_taps.split(","))
    # Tap 23 cannot move depth: layer4_rn's output is 100% negative and
    # ReLU(inplace=True) annihilates it through the residual. Asking for a depth
    # loss with only tap 23 refined would optimise a constant.
    use_depth = not args.no_depth_loss and any(t != 3 for t in refine_taps)
    if not args.no_depth_loss and not use_depth:
        print("refine_taps is {3} only -> depth loss disabled (tap 23 is inert for depth)")
    memory = SummaryMemory(
        dim=args.dim, num_slots=args.num_slots, num_heads=args.num_heads,
        write_layers=args.write_layers, read_layers=args.read_layers,
        write_mode=args.write_mode, scale_frames=args.scale_frames,
        sliding_window=args.sliding_window, frozen_state=args.frozen_state,
        refine_taps=refine_taps, share_read=args.share_read,
    ).to(device)
    print(f"trainable {memory.num_trainable()/1e6:.1f} M  schedule lag {memory.schedule.lag}  "
          f"refine_taps {refine_taps}  depth_loss {use_depth}  camera {args.with_camera}")
    if not use_depth and not args.with_camera:
        raise SystemExit("nothing to optimise: depth loss disabled and --with-camera not set")
    if args.with_camera and not args.force_untrusted_pose:
        bad = [c.name for c in clips
               if not ClipReader(c).meta.gt_pose_trusted]
        if bad:
            raise SystemExit(
                f"{len(bad)}/{len(clips)} clips mark their GT poses untrusted "
                f"(residual above {1.5} deg against the model's own trajectory, "
                f"vs the 0.58-0.92 deg it achieves on 7-Scenes/TUM). A pose loss "
                f"on that target pushes toward the wrong answer. Fix the transform "
                f"or pass --force-untrusted-pose deliberately. e.g. {bad[:3]}")

    weights = L.LossWeights(depth=args.w_depth, abs_pose=args.w_abs,
                            rel_pose=args.w_rel, alpha=args.alpha)
    opt = torch.optim.AdamW(memory.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, s / args.max_updates))))
    )

    step = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        memory.load_state_dict(state["memory"]); opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"]); step = state["step"]
        print(f"resumed at update {step}")

    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.arm,
                         id=f"{args.arm}-{args.seed}", resume="allow",
                         config=vars(args), dir=os.environ.get("WANDB_DIR", "."))

    t_start = time.time()
    while step < args.max_updates:
        reader = ClipReader(clips[rng.integers(len(clips))], device=device)
        H, W = reader.hw
        teacher_cache, teacher_c2w = None, None
        if args.with_camera:
            teacher_cache, _ = cb.build_teacher_cache(
                camera_head, reader.teacher_camera_tokens())
            # The window's other poses are the teacher's own, converted once per
            # clip rather than per step.
            with torch.no_grad():
                teacher_c2w = L.pose_enc_to_c2w(
                    torch.from_numpy(reader.cache.pose_enc).to(device), (H, W))

        state = memory.new_state(1)
        pending, n_sup = [], 0
        for i in range(len(reader)):
            j = memory.schedule.frame_for_step(i)
            x = reader.token_dict(i, refine_taps)
            wt = reader.tokens(j) if j is not None else None
            state, refined = memory.step(state, x, write_tokens=wt)

            if i % args.head_every == 0 and j is not None:
                parts = {}
                # Reading all four taps costs 16.5 MB/frame and this loop is
                # I/O bound; the camera head only needs tap 23, so a pose-only arm
                # reads a quarter as much.
                head_in = (reader.head_inputs(i, refined) if use_depth
                           else [reader.rebuild_tap(i, 3, refined[3]).unsqueeze(1)])
                if use_depth:
                    d, c = depth_head(head_in,
                                      images=torch.zeros(1, 1, 3, H, W, device=device),
                                      patch_start_idx=reader.meta.patch_start_idx)
                    parts.update(L.depth_loss(d[0, ..., 0], c[0], reader.gt_depth(i),
                                              weights, y_space=args.y_space_depth))
                if args.with_camera:
                    pose = cb.pose_at(camera_head, teacher_cache, i,
                                      head_in[-1][0, :, 0].float())
                    pc2w = L.pose_enc_to_c2w(pose, (H, W))
                    parts.update(L.abs_pose_loss(pc2w, reader.gt_c2w([i]), weights))
                    # Relative-pose term over the local window. Only frame i's pose
                    # is differentiable (the bridge replays teacher values for the
                    # rest), so `anchor` restricts the sum to pairs involving i --
                    # gradient-equivalent to the paper's all-pairs form and k times
                    # cheaper.
                    lo = max(0, i - args.sliding_window + 1)
                    win = torch.cat([teacher_c2w[lo:i], pc2w], dim=0)
                    parts.update(L.rel_pose_loss(
                        win, reader.gt_c2w(slice(lo, i + 1)), weights,
                        anchor=win.shape[0] - 1))
                pending.append(L.total_loss(parts, weights))
                n_sup += 1
                if run is not None and n_sup % 8 == 0:
                    run.log({f"loss/{k}": float(v) for k, v in parts.items()} |
                            {"state/norm": float(state.norm()),
                             "revisit/frame": reader.revisit(i)}, step=step)

            if (i + 1) % args.unroll == 0 and pending:
                loss = torch.stack(pending).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(memory.parameters(), args.grad_clip)
                opt.step(); sched.step(); step += 1
                if run is not None:
                    run.log({"loss/total": float(loss), "grad_norm": float(gn),
                             "lr": sched.get_last_lr()[0],
                             "sec_per_update": (time.time() - t_start) / max(step, 1)},
                            step=step)
                state = state.detach()
                pending = []
                if step % args.ckpt_every == 0:
                    torch.save({"memory": memory.state_dict(), "opt": opt.state_dict(),
                                "sched": sched.state_dict(), "step": step,
                                "args": vars(args)}, ckpt_path)
                if step >= args.max_updates:
                    break

    torch.save({"memory": memory.state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict(), "step": step, "args": vars(args)},
               ckpt_path)
    print(f"done at update {step} in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
