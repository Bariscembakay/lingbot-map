#!/usr/bin/env python3
"""Structural checks on `StateMemory`, runnable on CPU with a tiny config.

These are not accuracy tests. Each one asserts a property the design depends on
and that a plausible-looking loss curve would not reveal. V1 is the load-bearing
one: if the probe's gradient does not reach the write that ingested the queried
frame, the objective cannot teach recall and nothing else matters.

Usage:  validate_state.py [--only V1,V3] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory.cut3r_state import (  # noqa: E402
    DEC_DIM, TAP_DIM, StateMemory, grid_positions,
)

PH, PW = 4, 6          # tiny patch grid; the DPT pyramid halves and re-upsamples
PATCH = 14
HW = (PH * PATCH, PW * PATCH)
NSPECIAL = 6
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def tiny_model(device) -> StateMemory:
    torch.manual_seed(0)
    m = StateMemory(patch_size=PATCH, state_tokens=64, dec_depth=2,
                    dec_num_heads=4, state_dec_num_heads=4,
                    patch_start_idx=NSPECIAL)
    return m.to(device).train()


def fake_taps(n: int, device) -> list[torch.Tensor]:
    """One cached tap-23 tensor per frame, each independently differentiable."""
    return [torch.randn(1, NSPECIAL + PH * PW, TAP_DIM, device=device,
                        requires_grad=True) for _ in range(n)]


def fake_raymap(device) -> torch.Tensor:
    return torch.randn(1, 6, *HW, device=device)


def rollout(m, taps, device):
    state, spos = m.init_state(1, device)
    states = [state]
    for t in taps:
        state = m.write(state, spos, t, (PH, PW))
        states.append(state)
    return states, spos


# --------------------------------------------------------------------------
def v1_gradient_reaches_early_write(m, device) -> None:
    """A probe at time t must credit the write at step q, for q far behind t.

    Attribution works because each frame's tap is a separate leaf: if
    d(probe_loss)/d(tap_q) is non-zero, the gradient necessarily traversed the
    write at step q and every write between q and t.
    """
    T, q = 8, 0
    taps = fake_taps(T, device)
    states, spos = rollout(m, taps, device)
    out = m.probe(states[T], spos, fake_raymap(device), HW)
    out["pts3d_in_self_view"].square().mean().backward()

    grads = [(0.0 if t.grad is None else t.grad.abs().max().item()) for t in taps]
    ok = grads[q] > 0 and all(g > 0 for g in grads)
    check("V1 probe gradient reaches the write at step q << t",
          ok, f"lag {T - q} | grad@q={grads[q]:.3e} min over steps={min(grads):.3e}")


def v2_gradient_reaches_state_prior(m, device) -> None:
    m.zero_grad(set_to_none=True)
    taps = fake_taps(4, device)
    states, spos = rollout(m, taps, device)
    m.probe(states[-1], spos, fake_raymap(device), HW)["pts3d_in_self_view"] \
        .square().mean().backward()
    g = m.register_tokens.weight.grad
    check("V2 probe gradient reaches the state prior s0",
          g is not None and g.abs().max().item() > 0,
          f"max|grad|={0.0 if g is None else g.abs().max().item():.3e}")


def v3_state_actually_matters(m, device) -> None:
    """Zeroing the state must change the probe. A dead path trains happily."""
    taps = fake_taps(4, device)
    states, spos = rollout(m, taps, device)
    ray = fake_raymap(device)
    with torch.no_grad():
        live = m.probe(states[-1], spos, ray, HW)["pts3d_in_self_view"]
        dead = m.probe(torch.zeros_like(states[-1]), spos, ray, HW)["pts3d_in_self_view"]
    d = (live - dead).abs().mean().item()
    check("V3 zeroing the state changes the probe output", d > 1e-6,
          f"mean|live-dead|={d:.4e}")


def v4_head_output_shape(m, device) -> None:
    taps = fake_taps(2, device)
    states, spos = rollout(m, taps, device)
    with torch.no_grad():
        out = m.probe(states[-1], spos, fake_raymap(device), HW)
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    ok = all(v[1:3] == HW for v in shapes.values())
    check("V4 head output is exactly (H, W) after the terminal interpolate",
          ok, f"target={HW} got={ {k: v[1:3] for k, v in shapes.items()} }")


def v5_token_bookkeeping(m, device) -> None:
    ray = fake_raymap(device)
    tokens, pos = m.raymap(ray)
    n_patch = PH * PW
    ok_probe = tokens.shape[1] == 1 + n_patch and pos.shape[1] == 1 + n_patch
    wpos = grid_positions(PH, PW, NSPECIAL, 1, device)
    ok_write = wpos.shape[1] == NSPECIAL + n_patch
    # Every non-patch token shares one RoPE position, deliberately.
    ok_special = bool((wpos[0, :NSPECIAL] == -1).all())
    check("V5 token counts and special-token positions", ok_probe and ok_write and ok_special,
          f"probe={tokens.shape[1]}=1+{n_patch} write={wpos.shape[1]}={NSPECIAL}+{n_patch} "
          f"specials at -1: {ok_special}")


def v6_probe_cannot_see_the_taps(m, device) -> None:
    """With the state detached, a probe must have NO path to any tap.

    This is what makes the probe a measurement of the state and nothing else:
    the raymap path never touches the aggregator.
    """
    m.zero_grad(set_to_none=True)
    taps = fake_taps(4, device)
    states, spos = rollout(m, taps, device)
    out = m.probe(states[-1].detach(), spos, fake_raymap(device), HW)
    out["pts3d_in_self_view"].square().mean().backward()
    leaked = [i for i, t in enumerate(taps)
              if t.grad is not None and t.grad.abs().max().item() > 0]
    check("V6 probe reaches the taps ONLY through the state", not leaked,
          f"leaking taps: {leaked or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    device = torch.device(args.device)

    tests = {
        "V1": v1_gradient_reaches_early_write,
        "V2": v2_gradient_reaches_state_prior,
        "V3": v3_state_actually_matters,
        "V4": v4_head_output_shape,
        "V5": v5_token_bookkeeping,
        "V6": v6_probe_cannot_see_the_taps,
    }
    keep = [k.strip() for k in args.only.split(",") if k.strip()] or list(tests)
    for name in keep:
        m = tiny_model(device)      # fresh model per test, so grads never bleed
        tests[name](m, device)

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed"
          + (f" -- FAILED: {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
