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
from lingbot_map.memory.probe_data import (  # noqa: E402
    build_raymap, gt_pointmaps, ray_depth, relative_c2w,
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


def v7_raymap_geometry(m, device) -> None:
    """The raymap and the GT pointmap must describe the same rays.

    Checked positionally, not by magnitude: ||t_w2c|| == ||t_c2w||, so a norm
    comparison is blind to the exact inversion error that has bitten this repo
    before. Under the `true` convention every GT point must lie ON its ray, so
    the residual is a hard zero; under CUT3R's convention it must NOT, and the
    size of that violation is the quirk we deliberately inherit.
    """
    torch.manual_seed(0)
    h, w = 24, 32
    B = 3
    K = torch.tensor([[120.0, 0, w / 2], [0, 120.0, h / 2], [0, 0, 1]],
                     device=device).expand(B, 3, 3).contiguous()
    c2w_0 = torch.eye(4, device=device)[None]
    c2w_q = torch.eye(4, device=device).repeat(B, 1, 1)
    for i in range(B):                       # distinct rotations and translations
        a = 0.3 * (i + 1)
        c2w_q[i, :3, :3] = torch.tensor(
            [[torch.cos(torch.tensor(a)), 0, torch.sin(torch.tensor(a))],
             [0, 1, 0],
             [-torch.sin(torch.tensor(a)), 0, torch.cos(torch.tensor(a))]], device=device)
        c2w_q[i, :3, 3] = torch.tensor([0.4 * (i + 1), -0.2, 0.7 * (i + 1)], device=device)
    depth = 1.0 + torch.rand(B, h, w, device=device)

    x_self, x_world, valid = gt_pointmaps(depth, K, c2w_q, c2w_0)

    # (a) origin channel must equal the RELATIVE camera centre, positionally.
    rm_true = build_raymap(K, c2w_q, c2w_0, h, w, convention="true")
    t_rel = relative_c2w(c2w_q, c2w_0.expand(B, -1, -1))[:, :3, 3]
    o = rm_true[:, :3, 0, 0]
    err_o = (o - t_rel).abs().max().item()
    # what an inverted pose would have produced, for contrast
    t_bad = torch.linalg.inv(relative_c2w(c2w_q, c2w_0.expand(B, -1, -1)))[:, :3, 3]
    contrast = (t_bad - t_rel).abs().max().item()

    # (b) under `true`, every GT point lies on its ray: X = o + s*d, s > 0.
    d = rm_true[:, 3:].permute(0, 2, 3, 1)
    ovec = rm_true[:, :3].permute(0, 2, 3, 1)
    v = x_world - ovec
    s = (v * d).sum(-1, keepdim=True)
    resid = (v - s * d).norm(dim=-1)[valid]
    rel = (resid / v.norm(dim=-1)[valid].clamp_min(1e-9)).max().item()
    ahead = bool((s.squeeze(-1)[valid] > 0).all())

    # (c) X_self must round-trip to X_world through the relative pose.
    c2w = relative_c2w(c2w_q, c2w_0.expand(B, -1, -1))
    xs = x_self.reshape(B, -1, 3).transpose(1, 2)
    rt = (c2w[:, :3, :3] @ xs + c2w[:, :3, 3][:, :, None]).transpose(1, 2)
    err_rt = (rt.reshape(x_world.shape) - x_world).abs().max().item()

    # (d) CUT3R's convention must MISS the ray -- that is the inherited quirk.
    rm_c = build_raymap(K, c2w_q, c2w_0, h, w, convention="cut3r")
    dc = rm_c[:, 3:].permute(0, 2, 3, 1)
    ang = torch.rad2deg(torch.arccos(
        (dc * d).sum(-1).clamp(-1, 1)))[valid].median().item()

    ok = err_o < 1e-5 and contrast > 1e-3 and rel < 1e-4 and ahead and err_rt < 1e-4
    check("V7 raymap geometry: origins positional, GT points on the ray", ok,
          f"origin err {err_o:.2e} (inverted would be {contrast:.3f}) | "
          f"on-ray resid {rel:.2e} | s>0 {ahead} | self->world {err_rt:.2e} | "
          f"cut3r-vs-true median angle {ang:.2f} deg")


def v8_ray_depth_convention_free(m, device) -> None:
    """Ray depth must not depend on the direction-channel convention.

    This is what keeps the readable metric comparable across
    --raymap-convention, since only the origin channel enters it.
    """
    h, w = 16, 20
    K = torch.tensor([[100.0, 0, w / 2], [0, 100.0, h / 2], [0, 0, 1]],
                     device=device)[None]
    c2w_0 = torch.eye(4, device=device)[None]
    c2w_q = torch.eye(4, device=device)[None].clone()
    c2w_q[0, :3, 3] = torch.tensor([0.9, -0.1, 1.4], device=device)
    depth = 1.0 + torch.rand(1, h, w, device=device)
    _, x_world, _ = gt_pointmaps(depth, K, c2w_q, c2w_0)
    d1 = ray_depth(x_world, build_raymap(K, c2w_q, c2w_0, h, w, "cut3r"))
    d2 = ray_depth(x_world, build_raymap(K, c2w_q, c2w_0, h, w, "true"))
    err = (d1 - d2).abs().max().item()
    check("V8 ray depth is independent of the direction convention",
          err < 1e-6, f"max diff {err:.2e}")


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
        "V7": v7_raymap_geometry,
        "V8": v8_ray_depth_convention_free,
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
