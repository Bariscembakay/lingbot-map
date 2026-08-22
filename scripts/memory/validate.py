#!/usr/bin/env python3
"""The Phase-4 gate. Nothing runs on a dev set until this is green.

Each check exists because a specific silent failure is possible: a transposed
patch grid, a detached graph, a lag off by one, an unfrozen head, a
non-resumable run. All are cheap to test and expensive to discover in a loss
curve.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory import camera_bridge as cb                      # noqa: E402
from lingbot_map.memory import frozen                                   # noqa: E402
from lingbot_map.memory import losses as L                              # noqa: E402
from lingbot_map.memory.data import ClipReader, find_clips              # noqa: E402
from lingbot_map.memory.model import SummaryMemory                      # noqa: E402
from lingbot_map.memory.schedule import DISJOINT, OVERLAP, WriteSchedule, coverage_report  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            try:
                msg = fn(*a, **k)
                RESULTS.append((name, True, msg or ""))
            except AssertionError as e:
                RESULTS.append((name, False, str(e)))
            except Exception:
                RESULTS.append((name, False, traceback.format_exc(limit=3)))
        return wrapped
    return deco


@check("V1  cached taps -> frozen DPT head reproduces cached depth")
def v1(reader, depth_head, device):
    i = len(reader) // 2
    H, W = reader.hw
    taps = [reader._t(reader.cache.taps[i, t]).unsqueeze(0).unsqueeze(0) for t in range(4)]
    with torch.no_grad():
        d, _ = depth_head(taps, images=torch.zeros(1, 1, 3, H, W, device=device),
                          patch_start_idx=reader.meta.patch_start_idx)
    got = d[0, 0, ..., 0]
    want = reader._t(reader.cache.depth[i])[0] if reader.cache.depth[i].ndim == 3 \
        else reader._t(reader.cache.depth[i])
    rel = ((got - want).abs() / want.abs().clamp_min(1e-3)).median().item()
    assert rel < 5e-3, f"median relative error {rel:.2e} (fp16 cache tolerance)"
    return f"median rel err {rel:.2e}"


@check("V1b y-space inverse round-trips through the head's exp activation")
def v1b():
    # The depth head uses activation="exp", not the DPTHead default "inv_log",
    # so the pre-activation inverse is log(depth) and depth is strictly positive.
    d = torch.tensor([0.05, 0.3, 0.67, 1.58, 12.0])
    back = torch.exp(frozen.depth_to_preactivation(d))
    err = ((back - d).abs() / d).max().item()
    assert err < 1e-5, f"max relative err {err}"
    return f"max rel err {err:.2e}"


@check("V1d camera bridge reproduces teacher poses exactly, cache unmutated")
def v1d(device):
    head = frozen.build_camera_head().to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)
    toks = torch.randn(24, 2048, device=device)
    cache, teacher = cb.build_teacher_cache(head, toks)
    before = cache[0]["k_0"].shape
    errs = [(cb.pose_at(head, cache, i, toks[i].view(1, -1))[0] - teacher[i]).abs().max().item()
            for i in (1, 7, 23)]
    assert max(errs) == 0.0, f"errors {errs}"
    tok = toks[11].view(1, -1).clone().requires_grad_(True)
    cb.pose_at(head, cache, 11, tok).sum().backward()
    assert tok.grad is not None and tok.grad.abs().sum() > 0, "no gradient to the token"
    assert all(p.grad is None for p in head.parameters()), "gradient leaked into the frozen head"
    assert cache[0]["k_0"].shape == before, "teacher cache was mutated"
    return f"max err {max(errs)}"


@check("V2  patch grid is row-major (perturbation lands at the expected pixel)")
def v2(reader, depth_head, device):
    i = len(reader) // 2
    H, W = reader.hw
    ph, pw = reader.meta.patch_h, reader.meta.patch_w
    taps = [reader._t(reader.cache.taps[i, t]).unsqueeze(0).unsqueeze(0) for t in range(4)]
    with torch.no_grad():
        base, _ = depth_head(taps, images=torch.zeros(1, 1, 3, H, W, device=device),
                             patch_start_idx=reader.meta.patch_start_idx)
    r, c = ph // 4, pw // 2   # must differ, or the transposed candidate coincides
    # Perturb a tap that actually influences depth. Tap 3 is inert in the
    # published head (see V13), so perturbing it leaves only numerical noise and
    # the argmax lands wherever rounding puts it.
    probe = 2
    tok = reader.meta.patch_start_idx + r * pw + c
    taps[probe] = taps[probe].clone()
    # Scaled to the data (real tap activations have per-token norm ~460) and
    # applied as noise, not a uniform offset: the head LayerNorms its input, which
    # cancels a constant added across all channels of a token exactly.
    tokv = taps[probe][0, 0, tok]
    taps[probe][0, 0, tok] = tokv + 20.0 * tokv.std() * torch.randn_like(tokv)
    with torch.no_grad():
        pert, _ = depth_head(taps, images=torch.zeros(1, 1, 3, H, W, device=device),
                             patch_start_idx=reader.meta.patch_start_idx)
    diff = (pert - base)[0, 0, ..., 0].abs()
    # Average into the patch grid before locating the peak: the DPT head upsamples
    # and its fusion blurs, so a per-pixel argmax is noisy.
    gh, gw = H // ph, W // pw
    grid = diff[:ph * gh, :pw * gw].reshape(ph, gh, pw, gw).mean(dim=(1, 3))
    gr, gc = divmod(int(grid.argmax()), pw)
    py, px = int((gr + 0.5) * H / ph), int((gc + 0.5) * W / pw)
    ey, ex = int((r + 0.5) * H / ph), int((c + 0.5) * W / pw)
    # A column-major grid would put the peak at the transposed patch instead, so
    # compare the two candidates rather than using a loose radius.
    ty, tx = int((c + 0.5) * H / ph), int((r + 0.5) * W / pw)
    d_row = ((py - ey) ** 2 + (px - ex) ** 2) ** 0.5
    d_col = ((py - ty) ** 2 + (px - tx) ** 2) ** 0.5
    assert d_row < d_col, (f"peak ({py},{px}) is closer to the transposed location "
                           f"({ty},{tx}) than to row-major ({ey},{ex})")
    return f"peak ({py},{px}); row-major {d_row:.0f}px vs transposed {d_col:.0f}px"


@check("V5b disjoint schedule tiles the sequence: no gaps, no overlaps")
def v5b():
    rep = coverage_report(WriteSchedule(DISJOINT, 8, 64), 320)
    assert rep["steps_with_gap"] == 0 and rep["steps_with_overlap"] == 0, str(rep)
    ov = coverage_report(WriteSchedule(OVERLAP, 8, 64), 320)
    assert ov["steps_with_overlap"] > 0, "overlap mode should overlap"
    return f"lag {rep['lag']}, {rep['num_writes']} writes, first at step {rep['first_write_step']}"


@check("V9  at init every read is the identity (bit-identical to frozen lingbot-map)")
def v9(device):
    m = SummaryMemory(dim=1024, num_slots=512).to(device).eval()
    x = {t: torch.randn(1, 1005, 1024, device=device) for t in m.refine_taps}
    _, refined = m.step(m.new_state(1), x, write_tokens=x[m.refine_taps[-1]])
    bad = [t for t in m.refine_taps if not torch.equal(refined[t], x[t])]
    assert not bad, f"taps not identity at init: {bad}"
    return f"exact for taps {m.refine_taps}"


@check("V4  gradients: all trainable params get one, no frozen param does")
def v4(reader, depth_head, device):
    m = SummaryMemory(dim=1024, num_slots=256, write_layers=1, read_layers=1).to(device)
    i = len(reader) // 2
    H, W = reader.hw
    x = reader.token_dict(i, m.refine_taps)
    state, refined = m.step(m.new_state(1), x, write_tokens=x[m.refine_taps[-1]])
    d, c = depth_head(reader.head_inputs(i, refined),
                      images=torch.zeros(1, 1, 3, H, W, device=device),
                      patch_start_idx=reader.meta.patch_start_idx)
    L.depth_loss(d[0, ..., 0], c[0], reader.gt_depth(i))["depth"].backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    leaked = [n for n, p in depth_head.named_parameters() if p.grad is not None]
    # Two things are legitimately inert on a Loss-1 step and must not be flagged:
    # the write path (the read gate is zero-init, so step 0 sends it nothing) and
    # the raymap encoder (used only by Loss 2). V4b covers the latter.
    read_missing = [n for n in missing if "readers." in n]
    assert not read_missing, f"read params without finite grad: {read_missing[:4]}"
    assert not leaked, f"gradient leaked into the frozen head: {leaked[:4]}"
    return f"read path live; {len(missing)} inert (write path at step 0, raymap is Loss 2)"


@check("V4b Loss-2 path: raymap encoder receives gradient")
def v4b(reader, depth_head, device):
    m = SummaryMemory(dim=1024, num_slots=256, write_layers=1, read_layers=1).to(device)
    H, W = reader.hw
    out = m.read_at_camera(m.new_state(1), reader.pose_enc(0), H, W)
    out.square().mean().backward()
    dead = [n for n, p in m.raymap.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"raymap params without gradient: {dead}"
    return f"all {sum(1 for _ in m.raymap.parameters())} raymap params live"


@check("V4c camera path: refined tap 23 reaches the camera head and back")
def v4c(reader, device):
    camera = frozen.build_camera_head().to(device).eval()
    for p in camera.parameters():
        p.requires_grad_(False)
    m = SummaryMemory(dim=1024, num_slots=128, write_layers=1, read_layers=1,
                      refine_taps=(3,)).to(device)
    cache, _ = cb.build_teacher_cache(camera, reader.teacher_camera_tokens()[:16])
    i = 8
    _, refined = m.step(m.new_state(1), reader.token_dict(i, (3,)),
                        write_tokens=reader.tokens(i))
    tap23 = reader.rebuild_tap(i, 3, refined[3])
    assert tap23.shape[-1] == 2048, f"tap 23 is {tap23.shape[-1]}-d, the head needs 2048"
    pose = cb.pose_at(camera, cache, i, tap23[:, 0].float())
    assert pose.shape[-1] == 9, f"pose encoding is {pose.shape}"
    c2w = L.pose_enc_to_c2w(pose, reader.hw)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    L.abs_pose_loss(c2w, reader.gt_c2w([i]))["abs_pose"].backward()

    # Phase 1: the read gate is zero-init, so d(gate(x))/dx == gamma == 0 and the
    # whole gated branch is correctly gradient-free. Only gamma itself must move.
    gate = m.readers["3"].blocks[0].gate_cross.gamma
    assert gate.grad is not None and gate.grad.abs().sum() > 0, \
        "the read gate receives no gradient -- the camera path is disconnected"
    opt.step()
    assert gate.abs().sum() > 0, "gate did not move off zero"

    # Phase 2: with the gate nonzero, the branch itself must now be live.
    m.zero_grad(set_to_none=True)
    _, refined = m.step(m.new_state(1), reader.token_dict(i, (3,)),
                        write_tokens=reader.tokens(i))
    pose = cb.pose_at(camera, cache, i, reader.rebuild_tap(i, 3, refined[3])[:, 0].float())
    L.abs_pose_loss(L.pose_enc_to_c2w(pose, reader.hw),
                    reader.gt_c2w([i]))["abs_pose"].backward()
    dead = [n for n, p in m.readers.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"read params still inert after the gate opened: {dead[:3]}"
    return f"pose {tuple(pose.shape)}; gate opens then the branch goes live"


@check("V13 every refined tap actually influences its target loss")
def v13(reader, depth_head, device):
    """The check that would have caught the dead tap-23 depth branch.

    In the published head `d(depth)/d(tap23) == 0` exactly: layer4_rn's output is
    100% negative and ReLU(inplace=True) zeroes it through the residual. Refining
    a tap that cannot move the loss silently optimises a constant.
    """
    i = len(reader) // 2
    H, W = reader.hw
    influence = {}
    for t in range(4):
        taps = [reader.raw_tap(i, k) for k in range(4)]
        taps[t] = taps[t].clone().requires_grad_(True)
        d, _ = depth_head(taps, images=torch.zeros(1, 1, 3, H, W, device=device),
                          patch_start_idx=reader.meta.patch_start_idx)
        d.sum().backward()
        influence[t] = float(taps[t].grad.abs().max())
    live = [t for t, v in influence.items() if v > 0]
    dead = [t for t, v in influence.items() if v == 0]
    assert live, f"no tap influences depth at all: {influence}"
    assert dead == [3], (f"expected exactly tap 3 to be inert for depth, got dead={dead}. "
                         f"If this changed, refine_taps must change with it: {influence}")
    return (f"depth taps live {live}, inert {dead}; "
            f"grads " + " ".join(f"{t}:{v:.2e}" for t, v in influence.items()))


@check("V7  state norm is stable over a full clip of untrained writes")
def v7(reader, device):
    # The schedule lag is window-1, so a memory built with the default window 64
    # writes nothing inside a short clip and the test would pass vacuously.
    m = SummaryMemory(dim=1024, num_slots=512,
                      scale_frames=reader.meta.scale_frames,
                      sliding_window=reader.meta.kv_cache_sliding_window
                      ).to(device).eval()
    state = m.new_state(1)
    n_written = 0
    norms = []
    with torch.no_grad():
        for i in range(min(len(reader), 320)):
            j = m.schedule.frame_for_step(i)
            state, _ = m.step(state, reader.token_dict(i, m.refine_taps),
                              write_tokens=reader.tokens(j) if j is not None else None)
            n_written += j is not None
            norms.append(float(state.norm()))
    assert n_written >= 8, (f"only {n_written} writes in {len(norms)} steps -- the test "
                            f"would be vacuous; use a longer clip or a smaller window")
    # Drift is measured across *writes*, not against the init: the write ends in a
    # LayerNorm, so the first write moves the state onto the normalised scale once
    # and every step after that must stay there.
    written = norms[-n_written:]
    lo, hi = min(written), max(written)
    ratio = hi / max(lo, 1e-9)
    assert np.isfinite(hi) and ratio < 1.5, \
        f"norm drifts across writes: {lo:.1f}..{hi:.1f} (x{ratio:.2f})"
    return f"init {norms[0]:.1f}; across {n_written} writes {lo:.1f}..{hi:.1f} (x{ratio:.3f})"


@check("V8  overfits one frame to near-zero loss")
def v8(reader, depth_head, device):
    m = SummaryMemory(dim=1024, num_slots=128, write_layers=1, read_layers=1,
                      refine_taps=(0, 1, 2)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    i = len(reader) // 2
    H, W = reader.hw
    x, tgt = reader.token_dict(i, m.refine_taps), reader.gt_depth(i)
    first = last = None
    for it in range(60):
        _, refined = m.step(m.new_state(1), x, write_tokens=x[m.refine_taps[-1]])
        d, c = depth_head(reader.head_inputs(i, refined),
                          images=torch.zeros(1, 1, 3, H, W, device=device),
                          patch_start_idx=reader.meta.patch_start_idx)
        loss = L.depth_loss(d[0, ..., 0], c[0], tgt)["depth_val"]
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it == 0:
            first = float(loss)
        last = float(loss)
    assert last < 0.5 * first, f"loss {first:.4f} -> {last:.4f}, not learning"
    return f"depth_val {first:.4f} -> {last:.4f}"


@check("V10 fixed seed reproduces bitwise")
def v10(device):
    def run():
        torch.manual_seed(1234)
        m = SummaryMemory(dim=256, num_slots=64, num_heads=4,
                          write_layers=1, read_layers=1).to(device)
        g = torch.Generator(device=device).manual_seed(7)
        x = {t: torch.randn(1, 64, 256, device=device, generator=g) for t in m.refine_taps}
        s, r = m.step(m.new_state(1), x, write_tokens=x[m.refine_taps[-1]])
        return s, torch.cat([r[t] for t in m.refine_taps], dim=-1)
    a, b = run(), run()
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), "not reproducible"
    return "exact"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--heads", default="/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = torch.device(a.device)

    clips = find_clips(a.cache)
    if not clips:
        raise SystemExit(f"no clips under {a.cache}")
    reader = ClipReader(clips[0], device=device)
    print(f"clip {clips[0].name}: {len(reader)} frames, {reader.hw}, "
          f"P={reader.meta.num_tokens}, gt_scale={reader.meta.gt_scale:.3f}, "
          f"convention={reader.meta.gt_convention}")

    depth_head, _ = frozen.load_frozen(a.heads, device, need_camera=False)

    v1b(); v5b(); v9(device); v10(device); v1d(device)
    v1(reader, depth_head, device)
    v2(reader, depth_head, device)
    v4(reader, depth_head, device)
    v4b(reader, depth_head, device)
    v4c(reader, device)
    v13(reader, depth_head, device)
    v7(reader, device)
    v8(reader, depth_head, device)

    print()
    ok = 0
    for name, passed, msg in RESULTS:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  -- {msg}" if msg else ""))
        ok += passed
    print(f"\n{ok}/{len(RESULTS)} passed")
    sys.exit(0 if ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
