"""Pre-submit checks for --pts-frame / --fresh-last-conv (GPU)."""
import sys, types, torch
sys.path.insert(0, "scripts/memory")
from train_state import run_probe
from lingbot_map.memory.cut3r_state import LingbotFrozenHead
from lingbot_map.memory.lora import inject_conv_lora

dev = "cuda"
ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}")

# 1. geometry: the cam/world branches must invert the depth branch exactly
b, h, w = 3, 7, 9
R = torch.linalg.qr(torch.randn(b, 3, 3, device=dev))[0]
t = torch.randn(b, 3, device=dev)
c2w = torch.eye(4, device=dev).repeat(b, 1, 1); c2w[:, :3, :3] = R; c2w[:, :3, 3] = t
pix = torch.cat([torch.randn(b, h, w, 2, device=dev), torch.ones(b, h, w, 1, device=dev)], -1)
d = torch.einsum("bij,bhwj->bhwi", R, pix)
o = t[:, None, None, :].expand(b, h, w, 3)
z = torch.rand(b, h, w, device=dev) + 0.5
def fake(frame, out):
    m = types.SimpleNamespace(head=types.SimpleNamespace(pts_frame=frame),
                              head_type="smallread_lingbot")
    m.probe = lambda *a: out
    return m
c = torch.ones(b, h, w, device=dev)
ref = run_probe(fake(None, {"ray_depth": z, "conf": c}), None, None, None, (h, w), (o, d, c2w))
cam = run_probe(fake("cam", {"pts": ref["pts3d_in_self_view"], "conf": c}), None, None, None, (h, w), (o, d, c2w))
wld = run_probe(fake("world", {"pts": ref["pts3d_in_other_view"], "conf": c}), None, None, None, (h, w), (o, d, c2w))
check("depth: camera-frame point = z * pixel dir", torch.allclose(ref["pts3d_in_self_view"], z[..., None] * pix, atol=1e-5))
check("cam branch reproduces world points", torch.allclose(cam["pts3d_in_other_view"], ref["pts3d_in_other_view"], atol=1e-5))
check("world branch reproduces camera points", torch.allclose(wld["pts3d_in_self_view"], ref["pts3d_in_self_view"], atol=1e-5))

# 2. head surgery on the real trunk, after LoRA (as train_state orders it)
hw = (16 * 14, 16 * 14)
taps = [torch.randn(2, 256, 768, device=dev) for _ in range(4)]
for frame in (None, "cam", "world"):
    torch.manual_seed(0)
    head = LingbotFrozenHead().to(dev)
    inject_conv_lora(head.dpt, 16, 16.0)
    with torch.no_grad():
        _, c0 = head(taps, hw)
    head.reset_output(frame)
    last = head.dpt.scratch.output_conv2[-1]
    check(f"[{frame}] last conv is a plain trainable conv", type(last) is torch.nn.Conv2d
          and all(p.requires_grad for p in last.parameters()))
    with torch.no_grad():
        g0, c1 = head(taps, hw)
    shape = (2, *hw, 3) if frame else (2, *hw)
    check(f"[{frame}] output shape {tuple(g0.shape)}", tuple(g0.shape) == shape)
    check(f"[{frame}] cold start value (max|dev|={(g0 - (0.0 if frame else 1.0)).abs().max().item():.2e})", torch.allclose(g0, torch.full_like(g0, 0.0 if frame else 1.0), atol=0.1))
    check(f"[{frame}] confidence unchanged by surgery", torch.allclose(c0, c1, atol=1e-5))
    head.train()
    g, _ = head(taps, hw)
    (g - 2).square().mean().backward()
    # Existence is not enough: a dead activation hands back all-zero grads.
    gn = last.weight.grad[:-1].norm().item()
    check(f"[{frame}] geometry-channel grad is nonzero (|g|={gn:.3e})", gn > 1e-8)
    opt = torch.optim.SGD([last.weight, last.bias], lr=1e-2)
    opt.step()
    with torch.no_grad():
        g1, _ = head(taps, hw)
    check(f"[{frame}] geometry output moves after one step", (g1 - g0).abs().max().item() > 1e-6)
    trainable = [n for n, p in head.named_parameters() if p.requires_grad and p.grad is not None]
    frozen = [n for n, p in head.named_parameters() if not p.requires_grad and p.grad is not None]
    check(f"[{frame}] grads reach new conv + LoRA ({len(trainable)} tensors), 0 frozen ({len(frozen)})",
          any("output_conv2.2" in n for n in trainable) and any(".A." in n or n.endswith("A.weight") for n in trainable) and not frozen)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
