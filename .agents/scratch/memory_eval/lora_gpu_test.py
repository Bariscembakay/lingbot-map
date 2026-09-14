import torch, copy
from lingbot_map.heads.dpt_head import DPTHead
from lingbot_map.memory.lora import inject_conv_lora
dev = "cuda"
HALF = 1024
dpt = DPTHead(dim_in=2*HALF, patch_size=14, output_dim=2,
              activation="exp", conf_activation="expp1").eval().to(dev)
sd = torch.load("/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt",
                map_location="cpu", weights_only=False)
dpt.load_state_dict(sd["depth_head"])
dpt = dpt.to(dev)
toks = [torch.randn(1, 1, 16*16, 2*HALF, device=dev) for _ in range(4)]
imgs = torch.zeros(1, 1, 3, 16*14, 16*14, device=dev)
with torch.no_grad():
    z0, c0 = dpt(toks, imgs, patch_start_idx=0)
m = copy.deepcopy(dpt)
st = inject_conv_lora(m, 16, 16.0)
devs = {str(p.device) for p in m.parameters()}
print(f"param devices after injection: {devs}")
with torch.no_grad():
    z1, c1 = m(toks, imgs, patch_start_idx=0)
print(f"forward OK  max|dz|={(z1-z0).abs().max():.3e}  lora={st['lora_params']/1e6:.2f}M")
# backward must reach the LoRA params and nothing else
z2, _ = m(toks, imgs, patch_start_idx=0)
z2.sum().backward()
g = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is not None]
ng = [n for n, p in m.named_parameters() if (not p.requires_grad) and p.grad is not None]
print(f"grads on {len(g)} lora tensors; on {len(ng)} frozen tensors (must be 0)")
print("PASS" if devs == {"cuda:0"} and len(ng) == 0 and len(g) > 0 else "FAIL")
