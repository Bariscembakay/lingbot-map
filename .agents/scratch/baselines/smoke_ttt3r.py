"""TTT3R smoke: load CUT3R weights, run the ttt3r update rule on a few NRGBD
frames, print output shapes. Run from TTT3R/ with the cut3r env."""
import sys, glob, torch
sys.path.insert(0, ".")
from add_ckpt_path import add_path_to_dust3r

CKPT = "src/cut3r_512_dpt_4_64.pth"
add_path_to_dust3r(CKPT)
from src.dust3r.inference import inference_recurrent
from src.dust3r.model import ARCroco3DStereo
from demo import prepare_input

imgs = sorted(glob.glob("/data/NRGBD/breakfast_room/images/img*.png"),
              key=lambda p: int(p.split("img")[-1].split(".")[0]))[:8]
print(f"[smoke] {len(imgs)} frames")
views = prepare_input(img_paths=imgs, img_mask=[True]*len(imgs), size=512,
                      revisit=1, update=True, reset_interval=100)
model = ARCroco3DStereo.from_pretrained(CKPT).to("cuda")
model.config.model_update_type = "ttt3r"
model.eval()
outputs, state_args = inference_recurrent(views, model, "cuda")
print("[smoke] output type:", type(outputs))
pred = outputs["pred"] if isinstance(outputs, dict) and "pred" in outputs else outputs
try:
    p0 = pred[0]
    ks = list(p0.keys()) if hasattr(p0, "keys") else str(type(p0))
    print("[smoke] first pred keys:", ks)
    for k in ("pts3d_in_self_view", "pts3d_in_other_view", "conf"):
        if hasattr(p0, "keys") and k in p0:
            print(f"[smoke]   {k}: {tuple(p0[k].shape)}")
except Exception as e:
    print("[smoke] inspect failed:", e)
print("[smoke] TTT3R OK")
