"""ZipMap smoke: streaming (AR) model on a few NRGBD frames + verify the
state-query checkpoint loads. Run from ZipMap/ with the zipmap env."""
import glob, torch
from zipmap.models.ZipMap_AR import ZipMap as ZipMapAR
from zipmap.utils.load_fn import load_and_preprocess_images

imgs = sorted(glob.glob("/data/NRGBD/breakfast_room/images/img*.png"),
              key=lambda p: int(p.split("img")[-1].split(".")[0]))[:8]
print(f"[smoke] {len(imgs)} frames")
# model_config copied from demo_gradio_zipmap_streaming.py -- the online
# checkpoint does not embed its config.
model_config = {
    "img_size": 518, "patch_size": 14, "embed_dim": 1024,
    "enable_camera": False, "enable_camera_mlp": True,
    "enable_local_point": True, "enable_depth": True,
    "ttt_config": {
        "ttt_mode": True,
        "params": {"bias": True, "head_dim": 1024, "inter_multi": 2,
                   "base_lr": 0.01, "muon_update_steps": 5, "use_gate_fn": True},
        "window_size": 1,
    },
    "other_config": {
        "use_gradient_checkpointing_local_point": False,
        "use_gradient_checkpointing_depth": False,
        "affine_invariant": True,
    },
}
ckpt = torch.load("checkpoints/checkpoint_online.pt", map_location="cpu", weights_only=True)
sd = ckpt.get("model", ckpt)
model = ZipMapAR(**model_config)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"[smoke] online ckpt: {len(missing)} missing / {len(unexpected)} unexpected keys")
model = model.to("cuda").eval()
images = load_and_preprocess_images(imgs).to("cuda")
print("[smoke] images:", tuple(images.shape))
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    pred = model(images)
for k, v in pred.items():
    if isinstance(v, torch.Tensor):
        print(f"[smoke]   {k}: {tuple(v.shape)}")
del model; torch.cuda.empty_cache()
sq = torch.load("checkpoints/checkpoint_state_query.pt", map_location="cpu", weights_only=True)
sqd = sq.get("model", sq)
print(f"[smoke] state_query ckpt loads: {len(sqd)} tensors; sample keys:",
      list(sqd.keys())[:5])
print("[smoke] ZipMap OK")
