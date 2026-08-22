"""Cache invariants for the trajectory-memory context-token ablation.

Asserts, per frame, that the ablation touches the trajectory memory ONLY:
the resident store (anchor + sliding window) must keep every one of its 783
tokens per frame under every spec, while the memory store accumulates exactly
len(keep) tokens for each evicted frame and never sheds one.
"""
import sys
import torch
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.layers.attention import parse_memory_tokens, memory_keep_indices

H, W = 126, 154                 # 9x11 = 99 patches -> P = 6 + 99 = 105
SCALE, WINDOW = 2, 4            # eviction threshold = 6 frames
P = 105
FAIL = []

def check(cond, msg):
    if not cond:
        FAIL.append(msg)

def trace(spec, n_frames, keyframe_interval):
    """Run n_frames and return the per-frame (resident, memory) cache shapes."""
    torch.manual_seed(0)
    m = GCTStream(img_size=W, patch_size=14, enable_3d_rope=True, max_frame_num=64,
                  kv_cache_sliding_window=WINDOW, kv_cache_scale_frames=SCALE,
                  use_sdpa=True, kv_cache_memory_tokens=spec).eval()
    log = []
    real_forward = m.forward
    def spy(*a, **kw):
        out = real_forward(*a, **kw)
        c = m.aggregator.kv_cache
        res = tuple(c["k_0"].shape) if c.get("k_0") is not None else None
        mem = tuple(c["k_0_special"].shape) if c.get("k_0_special") is not None else None
        log.append((res, mem))
        return out
    m.forward = spy
    torch.manual_seed(1)
    with torch.no_grad():
        m.inference_streaming(torch.rand(1, n_frames, 3, H, W),
                              num_scale_frames=SCALE,
                              keyframe_interval=keyframe_interval)
    return log

print("=== behaviour-preservation: gather vs. the slice it replaces ===")
t = torch.randn(1, 16, 3, 6, 64)
kinds = parse_memory_tokens("camera,register,scale")
gathered = t[:, :, :, memory_keep_indices(kinds, 0, 4, 5), :]
check(torch.equal(gathered, t[:, :, :, 0:6, :]),
      "default spec gather != original contiguous slice 0:6")
print(f"  default spec keep={memory_keep_indices(kinds,0,4,5)} equals slice 0:6: "
      f"{torch.equal(gathered, t[:, :, :, 0:6, :])}")

print("\n=== per-frame invariants (scale=2, window=4 -> evict from frame 6) ===")
for spec in ["camera,register,scale", "camera,scale", "register,scale", "camera,register"]:
    kinds = parse_memory_tokens(spec)
    n_keep = len(memory_keep_indices(kinds, 0, 4, 5))
    log = trace(spec, 14, 1)

    # Phase 1 processes SCALE frames in one bidirectional block, then 1 frame/step.
    keyframes_after = [SCALE] + [SCALE + i for i in range(1, len(log))]
    prev_mem = 0
    for step, ((res, mem), kf) in enumerate(zip(log, keyframes_after)):
        res_frames, res_tok = res[2], res[3]
        mem_frames = mem[2] if mem else 0
        mem_tok = mem[3] if mem else None

        check(res_tok == P, f"{spec} step{step}: resident tokens/frame {res_tok} != {P} "
                            f"-- window/anchor frames were modified")
        check(res_frames == min(kf, SCALE + WINDOW),
              f"{spec} step{step}: resident frames {res_frames} != {min(kf, SCALE+WINDOW)}")
        check(mem_frames == max(0, kf - (SCALE + WINDOW)),
              f"{spec} step{step}: memory frames {mem_frames} != {max(0, kf-(SCALE+WINDOW))}")
        if mem:
            check(mem_tok == n_keep,
                  f"{spec} step{step}: memory tokens/frame {mem_tok} != {n_keep}")
        check(mem_frames + res_frames == kf,
              f"{spec} step{step}: {mem_frames}+{res_frames} != {kf} frames processed "
              f"-- a frame is double-counted or lost")
        check(mem_frames >= prev_mem, f"{spec} step{step}: memory shrank")
        prev_mem = mem_frames

    final_res, final_mem = log[-1]
    print(f"  {spec:24s} keep={n_keep}  resident={final_res[2]}x{final_res[3]}  "
          f"memory={final_mem[2]}x{final_mem[3]}  (accumulates, never evicted)")

print("\n=== non-keyframes must persist to neither store ===")
log_k1 = trace("camera,register,scale", 14, 1)
log_k2 = trace("camera,register,scale", 14, 2)
r1, m1 = log_k1[-1]
r2, m2 = log_k2[-1]
# interval 2 over frames 2..13 keyframes at 2,4,6,8,10,12 -> 6 keyframes + 2 anchors = 8
check(m2[2] < m1[2], "keyframe_interval=2 did not reduce the memory frame count")
print(f"  interval=1: memory={m1[2]} frames   interval=2: memory={m2[2]} frames "
      f"(non-keyframes correctly not persisted)")

print("\n=== unbounded growth: memory frames scale with sequence length ===")
for n in (8, 12, 20):
    log = trace("camera,scale", n, 1)
    mem = log[-1][1]
    expect = n - (SCALE + WINDOW)
    check(mem[2] == expect, f"n={n}: memory {mem[2]} != {expect}")
    print(f"  {n:2d} frames -> resident {log[-1][0][2]} (capped at {SCALE+WINDOW}), "
          f"memory {mem[2]} (= n - {SCALE+WINDOW})")

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}):")
    for f in FAIL[:20]:
        print("  -", f)
    sys.exit(1)
print("ALL INVARIANTS HOLD")
