import sys
import time
import torch

# Default 40% VRAM reservation is fine for short jobs (Table 1's 300-500
# frame runs) but competes with Long3D's much larger KV cache (total_budget
# up to 1.2M tokens across thousands of frames) on a 44GB card -- pass a
# smaller fraction (e.g. 0.05) as argv[1] for those jobs.
VRAM_FRACTION = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4

# How long each matmul burst should keep the GPU visibly busy. A FIXED matrix
# size (the old approach) breaks across GPU generations: the same 1024x1024
# matmul that's a real ~0.2s burst on an A6000 finishes in well under a
# millisecond on an H200, so the cluster's idle-GPU monitor (which samples
# utilization over some window) sees ~0% activity and pages anyway even
# though the script is "working". Instead we calibrate the matrix size PER
# GPU at startup so a burst reliably spans this many seconds regardless of
# how fast the card is.
TARGET_BURST_SECONDS = 0.3
SLEEP_SECONDS = 2
MIN_WORK_SIZE = 256
MAX_WORK_SIZE = 16384  # cap so calibration can't run away on absurdly fast cards


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def time_matmul(device: torch.device, size: int, reps: int = 3) -> float:
    """Median wall-clock seconds for one size x size matmul on device."""
    a = torch.randn(size, size, device=device)
    out = torch.empty_like(a)
    torch.mm(a, a, out=out)
    torch.cuda.synchronize(device)  # warm up (first call pays kernel-launch overhead)

    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        torch.mm(a, a, out=out)
        torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def calibrate_work_size(device: torch.device) -> int:
    """Pick a matrix size so a single matmul takes ~TARGET_BURST_SECONDS on
    THIS gpu. Matmul FLOPs scale as size^3, so one timing sample is enough to
    extrapolate; one correction pass fixes the extrapolation error from
    fixed per-call overhead (kernel launch, sync) not scaling with size."""
    probe_size = 512
    elapsed = time_matmul(device, probe_size)
    elapsed = max(elapsed, 1e-6)  # guard against a suspiciously-fast first sample

    size = int(probe_size * (TARGET_BURST_SECONDS / elapsed) ** (1 / 3))
    size = max(MIN_WORK_SIZE, min(size, MAX_WORK_SIZE))

    # One correction pass: re-measure at the extrapolated size and nudge
    # again, since fixed overhead makes the size^3 extrapolation imperfect
    # (worse the smaller/faster the initial probe was).
    elapsed2 = time_matmul(device, size)
    if elapsed2 > 1e-6:
        size = int(size * (TARGET_BURST_SECONDS / elapsed2) ** (1 / 3))
        size = max(MIN_WORK_SIZE, min(size, MAX_WORK_SIZE))

    return size


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise SystemExit("No CUDA GPUs are visible")

    allocations = []
    work = []

    for index in range(device_count):
        device = torch.device(f"cuda:{index}")
        properties = torch.cuda.get_device_properties(device)
        target_bytes = int(properties.total_memory * VRAM_FRACTION)

        # uint8 uses exactly one byte per element, making the target predictable.
        allocation = torch.empty(target_bytes, dtype=torch.uint8, device=device)
        allocation.zero_()  # Touch the allocation before reporting readiness.
        allocations.append(allocation)

        work_size = calibrate_work_size(device)
        matrix = torch.randn(work_size, work_size, device=device)
        work.append((matrix, torch.empty_like(matrix)))

        measured = time_matmul(device, work_size)
        print(
            f"GPU {index}: {properties.name} - allocated "
            f"{gib(target_bytes):.2f} GiB / {gib(properties.total_memory):.2f} GiB "
            f"({VRAM_FRACTION:.0%}); keep-alive burst calibrated to "
            f"{work_size}x{work_size} matmul (~{measured*1000:.0f} ms/call, "
            f"target {TARGET_BURST_SECONDS*1000:.0f} ms)",
            flush=True,
        )

    for index in range(device_count):
        torch.cuda.synchronize(index)

    print(f"GPU keep-alive started on {device_count} visible GPU(s)", flush=True)

    while True:
        for index, (matrix, output) in enumerate(work):
            with torch.cuda.device(index):
                # Issue matmuls back-to-back until the calibrated burst
                # duration has actually elapsed on this device -- covers
                # drift from thermal throttling or contention with the main
                # job's own GPU work, not just the one-time calibration.
                start_time = time.time()
                while time.time() - start_time < TARGET_BURST_SECONDS:
                    torch.mm(matrix, matrix, out=output)
                torch.cuda.synchronize(index)

        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
