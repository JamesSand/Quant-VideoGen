"""Hold GPU memory until released, then decay gradually.

Grabs (free - headroom) GiB on the given GPU in 1-GiB chunks, then waits for
the release file to appear. Once it appears, frees DECAY_GB every DECAY_SEC
seconds so a co-scheduled training job never sees a large free window while
our model load ramps up. Exits when empty.

Usage: python repro/gpu_sentinel.py <cuda_device_index_within_visible> <release_file>
Env: SENTINEL_HEADROOM_GB (default 3), SENTINEL_DECAY_GB (default 2),
     SENTINEL_DECAY_SEC (default 20)
"""

import os
import sys
import time

import torch

dev_idx = int(sys.argv[1])
release_file = sys.argv[2]
headroom = float(os.environ.get("SENTINEL_HEADROOM_GB", "3"))
decay_gb = float(os.environ.get("SENTINEL_DECAY_GB", "2"))
decay_sec = float(os.environ.get("SENTINEL_DECAY_SEC", "20"))

torch.cuda.set_device(dev_idx)
GiB = 1024**3
chunks = []
while True:
    free, _total = torch.cuda.mem_get_info(dev_idx)
    if free < (headroom + 1.0) * GiB:
        break
    try:
        chunks.append(torch.empty(GiB, dtype=torch.uint8, device=f"cuda:{dev_idx}"))
    except torch.OutOfMemoryError:
        break
print(f"[sentinel gpu{dev_idx}] holding {len(chunks)} GiB, waiting for {release_file}", flush=True)

while not os.path.exists(release_file):
    time.sleep(2)

print(f"[sentinel gpu{dev_idx}] release requested, decaying {decay_gb} GiB / {decay_sec}s", flush=True)
while chunks:
    for _ in range(int(decay_gb)):
        if chunks:
            chunks.pop()
    torch.cuda.empty_cache()
    time.sleep(decay_sec)

print(f"[sentinel gpu{dev_idx}] empty, exiting", flush=True)
