"""Precompute per-frame PSNR/SSIM/LPIPS arrays for every method-vs-bf16 pair.

LPIPS follows the PAPER convention deliberately: frames in [0,1] fed to lpips(vgg)
WITHOUT the library's [-1,1] normalization (replicates official metric.py:131).
Numbers are NOT comparable to standard-normalized LPIPS from other papers.
Arrays computed before 2026-07-16 used [-1,1] normalization — do not mix.

Output: repro/backup/protosearch/<name>.npz with keys psnr, ssim, lpips (float64,
one value per frame index, full video length, frame 0 included).
"""

import json
import os
import sys

sys.path.insert(0, "experiments/LongCat")
import imageio
import lpips as lpips_mod
import numpy as np
import torch
import torch.nn.functional as F

OUT = "repro/backup/protosearch"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"
lp = lpips_mod.LPIPS(net="vgg").to(dev)


def calc_ssim(a, b):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(a, 11, 1, 5)
    mu2 = F.avg_pool2d(b, 11, 1, 5)
    s1 = F.avg_pool2d(a * a, 11, 1, 5) - mu1**2
    s2 = F.avg_pool2d(b * b, 11, 1, 5) - mu2**2
    s12 = F.avg_pool2d(a * b, 11, 1, 5) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1 + s2 + C2))
    return m.mean().item()


def load(p):
    # keep frames on CPU; move to GPU per frame (local GPU has little free VRAM)
    return [torch.tensor(f, dtype=torch.float32).permute(2, 0, 1) / 255
            for f in imageio.get_reader(p)]


PAIRS = {
    # LC seg1 pairs (vs bf16 seg1, 113 frames)
    "lc_quarot_int2_asym_b16": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_quarot_int2_asym_b16/1-0/segment_1.mp4"),
    "lc_quarot_int2_sym_b16": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_quarot_int2_sym_b16/1-0/segment_1.mp4"),
    "lc_quarot_int2_asym_b128": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_quarot_int2_asym_b128/1-0/segment_1.mp4"),
    "lc_quarot_int4_asym_b16": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_quarot_int4_asym_b16/1-0/segment_1.mp4"),
    "lc_quarot_int4_sym_b16": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_quarot_int4_sym_b16/1-0/segment_1.mp4"),
    "lc_quarot_int4_asym_b128": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_quarot_int4_asym_b128/1-0/segment_1.mp4"),
    "lc_rtn_int2_b16": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_rtn_int2_b16/1-0/segment_1.mp4"),
    "lc_rtn_int4_b16": ("results/longcat/bf16/1-0/segment_1.mp4", "results/quarot/lc_rtn_int4_b16/1-0/segment_1.mp4"),
    "lc_qvgpro_int2": ("results/longcat/bf16/1-0/segment_1.mp4", "results/diag/pro_int2/1-0/segment_1.mp4"),
    # LC QVG full-length pairs (vs bf16 segment_10, 293 frames)
    "lc_qvg_int2_rngiso": ("results/longcat/bf16/1-0/segment_10.mp4", "results/longcat_rngiso/triton-nstages-kmeans-int2_64/kc_256_vc_256/nstages_1_iters_100/1-0/segment_10.mp4"),
    "lc_qvg_int4_rngiso": ("results/longcat/bf16/1-0/segment_10.mp4", "results/longcat_rngiso/triton-nstages-kmeans-int4_64/kc_256_vc_256/nstages_1_iters_100/1-0/segment_10.mp4"),
    "lc_qvg_int2_released": ("results/longcat/bf16/1-0/segment_10.mp4", "results/longcat/triton-nstages-kmeans-int2_64/kc_256_vc_256/nstages_1_iters_100/1-0/segment_10.mp4"),
    "lc_qvg_int4_released": ("results/longcat/bf16/1-0/segment_10.mp4", "results/longcat/triton-nstages-kmeans-int4_64/kc_256_vc_256/nstages_1_iters_100/1-0/segment_10.mp4"),
    # HY pairs (vs matched bf16, 189 frames)
    "hy_qvg_int2": ("results/hyworldplay/bf16_matched/0-0.mp4", "results/hyworldplay/triton-nstages-kmeans-int2_64/kc_256_vc_256_nstages_1/0-0.mp4"),
    "hy_qvg_int4": ("results/hyworldplay/bf16_matched/0-0.mp4", "results/hyworldplay/triton-nstages-kmeans-int4_64/kc_256_vc_256_nstages_1/0-0.mp4"),
    "hy_quarot_int2": ("results/hyworldplay/bf16_matched/0-0.mp4", "results/quarot/hy_quarot_int2_asym_b16/0-0.mp4"),
    "hy_quarot_int4": ("results/hyworldplay/bf16_matched/0-0.mp4", "results/quarot/hy_quarot_int4_asym_b16/0-0.mp4"),
    "hy_rtn_int2": ("results/hyworldplay/bf16_matched/0-0.mp4", "results/quarot/hy_rtn_int2_b16/0-0.mp4"),
    "hy_rtn_int4": ("results/hyworldplay/bf16_matched/0-0.mp4", "results/quarot/hy_rtn_int4_b16/0-0.mp4"),
}

index = {}
for name, (p1, p2) in PAIRS.items():
    if not (os.path.exists(p1) and os.path.exists(p2)):
        print(f"SKIP {name} (missing file)")
        continue
    if os.path.exists(f"{OUT}/{name}.npz"):
        d = np.load(f"{OUT}/{name}.npz")
        index[name] = {"frames": int(len(d["psnr"])), "src": p2}
        print(f"{name}: cached")
        continue
    v1, v2 = load(p1), load(p2)
    n = min(len(v1), len(v2))
    psnr, ssim, lpv = [], [], []
    with torch.no_grad():
        for i in range(n):
            a, b = v1[i].to(dev), v2[i].to(dev)
            mse = ((a - b) ** 2).mean().item()
            psnr.append(10 * np.log10(1 / mse) if mse > 0 else np.inf)
            ssim.append(calc_ssim(a.unsqueeze(0), b.unsqueeze(0)))
            lpv.append(lp(a.unsqueeze(0), b.unsqueeze(0)).item())  # PAPER convention: feed [0,1] directly (metric.py:131 quirk), adopted as project standard 2026-07-16
    np.savez(f"{OUT}/{name}.npz", psnr=np.array(psnr), ssim=np.array(ssim), lpips=np.array(lpv))
    index[name] = {"frames": n, "src": p2}
    print(f"{name}: {n} frames  psnr[93..96]={[round(x,2) for x in psnr[93:97]]}" if n > 96 else f"{name}: {n} frames")

json.dump(index, open(f"{OUT}/index.json", "w"), indent=1)
print("DONE", len(index), "pairs")
