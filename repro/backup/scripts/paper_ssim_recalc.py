"""Recompute per-frame SSIM with the paper's exact implementation
(metric.py calculate_ssim: 11x11 avg_pool local windows) and update the
protosearch npz arrays in place (adds/overwrites key 'ssim_paper').

Usage: paper_ssim_recalc.py <ref_video> <n_frames> <test_video:npz_tag> ...
"""
import sys
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

dev = "cuda"

def paper_ssim(img1, img2):  # verbatim from experiments/LongCat/longcat_video/utils/metric.py
    C1 = 0.01**2
    C2 = 0.03**2
    mu1 = F.avg_pool2d(img1, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(img2, kernel_size=11, stride=1, padding=5)
    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=11, stride=1, padding=5) - mu1**2
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=11, stride=1, padding=5) - mu2**2
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=11, stride=1, padding=5) - mu1 * mu2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())

ref_path, n_frames = sys.argv[1], int(sys.argv[2])
ref_frames = list(iio.imiter(ref_path, plugin="pyav"))[:n_frames]
for spec in sys.argv[3:]:
    test_path, tag = spec.rsplit(":", 1)
    vals = []
    for i, tf in enumerate(iio.imiter(test_path, plugin="pyav")):
        if i >= n_frames: break
        a = torch.from_numpy(np.asarray(ref_frames[i])).float().permute(2,0,1).unsqueeze(0).to(dev)/255
        b = torch.from_numpy(np.asarray(tf)).float().permute(2,0,1).unsqueeze(0).to(dev)/255
        vals.append(paper_ssim(a, b))
    S = np.array(vals)
    f = f"repro/backup/protosearch/sf_{tag}.npz"
    d = dict(np.load(f))
    d["ssim_paper"] = S
    np.savez(f, **d)
    w = slice(23, 36)
    print(f"{tag}: paper-SSIM @[23,36)={S[w].mean():.4f}  首帧后全程={S[1:].mean():.4f}  "
          f"(旧全局SSIM @[23,36)={np.load(f)['ssim'][w].mean():.4f})")
