"""SF reference-based metrics: per-frame PSNR/SSIM/LPIPS(paper conv) vs BF16.
Same prompt/seed causal runs: frames identical until the first quantization
event; report first divergent frame, first-affected-frame metrics, and means.
"""
import sys
import imageio.v3 as iio
import numpy as np
import torch
import lpips as lpips_mod

sys.path.insert(0, "experiments/LongCat")
dev = "cuda"
lp = lpips_mod.LPIPS(net="vgg").to(dev)

def calc_ssim(a, b):
    # paper 口径（metric.py calculate_ssim 原版）：11x11 局部窗 avg_pool SSIM。
    # 曾用整帧单窗的"全局 SSIM"——严重虚高（0716 勘误），0716 前的 npz ssim 数组作废。
    import torch.nn.functional as F
    C1, C2 = 0.01**2, 0.03**2
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu1, mu2 = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
    s1 = F.avg_pool2d(a*a, 11, 1, 5) - mu1**2
    s2 = F.avg_pool2d(b*b, 11, 1, 5) - mu2**2
    s12 = F.avg_pool2d(a*b, 11, 1, 5) - mu1*mu2
    return float((((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean())

ref_path, n_frames = sys.argv[1], int(sys.argv[2])
ref_frames = list(iio.imiter(ref_path, plugin="pyav"))[:n_frames]
for test_path in sys.argv[3:]:
    psnr, ssim, lpv = [], [], []
    first_div = None
    for i, tf in enumerate(iio.imiter(test_path, plugin="pyav")):
        if i >= n_frames: break
        a = torch.from_numpy(np.asarray(ref_frames[i])).float().permute(2,0,1)/255
        b = torch.from_numpy(np.asarray(tf)).float().permute(2,0,1)/255
        mse = float(((a-b)**2).mean())
        p = 10*np.log10(1/mse) if mse > 0 else np.inf
        if first_div is None and mse > 0: first_div = i
        psnr.append(p); ssim.append(calc_ssim(a, b))
        with torch.no_grad():
            lpv.append(float(lp(a.unsqueeze(0).to(dev), b.unsqueeze(0).to(dev))))  # paper 口径: [0,1] 直喂
    P, S, L = np.array(psnr), np.array(ssim), np.array(lpv)
    fd = first_div if first_div is not None else -1
    tag = test_path.split('/')[-2] if '/' in test_path else test_path
    np.savez(f"repro/backup/protosearch/sf_{tag}.npz", psnr=P, ssim=S, lpips=L)
    fin = np.isfinite(P)
    print(f"{test_path}")
    print(f"  首个分歧帧: {fd}  | 首分歧帧指标: PSNR={P[fd]:.3f} SSIM={S[fd]:.4f} LPIPS={L[fd]:.4f}")
    print(f"  全视频均值(仅分歧后): PSNR={P[fd:][fin[fd:]].mean():.3f} SSIM={S[fd:].mean():.4f} LPIPS={L[fd:].mean():.4f}")
