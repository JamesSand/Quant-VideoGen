"""qvg4pt fp8 终值逐 prompt 落盘:f93-only,协议逐行照抄 repro/0720/score_fp8.py(CPU)。"""
import os, glob, json
import numpy as np, torch, lpips, imageio.v3 as iio
import torch.nn.functional as F
os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
lp = lpips.LPIPS(net="vgg")
def calc_ssim(a, b):
    C1, C2 = 0.01**2, 0.03**2
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu1, mu2 = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
    s1 = F.avg_pool2d(a*a, 11, 1, 5) - mu1**2
    s2 = F.avg_pool2d(b*b, 11, 1, 5) - mu2**2
    s12 = F.avg_pool2d(a*b, 11, 1, 5) - mu1*mu2
    return float((((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean())
def frame93(path):
    for i, fr in enumerate(iio.imiter(path, plugin="pyav")):
        if i == 93:
            return torch.from_numpy(np.asarray(fr)).float().permute(2, 0, 1) / 255
    return None
out = {}
for p in range(1, 11):
    ref = glob.glob(f"results/multiprompt/mp100/lc/bf16_rep0/p{p}/*/segment_1.mp4")[0]
    tst = glob.glob(f"results/multiprompt/mp100/lc/qvg4pt_rep0/p{p}/*/segment_1.mp4")[0]
    a, b = frame93(ref), frame93(tst)
    mse = float(((a - b) ** 2).mean())
    with torch.no_grad():
        l = float(lp(a.unsqueeze(0), b.unsqueeze(0)))
    out[f"p{p}"] = {"psnr": round(10 * np.log10(1 / mse), 2),
                    "ssim": round(calc_ssim(a, b), 4), "lpips": round(l, 4)}
    print(p, out[f"p{p}"], flush=True)
m = {k: round(float(np.mean([v[k] for v in out.values()])), 4) for k in ("psnr", "ssim", "lpips")}
res = {"arm": "qvg4pt (fp8 合法元数据口径终值)", "protocol": "LC f93 vs bf16_rep0, score_fp8.py 同款",
       "per_prompt": out, "mean": m}
json.dump(res, open("repro/0721/qvg4pt-fp8-score.json", "w"), indent=1, ensure_ascii=False)
print("MEAN", m)
