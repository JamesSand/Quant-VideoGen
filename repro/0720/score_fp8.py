#!/usr/bin/env python3
"""MP100 scoring shard: ref metrics (LC f93 / HY [13:]) + vbench4 for all arms.
Usage: score100.py <shard> <nshards>   (run one per GPU with CUDA_VISIBLE_DEVICES)
Caches: repro/0718/npz/*.npz (per-key) and vbench4_shard<i>.json (merged later).
"""
import os, sys, glob, subprocess

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
PY = ".venv/bin/python"
os.environ["VBENCH4_CACHE"] = f"repro/0718/npz/vbench4_shard{SHARD}.json"

LC_ARMS = ["bf16", "rtn", "kivi", "quarot", "qvg", "pcakaxvaxfp8"]
SF_ARMS = ["bf16", "rtn", "kivi", "quarot", "qvg", "pcaa128kaxvaxfp8"]
HY_ARMS = ["bf16", "rtn", "kivi", "quarot", "qvg", "pcav90kpternkaxkb64fp8"]

def g1(p):
    g = glob.glob(p); return g[0] if g else None

jobs = []  # (kind, payload)
for p in range(1, 101):
    ref = g1(f"results/multiprompt/mp100/lc/bf16_rep0/p{p}/*/segment_1.mp4")
    for arm in LC_ARMS:
        v = g1(f"results/multiprompt/mp100/lc/{arm}_rep0/p{p}/*/segment_1.mp4")
        if not v: continue
        jobs.append(("vb", (v, 0)))
        if arm != "bf16" and ref:
            jobs.append(("ref", (ref, v, f"mp100_lc_p{p}_{arm}")))
    for arm in SF_ARMS:
        v = g1(f"results/multiprompt/mp100/sf/{arm}_rep0_f180/p{p}/*.mp4")
        if v: jobs.append(("vb", (v, 700)))
for s in range(10):
    ref = f"results/multiprompt/hy/bf16_s{s}/0-{s}.mp4"
    for arm in HY_ARMS:
        v = f"results/multiprompt/hy/{arm}_s{s}/0-{s}.mp4"
        if not os.path.exists(v): continue
        jobs.append(("vb", (v, 0)))
        if arm != "bf16" and os.path.exists(ref):
            jobs.append(("ref", (ref, v, f"hy_s{s}_{arm}")))

mine = [j for i, j in enumerate(jobs) if i % NSH == SHARD]
vb_by_mf = {}
ref_jobs = []
for kind, pl in mine:
    if kind == "vb": vb_by_mf.setdefault(pl[1], []).append(pl[0])
    else: ref_jobs.append(pl)

# ref metrics via a tiny inline runner (reuses sweep_stats helper math)
if ref_jobs:
    import numpy as np, torch, lpips, imageio.v3 as iio
    import torch.nn.functional as F
    lp = lpips.LPIPS(net="vgg").to("cuda")
    def calc_ssim(a, b):
        C1, C2 = 0.01**2, 0.03**2
        a, b = a.unsqueeze(0), b.unsqueeze(0)
        mu1, mu2 = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
        s1 = F.avg_pool2d(a*a, 11, 1, 5) - mu1**2
        s2 = F.avg_pool2d(b*b, 11, 1, 5) - mu2**2
        s12 = F.avg_pool2d(a*b, 11, 1, 5) - mu1*mu2
        return float((((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean())
    for ref, test, key in ref_jobs:
        f = f"repro/0718/npz/{key}.npz"
        if os.path.exists(f): continue
        try:
            R = list(iio.imiter(ref, plugin="pyav"))
            P, S, L = [], [], []
            for i, tf in enumerate(iio.imiter(test, plugin="pyav")):
                if i >= len(R): break
                a = torch.from_numpy(np.asarray(R[i])).float().permute(2,0,1)/255
                b = torch.from_numpy(np.asarray(tf)).float().permute(2,0,1)/255
                mse = float(((a-b)**2).mean())
                P.append(10*np.log10(1/mse) if mse > 0 else 99.0)
                S.append(calc_ssim(a, b))
                with torch.no_grad():
                    L.append(float(lp(a.unsqueeze(0).to("cuda"), b.unsqueeze(0).to("cuda"))))
            np.savez(f, psnr=np.array(P), ssim=np.array(S), lpips=np.array(L))
            print("ref done", key, flush=True)
        except Exception as e:
            print("ref FAIL", key, e, flush=True)

for mf, vids in vb_by_mf.items():
    for i in range(0, len(vids), 25):
        cmd = [PY, "repro/0718/scripts/vbench4.py"] + vids[i:i+25]
        if mf: cmd += ["--max-frames", str(mf)]
        subprocess.run(cmd)
print(f"SHARD {SHARD} DONE")
