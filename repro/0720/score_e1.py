#!/usr/bin/env python3
"""E1 scoring (paper-diff-plan): rtn3ptfp8 / kivipaperfp8 / kivi3ptfp8 x10 prompts.
Ref metrics (f93 PSNR/SSIM/LPIPS vs bf16) + vbench4. Single GPU, ~30 videos.
Usage: score_e1.py <shard> <nshards>
"""
import os, sys, glob, subprocess

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
PY = ".venv/bin/python"
os.environ["VBENCH4_CACHE"] = f"repro/0718/npz/vbench4_e1_shard{SHARD}.json"

ARMS = ["rtn3ptfp8", "kivipaperfp8", "kivi3ptfp8"]

def g1(p):
    g = glob.glob(p); return g[0] if g else None

jobs = []
for p in range(1, 11):
    ref = g1(f"results/multiprompt/mp100/lc/bf16_rep0/p{p}/*/segment_1.mp4")
    for arm in ARMS:
        v = g1(f"results/multiprompt/mp100/lc/{arm}_rep0/p{p}/*/segment_1.mp4")
        if not v: continue
        jobs.append(("vb", (v, 0)))
        if ref:
            jobs.append(("ref", (ref, v, f"mp100_lc_p{p}_{arm}")))

mine = [j for i, j in enumerate(jobs) if i % NSH == SHARD]
vb_by_mf = {}
ref_jobs = []
for kind, pl in mine:
    if kind == "vb": vb_by_mf.setdefault(pl[1], []).append(pl[0])
    else: ref_jobs.append(pl)

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
