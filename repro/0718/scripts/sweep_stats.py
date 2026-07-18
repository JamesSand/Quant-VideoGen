#!/usr/bin/env python3
"""Score sweep-1 variants against QVG (and the default pca arm), shared caches."""
import os, glob, json, subprocess, re
import numpy as np

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
NPZD = "repro/0718/npz"; PY = ".venv/bin/python"

import importlib.util
# reuse ref_metrics/vbench from stats.py without executing its main body: inline copies
_lp = None
def lp():
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net="vgg").to("cuda")
    return _lp

def calc_ssim(a, b):
    import torch.nn.functional as F
    C1, C2 = 0.01**2, 0.03**2
    a, b = a.unsqueeze(0), b.unsqueeze(0)
    mu1, mu2 = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
    s1 = F.avg_pool2d(a*a, 11, 1, 5) - mu1**2
    s2 = F.avg_pool2d(b*b, 11, 1, 5) - mu2**2
    s12 = F.avg_pool2d(a*b, 11, 1, 5) - mu1*mu2
    return float((((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean())

def ref_metrics(ref_path, test_path, key):
    f = f"{NPZD}/{key}.npz"
    if os.path.exists(f):
        d = np.load(f); return d["psnr"], d["ssim"], d["lpips"]
    import imageio.v3 as iio, torch
    ref = list(iio.imiter(ref_path, plugin="pyav"))
    P, S, L = [], [], []
    for i, tf in enumerate(iio.imiter(test_path, plugin="pyav")):
        if i >= len(ref): break
        a = torch.from_numpy(np.asarray(ref[i])).float().permute(2,0,1)/255
        b = torch.from_numpy(np.asarray(tf)).float().permute(2,0,1)/255
        mse = float(((a-b)**2).mean())
        P.append(10*np.log10(1/mse) if mse > 0 else 99.0)
        S.append(calc_ssim(a, b))
        with torch.no_grad():
            L.append(float(lp()(a.unsqueeze(0).to("cuda"), b.unsqueeze(0).to("cuda"))))
    P, S, L = map(np.array, (P, S, L))
    np.savez(f, psnr=P, ssim=S, lpips=L)
    return P, S, L

_vbf = f"{NPZD}/vbench.json"
_vb = json.load(open(_vbf)) if os.path.exists(_vbf) else {}
def vbench(path):
    if path in _vb: return _vb[path]
    r = subprocess.run([PY, "repro/backup/scripts/vbench_iq.py", path], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith(path)]
    if not line: raise RuntimeError(f"vbench fail {path}: {r.stderr[-200:]}")
    d = {m.group(1): float(m.group(2)) for m in re.finditer(r'(\d+)f=([\d.]+)', line[0])}
    d['full'] = float(re.search(r'full=([\d.]+)', line[0]).group(1))
    _vb[path] = d; json.dump(_vb, open(_vbf, "w"), indent=0)
    return d

def sign_test(diffs, direction=1):
    from math import comb
    d = [x for x in diffs if x != 0]
    n = len(d); k = sum(1 for x in d if x*direction > 0)
    return (sum(comb(n, i) for i in range(k, n+1)) / 2**n if n else 1.0), k, n

# ---------- LC: variants vs qvg on f93 ----------
print("== LC variants (f93 PSNR vs QVG mean-of-3) ==")
for var in ("pca", "pcar6"):
    diffs, p9val = [], None
    for p in range(1, 11):
        ref = glob.glob(f"results/multiprompt/lc/bf16_rep0/p{p}/*/segment_1.mp4")
        tv = glob.glob(f"results/multiprompt/lc/{var}_rep0/p{p}/*/segment_1.mp4")
        if not ref or not tv: continue
        qs = []
        for r in range(3):
            tq = glob.glob(f"results/multiprompt/lc/qvg_rep{r}/p{p}/*/segment_1.mp4")
            if tq:
                Pq, _, _ = ref_metrics(ref[0], tq[0], f"lc_p{p}_qvg{r}")
                qs.append(Pq[93])
        Pv, Sv, Lv = ref_metrics(ref[0], tv[0], f"lc_p{p}_{var}0")
        d = Pv[93] - np.mean(qs); diffs.append(d)
        if p == 9: p9val = (Pv[93], d)
    d = np.array(diffs); pv, k, n = sign_test(d)
    print(f"{var}: mean Δ {d.mean():+.3f} ± {d.std():.3f}, win {k}/{n}, p={pv:.4f}, p9 f93={p9val}")

# ---------- SF: variants VBench700 vs qvg ----------
print("\n== SF variants (VBench700 vs QVG mean-of-3) ==")
qvg700 = {}
for p in range(1, 11):
    vals = []
    for r in range(3):
        g = glob.glob(f"results/multiprompt/sf/qvg_rep{r}_f180/p{p}/*.mp4")
        if g: vals.append(vbench(g[0])["700"])
    if vals: qvg700[p] = np.mean(vals)
for var in ("pca", "pcaa128", "pcar6", "pcavmean", "pcar6vmean"):
    diffs = []
    for p in range(1, 11):
        g = glob.glob(f"results/multiprompt/sf/{var}_rep0_f180/p{p}/*.mp4")
        if g and p in qvg700: diffs.append(vbench(g[0])["700"] - qvg700[p])
    if not diffs: print(f"{var}: no data"); continue
    d = np.array(diffs); pv, k, n = sign_test(d)
    print(f"{var}: mean Δ {d.mean():+.3f} ± {d.std():.3f}, win {k}/{n}, p={pv:.4f}")

# ---------- HY: variants ref-metrics + VBench vs qvg ----------
print("\n== HY variants (frames[13:] + VBench vs QVG, 5 seeds) ==")
for var in ("pca", "pcav90", "pcav00"):
    dp, dv, ds, dl = [], [], [], []
    for s in range(5):
        ref = f"results/multiprompt/hy/bf16_s{s}/0-{s}.mp4"
        tq = f"results/multiprompt/hy/qvg_s{s}/0-{s}.mp4"
        tv = f"results/multiprompt/hy/{var}_s{s}/0-{s}.mp4"
        if not all(os.path.exists(x) for x in (ref, tq, tv)): continue
        Pq, Sq, Lq = ref_metrics(ref, tq, f"hy_s{s}_qvg")
        Pv, Sv, Lv = ref_metrics(ref, tv, f"hy_s{s}_{var}")
        dp.append(Pv[13:].mean() - Pq[13:].mean()); ds.append(Sv[13:].mean() - Sq[13:].mean())
        dl.append(Lv[13:].mean() - Lq[13:].mean()); dv.append(vbench(tv)["full"] - vbench(tq)["full"])
    if not dp: print(f"{var}: no data"); continue
    dp, ds, dl, dv = map(np.array, (dp, ds, dl, dv))
    print(f"{var}: ΔPSNR {dp.mean():+.3f} (win {sum(x>0 for x in dp)}/5)  ΔSSIM {ds.mean():+.4f} ({sum(x>0 for x in ds)}/5)  "
          f"ΔLPIPS {dl.mean():+.4f} ({sum(x<0 for x in dl)}/5)  ΔVB {dv.mean():+.2f} ({sum(x>0 for x in dv)}/5)")
