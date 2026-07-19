#!/usr/bin/env python3
"""MP100 aggregation: merge shard caches -> per-arm means -> Table-1 markdown +
best-verdict (is Budget-PCA best among quant methods in every column?)."""
import os, glob, json
import numpy as np

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
NPZ = "repro/0718/npz"

vb = {}
for f in sorted(glob.glob(f"{NPZ}/vbench4_shard*.json")) + [f"{NPZ}/vbench4.json"]:
    if os.path.exists(f): vb.update(json.load(open(f)))

def g1(p):
    g = glob.glob(p); return g[0] if g else None

def vb_get(path, mf):
    return vb.get(f"{path}::{mf}")

BPE = {"rtn": "2.25", "kivi": "2.25", "quarot": "2.25", "qvg": "2.326",
       "pcakaxvax": "**2.3125**", "pcaa128kaxvax": "**2.3125**", "pcav90kpternkaxkb64": "**2.29**", "bf16": "16"}
LABEL = {"bf16": "BF16 KV(参考)", "rtn": "RTN", "kivi": "KIVI", "quarot": "QuaRot",
         "qvg": "QVG", "pcakaxvax": "**Budget-PCA(r4+双通道轴)**",
         "pcaa128kaxvax": "**Budget-PCA(r4+双通道轴)**", "pcav90kpternkaxkb64": "**Budget-PCA(K9:0/V9:0+K通道轴B64+KP三值)**"}

def collect(model):
    rows, ns = {}, {}
    if model == "lc":
        arms = ["bf16", "rtn", "kivi", "quarot", "qvg", "pcakaxvax"]
        for arm in arms:
            ref_vals, vbs = [], []
            for p in range(1, 101):
                v = g1(f"results/multiprompt/mp100/lc/{arm}_rep0/p{p}/*/segment_1.mp4")
                if not v: continue
                d = vb_get(v, 0)
                if d: vbs.append((d["BC"], d["IQ"], d["SC"], d["AQ"]))
                if arm != "bf16":
                    f = f"{NPZ}/mp100_lc_p{p}_{arm}.npz"
                    if os.path.exists(f):
                        z = np.load(f)
                        if len(z["psnr"]) > 93:
                            ref_vals.append((z["psnr"][93], z["ssim"][93], z["lpips"][93]))
            rows[arm] = (np.array(ref_vals) if ref_vals else None,
                         np.array(vbs) if vbs else None)
            ns[arm] = (len(ref_vals), len(vbs))
    elif model == "sf":
        arms = ["bf16", "rtn", "kivi", "quarot", "qvg", "pcaa128kaxvax"]
        for arm in arms:
            vbs = []
            for p in range(1, 101):
                v = g1(f"results/multiprompt/mp100/sf/{arm}_rep0_f180/p{p}/*.mp4")
                if v:
                    d = vb_get(v, 700)
                    if d: vbs.append((d["BC"], d["IQ"], d["SC"], d["AQ"]))
            rows[arm] = (None, np.array(vbs) if vbs else None)
            ns[arm] = (0, len(vbs))
    else:
        arms = ["bf16", "rtn", "kivi", "quarot", "qvg", "pcav90kpternkaxkb64"]
        for arm in arms:
            ref_vals, vbs = [], []
            for s in range(10):
                v = f"results/multiprompt/hy/{arm}_s{s}/0-{s}.mp4"
                if not os.path.exists(v): continue
                d = vb_get(v, 0)
                if d: vbs.append((d["BC"], d["IQ"], d["SC"], d["AQ"]))
                if arm != "bf16":
                    f = f"{NPZ}/hy_s{s}_{arm}.npz"
                    if os.path.exists(f):
                        z = np.load(f)
                        ref_vals.append((z["psnr"][13:].mean(), z["ssim"][13:].mean(), z["lpips"][13:].mean()))
            rows[arm] = (np.array(ref_vals) if ref_vals else None,
                         np.array(vbs) if vbs else None)
            ns[arm] = (len(ref_vals), len(vbs))
    return rows, ns

def fmt_table(model, title, rows, ns):
    out = [f"## {title}\n",
           "| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | BG Consist.↑ | Image Quality↑ | Subj Consist.↑ | Aesthetic↑ | BPE↓ |",
           "|---|---|---|---|---|---|---|---|---|"]
    quant_arms = [a for a in rows if a != "bf16"]
    ours = quant_arms[-1]
    # column bests among quant arms
    def mean_of(arm, which, idx):
        r, v = rows[arm]
        x = r if which == "ref" else v
        return None if x is None or len(x) == 0 else float(x[:, idx].mean())
    best = {}
    for col, which, idx, direc in (("psnr","ref",0,1),("ssim","ref",1,1),("lpips","ref",2,-1),
                                    ("bc","vb",0,1),("iq","vb",1,1),("sc","vb",2,1),("aq","vb",3,1)):
        vals = {a: mean_of(a, which, idx) for a in quant_arms}
        vals = {a: v for a, v in vals.items() if v is not None}
        best[col] = max(vals, key=lambda a: vals[a]*direc) if vals else None
    from math import comb
    def sign_p(diffs, direction=1):
        d=[x for x in diffs if x!=0]
        n=len(d); k=sum(1 for x in d if x*direction>0)
        return (sum(comb(n,i) for i in range(k,n+1))/2**n if n else 1.0), n
    verdict = []
    for arm in rows:
        r, v = rows[arm]
        def cell(col, which, idx, nd):
            m = mean_of(arm, which, idx)
            if m is None: return " "
            s = f"{m:.{nd}f}"
            return f"**{s}**" if best.get(col) == arm else s
        if arm == "bf16":
            ref_cells = ["—", "—", "—"]
        else:
            ref_cells = [cell("psnr","ref",0,2), cell("ssim","ref",1,4), cell("lpips","ref",2,4)]
        vb_cells = [cell("bc","vb",0,2), cell("iq","vb",1,2), cell("sc","vb",2,2), cell("aq","vb",3,2)]
        out.append(f"| {LABEL[arm]} | {ref_cells[0]} | {ref_cells[1]} | {ref_cells[2]} | "
                   f"{vb_cells[0]} | {vb_cells[1]} | {vb_cells[2]} | {vb_cells[3]} | {BPE[arm]} |")
    COLIDX = {"psnr":("ref",0,1),"ssim":("ref",1,1),"lpips":("ref",2,-1),
              "bc":("vb",0,1),"iq":("vb",1,1),"sc":("vb",2,1),"aq":("vb",3,1)}
    for col in best:
        if best[col] and best[col] != ours:
            which, idx, direc = COLIDX[col]
            ro, rb = rows[ours], rows[best[col]]
            xo = (ro[0] if which=="ref" else ro[1])
            xb = (rb[0] if which=="ref" else rb[1])
            tie = ""
            if xo is not None and xb is not None and len(xo)==len(xb):
                dif = (xo[:,idx]-xb[:,idx])*direc
                p_hi, n = sign_p(dif, 1)
                p_lo, _ = sign_p(dif, -1)
                p2 = min(1.0, 2*min(p_hi, p_lo))   # two-sided
                lab = '统计平局' if p2 > 0.05 else '显著更差'
                tie = f" (Δ{dif.mean():+.3f}, 双侧p={p2:.3f}, {lab})"
            verdict.append(f"{model}:{col} best={best[col]}{tie}")
    out.append(f"\n覆盖数 n(ref/vb): " + ", ".join(f"{a}={ns[a][0]}/{ns[a][1]}" for a in rows))
    return "\n".join(out), verdict

sections, all_verdicts = [], []
for model, title in (("lc", "LongCat-Video-13B(100 random prompts × f93)"),
                     ("hy", "HY-WorldPlay-8B(10 seeds × frames[13:])"),
                     ("sf", "Self-Forcing-Wan(100 random prompts × 700 帧窗)")):
    rows, ns = collect(model)
    sec, verd = fmt_table(model, title, rows, ns)
    sections.append(sec); all_verdicts += verd

hdr = ("# MP100 定案表(QVG Table-1 指标集,MovieGen 随机 100 prompts,seed=42)\n\n"
       "> 每列在量化方法间的最优加粗;BF16 为参考不参与排名。协议:LC f93、HY [13:] 均值、\n"
       "> SF 700 帧窗;VBench 四维为 CLIP-B/32 / MUSIQ / DINO-B/16 / CLIP-L/14+LAION 口径。\n")
verdict_txt = ("\n## 胜负判定(goal:我们必须每列最好)\n\n" +
               ("**全部列均为 Budget-PCA 最优 ✓**" if not all_verdicts else
                "未达最优的列:\n" + "\n".join(f"- {v}" for v in all_verdicts)))
open("repro/0718/mp100-table.md", "w").write(hdr + "\n\n".join(sections) + verdict_txt + "\n")
print(hdr + "\n\n".join(sections) + verdict_txt)
