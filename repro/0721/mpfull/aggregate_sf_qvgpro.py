#!/usr/bin/env python3
"""聚合 SF qvg-nom 判决:qvgprot/qvgproc vs mp100 既有臂(bf16/kivi/qvg/pcaa128)。
四轴 VBench 均值 + 对 ours 的逐 prompt 配对符号检验。→ repro/0721/sf-qvgpro-vbench.md"""
import os, glob, json, math

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
cache = {}
for f in sorted(glob.glob("repro/0718/npz/vbench4_shard*.json")) + \
         sorted(glob.glob("repro/0718/npz/vbench4_sfq_shard*.json")) + \
         ["repro/0718/npz/vbench4.json"]:
    if os.path.exists(f):
        cache.update(json.load(open(f)))

ARMS = ["bf16", "kivi", "qvg", "pcaa128", "qvgprot", "qvgproc"]
AX = ["BC", "IQ", "SC", "AQ"]
per = {a: {} for a in ARMS}  # arm -> p -> {BC..}
for a in ARMS:
    for p in range(1, 101):
        g = glob.glob(f"results/multiprompt/mp100/sf/{a}_rep0_f180/p{p}/*.mp4")
        if g and f"{g[0]}::700" in cache:
            per[a][p] = cache[f"{g[0]}::700"]

def sign_test(w, l):
    n = w + l
    if n == 0: return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(w, l) + 1)) / 2**n)

L = ["# SF qvg-nom 判决:VBench 四轴(700 帧窗,mp100 同协议)", ""]
L.append("| Arm | n | BC | IQ | SC | AQ |")
L.append("|---|---|---|---|---|---|")
for a in ARMS:
    d = per[a]
    if not d: continue
    m = {x: sum(v[x] for v in d.values()) / len(d) for x in AX}
    L.append(f"| {a} | {len(d)} | {m['BC']:.2f} | {m['IQ']:.2f} | {m['SC']:.2f} | {m['AQ']:.2f} |")
L += ["", "## 配对符号检验(vs ours=pcaa128,同 prompt 双向)", "",
      "| 对比 | 轴 | win/tie/loss(对方视角) | p(双侧) |", "|---|---|---|---|"]
for a in ["qvgprot", "qvgproc"]:
    common = sorted(set(per[a]) & set(per["pcaa128"]))
    for x in AX:
        w = sum(1 for p in common if per[a][p][x] > per["pcaa128"][p][x])
        l = sum(1 for p in common if per[a][p][x] < per["pcaa128"][p][x])
        t = len(common) - w - l
        L.append(f"| {a} vs ours | {x} | {w}/{t}/{l} | {sign_test(w, l):.4f} |")
open("repro/0721/sf-qvgpro-vbench.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
