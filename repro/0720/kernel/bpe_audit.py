#!/usr/bin/env python3
"""BPE hard audit: encode REAL dumped chunks with the real kernel, count stored
bytes per component, assert <= 2.326; and verify fake-path (table-generating)
vs kernel dequant equivalence on the same chunks.

Output: repro/0720/bpe-audit.md + hard exit(1) on any violation.
"""
import os, sys
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, "repro/0720/kernel")
sys.path.insert(0, "repro/backup/scripts")
from bp_quant import bp_encode_fast, bp_decode, bp_bytes

BUDGET = 2.326
rows, fails = [], []

def comp_bits(d, numel):
    """Per-component bits/elem from actual tensor bytes."""
    def b(*keys):
        n = 0
        for k in keys:
            t = d.get(k)
            if torch.is_tensor(t): n += t.numel() * t.element_size()
        return n * 8 / numel
    scalars = 4 * 8 / numel * sum(1 for k in ("f_cs", "f_cz", "f_sc", "f_zp") if k in d)
    return dict(codes=b("res_pack"), scales=b("res_scale", "res_zp"),
                coef=b("coef_pack"), coef_meta=b("coef_scale", "coef_zp"),
                factors=b("f_exp", "f_sc_t", "f_zp_t") + scalars,  # 归一化因子入账(0720 外审勘误)
                amort=b("mu", "basis"))

def audit_tensor(name, x, cfgs, assert_budget=True):
    """cfgs: list of (slice, encode_kwargs) — halves for HY."""
    total_bytes, numel = 0, x.numel()
    parts = []
    for sl, kw in cfgs:
        xs = x[..., sl].contiguous()
        d = bp_encode_fast(xs, **kw)
        total_bytes += bp_bytes(d)
        parts.append((d, xs.numel()))
    bpe = total_bytes * 8 / numel
    comp = {}
    for d, n in parts:
        for k, v in comp_bits(d, numel).items():
            comp[k] = comp.get(k, 0) + v * 1  # already normalized by full numel? fix below
    # recompute per-part correctly
    comp = {}
    for d, n in parts:
        for k, v in comp_bits(d, x.numel()).items():
            comp[k] = comp.get(k, 0) + v
    ok = bpe <= BUDGET
    rows.append((name, bpe, comp, ok))
    return bpe

LC_CFG = [(slice(0, 128), dict(r=4, grid="asym", block=128, axis="channel"))]
SF_CFG = LC_CFG
HYK_CFG = [(slice(0, 128), dict(r=9, grid="asym", block=64, axis="channel")),
           (slice(128, 256), dict(r=0, grid="ternary", block=64, axis="channel"))]
HYV_CFG = [(slice(0, 128), dict(r=9, grid="asym", block=128, axis="token")),
           (slice(128, 256), dict(r=0, grid="asym", block=128, axis="token"))]

for model, cfgk, cfgv, n_ch in (("lc", LC_CFG, LC_CFG, 3), ("sf", SF_CFG, SF_CFG, 3), ("hy", HYK_CFG, HYV_CFG, 3)):
    for ci in range(n_ch):
        f = f"repro/0720/chunks/{model}/chunk_{ci:03d}.pt"
        if not os.path.exists(f): continue
        d = torch.load(f, map_location="cuda")
        bk = audit_tensor(f"{model} chunk{ci} K", d["k"].float(), cfgk, assert_budget=False)
        bv = audit_tensor(f"{model} chunk{ci} V", d["v"].float(), cfgv, assert_budget=False)
        cache_bpe = (bk + bv) / 2
        ok = cache_bpe <= BUDGET
        rows.append((f"{model} chunk{ci} **KV合并(判定口径)**", cache_bpe, {"codes":0,"scales":0,"coef":0,"coef_meta":0,"factors":0,"amort":0}, ok))
        if not ok: fails.append(f"{model} chunk{ci}: cache BPE {cache_bpe:.4f} > {BUDGET}")

# fake-vs-kernel equivalence on real chunks (final configs)
import importlib
os.environ.update(PCA_FP8SIM="1", PCA_COEFF_BITS="2", PCA_V_MODE="pca")
eq_rows = []
for model, env, enc in (
    ("lc", dict(PCA_R="4", PCA_RES_GRID="asym", PCA_RES_BLOCK="128",
                PCA_RES_AXIS_K="channel", PCA_RES_AXIS_V="channel"),
     lambda k: bp_encode_fast(k, r=4, grid="asym", block=128, axis="channel")),
    ("sf", dict(PCA_R="4", PCA_RES_GRID="asym", PCA_RES_BLOCK="128",
                PCA_RES_AXIS_K="channel", PCA_RES_AXIS_V="channel"),
     lambda k: bp_encode_fast(k, r=4, grid="asym", block=128, axis="channel")),
):
    for k_, v_ in list(env.items()):
        os.environ[k_] = v_
    import pca_quant
    importlib.reload(pca_quant)
    d = torch.load(f"repro/0720/chunks/{model}/chunk_000.pt", map_location="cuda")
    k = d["k"].float()
    fake, _ = pca_quant.pca_fake_quant_kv(k, k.clone())
    kern = bp_decode(enc(k), dtype=torch.float32)
    rf = float((fake.float() - k).norm() / k.norm())
    rk = float((kern.float() - k).norm() / k.norm())
    eq_rows.append((model, rf, rk, abs(rf - rk) / rf))
    if abs(rf - rk) / rf > 0.02:
        fails.append(f"{model}: fake/kernel relL2 mismatch {rf:.5f} vs {rk:.5f}")

out = ["# BPE 硬审计(真 kernel 逐字节;预算 2.326)\n",
       "| 张量 | 实测 BPE | codes | scales | coef | coef_meta | 归一因子 | μ/基摊销 | 合规 |",
       "|---|---|---|---|---|---|---|---|---|"]
for name, bpe, c, ok in rows:
    out.append(f"| {name} | **{bpe:.4f}** | {c['codes']:.3f} | {c['scales']:.3f} | "
               f"{c['coef']:.4f} | {c['coef_meta']:.4f} | {c.get('factors', 0):.4f} | "
               f"{c['amort']:.4f} | {'✓' if ok else '✗ 超预算'} |")
out.append("\n## fake 路径(出表)vs kernel(实测)一致性(同 chunk relL2)\n")
out.append("| 模型 | fake relL2 | kernel relL2 | 相对偏差 |")
out.append("|---|---|---|---|")
for m, rf, rk, dd in eq_rows:
    out.append(f"| {m} | {rf:.5f} | {rk:.5f} | {dd*100:.2f}% |")
out.append(f"\n**审计结论:{'全部合规 ✓' if not fails else '发现违规: ' + '; '.join(fails)}**")
open("repro/0720/bpe-audit.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
sys.exit(1 if fails else 0)
