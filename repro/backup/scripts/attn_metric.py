"""Attention-output error metric (Q3): rel-L2 of Attn(Q,K,V) vs Attn(Q,K_hat,V_hat)
on dumped (q,k,v) triples, with random relative-RoPE rotations applied to q so the
metric sees the pair-mixing effect that plain tensor MSE misses.

Validation criterion: must retrodict the KNOWN HY generation ranking
    dict(18.84) > N4(18.15) > cube(17.3) > KLT-lm_mm(16.6)
which tensor rel-L2 orders INCORRECTLY (KLT best). Usage: attn_metric.py <dump_dir>
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PCA_R", "4")
os.environ.setdefault("PCA_RES_GRID", "asym")
os.environ.setdefault("PCA_RES_BLOCK", "128")
os.environ.setdefault("PCA_V_MODE", "pca")
os.environ.setdefault("PCA_KLT_GRID", "lm_mm")
os.environ.setdefault("PCA_KLT_BMIN", "1")

import pca_quant as pq
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)


def rand_rope(q):
    """q: [n, H, D] -> random per-pair relative rotation (interleaved pairing)."""
    n, H, D = q.shape
    th = torch.rand(n, 1, D // 2, device=q.device) * 6.28318
    c, s = th.cos(), th.sin()
    x = q.view(n, H, D // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    out = torch.empty_like(x)
    out[..., 0] = x1 * c - x2 * s
    out[..., 1] = x1 * s + x2 * c
    return out.view(n, H, D)


def attn_out(q, k, v):
    """q [n,H,D]; k,v [H,S,D] -> [n,H,D] attention output (chunk-restricted)."""
    logits = torch.einsum("nhd,hsd->nhs", q, k) / (k.shape[-1] ** 0.5)
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("nhs,hsd->nhd", p, v)


def sas_recon(x, K=256, iters=20):
    a, c = batch_kmeans_Euclid(x, K, max_iters=iters)[:2]
    sm = torch.gather(c, 1, a.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    return sm + pq._asym_quant_lastdim_grouped(x - sm, 2, 128)


def quantize(name, x):
    """x: [H, S, 128] (rope half). Returns quantized recon."""
    if name == "n4":
        return pq.pca_fake_quant(x.unsqueeze(0), 4)[0]
    if name == "dict":
        return sas_recon(x)
    if name == "klt":
        return pq._klt_fake_quant(x.unsqueeze(0))[0]
    if name == "cube":
        old = pq.PCA_CUBE_C, pq.RES_BLOCK
        pq.PCA_CUBE_C = 32
        out = pq._cube_fake_quant(x.unsqueeze(0))[0]
        pq.PCA_CUBE_C = old[0]
        return out
    raise ValueError(name)


METHODS = ["dict", "n4", "cube", "klt"]

rows = {m: [] for m in METHODS}
for f in sorted(glob.glob(f"{sys.argv[1]}/layer_*.pt")):
    d = torch.load(f, map_location=dev)
    if d.get("q") is None or d["q"].shape[0] == 0:
        continue
    q = rand_rope(d["q"].to(dev).float()[:512])           # [n, 24, 128]
    k = d["k"].float()[0][..., :128].contiguous()          # rope half [24, S, 128]
    v = d["v"].float()[0][..., :128].contiguous()
    ref = attn_out(q, k, v)
    for m in METHODS:
        kq = quantize(m, k)
        vq = quantize(m, v)
        e = (attn_out(q, kq, vq) - ref).norm() / ref.norm()
        rows[m].append(e.item())
    lay = f.split("_")[-1].split(".")[0]
    print(f"L{lay}: " + "  ".join(f"{m}={rows[m][-1]:.4f}" for m in METHODS), flush=True)

import numpy as np
print("MEAN: " + "  ".join(f"{m}={np.mean(rows[m]):.4f}" for m in METHODS))
print("生成排序参照: dict(18.84) > n4(18.15) > cube(17.3) > klt(16.6) —— 数值越小应越靠前")
