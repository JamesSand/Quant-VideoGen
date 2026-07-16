"""Reproduce QVG Fig.6 (SAS quantization-MSE reduction) on real SF KV, and test
whether PCA-KV (N4) shows the same kind of reduction.

Protocol per (tensor in {K,V}, grid in {INT2 ternary B64, INT4 sym15 B64}):
  MSE_ratio = MSE(quant(raw)) / MSE(smooth_then_quant reconstruction)
Smoothers: SAS = k-means-256 centroid subtraction (repo's own batch_kmeans_Euclid,
max_iters=100); PCA = mean + top-r self-cov basis subtraction (r=4/8, coef fp
here — isolating smoothing power; N4-full adds coef 2bit + asym B128 residual).
Data: sf_qkv.pt layers {0,15,29}, mid window (9360 tokens, 12 heads).
"""
import os, sys
import torch

sys.path.insert(0, os.getcwd())
from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)

d = torch.load("results/kvplot/sf_qkv.pt", map_location="cpu", weights_only=False)
slots = d["slots"]

def get(layer, t):
    fr = sorted(f for (l, f) in slots if l == layer and 87 <= f <= 92)
    return torch.cat([slots[(layer, f)][t] for f in fr], dim=1)[0].float()  # [S,H,D]

def q_ternary(x, B=64):
    S = x.shape
    xb = x.reshape(*S[:-1], S[-1]//B, B)
    sc = xb.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    return (torch.clamp(torch.round(xb/sc), -1, 1)*sc).reshape(S)

def q_sym15(x, B=64):  # INT4 symmetric, max_int=7 (repo grid)
    S = x.shape
    xb = x.reshape(*S[:-1], S[-1]//B, B)
    sc = (xb.abs().amax(-1, keepdim=True)/7).clamp_min(1e-8)
    return (torch.clamp(torch.round(xb/sc), -7, 7)*sc).reshape(S)

def q_asym(x, bits, B):
    S = x.shape
    xb = x.reshape(*S[:-1], S[-1]//B, B)
    mn, mx = xb.amin(-1, keepdim=True), xb.amax(-1, keepdim=True)
    sc = ((mx-mn)/(2**bits-1)).clamp_min(1e-8)
    return (torch.clamp(torch.round((xb-mn)/sc), 0, 2**bits-1)*sc+mn).reshape(S)

def sas_smooth(x):  # x [H,S,D] -> residual after kmeans-256 centroid subtraction, + centroids recon part
    ids, cents, _, _ = batch_kmeans_Euclid(x, n_clusters=256, max_iters=100)
    assigned = torch.gather(cents, 1, ids.unsqueeze(-1).expand(-1, -1, x.shape[-1]).long())
    return x - assigned, assigned

def pca_smooth(x, r):  # x [H,S,D] -> residual after mean+top-r subtraction, + lowrank part
    mu = x.mean(1, keepdim=True)
    xc = x - mu
    cov = torch.einsum("hsd,hse->hde", xc, xc) / x.shape[1]
    _, vecs = torch.linalg.eigh(cov)
    Vr = vecs[:, :, -r:]
    c = torch.einsum("hsd,hdr->hsr", xc, Vr)
    low = torch.einsum("hsr,hdr->hsd", c, Vr)
    return xc - low, mu + low, (c, Vr, mu)

def mse(a, b): return float(((a-b)**2).mean())

print("layer,tensor,grid,raw_mse,sas_mse,sas_x,pca4_mse,pca4_x,pca8_mse,pca8_x,n4full_mse,n4full_x")
for L in (0, 15, 29):
    for tname in ("k", "v"):
        X = get(L, tname).permute(1, 0, 2).contiguous().to(dev)   # [H,S,D]
        for gname, qf in (("INT2", q_ternary), ("INT4", q_sym15)):
            raw = mse(qf(X), X)
            res_s, cent = sas_smooth(X)
            sas = mse(cent + qf(res_s), X)
            res4, low4, (c4, V4, mu4) = pca_smooth(X, 4)
            pca4 = mse(low4 + qf(res4), X)
            res8, low8, _ = pca_smooth(X, 8)
            pca8 = mse(low8 + qf(res8), X)
            # N4 full recipe (INT2 row only meaningful): coef asym2 per-token + residual asym2 B128
            if gname == "INT2":
                ch = q_asym(c4, 2, c4.shape[-1])
                lowq = torch.einsum("hsr,hdr->hsd", ch, V4)
                resq = X - mu4 - lowq
                n4 = mse(mu4 + lowq + q_asym(resq, 2, 128), X)
            else:
                n4 = float("nan")
            print(f"L{L},{tname.upper()},{gname},{raw:.5f},{sas:.5f},{raw/sas:.2f},"
                  f"{pca4:.5f},{raw/pca4:.2f},{pca8:.5f},{raw/pca8:.2f},"
                  f"{n4:.5f},{raw/n4:.2f}" if n4==n4 else
                  f"L{L},{tname.upper()},{gname},{raw:.5f},{sas:.5f},{raw/sas:.2f},"
                  f"{pca4:.5f},{raw/pca4:.2f},{pca8:.5f},{raw/pca8:.2f},nan,nan")
