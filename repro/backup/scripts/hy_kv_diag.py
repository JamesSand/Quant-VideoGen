"""HY KV tensor-level diagnosis: where does QVG-SAS beat N4 on HY?
Per dumped layer, per side (K/V): rel-L2 of
  N4   = mean + top-4 PCA + 2-bit coef + asym 2-bit B128 residual
  SAS  = kmeans-256 centroid subtraction + ternary B64 residual (official INT2 grid)
Usage: hy_kv_diag.py <dump_dir>
"""
import glob
import os
import sys

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("PCA_R", "4")
os.environ.setdefault("PCA_COEFF_BITS", "2")
os.environ.setdefault("PCA_RES_GRID", "asym")
os.environ.setdefault("PCA_RES_BLOCK", "128")
os.environ.setdefault("PCA_V_MODE", "pca")

from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
import pca_quant as pq

dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(0)


def q_ternary(x, B=64):
    S = x.shape
    xb = x.reshape(*S[:-1], S[-1] // B, B)
    sc = xb.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    return (torch.clamp(torch.round(xb / sc), -1, 1) * sc).reshape(S)


def sas_recon(x, K=256, iters=20):
    """x: [1, H, S, D] -> SAS smoothed+ternary reconstruction."""
    xf = x[0].float()                                   # [H, S, D]
    assign, cent = batch_kmeans_Euclid(xf, K, max_iters=iters)[:2]
    sm = torch.gather(cent, 1, assign.unsqueeze(-1).expand(-1, -1, xf.shape[-1]))
    res = xf - sm
    return (sm + q_ternary(res, 64)).unsqueeze(0)


def n4_recon(x):
    return pq.pca_fake_quant(x.float(), int(os.environ["PCA_R"]))


def rel_l2(a, b):
    return (a - b).norm().item() / b.norm().item()


rows = []
for f in sorted(glob.glob(f"{sys.argv[1]}/layer_*.pt")):
    layer = int(f.split("_")[-1].split(".")[0])
    d = torch.load(f, map_location=dev)
    for side in ("k", "v"):
        x = d[side].float()
        n4 = rel_l2(n4_recon(x), x)
        sas = rel_l2(sas_recon(x), x)
        rows.append((layer, side, n4, sas))
        print(f"L{layer:02d} {side.upper()}: N4={n4:.4f}  SAS={sas:.4f}  "
              f"{'SAS wins' if sas < n4 else 'N4 wins'} ({n4 / max(sas, 1e-9):.2f}x)",
              flush=True)

import numpy as np
for side in ("k", "v"):
    r = [(a, b) for l, s, a, b in rows if s == side]
    n4m = np.mean([a for a, b in r]); sasm = np.mean([b for a, b in r])
    print(f"MEAN {side.upper()}: N4={n4m:.4f} SAS={sasm:.4f}")
