#!/usr/bin/env python3
"""H1 修正版之我方侧:同一批 chunk 上测 Budget-PCA 的减法消掉能量与最终 relL2²,
与 h1_real_path.py 同协议(全部 8 chunk),终版配置(K 张量侧)。
输出并入 h1_h2_data.npz 的 {model}_ourspath(列:chunk_id, removed, relL2sq_final)。
"""
import os, sys
import numpy as np
import torch

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
sys.path.insert(0, ".")
sys.path.insert(0, "repro/backup/scripts")

CFG = {
    "lc": dict(PCA_R="4", PCA_RES_GRID="asym", PCA_RES_BLOCK="128", PCA_RES_AXIS_K="channel"),
    "sf": dict(PCA_R="4", PCA_RES_GRID="asym", PCA_RES_BLOCK="128", PCA_RES_AXIS_K="channel"),
    "hy": dict(PCA_R="4", PCA_HALF_R_K="9,0", PCA_RES_GRID="asym", PCA_RES_BLOCK="128",
               PCA_RES_AXIS_K="channel", PCA_RES_BLOCK_K="64",
               PCA_RES_GRID_KP="ternary", PCA_RES_BLOCK_KP="64"),
}

res = dict(np.load("repro/0720/why/h1_h2_data.npz"))
for model in ("lc", "sf", "hy"):
    for k_env in list(os.environ):
        if k_env.startswith("PCA_"):
            del os.environ[k_env]
    os.environ.update(CFG[model])
    import importlib
    import pca_quant
    importlib.reload(pca_quant)
    rows = []
    for ci in range(8):
        k = torch.load(f"repro/0720/chunks/{model}/chunk_{ci:03d}.pt", map_location="cuda")["k"].float()
        tot = float((k ** 2).sum())
        # 消掉的能量 = μ + top-r 低秩重建部分(与质心减法同角色)
        if model == "hy":
            Xr, Xp = k[..., :128], k[..., 128:]
            parts = []
            for Xh, r in ((Xr, 9), (Xp, 0)):
                mu = Xh.mean(dim=2, keepdim=True)
                Xc = (Xh - mu).float()
                if r > 0:
                    B, H, S, D = Xc.shape
                    Xf = Xc.reshape(B * H, S, D)
                    cov = torch.einsum("bsd,bse->bde", Xf, Xf) / S
                    ev, V = torch.linalg.eigh(cov)
                    Vr = V[..., -r:]
                    low = torch.einsum("bsd,bdr,ber->bse", Xf, Vr, Vr).reshape(B, H, S, D)
                else:
                    low = torch.zeros_like(Xc)
                parts.append(mu + low)
            sub = torch.cat(parts, dim=-1)
        else:
            mu = k.mean(dim=2, keepdim=True)
            Xc = k - mu
            B, H, S, D = k.shape
            Xf = Xc.reshape(B * H, S, D)
            cov = torch.einsum("bsd,bse->bde", Xf, Xf) / S
            ev, V = torch.linalg.eigh(cov)
            Vr = V[..., -4:]
            low = torch.einsum("bsd,bdr,ber->bse", Xf, Vr, Vr).reshape(B, H, S, D)
            sub = mu + low
        removed = 1 - float(((k - sub) ** 2).sum()) / tot
        khat, _ = pca_quant.pca_fake_quant_kv(k.to(torch.bfloat16), k.to(torch.bfloat16))
        relsq = float(((k - khat.float()) ** 2).sum()) / tot
        rows.append((ci, removed, relsq))
        print(f"{model} chunk{ci:03d}: 减法消掉 {removed*100:.2f}%  最终 relL2² {relsq*100:.3f}%"
              f"  残差回收率 {(1 - relsq/(1-removed))*100:.1f}%", flush=True)
    res[f"{model}_ourspath"] = np.array(rows)
np.savez("repro/0720/why/h1_h2_data.npz", **res)
print("saved h1_h2_data.npz:{model}_ourspath")
