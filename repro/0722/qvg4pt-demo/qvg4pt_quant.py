"""qvg4pt 量化实现(自包含拷贝,0722 demo 用)。

数值路径与 repro/backup/scripts/pca_quant.py 逐行一致(fp8 合法元数据口径,
即 0721 勘误后的终值口径):
  - _fp8                       = pca_quant.py:70-79 逐字
  - _asym_quant_lastdim_grouped = pca_quant.py:89-120 逐字(含 N8 mse 路径)
  - qvg4pt_fake_quant_kv        = pca_quant.py:646-662 的 _qvg4_fake_quant_kv,
                                  仅改名去下划线
与原文件的唯一行为差异:PCA_FP8SIM 默认 "1"(demo 固定跑合法口径;原模块默认
"0" 由 campaign.sh 显式置 1)。kmeans 用 QVG 自己发布的库
quant_videogen/kmeans/kmeans_euclid.py(不复制,保持"减法一个字不动")。
"""
import os

import torch

PCA_FP8SIM = os.environ.get("PCA_FP8SIM", "1") == "1"
PCA_RES_MSEOPT = os.environ.get("PCA_RES_MSEOPT", "0") == "1"
_MSE_RATIOS = (1.0, 0.92, 0.85, 0.78, 0.70, 0.62)


def _fp8(t, per_row=False):
    # normalized fp8 storage. per_row=True (channel-axis residual scales,
    # t: [BH, D, nblk, 1]): one bf16 factor per channel (amortized ~0) —
    # LC's cross-channel scale dynamic range exceeds E4M3's 2^18 and a global
    # factor underflows small-variance channels (the 0720 -3.6dB regression).
    if per_row:
        f = t.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    else:
        f = t.abs().amax().clamp_min(1e-12)
    return (t / f).to(torch.float8_e4m3fn).to(t.dtype) * f


def _asym_quant_lastdim_grouped(x, bits, group, mse_opt=None, fp8_per_row=False):
    """Asymmetric fake-quant with one (scale, zero) per group along the last dim."""
    if mse_opt is None:
        mse_opt = PCA_RES_MSEOPT
    S = x.shape
    xg = x.reshape(*S[:-1], S[-1] // group, group)
    mn = xg.amin(dim=-1, keepdim=True)
    mx = xg.amax(dim=-1, keepdim=True)
    if not mse_opt:
        scale = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8)
        if PCA_FP8SIM:
            scale = _fp8(scale, per_row=fp8_per_row).clamp_min(1e-8)
            mn = _fp8(mn, per_row=fp8_per_row)
        q = torch.clamp(torch.round((xg - mn) / scale), 0, 2 ** bits - 1)
        return (q * scale + mn).reshape(S)
    ctr = (mx + mn) / 2
    half0 = (mx - mn) / 2
    best_err = None
    best = None
    for r in _MSE_RATIOS:
        lo = ctr - half0 * r
        scale = (half0 * 2 * r / (2 ** bits - 1)).clamp_min(1e-8)
        q = torch.clamp(torch.round((xg - lo) / scale), 0, 2 ** bits - 1)
        deq = q * scale + lo
        err = (deq - xg).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best, best_err = deq, err
        else:
            better = err < best_err
            best = torch.where(better, deq, best)
            best_err = torch.minimum(best_err, err)
    return best.reshape(S)


def qvg4pt_fake_quant_kv(k, v):
    """反事实臂(0721):QVG 原装 kmeans 减法(per-head 全 D 维,K=256,LC 官配
    iters=100)不动,残差格由其三电平对称换成【四电平非对称 B64 + fp8 s/z】
    ——回答"QVG 若把 2-bit 的四个码字用满会怎样"。BPE 代价:残差 s/z 从
    0.125 涨到 0.25(B64 asym 双参数),质心/索引账不变。"""
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    iters = int(os.environ.get("PCA_QVG4_ITERS", "100"))
    outs = []
    for x in (k, v):
        B, H, S, D = x.shape
        X = x.float().view(B * H, S, D).contiguous()
        lab, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=iters)
        g = torch.gather(cent, 1, lab.long().unsqueeze(-1).expand(-1, -1, D))
        res = (X - g).view(B, H, S, D)
        rq = _asym_quant_lastdim_grouped(res, 2, 64, mse_opt=False, fp8_per_row=True)
        outs.append((g.view(B, H, S, D) + rq).to(x.dtype))
    return outs[0], outs[1]
