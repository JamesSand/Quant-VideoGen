# PCA-Grid Hash 预注册三指标闸门

本文件在候选端到端结果产生前固定判定规则。比较对象始终是发布代码的
QVG real implementation；双方按实际存储逐字节记账。

## G0：协议

- 硬件：同一 H100 80GB、同一进程、同一输入 chunk。
- 时延：CUDA event；5 次 warmup，50 次 measurement（LC QVG 可降至 10 次）；
  排除首次 `torch.compile`，报告 median / p10 / p90。
- 总时延：一次 K+V encode 加一次 K+V decode；不能只报 encode。
- 数值：必须从实际 packed payload 解码后计算，不使用 fake-quant 代替。

## G1：实际 all-in BPE

| 模型 | QVG 实测上限 |
|---|---:|
| LC | 2.4639 |
| SF | 2.4063 |
| HY（K/V 平均） | 3.3199 |

逐字节包含 table、labels、residual codes、scale、zero-point、normalization、
padding 和任何 selector。任一模型超过对应 QVG 即失败。

## G2：chunk 级重建

- 三模型各 8 个真实 chunk，K/V 分开统计 stored-value MSE。
- 每个模型 K、V 的总 SSE / 总信号能量均不得高于 QVG。
- 灾难层保护：任一 `(chunk,K/V)` 的 MSE 不得超过 QVG 的 1.5 倍。
- MSE 只负责筛选 codec，不替代端到端质量结论。

## G3：kernel 总时延

- LC、SF、HY 各自 8 个 paired chunk 均要求 candidate median
  `(encode+decode)` 小于 QVG。
- 8 个 paired speedup 的 bootstrap 95% 下界必须大于 1.00。
- SF 是主要否决项；只要 SF 未严格更快，不能宣称满足速度硬要求。

当前 QVG 三 chunk 参考总时延：

| 模型 | encode ms | decode ms | total ms |
|---|---:|---:|---:|
| LC | 177.95 | 0.39 | 178.34 |
| SF | 3.11 | 0.233 | 3.343 |
| HY | 3.06 | 0.227 | 3.287 |

## G4：paired 端到端质量

候选相对 QVG 的差值使用 paired TOST，`alpha=0.05`、90% CI。若一侧优效检验
已证明候选更好，也记为通过。实用等价 margin 预注册为：

| 指标 | 不劣 margin |
|---|---:|
| PSNR | -0.10 dB |
| SSIM | -0.002 |
| LPIPS | +0.003 |
| BG Consistency | -0.20 point |
| Image Quality | -0.30 point |
| Subject Consistency | -0.30 point |
| Aesthetic | -0.30 point |

LC/HY 的 PSNR、SSIM、LPIPS 和 SF 的四项 VBench 为 primary gates；其余
VBench 是 guardrail，同样不得越过 margin。Canary 仅用于提前淘汰，最终结论
必须来自 MP100（HY 保持既有 10-seed paired 协议）。

## 总判定

只有 G1、G2、G3、G4 全部通过，才称“三项硬要求达成”。未通过时必须明确记录
失败闸门，不能把条件最优定理、较低 tensor MSE 或单独 encode 加速替代总判定。
