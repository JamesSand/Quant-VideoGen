# Kernel 速度：同一块 KV cache 上 QVG / QVG-Pro / PCA-KV N4 的编解码耗时

**设置**：同一块真实 LongCat K chunk（`lc_kv.pt` layer24，bf16 [1,32,S,128]；采集截断 12k，
平铺补到真实 chunk 尺寸 S=29,640），H100 单卡，warmup 后取 3 次均值。
QVG/QVG-Pro 走**发布版 triton 真路径**（`triton_prq_quantize_tensor`），N4 是**朴素 torch
fp32**（未写任何自定义 kernel、未打包位）。"全模型"= ×96（48 层 × K/V）。

## Encode（压缩一个 chunk）

| 方法 | ms/层张量 | 全模型/段 | 相对 N4 |
|---|---:|---:|---:|
| QVG 流式配置（S=1, B=64, **iters=2**；SF/HY 实际所用） | **2.8** | **0.27 s** | 快 23× |
| QVG LC 配置（S=1, B=64, **iters=100**） | 66.4 | 6.37 s | ≈打平 |
| **PCA-KV N4**（mean+PCA r=4+系数2bit+非对称残差 B128，朴素 torch） | 64.9 | 6.23 s | 1× |
| QVG-Pro（S=4, B=16, iters=100） | 209.6 | 20.12 s | 慢 3.2× |

## Decode（读路径重建，每次 attention 读付一次）

| 方法 | ms/层张量 | 全模型×56步/段 | 备注 |
|---|---:|---:|---|
| QVG（融合 triton kernel） | **0.2** | ~1.1 s | 拆位+scale+质心累加一个 kernel 搞定 |
| N4（朴素 torch，未打包） | 2.2 | ~12 s | 系数 GEMM+逐元素反量化未融合；相对段生成 ~190s 仍是零头 |

## 读数

1. **迭代次数就是 k-means 的全部命门**：iters=2 的流式配置比 iters=100 快 24×——QVG 在
   SF/HY 上的 encode 其实非常便宜（0.27s/段级别），paper 的质心热启动优化确实值。
2. **N4 的朴素 torch 与重度优化的 QVG-100 打平**（64.9 vs 66.4 ms）、比 QVG-Pro 快 3.2×。
   零自定义 kernel 达到这个位置，且大头（协方差 GEMM）仍是 fp32 关 TF32——换 bf16/TF32
   或写融合 kernel 都有数倍余量。但**对流式 iters=2 的 QVG，N4 现状慢 23×**——若要在
   SF/HY 流式场景争 encode 速度，需把 cov 换低精度 GEMM（预计 ~4-8×）或增量更新协方差。
3. **Decode 双方都可忽略**（≤12s vs 段生成 ~190s），QVG 的融合 kernel 比我们未融合版快
   11×——生产化 N4 需要一个同款融合 decode（工作量与其 accumulate.py 相当）。
4. **勘误**：此前"PCA encode 比 k-means 快 2.4×（5.34 vs 12.7s）"的说法不准——12.7s 是
   管线内 cuda-event 计时（含额外开销），同 chunk 干净对比下 QVG-100 与 N4 是打平。
   本文数字为准。

复现：`repro/backup/scripts/kernel_speed.py`（source env_fix.sh 后单卡运行）。
