# Plan：PCA-KV N4 的真 INT2 实现（位打包 + 真实 cache + kernel）

目标：把 N4（K/V 双侧 mean+PCA r=4 + 非对称 2bit 残差 B=128）从 fake-quant 原型变成
**真实打包存储的量化 cache**，拿到 QVG 同款的三个系统数字：显存、长度极限、端到端速度。
可行性判断：**高**——所有重原语（batched GEMM/eigh）现成，自定义 kernel 只有
pack/unpack+dequant，比 QVG 的 k-means 栈（迭代 assign/update/permute/gather）简单。

## 1. 存储格式（每 head、每 chunk，D=128）

| 项 | 格式 | 每 token 开销 |
|---|---|---|
| 字典 mu + V₄ | bf16 [128] + [128×4]（5 向量 = 1.25 KB/头/chunk） | ~0（按 chunk 摊） |
| 系数 | 4×2bit 打包 = **1 字节** + scale/zp 各 FP8 = 2 字节 | 3 B |
| 残差 | 128×2bit 打包 = **32 字节** + scale/zp 各 FP8 = 2 字节（块=整头） | 34 B |
| 合计 | | **37 B/token/头** vs bf16 256 B → **6.92×，BPE 2.3125** |

诚实记账修正：fake 版账面 2.253 没算系数 zp 的 8bit；真实现 = **2.3125**，仍 < QVG 2.326。

## 2. 里程碑

- **M0 数值预检（0.5 天，先行避坑）**：fake-quant 加"FP8 scale/zp 模拟"臂——fake 版
  的 scale 是 fp32，真实现必须 FP8；先验证质量损失（预期 <0.1 dB，QVG 同款设计），
  出问题在这层修（如 scale 用 FP16）。
- **M1 编码 + 打包 + cache 管道（1-1.5 天）**：encode = torch 现成算子（mean/协方差
  GEMM/eigh/投影）+ 一个 triton pack kernel（量化+2bit 打包，模板=官方 `quant_pack.py`）；
  新 quant_type `pca-int2` 走 `compress_kv_cache` 返回 dict → 复用现有 `store_quantized`
  dict 路径（三模型的 dict 管道是官方踩熟的，绕开我们撞过的 fake 张量布局坑）；
  decode 先用未融合 torch（实测 2.2ms/层，能用）。**交付：真实显存数字 + SF/LC 长度极限重测**。
- **M2 融合 decode kernel（1 天）**：一个 triton kernel 完成拆位→系数 dequant→
  c·V₄ᵀ（每元素 4 次 FMA）→残差 dequant→+mu。对标官方 `accumulate.py`（我们无 gather，
  更规则）。**交付：端到端速度三方对比（bf16/QVG/N4），检验访存收益是否同为 −20% 级**。
- **M3 基准与文档（0.5 天）**：显存/长度/速度/质量四表 + REPRODUCE 记录；
  质量回归验证（真实现 vs fake 版 PSNR 差应 <0.1 dB）。
- **M4 流式 encode 热启动（1 天，可选/研究性）**：基跨 chunk 复用 + 每 K chunk 重算
  + 协方差 bf16 GEMM——目标从 6.2s 逼近流式 QVG 的 0.27s；失败退路 = 保持 6s
  （QVG-100 档，LC 场景够用）。

总计 3~4.5 天。M0-M3 无研究风险，M4 有（子空间漂移速度未测）。

## 3. 难点清单（讨论用）

1. **FP8 scale 的质量代价**（M0 裁决）——fake 版用 fp32 scale 是隐性放水，真实现对齐
   QVG 的 E4M3 惯例后 +2.9 dB 还剩多少？预期损失很小但必须实测。
2. **eigh 的算力形态**：`torch.linalg.eigh` 算全部 128 个特征对而我们只要 top-4；
   batched cuSOLVER 在 128×128 是毫秒级、LC 一段 96 个矩阵不构成瓶颈——**先不优化**；
   若 M4 流式场景吃紧再换幂迭代/LOBPCG（只算 top-r，能省 ~10×）。
3. **小 chunk 的字典密度**：SF 每 16 latents 一个 chunk → 字典 1.25KB×12头×2×30层
   ≈ 0.9MB/chunk，长视频累计可观但仍 ≪ 数据本体（记账里已含）；HY 4-latent chunk
   密度×4——必要时 HY 用跨 chunk 共享基（M4 的机制顺带解决）。
4. **读路径峰值**：与 QVG 相同的"全量反量化回 bf16 再 attention"瞬态——长度极限
   预期与 QVG 同受此限（~3× 而非 7×），这是结构性问题、两家平等，不在本计划内解。
5. **HY 的 256 维拼接 cache**（rope‖prope）：PCA 直接在 256 维上做还是拆两半各做？
   建议拆半（各 128 维、各自协方差），实现最简且与两分支语义对齐——HY 集成放 M3 之后。

## 4. 不做什么

- 不动注意力 kernel（继续 bf16 SDPA，与 QVG 同假设）
- 不做训练侧任何事
- M4 之前不承诺流式 encode 性能
