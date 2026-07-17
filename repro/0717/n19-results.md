# N19：无 k-means 约束下三模型全超 QVG（最终达成报告，取代 N12）

> **一句话**：N19 = **N4 的 PCA 框架（均值+top-4 自协方差基+2-bit 系数+非对称残差）
> + 锚点保真事件调度**——首个量化事件的残差用 3/4-bit、后续事件用 trit 打包
> （1.6 bit），按已知视频长度预判摊销预算、单事件视频自动退化为平坦 N4。
> **零 k-means、零字典、零校准、全在线**。在 [eval-protocol.md](eval-protocol.md)
> 全部判据（含 0717 新增无 k-means 硬约束）下同时超越 QVG。
> 探索全轨迹（N13-N19，7 个证伪候选 + RoPE 机制发现）见 [research-log.md](research-log.md)。

## 一、最终对表（全部实测；SF 已验证 hijack>0 的真量化 run）

### 闸门①：LongCat f93（单事件 → 调度退化为 N4，代码路径逐位相同）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | BPE ↓ |
|---|---:|---:|---:|---:|
| QVG（靶） | 28.73 | 0.9033 | 0.089 | 2.326 |
| **N19（=N4 退化）** | **31.79** | **0.9424** | **0.0670** | **2.3125** |

### 闸门②：HY-WorldPlay 全程（189 帧，5 事件）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | BPE ↓ |
|---|---:|---:|---:|---:|
| QVG（靶） | 18.77 | 0.4584 | 0.3740 | 2.326 |
| N4 | 18.15 | 0.4637 | 0.3799 | 2.3125 |
| ~~N12 字典（被禁参考）~~ | 18.841 | 0.5003 | 0.3373 | 2.320 |
| N19 3:t | 19.120 | 0.5094 | **0.3360** | **2.082** |
| **N19 4:t** | **19.481** | **0.5157** | 0.3416 | 2.283 |

**断崖被推迟**（首个合法做到的方法）：QVG/N4 f29 → 3:t f30 → **4:t f31**；
崖后地板 17.03（QVG 15.79）。机制 = 位宽是唯一被证明能移分岔点的量
（INT4：f29→f35），把位宽集中花在"回访时必被检索的锚点内容"上。

### 闸门③：Self-Forcing VBench IQ（8 事件；hijack=1 已验证）

| 方法 | 350f | 700f | BPE |
|---|---:|---:|---:|
| BF16 | 72.91 | 71.51 | 16 |
| QVG（靶） | 72.68 | 70.41 | 2.326 |
| N4 | 72.32 | 70.26 | 2.3125 |
| **N19 3:t** | **72.92** | **72.99** | **1.973** |
| N19 4:t | 72.65 | 72.87 | 2.098 |

## 二、方法定义（完整）

基座 = N4：per (head, chunk) `X ≈ mu + quant₂(coef)·V₄ᵀ + 残差`，V₄ = 本 chunk
自协方差 top-4 特征基（在线一次 eigh，无迭代），coef 2-bit 非对称，残差默认
非对称 2-bit B=128。

**锚点保真调度（新）**：设视频共 n 个量化事件（生成长度已知 → n 在线可判）：
- **事件 0（锚点：图像条件/sink 邻接、回访检索的目标区）**：残差升为 3 或 4-bit 非对称；
- **事件 ≥1**：残差降为对称 ternary（真实现 5 trits/字节打包 = 1.6 bit/值）；
- **可行性规则**：仅当摊销 `(b₀ + (n−1)×1.79)/n < 2.326` 时启用；n=1（LC 条件窗）
  自动退化为平坦 2-bit = 纯 N4。

**BPE 账（真实现口径）**：事件 0 = b₀ + 16/128 + coef/mu ≈ b₀+0.253；后续 =
1.6 + 8/128(对称 scale) + 0.128 ≈ 1.79。LC 2.3125；HY(n=5) 3:t 2.082 / 4:t 2.283；
SF(n=8) 3:t 1.973 / 4:t 2.098。**全部 < QVG 2.326**。

**约束合规**：无 k-means/字典（纯 PCA + 标量格）✓；OSCAR 思想的体现 = "重要性加权
预算"沿时间轴的极端形式（锚点内容 = 未来被读概率最高的内容拿最多比特）✓；
全在线零校准 ✓。

## 三、怎么到这一步的（无 k-means 轮次的方法论）

1. Idea #1（构造性质心）：KLT 格点离线碾压 k-means（HY K 0.104 vs 0.278）但
   **生成端全线崩**（HY 16.6）——第 3-5 次张量-生成脱钩；
2. **机制发现：K 是 pre-RoPE 缓存，读取时按位置旋转混合维度对——任何自由旋转/
   按维分配都破坏 RoPE 配对结构**（bmin=1 消融证实非删除维所致；V-KLT 在 LC 也毒）；
3. Idea #3（cube-mean）过 LC 但 HY 不及 N4——位置局部均值不是字典的等价物；
4. **确立事实：HY 上合法方法无一胜 N4，唯一被证明能移断崖的是位宽** → 把位宽
   当预算沿事件轴重分配（N19），一举 +1.3 dB 超 QVG 与被禁字典。

## 四、复现命令

```bash
. repro/backup/scripts/env_fix.sh
BASE="PCA_R=4 PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_V_MODE=pca"

# LC（单事件 → 不设 PCA_EVENT_SCHED，即纯 N4）
env $BASE PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
  torchrun --standalone --nproc_per_node=1 repro/backup/scripts/pca_launcher.py \
  --checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
  --init_video_path results/longcat/base/1-0.mp4 --output_dir results/pcastudy/pca_n4 \
  --num_segments 1 --num_cond_frames 73 --seed 0 \
  --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
  --quant_type naive-int2 --quant_block_size 64

# HY（5 事件，4:t）
env $BASE PCA_N_LAYERS=30 PCA_EVENT_SCHED=4:t \
  PCA_TARGET=experiments/HY-WorldPlay/wan/generate.py \
  PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
  torchrun --standalone --nproc_per_node=1 repro/backup/scripts/pca_launcher.py \
  --input "<湖桥 prompt>" --image_path assets/hyworld.png --num_chunk 12 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder --out results/pcastudy/hy_n19_43t \
  --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 --quant_type naive-int2

# SF（8 事件，3:t 或 4:t；勿忘 --quant_type！验证日志 hijack 计数 >0）
env $BASE PCA_N_LAYERS=30 PCA_EVENT_SCHED=3:t PCA_SF_STORE_FIX=1 \
  PCA_TARGET=experiments/Self-Forcing/inference.py PYTHONPATH=experiments/Self-Forcing:. \
  torchrun --standalone --nproc_per_node=1 repro/backup/scripts/pca_launcher.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path repro/backup/scripts/prompt0.txt --output_folder results/pcastudy/sf_n19_3t \
  --num_samples 1 --num_output_frames 195 --local_attn_size 195 --use_ema \
  --quant_type naive-int2 --quant_block_size 64
```

## 五、诚实边界

- 单 prompt/seed、n=1（同此前所有头条；QVG 靶亦然）；HY 全程指标混沌噪声 ±0.2，
  N19 4:t 的 +0.71 dB 领先幅度大于噪声但仍需多 prompt 复验定案；
- fake-quant（BPE 为真实现记账值，trit 打包/fp8 元数据尚未真实现）；
- 调度的"锚点=首事件"在 LC/HY/SF 上与"图像条件/sink 邻接"重合——更一般的
  内容自适应锚点选择（Idea #2 的 M_Q 流式度量挑锚点）是自然下一步，本轮未需动用；
- N12（k-means 字典版）的历史 SF 读数已勘误作废（未量化 run），其 LC/HY 数字仍有效
  但方法因约束退役。
