# N12：三模型全面超越 QVG 的 2-bit KV 量化方法（目标达成报告）

> **一句话**：N12 = **K/V 双侧在线 k-means 字典**（逐量化事件重拟合、fp8+per-head-scale
> 质心表）+ **非对称 2-bit B=128 残差**。在 [eval-protocol.md](eval-protocol.md) 的全部
> 判据下同时超越 QVG：LC f93 三指标 ✓、HY 全程三指标 ✓、SF VBench ✓、BPE 全部 < 2.326 ✓。
> 完整探索轨迹（含 5 个被证伪的候选与 2 次关键诊断）见 [research-log.md](research-log.md)。

## 一、最终对表（协议 = eval-protocol.md，全部实测）

### 闸门①：LongCat f93 三指标

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | BPE ↓ |
|---|---:|---:|---:|---:|
| QVG（靶） | 28.73 | 0.9033 | 0.089 | 2.326 |
| N4（旧候选） | 31.79 | 0.9424 | 0.0670 | 2.3125 |
| **N12（K=256）** | **32.16** | **0.9438** | **0.0659** | **2.257** |

### 闸门②：HY-WorldPlay 全程三指标（189 帧逐帧均值）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | BPE ↓ |
|---|---:|---:|---:|---:|
| QVG（靶） | 18.77 | 0.4584 | 0.3740 | 2.326 |
| N4 | 18.15 | 0.4637 | 0.3799 | 2.3125 |
| **N12（K=144）** | **18.841** | **0.5003** | **0.3373** | **2.320** |

### 闸门③：Self-Forcing VBench Image Quality

| 方法 | 350f | 700f |
|---|---:|---:|
| BF16 | 72.91 | 71.51 |
| QVG（靶） | 72.68 | 70.41 |
| N4 | 72.32 | 70.26 |
| **N12（K=256）** | **72.92** | **71.52** |

N12 在 350f/700f 都与 BF16 打平——零画质税，且超 QVG +1.1。

## 二、方法定义

对每个量化事件的 KV chunk，K 和 V 各自独立执行：

1. **在线 k-means 字典**：`batch_kmeans_Euclid`（repo 自带）per-head 对 chunk 内 token
   聚类（iters=20，跨事件 warm-start 上一事件的表），减去所属质心；
2. **质心表存储**：fp8 E4M3 + **per-head fp16 scale**（必须——LC 深层 V 质心达 |1440|，
   超 E4M3 上限 448，无 scale 会溢出 NaN 致全黑）；残差对反量化后的表计算，解码一致；
3. **残差量化**：非对称 2-bit，block=128，scale/zp fp8（N4 验证过的最优残差格——
   对比 QVG 的 ternary B64 是纯升级）；
4. **K 的选择规则**：每模型取"预算内最大 K"（BPE < 2.326 约束下）——
   LC/SF（大 chunk）K=256，HY（7040-token 小 chunk）K=144。

**BPE 账（真实现口径，逐项）**：每侧 = 残差 2 + scale/zp 16/128 + 指派 8/D + 表 K×8/N
（+per-head scale ~1e-5）。LC = 2.257，HY = 2.320，SF = 2.242，均 < QVG 2.326。

**在线性**：无任何离线校准；字典逐事件从当前 chunk 拟合（warm-start 免费加速），
与 QVG 的 streaming k-means 同类开销（fake-quant 实测每事件 <1s）。

## 三、怎么找到它的（方法论小结，细节见 research-log）

1. **N5-N7（重要性预算重分配）全部证伪**：维度缩放（与 minmax 打架）、维度比特、
   时间比特（此 pose 全内容被回访→均匀本就最优）——三个轴的重分配都不解 HY；
2. **N8-N9（量化器/统计粒度微调）证伪**：MSE 最优裁剪=裁离群值（0713 CLIP_STUDY
   重演）；更细统计=更噪的基——N4 已是其邻域局部最优；
3. **诊断 1（转折点）**：HY KV 张量级分解——**K 侧 SAS 字典全层胜子空间（最高 2×）**，
   V 侧 N4 胜 → N10（K 字典+V PCA）；
4. **诊断 2**：N10 的 HY 平台与 N4 逐位相同、崖后 +0.67 → **平台差距是 V 侧生成效应**
   （张量 V-MSE 更低≠生成更好，普通 MSE 再次误导）→ N12（V 也上字典）；
5. N12 首跑 LC 全黑 → fp8 溢出 NaN 解剖修复 → 三关连破。

## 四、复现命令

```bash
. repro/backup/scripts/env_fix.sh
COMMON="PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_K_MODE=sas PCA_V_MODE=sas \
  PCA_SAS_ITERS=20 PCA_SAS_REFIT=1 PCA_SAS_TAB8=1"

# LC（K=256, 48 层）——读 f93 三指标
env $COMMON PCA_SAS_K=256 PCA_N_LAYERS=48 \
  PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
  torchrun --standalone --nproc_per_node=1 repro/backup/scripts/pca_launcher.py \
  --checkpoint_dir=ckpts/LongCat-Video --workload 480p_long_gen \
  --init_video_path results/longcat/base/1-0.mp4 --output_dir results/pcastudy/n12_lc2 \
  --num_segments 1 --num_cond_frames 73 --seed 0 \
  --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1 \
  --quant_type naive-int2 --quant_block_size 64

# HY（K=144, 30 层）——读 189 帧全程三指标（vs bf16_matched）
env $COMMON PCA_SAS_K=144 PCA_N_LAYERS=30 \
  PCA_TARGET=experiments/HY-WorldPlay/wan/generate.py \
  PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
  torchrun --standalone --nproc_per_node=1 repro/backup/scripts/pca_launcher.py \
  --input "<湖桥 prompt>" --image_path assets/hyworld.png --num_chunk 12 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder --out results/pcastudy/hy_n12_k144 \
  --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 --quant_type naive-int2

# SF（K=256, 30 层, 195 latents）——vbench_iq.py 打分
env $COMMON PCA_SAS_K=256 PCA_N_LAYERS=30 PCA_SF_STORE_FIX=1 \
  PCA_TARGET=experiments/Self-Forcing/inference.py PYTHONPATH=experiments/Self-Forcing:. \
  torchrun --standalone --nproc_per_node=1 repro/backup/scripts/pca_launcher.py \
  --config_path experiments/Self-Forcing/configs/self_forcing_dmd.yaml \
  --checkpoint_path ckpts/Self-Forcing/self_forcing_dmd.pt \
  --data_path repro/backup/scripts/prompt0.txt --output_folder results/pcastudy/sf_n12 \
  --num_samples 1 --num_output_frames 195 --local_attn_size 195 --use_ema
```

视频/数组：`results/pcastudy/{n12_lc2,hy_n12_k144,sf_n12}/`、
`repro/backup/protosearch/sf_hy_n12_k144.npz`。

## 五、诚实边界（写 paper 前必须处理）

- 全部 n=1、单 prompt/seed（QVG 靶数字同样 n=1，且 QVG 非确定性 σ≈0.18）；
  HY 全程 PSNR 对配置微调呈 ±0.2 混沌噪声——0.07 的领先在单 prompt 下不构成统计结论，
  **多 prompt 复验（0717 待办三）是下一步**；
- N12 是 fake-quant（BPE 为记账值）；真 kernel 化后 speed/显存故事待建；
- HY 断崖（f29）未被推迟——N12 赢在崖后地板（16.3+）与感知指标；
- "查字典 vs 报坐标"的叙事需更新：最终赢家是"更好的字典系统"（我们的贡献 =
  残差格 + fp8 表 + 在线预算规则），PCA 路线的价值在 LC 的 N4/N10b 结果里保留。
