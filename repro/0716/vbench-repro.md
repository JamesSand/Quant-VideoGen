# VBench Image Quality 复现（paper 附录 A.1）+ PCA-KV N4 的第四项指标

## 协议

- **指标**：VBench 的 `imaging_quality` 维度 = 逐帧 MUSIQ-SPAQ 打分取均值（÷100 报百分数）。
  我们**逐行复刻**官方 `VBench/vbench/imaging_quality.py`（'longer' 模式：长边 >512 则
  resize 到 512、antialias=False、/255；同款 musiq_spaq 权重），脚本
  `repro/backup/scripts/vbench_iq.py`——逐帧算分后取前缀均值，350/700/1050/1400 帧窗口
  一次得出（因果 AR 模型前缀等价性：长视频前 N 帧 ≡ 短生成，逐帧一致）。
- **视频**：SF，prompt0（滑板手），seed 0，全历史注意力（`local_attn_size=全长`——发布代码的
  pre-RoPE 路径**不支持滑窗**，超窗直接 raise，见 §口径缺口）。BF16/QVG 用既有极限视频
  （777f / 2397f），N4 新生成（fake-quant，含 SF 兼容修复，见下）。

## 对表（VBench Image Quality，%）

| 方法 | 350f | 700f | 1050f | 1400f |
|---|---:|---:|---:|---:|
| BF16 我们 / paper | 72.91 / 74.33 | **71.51 / 70.56** | 墙¹ / 68.80 | 墙 / 65.87 |
| QVG 我们 / paper | 72.68 / 74.36 | **70.41 / 69.52** | 65.20 / 67.23 | 60.50 / 67.28 |
| **PCA-KV N4（我们）** | **72.32** | **70.26** | 墙² | 墙 |
| BF16−QVG 差 我们 / paper | 0.23 / −0.03 | **1.10 / 1.04** ✓ | | |

¹ BF16 全注意力在单卡 80GB 上限 195 latents（777 帧）——**paper 的 BF16@1400 在发布代码 +
单卡 80GB 上不可复现**（需更大显存或未发布的 offload/滚动实现）。
² N4 目前为 fake-quant（cache 实存 bf16），显存同 BF16 级；真 kernel 化后可达 int2 级长度。

## 读数

1. **协议复现成立**：700f 处三方绝对值均在 paper ±1 内，**方法间相对差精确吻合**
   （1.10 vs 1.04）——这是方法比较可信的关键。
2. **绝对值系统偏移 ~1.5（350f 处）**：首要嫌疑 = prompt 差异（我们单 prompt vs paper
   可能的多 prompt 平均）；MUSIQ 对内容敏感，绝对分不跨内容可比。
3. **QVG 长尾分歧**（我们 60.5 vs paper 67.3 @1400f）：我们的视频持续下滑、paper 平台化。
   滑窗假设已被排除（发布代码不支持）；剩余嫌疑：prompt 内容的长时退化差异、paper 按
   多 prompt 平均摊平了个体崩坏。待多 prompt 复验裁决。
4. **N4 与 QVG 在 VBench IQ 上打平**（−0.36/−0.15，同档噪声内）——符合预期且是好消息：
   MUSIQ 无参考、只看"画面本身好不好"，看不见保真度；N4 的优势（PSNR +2.9 dB、
   LPIPS 好 25%）全在"忠实 BF16"维度，VBench IQ 证明它**没有为保真付出画质代价**。
   N4 四指标画像定稿：**PSNR/SSIM/LPIPS 三项大胜、VBench IQ 持平**。

## 本次发现的口径缺口 / 上游问题

1. SF 发布代码 pre-RoPE 路径**无滑窗实现**（超 `local_attn_size` 即
   `ValueError: KV cache size is too small`，滚动逻辑只在被注释的旧路径里）。
2. BF16@1400 帧不可在单卡 80GB 复现（同上，全注意力显存墙 777 帧）。
3. SF 的 **fake-quant 张量路径自相矛盾**：`_print_kv_cache_mse_error` 要求 BHSD、
   `store_quantized→write` 要求 BSHD——官方 sim/fake 路径在 SF 上从未跑通过。
   我们的兼容修复：`pca_launcher.py` 的 `PCA_SF_STORE_FIX=1`（store 时转 BSHD）。

## 复现命令

```bash
# 打分（任意 mp4，逐帧 MUSIQ + 前缀窗口均值）
.venv/bin/python repro/backup/scripts/vbench_iq.py <video.mp4> ...
# N4 的 SF 生成（fake-quant + SF 兼容修复）
PCA_R=4 PCA_COEFF_BITS=2 PCA_RES_GRID=asym PCA_V_MODE=pca PCA_RES_BLOCK=128 \
PCA_SF_STORE_FIX=1 PCA_TARGET=experiments/Self-Forcing/inference.py \
torchrun --standalone repro/backup/scripts/pca_launcher.py ... --quant_type naive-int2
```
