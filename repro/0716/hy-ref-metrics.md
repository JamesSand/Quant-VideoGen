# HY-WorldPlay 三指标：paper 协议考古 + 平台期结论（含一次协议级修正）

> **本文档经历一次重大修正**：初版用 [23,36) 跨崖窗口得出"N4 与 QVG 打平、SSIM/LPIPS
> 反超"——该窗口横跨漂移断崖，是混合平均伪影，连同错误的"全局 SSIM"实现一并撤回。
> 以下为修正后的完整版本。

## 一、paper 的 HY 到底怎么测的（考古结论）

paper §5.1 Metrics 原文只对 LC 指定了相似度协议（"For the similarity experiments on
LongCat-Video-13B, we report the number of the first generated chunk as the content
starts to diverge..."）；**HY 的 PSNR/SSIM/LPIPS 协议在 paper 中未指定**。考古证据链：

1. **全程均值不可能**：13 方法×位宽全程实测 12-22 dB（0/13 对上，原始复现结论）；
   确定性的 N4 全程同样 ~18 dB，证明 chaos 是自回归反馈的真混沌而非 RNG 分歧；
   paper 版 SSIM 下全程只有 0.46（paper 报 0.882）。
2. **发布 repo 无法复现 paper 的 HY 对比**：官方 `run_bf16.sh` 与 `run_qvg.sh` 配置
   根本不配对（memory 56/52 vs 48/44；num_chunk 14 vs 12→帧数都不同；pose 7 段 vs
   6 段）——paper 的 HY 参考管线未完整发布。paper 正文还说 HY chunk=12 帧，发布代码
   实为 16 帧/chunk。
3. **逐帧结构 = 平台 + 断崖**（我们的 QVG run，vs 配置匹配的 BF16 参考）：
   帧 1-28 平台（PSNR 33.6-37.2，缓降），帧 29 断崖（25.4），帧 30 起饱和 ~19 dB
   （SSIM 0.55，内容分岔）。断崖位置 ≈ pose 从 `w-8` 切 `s-8` 的**回访点**——镜头
   回头、需要从量化 memory 检索旧内容的时刻。
4. **paper 三元组 = 跨崖窗口平均的形状**：联合扫描显示 [20,32) 窗口给
   31.12 / 0.8823 / 0.0995——SSIM、LPIPS 与 paper 的 0.882/0.094 几乎精确重合，
   PSNR 高 2 dB。断崖的确切位置取决于 chunk 尺寸（他们 12 帧 vs 发布代码 16 帧）、
   pose 脚本与 k-means 非确定性——**paper 的 29.174/0.882/0.094 与"横跨其内部 run
   断崖的早期窗口平均"一致，但发布代码无法逐位复现**。

旁证：LC 列在 paper 原版 SSIM 实现下**精确 match**（QVG f93 = 28.73/0.9033 vs paper
28.716/0.909）——管线对齐无问题，分歧被隔离在 HY 的未发布协议上。

## 二、SSIM 实现勘误（影响 SF/HY 新数组）

`sf_ref_metrics.py` 曾用整帧单窗的"全局 SSIM"（一个 mean/var），读数严重虚高
（HY 窗口 0.966 vs 真实 0.761）。paper 的 metric.py 是 **11×11 avg_pool 局部窗**。
脚本已修复；`precompute_arrays.py` 一直是对的（原始 REPORT 的 hy_*/lc_* 数组不受影响）；
受影响数组已补 `ssim_paper` 键。PSNR/LPIPS 不受影响。

## 三、修正后的 HY 结论（平台期协议）

协议：**平台期 [1,断崖) 逐帧均值 + 断崖帧位置**。断崖 = 首个 PSNR<28 的帧。
两方法断崖同在帧 29（同一 pose 回访点触发）。参考 = 配置匹配的 BF16
（`results/hyworldplay/bf16_matched/`，189 帧）；LPIPS paper 口径；SSIM paper 实现。

| 方法（HY INT2） | 平台 PSNR ↑ | 平台 SSIM ↑ | 平台 LPIPS ↓ | 断崖帧 |
|---|---:|---:|---:|---:|
| QVG | **35.11** | **0.9655** | **0.0544** | 29 |
| PCA-KV N4 | 31.98 | 0.9439 | 0.0770 | 29 |

崖后（内容分岔区，仅参考）：QVG 15.79/0.370/0.432，N4 15.62/0.380/0.435；
N4 崖后头几帧衰减更缓（21-22 dB vs QVG 19.6）——初版"打平"正是这段在跨崖窗口里
抵消了平台劣势。

**诚实结论：HY 上 N4 输 QVG 3.1 dB**（平台期）。三模型画像修正为：
**LC 大胜（+3.1 dB）/ SF 起点打平 / HY 落败（−3.1 dB）**。
现成嫌疑：HY 是 256 维头（LC/SF 128 维），N4 的 r=4 未调参——相对子空间只有 LC 的
一半；候选补救 = r=8 或 128 维半头分裂（见 [n4-int2-impl-plan.md](n4-int2-impl-plan.md)
HY 决策点），未跑，待拍板。

## 四、数据与复现

- 逐帧数组：`repro/backup/protosearch/sf_kc_256_vc_256_nstages_1.npz`（QVG）、
  `sf_hy_n4.npz`（N4）——`ssim` 键为作废的全局版，用 **`ssim_paper`** 键
- 生成命令与两个坑（PYTHONPATH 加 `wan`；HY 严禁 `PCA_SF_STORE_FIX`——HY 原生 BHSD，
  permute 会把 24 头当 token 数）：

```bash
. repro/backup/scripts/env_fix.sh
CUDA_VISIBLE_DEVICES=7 PCA_R=4 PCA_COEFF_BITS=2 PCA_RES_GRID=asym PCA_V_MODE=pca PCA_RES_BLOCK=128 \
PCA_TARGET=experiments/HY-WorldPlay/wan/generate.py \
PYTHONPATH=experiments/HY-WorldPlay:experiments/HY-WorldPlay/wan \
.venv/bin/torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
  --input "<湖桥 prompt>" --image_path assets/hyworld.png --num_chunk 12 \
  --pose "w-8,s-8,a-8,d-8,up-8,down-8" \
  --ar_model_path ckpts/HY-WorldPlay/wan_transformer \
  --ckpt_path ckpts/HY-WorldPlay/wan_distilled_model/model.pt \
  --offload_text_encoder --out results/pcastudy/hy_n4 \
  --memory_frames 48 --temporal_context_size 44 --pred_latent_size 4 --quant_type naive-int2
# 逐帧指标（修复后的 paper 版 SSIM）
.venv/bin/python repro/backup/scripts/sf_ref_metrics.py \
  results/hyworldplay/bf16_matched/0-0.mp4 189 results/pcastudy/hy_n4/0-0.mp4
# 旧数组补 ssim_paper 键
.venv/bin/python repro/backup/scripts/paper_ssim_recalc.py \
  results/hyworldplay/bf16_matched/0-0.mp4 189 "<video>:<npz_tag>"
```
