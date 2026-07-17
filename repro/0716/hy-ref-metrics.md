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
   实为 16 帧/chunk；**且 12 帧/chunk 在发布代码里结构性跑不通**（0717 实测）：
   `generate.py:205` 强制 memory−context=pred_latent_size（pred=3 时配额=3），而
   `select_mem_frames_wan`（utils.py）的 memory 帧按 4 帧块分配（配额必须是 4 的倍数）
   ——两断言在 pred=3 下不可同时满足。**最终判定：§5.2 的 "12 and 16 frames" 是
   HY/SF 交叉写反的笔误**（代码实际 HY=16、SF=12，且作者用这份代码也跑不了 HY@12）
   ——即我们的 16 帧配置就是 paper 实际的 chunk 尺寸，chunk 粒度嫌疑排除；HY 复现
   缺口收敛到帧范围/pose/参考配置三项（见 issue 草稿）。
3. **逐帧结构 = 平台 + 断崖**（我们的 QVG run，vs 配置匹配的 BF16 参考）：
   帧 1-28 平台（PSNR 33.6-37.2，缓降），帧 29 断崖（25.4），帧 30 起饱和 ~19 dB
   （SSIM 0.55，内容分岔）。断崖位置 ≈ pose 从 `w-8` 切 `s-8` 的**回访点**——镜头
   回头、需要从量化 memory 检索旧内容的时刻。逐边界验证：六段 pose 的五个切换点里
   **只有第一个（w→s，帧 29）产生断崖**（34.5→19.6），其余四个切换全平（±0.5 dB）
   ——分岔是**吸收态**：AR 模型以自己已错位的历史为条件，轨迹不会重新对齐，后续
   切换没有对齐基线可跌，PSNR 恒在"同源场景两段不相关视频"的 14-18 dB 地板上。
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

## 三、修正后的 HY 结论（两段协议，用户 2026-07-17 定稿）

协议：**drop 前段（帧 [1,断崖)）与 drop 后段（帧 [断崖,189)）分别报三指标，
外加断崖帧位置**。断崖 = 首个 PSNR<28 的帧（平台 33+、崖底 20-，28 为干净分界）。
参考 = 配置匹配的 BF16（`results/hyworldplay/bf16_matched/`，189 帧）；
LPIPS paper 口径；SSIM paper 实现。

| 方法 | 断崖帧 | drop 前 PSNR/SSIM/LPIPS | drop 后 PSNR/SSIM/LPIPS |
|---|:---:|---|---|
| QVG INT2 | 29 | **35.11 / 0.9655 / 0.0544** | 15.79 / 0.3696 / 0.4321 |
| PCA-KV N4（INT2） | 29 | 31.98 / 0.9439 / 0.0770 | 15.62 / 0.3797 / 0.4350 |
| QVG INT4 | **35** | 35.14 / 0.9640 / 0.0500 | 19.72 / 0.6638 / 0.2219 |

paper Table 1 对照：INT2 QVG = 29.174 / 0.882 / 0.094；INT4 QVG = **34.454 / 0.954 / 0.051**。

三个读数：

1. **INT4 的 drop 前段与 paper 三指标几乎精确吻合**（35.14/0.964/0.050 vs
   34.454/0.954/0.051，差 +0.7 dB / +0.010 / −0.001）——paper 的 HY INT4 数字实质
   就是"未分岔段保真度"；INT2 的 paper 值落在我们两段之间，与"作者 INT2 run 分岔
   更晚、全程均值被崖后拉低到 29"一致（见 §一）。
2. **崖后段测的是内容分岔的噪声地板，不是量化质量**——两方法崖后打平
   （15.79 vs 15.62）；N4 崖后头几帧衰减更缓（21-22 dB vs QVG 19.6），初版"打平"
   正是这段在跨崖窗口里抵消了平台劣势。
3. **INT4 的隐藏卖点**：断崖比 INT2 晚 6 帧（35 vs 29）且崖后退化温和得多
   （SSIM 0.66 vs 0.37）——位宽不仅提平台保真，还推迟/减缓轨迹分岔。

**诚实结论：HY 上 N4 drop 前段输 QVG 3.1 dB**（断崖帧相同）。三模型画像修正为：
**LC 大胜（+3.1 dB）/ SF 起点打平 / HY 落败（−3.1 dB）**。

### 补救实验（0717 凌晨）：两个调参假设双双证伪

"256 维头未调参"的嫌疑已测试——两个 arm（BPE 代价相同：8 coef/token）：

| arm | 配置 | 平台 [1,29) | 判定 |
|---|---|---|---|
| N4 r=4（基线） | r=4 直接在 256 维上 | 31.98 / 0.9439 / 0.0770 | — |
| N4 r=8 | 秩翻倍 | 31.99 / 0.9439 / 0.0768 | 无差异 |
| N4 split128 | 切成 2×128 维子头、各 r=4（与 LC 同构，`PCA_SPLIT_D=128`） | 31.98 / 0.9438 / 0.0767 | 无差异 |

逐帧数组确有差异（个别帧 ±2 dB），平台均值收敛同一水平——**HY 的瓶颈不在子空间
秩/粒度**。结合 [mse-reduction.md](mse-reduction.md) 的张量级发现（K 侧中层 SAS 的
256 原子字典本就胜过 4 维子空间），HY 的差距更可能来自 K 侧结构对"字典型"方法的
偏好。下一步候选（待讨论）：误差在 HY 上按 层×头×K/V 的分解定位、K/V 不对称预算、
残差位宽/块尺寸扫描。数组：`sf_hy_n4_r8.npz`、`sf_hy_n4_split128.npz`。

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
