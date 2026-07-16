# HY-WorldPlay 三指标：QVG match paper + PCA-KV N4 同协议对比

## 协议（既定，非本日新立）

- **起点窗口 [23,36) 帧均值** —— 原始复现建立的 paper-match 口径
  （[REPORT.md](../backup/REPORT.md) §起点窗口协议，13/13 方法×位宽进 ±2.6 dB）。
  HY 第 1 个 chunk 量化尚未生效（34-38 dB 编码噪声底），字面首帧读不到 INT2 误差，
  故 HY 的"漂移起点"= 量化刚生效后的 [23,36) 窗口（LC 才是字面 frame 93 首帧，
  见 [0713 报告](../0713/report-0713.md)）。
- 参考 = BF16 同 prompt/seed/config（`results/hyworldplay/bf16_matched/0-0.mp4`，189 帧）；
  LPIPS 一律 paper 口径（[0,1] 直喂 vgg）。
- N4 配置与 LC 完全相同（r=4 双侧 PCA、coef 2-bit asym、残差 2-bit asym B=128），
  **未对 HY 的 256 维头重调**（LC/SF 是 128 维；秩 4/256 的相对子空间只有 LC 的一半）。

## 对表（HY INT2，[23,36) 窗口均值）

| 方法 | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| paper Table 1（QVG） | 29.174 | 0.882 | 0.094 |
| QVG（我们，同窗口，n=1） | 26.782 | 0.9663 | 0.1597 |
| **PCA-KV N4（我们，同窗口，n=1）** | 26.207 | **0.9765** | **0.1427** |

- **QVG match 判定**：PSNR −2.39 dB，在原始复现 13/13@±2.6 dB 的判定标准内（当时同表：
  RTN +1.33、QuaRot +2.55）——match 成立。SSIM/LPIPS 与 paper 绝对值系统偏移
  （参考视频与评测链路不同源），只做方法间同协议对比，不与 paper 横比。
- **N4 vs QVG**：PSNR −0.58 dB、SSIM +0.010、LPIPS 好 11%。**HY 上两者大致打平、
  感知指标略优**——不同于 LC 的 +2.9 dB 大胜。全程均值（chaos 区，仅参考）：
  QVG 18.77/0.8434/0.3740 vs N4 18.15/0.8335/0.3799，同样打平。

## 读数

1. N4 未调 256 维头就已与 QVG 打平——LC 大胜、SF/HY 打平，**没有一处输**。
2. HY 的 PSNR 微差与 SSIM/LPIPS 反超并存：残差非对称量化保住了感知结构，
   逐像素误差略高。r 或残差块对 256 维头的适配（如 r=8 或 128 维半头分裂，
   见 [n4-int2-impl-plan.md](n4-int2-impl-plan.md) 的 HY 决策点）是现成的提升抓手。
3. 单 prompt/seed、n=1（QVG 非确定性 σ≈0.18 dB）——结论方向可用，写 paper 需多 prompt 复验。

## 复现

```bash
. repro/backup/scripts/env_fix.sh
# 生成（无 PCA_SF_STORE_FIX——HY 的 cache 布局本来就是 BHSD，SF 补丁只适用于 SF）
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
# 逐帧三指标 + 窗口均值
.venv/bin/python repro/backup/scripts/sf_ref_metrics.py \
  results/hyworldplay/bf16_matched/0-0.mp4 189 results/pcastudy/hy_n4/0-0.mp4
python - <<'EOF'
import numpy as np
for f in ['sf_kc_256_vc_256_nstages_1','sf_hy_n4']:
    d = np.load(f'repro/backup/protosearch/{f}.npz'); w = slice(23,36)
    print(f, d['psnr'][w].mean(), d['ssim'][w].mean(), d['lpips'][w].mean())
EOF
```

坑两则：① `ModuleNotFoundError: No module named 'models'` → 必须把
`experiments/HY-WorldPlay/wan` 加进 PYTHONPATH（runpy 不加目标脚本目录）；
② 带 `PCA_SF_STORE_FIX=1` 会炸 `data tokens (24)` 断言——HY read/write 均为
BHSD，permute 会把头维当 token 维。
