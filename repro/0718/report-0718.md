# Report 0718 — Multi-prompt campaign：从 n=1 到带显著性的定案（+ 一次翻盘、一次勘误、一个结构性发现）

> 三件套之一。明细：[multi-prompt-results.md](multi-prompt-results.md)、
> 计划：[multi-prompt-plan.md](multi-prompt-plan.md)、复现：[REPRODUCE.md](REPRODUCE.md)。

## 一句话

单日 ~420 次生成（10-20 prompts / 10 seeds、8×H100 就地并行），把 Budget-PCA 的
头条从 n=1 升到配对显著：**LC 三指标胜 9/10（p=0.011）、SF VBench700 胜 15/20
（p=0.021，held-out 泛化）**；HY SSIM/LPIPS 稳胜（9/10、10/10），PSNR 持平，
VBench 确立为结构性权衡（忠实 vs MUSIQ 伪影偏好），8 配置 × 10 seeds 无双赢解。

## 三个剧情转折

1. **SF 翻盘**：基线（0717 定的三值 B64）多 prompt 上输 QVG（3/10，−1.46）——
   旧单 prompt headline 不泛化。扫描发现 **asym B128 才是对的格**（+0.526，15/20，
   p=0.021），且与 LC 同格 → 方法更统一（三值特例取消）；
2. **记账勘误（自查抓获）**：一度"定案 LC r6"，但每秩 +0.0156 BPE，r6=2.344 超线
   非法（r5=2.328 微超线在 research-log #35a 早有记录）。LC 回 r4——它本来就赢。
   教训：任何配置变更先过一遍 BPE 账，历史记录就是护栏；
3. **HY 的 prope 半区第二次兑现 + 权衡前沿**：K-prope 残差换三值 = 参考指标不掉
   （SSIM/LPIPS 两个 seed 集都赢）且省 BPE（2.320→2.258）；但 VBench 缺口
   （−0.78）被证明是结构性的——QVG 的 VBench 常高于 BF16 无损参考，MUSIQ 奖励
   其字典锐化伪影，忠实型方法被 BF16 自身分数封顶。

## 终版配置（全部 BPE < QVG 2.326）

```bash
# LC:  PCA_R=4 PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_V_MODE=pca              (2.3125)
# SF:  同 LC + PCA_SF_STORE_FIX=1                                              (2.3125)
# HY:  PCA_R=4 PCA_HALF_R_K=9,0 PCA_HALF_R_V=9,0 PCA_RES_GRID=asym \
#      PCA_RES_BLOCK=128 PCA_RES_GRID_KP=ternary PCA_RES_BLOCK_KP=64 PCA_V_MODE=pca  (2.258)
```

## 附带产出

- 协议修正四则（QVG 重复=同 seed、SF latent 帧语义、HY [13:] 窗口、Q3 双边受阻）；
- `pca_quant.py` 新基建：每张量/每半区残差格覆盖（`PCA_RES_GRID_K/V/KP/VP`）、zero 格；
- campaign 基建：`campaign.sh`（单任务入口）+ `gpu_queue.sh`（8 卡队列+显存守卫）+
  三个统计脚本，全部可复用于下一次扫描；
- 基础设施教训两则：torchrun 崩溃残留 NCCL 僵尸的识别（kill 前必须独立读证据——
  误杀过一次 LC base）、per-job triton cache 撑爆配额（20G，已清+勿再复制）。

## 待用户仲裁

HY 的"更好"以什么为准：参考三指标（我们稳胜）还是 VBench-IQ（奖励失真，无解）？
若协议只认 VBench,备选解 `全三值`(VB +0.11 但参考指标全负)在案。
