# 研究日志：BPE<QVG 且三模型全超 QVG（协议见 eval-protocol.md）

每次尝试一行；判定按 eval-protocol.md 的门控（LC → HY → SF）。
基线靶子：LC f93 QVG=28.73/0.9033/0.089；HY 全程 QVG=18.77/0.4584/0.3740；
SF VBench700 QVG=70.41。N4 候选现状：LC 31.79/0.9424/0.067 ✓ / HY 18.15 ✗ / SF 70.26 ✗。

## 检索结论（2026-07-17）

- **MixKVQ**（arXiv 2512.19206）：per-channel 显著性 = E|Q_d|×K 量化难度，三档混精度，
  **online 零校准**，开销 2.17%——直接验证"Q 幅度加权、在线"可行。
- **Block-GTQ**（arXiv 2606.24033）：RoPE 2D 块能量分 = (E‖q‖²+E‖k‖²)/2，
  贪心比特分配 J=Σ s·4^{-b} 可证最优；校准式，但统计量可在线免费获得；
  证实 RoPE 配对粒度。
- OSCAR（2605.17757）：重要性=旋转到 Q/注意力加权谱；离线校准——只借思想。

## 方法 N5：N4 + 在线 Q 能量加权 K 量化

生成时在 q_norm 出口累计 per(层,头,维) E[q²]（零额外前向）；量化 K 时
`s_d=(E[q_d²]/mean)^{α/2}`（几何均值归一），k′=k⊙s，N4 管线作用于 k′，解码除回。
效果 = PCA 基与残差量化器在注意力 logit 加权误差下工作。V 不动。
真实现代价：每(头,chunk) 一个 D 维 fp16 向量（与 mu 同级，BPE +≈0.005，仍<2.326）。
实现：`pca_quant.py`（PCA_QW_ALPHA + provider）、`pca_launcher.py`（LC q_norm 包装）。

## 尝试记录

| # | 方法/配置 | 读数 (PSNR/SSIM/LPIPS) | 判定 | 备注 |
|---|---|---|---|---|
| 0 | N4 基线（r4/coef2/asym/vpca/B128） | LC f93 31.79 / 0.9424 / 0.0670 | LC ✓，HY ✗（18.15），SF ✗（70.26） | 出发点 |
| 1-2 | ~~N5 α=0.5/1.0 @LC~~ **作废** | 31.79-31.80（≈N4） | 无效读数 | **QW 未激活**：LC 在 prefill 后立即量化条件窗，惰性 hook 错过唯一量化事件 → 静默回退纯 N4（差异是运行噪声）。教训：静默回退必须有 WARNING（已加） |
| 2b | （bug 修复）HY 首轮三臂崩溃 | — | — | diffusers 按签名过滤 processor kwargs，包装函数吞了 viewmats → 已用 __signature__ 镜像修复 |
| 3 | N5 α=1.0 pair=half @LC（真激活） | LC f93 28.88 / 0.9049 / 0.0932 | **✗ 淘汰** | s 达 54×，过度倾斜毁整体保真——α=1.0 出局 |
| 4 | **N5 α=0.5 pair=half @LC（真激活）** | **LC f93 31.86 / 0.9417 / 0.0672** | **闸门① ✓**（超 QVG；≈N4，PSNR +0.07） | s 范围 0.31-7.4，温和倾斜；进 HY |
| 5 | N5 α=0.5 pair=interleave @HY | HY 全程 17.30 / 0.4458 / 0.4222 | **闸门② ✗ → N5 判死** | 三臂全低于 N4/QVG（α1.0=17.11，无配对=16.40）。失败机制：对角缩放与残差 per-block minmax 打架——拉伸撑大块内 range，低权维步长爆炸 |
| 6 | **N6：同预算按维比特重分配**（共享 scale 无 range 拉伸；贪心 Block-GTQ 规则，b∈[1,4]，Σ=2D；BPE 严格不变） | LC 跑中 | — | 单测：加权 MSE 比均匀好 2.2×；N5 失败机制的直接对症解 |
| 2c | （bug 修复 ×2）HY head_dim 打包 | — | — | ①HY cache head_dim = 模型×2（rope+prope 拼接）→ 重要性平铺；②LC 量化在 prefill 后即刻发生 → q_norm 包装移到 __init__ |
