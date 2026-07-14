# Self-Forcing：KV 值 + QKV norm，三个分析角度

- **对象**：Self-Forcing（Wan 1.3B），180 帧 bf16 生成中现场捕获
- **指标**：① K/V 的**值**（token×channel 热图）② Q/K/V 的 **norm** 分布
- **角度**：视频前后 ／ chunk 内部 ／ layer 深浅
- 采集：layer {0,15,29} × 时间窗 {帧0-5, 87-92, 174-179}，K 为 pre-RoPE 原始值，Q 取每
  block 最后去噪步；dump `results/kvplot/sf_qkv.pt`；绘图 `plot_sf_kv_summary.py`（值）+
  `plot_qkv_anatomy.py`（norm）

---

## 角度一：视频前后（开头 / 中段 / 结尾，L15）

**【KV 值】** 三个时间窗的 K 热图几乎是同一张图——通道墙位置/强度全程不变；V 同样无变化：

![kv time](figs/sf_kv_time.png)

**【QKV norm】** 三窗箱形几乎重合（Q 14.7→15.4、K 14.4→14.9、V ≈11.8，变化 <5%）：

![qkv norm time](figs/qkv_time.png)

**结论**：KV 分布不随视频进度漂移。一套量化参数/质心全程适用；"视频越长越难压"不成立。

## 角度二：chunk 内部（一个 3-latent 生成 block）

**【KV 值】** 单 block 的 K 热图：通道墙笔直贯穿两条帧边界（黑虚线），块内均质；
左图 K/V norm 逐位置曲线：仅帧起点后 ≤10% 抬升：

![kv chunk](figs/sf_kv_chunk.png)

**【QKV norm】** Q/K/V 三线的块内与帧内逐位置曲线（相对中位数）——Q 最平，K 帧起点小峰，
V 噪声 ±20% 无位置规律：

![qkv norm chunk](figs/qkv_chunk.png)

**结论**：chunk 内部没有 sink 式特殊 token，不需要按位置区别对待。

## 角度三：layer 深浅（L0 / L15 / L29，中段窗）

**【KV 值】** K 随深度恶化：L0 大片灰底几根细墙 → L15 墙变宽密 → **L29 墙又密又猛**；
V 三层始终均匀噪声。底排：K norm 箱线（L29 悬空一簇 ≈105）、V norm（深度不敏感）、
absmax 柱状（K: 9.4→13.1→**93.5**）：

![kv depth](figs/sf_kv_depth.png)

**【QKV norm】** Q/K/V 分层箱线——Q 和 K 在 L29 都长出大离群（Q 拖尾至 ~78，K 脱离簇
103-107 vs 主体 15），V 三层无差：

![qkv norm depth](figs/qkv_depth.png)

**结论**：量化难度集中在**末层的 K（和 Q）**；V 对深度完全不敏感。

---

## 统计总表（median norm ｜ token-norm 极值比 ｜ absmax）

| Layer | 窗 | Q | K | V |
|---|---|---|---|---|
| L0 | begin | 11.1 ｜ 1.32× ｜ 10.4 | 7.7 ｜ 2.04× ｜ 9.3 | 12.1 ｜ 2.82× ｜ 10.8 |
| L0 | mid | 11.1 ｜ 1.21× ｜ 10.2 | 7.6 ｜ 1.77× ｜ 9.4 | 12.4 ｜ 2.69× ｜ 10.5 |
| L0 | end | 10.9 ｜ 1.28× ｜ 10.2 | 7.1 ｜ 1.79× ｜ 9.2 | 12.2 ｜ 2.33× ｜ 10.5 |
| L15 | begin | 14.7 ｜ 1.30× ｜ 16.4 | 14.4 ｜ 1.35× ｜ 11.9 | 11.8 ｜ 3.43× ｜ 8.6 |
| L15 | mid | 15.3 ｜ 1.22× ｜ 14.1 | 14.7 ｜ 1.35× ｜ 13.1 | 11.8 ｜ 2.65× ｜ 7.5 |
| L15 | end | 15.4 ｜ 1.21× ｜ 14.3 | 14.9 ｜ 1.34× ｜ 12.6 | 11.6 ｜ 2.75× ｜ 8.1 |
| L29 | begin | 15.1 ｜ 1.35× ｜ **71.0** | 14.5 ｜ 1.27× ｜ **94.0** | 12.1 ｜ 2.76× ｜ 10.5 |
| L29 | mid | 16.8 ｜ 1.39× ｜ **64.5** | 14.9 ｜ 1.22× ｜ **93.5** | 12.8 ｜ 2.39× ｜ 10.1 |
| L29 | end | 16.3 ｜ 1.33× ｜ **62.0** | 14.8 ｜ 1.18× ｜ **93.5** | 15.0 ｜ 2.21× ｜ 10.0 |

## 三条可动手的推论

1. **按深度重分配量化预算**：末层 K 的 absmax 是中间层 7 倍（93.5 vs 13.1），同样块结构下
   scale 被撑大 7 倍——末层用细块/高位宽/离群旁路、浅层更狠，是免费的质量空间。
2. QVG 的 token 维 k-means 恰好能把 L29 的高 norm 簇吸进专属质心（簇状离群是 k-means
   舒适区）——per-token 轴无离群保护也能工作的隐性原因。
3. per-channel 路线（KIVI/OScaR）搬到视频模型，风险点在末层：高 norm token 簇会撑爆跨
   token 共享的 scale——OScaR 的 Omni-Token Scaling 在末层有真实用武之地（中间层无 TNI，
   见 [kv-distributions.md](kv-distributions.md)）。

局限：单模型（SF）、单 prompt、每窗 2 block、Q 只取最后去噪步；LC/HY 同款整理待采
（采集器已通用化，各一条 pod）。
