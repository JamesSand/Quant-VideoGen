# Report 2026-07-13：首帧口径下的全方法量化对比

> 协议：**首个生成帧 PSNR**——LongCat seg1 输出的帧索引 93（93 帧共享初始视频之后的第一个新生成帧），量化 run vs BF16 run，同 prompt（滑板手, prompt_idx=1）/同 seed(0)/同评测器。该口径测的是"量化误差刚注入注意力、未被自回归混沌放大"的纯净信号，也是与 paper Table 1 数字吻合的口径（见 [REPORT.md](REPORT.md) §一.3）。
> QuaRot 为本项目移植（官方无发布实现，`quarot_quant.py`：Hadamard 旋转 + 分块 RTN，无 clip，等价官方默认 clip_ratio=1.0）；RTN 为仓库自带 `naive-int*`；QVG/QVG-Pro 为论文原方法。

## 总表（LongCat，frame 93，dB）

| 位宽 | 方法/变体 | 本地实测 | Paper Table 1 | 名义压缩率 | 备注 |
|---|---|---:|---:|---:|---|
| **INT2** | QVG-Pro（S=4, B=16） | **31.03** | 30.376 | 4.97× | 全场最高 |
| | QuaRot 非对称 B16 | **30.38** | 21.573 | 6.40× | ⚠️ 高出 paper 的 QuaRot 8.8 dB |
| | QVG（发布版 / rngiso） | 28.72 / 29.34 | 28.716 | 6.89× | k-means 非确定性 ±0.6 |
| | QuaRot 非对称 B128 | 24.55 | — | ~7.3× | |
| | RTN B16 | 21.85 | 20.872 | 6.40× | |
| | QuaRot 对称 B16 | 21.42 | (21.573) | 6.40× | 与 paper 的 QuaRot 值吻合 |
| **INT4** | RTN B16 | **35.23** | 32.984 | 3.55× | INT4 最高 |
| | QuaRot 对称 B16 | 34.01 | (33.744) | 3.55× | |
| | QVG（三次独立测量） | 33.71 / 33.54 / 35.84 | 37.141 | 3.72× | 波动 ±2.3（质心 atomic_add 非确定性） |
| | QuaRot 非对称 B16 | 33.04 | 33.744 | 3.55× | |
| | QuaRot 非对称 B128 | 30.72 | — | | |

## 三个读数

1. **INT2：标准非对称 QuaRot 是被 paper 严重低估的强基线**（本地 30.38 vs paper 声称 21.57），在首帧口径下略胜 QVG（28.7~29.3）约 1~1.7 dB，压缩率相近（6.40× vs 6.89×）。paper 的 QuaRot 数字只与**对称量化**变体吻合（21.42）——作者的私有移植大概率用了对称量化。"QVG 显著优于 QuaRot"这一 Table 1 的核心对比仅在该弱化版基线下成立。
2. **INT4：全部方法挤在 33-35 dB 窄带**，方法间排序被 QVG 自身 ±2.3 dB 的运行间波动（k-means 质心 atomic_add 非确定性）淹没；确定性方法中 RTN B16（35.23）最高——15 个量化等级 + 细粒度 scale 已足够，旋转与聚类均不再提供优势。
3. **QVG 无争议的优势档位是 QVG-Pro**（31.03，INT2 最高），但其压缩率也最低（4.97×）。

## 关联研究

- [CLIP_STUDY.md](CLIP_STUDY.md)：纯丢弃式离群值裁剪严格有害（裁 0.1% 即损失 2.6~7.2 dB）→ 离群值是信号
- [ROPE_DISPERSION.md](ROPE_DISPERSION.md)：旋转类操作为何伤害视频 KV 的可聚类结构（推导+实测）
- [REPORT.md](REPORT.md)：复现总报告（协议定位、压缩率验证、长度极限）

## 数据来源

全部数值取自 `repro/protosearch/*.npz` 逐帧数组（frame 93），对应视频在 `results/{longcat,longcat_rngiso,quarot,clipstudy,diag}/`；QVG 第三次测量来自 clip 扫描的 p=100 对照臂。
