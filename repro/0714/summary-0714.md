# 0714 Summary

今天的 update，五件事。细节全部在各专题 md 里，这里只放结论和关键数字。

## 1. BPE：他们的账其实是诚实的

精确公式 `BPE = r + 8/B + S·8/128 + S·(256·128·16)/(N·128)`，与 README 实测显存**逐字节对上**
（67.318 MB/层 vs 日志 67.32）。paper Table 1 的全部压缩率都能被公式重现——质心、索引、
scale 全计入，账目结构没毛病。仅有的两个小瑕疵：①Table 1 的 6.94× 隐含 ~35k token 的
chunk，发布配置实际 29,640 → 只有 6.88×（高报 0.8%）；②附录 Table 5 的 K 扫描漏记了
scale 项。另有免费空间：三元残差按 trit 打包可再赚 16-22%（6.88×→~8.3×，零质量代价）。

**各方法 BPE 一览**（INT2，块级标量记 FP8；QVG 系含索引+质心、按 LC 发布 chunk 29,640 token 分摊；QuaRot 系与序列长度无关）：

| 方法 | 账单（bit/元素） | **BPE** | 压缩率 | frame-93 PSNR |
|---|---|---:|---:|---:|
| QVG（S=1, B=64） | 2 + scale 0.125 + 索引 0.0625 + 质心 0.138 | **2.326** | 6.88× | 28.88 |
| QVG-Pro（S=4, B=16） | 2 + 0.5 + 0.25 + 质心 0.553 | **3.30** | 4.85× | **31.04** |
| QuaRot 非对称 B=64 | 2 + scale 0.125 + zero-point 0.125 | **2.25** | 7.11× | 28.85 |
| QuaRot 非对称 B=16 | 2 + 0.5 + 0.5 | **3.0** | 5.33× | 30.38 |
| QuaRot 对称 B=64 | 2 + 0.125（无 zero-point） | **2.125** | 7.53× | 19.14 |
| QuaRot 对称 B=16 | 2 + 0.5 | **2.5** | 6.40× | 21.42 |

两个读法：①同预算档位各有胜者——2.25~2.33 档 QuaRot 非对称 B64 与 QVG 打平（28.85 vs 28.88），3.0~3.3 档 QVG-Pro 胜非对称 B16 0.66 dB；②对称行省下 zero-point 的 0.125~0.5 bit，代价是 9 个 dB——全表性价比最差的钱。
→ 细节：[report-0714.md](report-0714.md) §2

## 2. Quant block 大小：QVG 的 k-means 已经把残差 smooth 得够好，B 不敏感

QVG 发布配置是 **B=64**。九宫格 B 扫描（INT2，frame-93 口径）：

| B | 16 | 64 | 128 |
|---|---:|---:|---:|
| QVG (S=1) | 30.96 | 28.88 | **28.41** |
| QuaRot 非对称 | 30.38 | 28.85 | **24.54** |

关键对比在粗块端：**64→128 QVG 只掉 0.47 dB，QuaRot 崩 4.31 dB（9 倍差距）**——质心减掉
结构性尖刺后，残差对 scale 粒度不再敏感；QuaRot 面对的原始 KV 则块越粗越被离群拖垮。
（细 scale 依然值钱：16→64 QVG 也有 −2.08，但那属于"锦上添花"而非"生死线"。）
副产物：QVG-Pro 的 +2.16 dB 优势里 S=4 只贡献 +0.08，其余全来自 B=64→16。
→ 细节：[b-sweep.md](b-sweep.md)

## 3. 按 paper 口径的速度实测（H100，同 GPU 上 bf16/QVG 各跑完整生成）

paper 只声称"INT2 比 bf16 慢不超过 1.5~4.3%"。我们按它自己的方法实测，**三个模型全是
负开销（INT2 更快）**：

| 模型（发布工作量） | bf16 端到端 | QVG INT2 端到端 | 差 |
|---|---:|---:|---:|
| LongCat（10 段全量） | 2977 s | 2385 s | **−19.9%** |
| Self-Forcing（180 latent） | 605 s | 575 s（稳态；含编译的首遍 599 s） | **−5.0%** |
| HY-WorldPlay（12 chunk 匹配几何） | 635 s | 200 s（稳态；首遍 301 s） | −68% ⚠️ 数字异常大，待逐 chunk 核验 |

加速来源是访存不是计算（cache 22.3GB→3.2GB 的搬运量差）。另一个对齐结论：附录 C 的
"SF 180 帧 43s" 之谜已解——发布脚本实际生成 717 帧（4 倍工作量），按块折算后同量级。
→ 方法口径：[report-0714.md](report-0714.md) §3（实测数字待补入 §3.5）

## 4. OScaR 的 token norm 分析搬到视频上：TNI 不存在，原因正是你说的那个

**video 的 self-attention 序列和 LLM 的长得不一样**：LLM 里 prompt（含 BOS/标点——sink 的
载体）和生成 token 在同一条 self-attn 序列里；视频 DiT 把文本挪去了 cross-attention
（512 个 text token，静态小 buffer，不量化），self-attn cache 里**只有视频 patch token**，
经典 sink 无处可长。实测印证：三模型 K 的 token norm 极值比仅 1.03×（LC）/1.27×（HY）/
1.41×（SF），逐位置箱线图无任何低 norm 离群 token。视频真正的离群换了轴：**SF L29 的
H9 整头 norm≈105**（本质是 ch95/ch49 两根巨型通道：W_k 能量集中 16.6× × g 增益 5.4 对齐
相乘）——head/channel 级，token 级手术（OScaR 的 OTS）在这治不了也不需要。
勘误一条：QK-Norm 挡不住 TNI（Qwen3 有 QK-Norm 仍被 OScaR 观察到 sink），它只负责压窄。
→ 细节：[kv-distributions.md](kv-distributions.md)、[qkv-anatomy.md](qkv-anatomy.md)、
[qk-norm.md](qk-norm.md)

## 5. Normalization / DeltaQuant 的 temporal activation 分析

DeltaQuant（CVPR26，今日入库）诊断了 SVDQuant 直搬视频失败的原因：**激活的离群通道和
幅值随去噪 timestep 剧烈变化**，离线校准的静态 smoothing 按某一步调好、在其他步反而帮倒忙
（其 Fig.4）——与你自己的观察一致（不同 timestep 下 activation 差距很大）。
和我们 KV 侧结论的分界要划清：KV cache 沿**视频位置**轴是平稳的（qkv-anatomy 角度一），
但 **timestep 轴**是另一回事——我们的 Q/激活只取过最后一个去噪步，跨步演化未测，
已列入待办（对注意力输出 proxy 和任何激活量化都重要）。
→ 论文条目：[../../papers/README.md](../../papers/README.md)；待办：[qkv-anatomy.md](qkv-anatomy.md) §6

---

遗留待收尾：HY 速度数字核验、实测结果补入 report §3.5、0714 三件套（HANDOFF/REPRODUCE）。
