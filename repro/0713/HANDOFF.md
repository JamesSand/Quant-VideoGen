# 0713 Handoff — QuaRot+Clip 扫描 / 方差研究 / Clip 张量刨析

> 日报正文：[report-0713.md](report-0713.md)（总表、扫描数据、可视化全在里面）
> 上游背景：[../backup/REPORT.md](../backup/REPORT.md)（复现总报告）、[../backup/CLIP_STUDY.md](../backup/CLIP_STUDY.md)、[../backup/ROPE_DISPERSION.md](../backup/ROPE_DISPERSION.md)

## 今日完成（按时间序）

| # | 工作 | 关键结果 | 产物 |
|---|---|---|---|
| 1 | 首帧口径全方法对比表 | INT2 下非对称 QuaRot（30.38）被 paper 低估 8.8 dB；paper 的 QuaRot 数字与对称变体（21.42）吻合 → 其私有移植疑用对称量化 | report-0713.md 总表 |
| 2 | **QuaRot+旋转后裁剪扫描**（20 run：ratio 0.85-0.99 ×6 + pct 99.0-99.9 ×4 × 两位宽） | 用户假设方向证实：最温和收缩 r0.99 双位宽小赚（INT2 +0.30→**30.68**；INT4 +0.44→33.48）；**意外亮点 INT4 pct99.7 = 34.92（+1.88）**；深裁有害但主体不毁容（旋转保护效应，对比原始 KV 裁剪的"幽灵化"） | report 扫描节 + `results/qclip/` |
| 3 | **INT2 方差研究**（6 方法 × 3 次独立运行） | 确定性方法 σ≤0.003（测量管线零噪声实证）；QVG σ=0.18（atomic_add）；QVG-Pro 意外极稳 σ=0.005（4 阶段平均稀释噪声）；排序稳健：QVG-Pro 31.04 > QuaRot+clip 30.68 > QuaRot 30.38 > QVG 28.88 | report 方差节 + `results/varstudy/` |
| 4 | 总表覆盖为 3 次均值 mean±std | INT2 全列统计化；INT4 QVG = 34.36±1.28（长尾大） | report 总表 |
| 5 | 首帧可视化 | INT2/INT4 各 12 格带 PSNR 标注的 contact sheet + 22 张单帧，内嵌 report | `first_frames/` |
| 6 | **Clip 张量刨析**（真实 SF KV） | K 的 max：18.0 →(旋转)9.5 →(p99.7)4.7；MSE 分解：幸存值恒赚 ~1.6%、被裁 0.3% 恒亏，净号由量化噪声地板决定（INT2 地板高→net 赚，INT4 地板低→net 亏 19.7%）；**发现张量 MSE 与视频 PSNR 方向相反的悖论（INT4 p99.7）** | report 刨析节 |
| 7 | 概念澄清（讨论，未入 report） | QVG 的"INT2"残差实为**三元量化**（{−1,0,+1}，−2 码位弃用）→ 为 4 码位付费只用 3 个，trit 打包可免费把压缩率 6.89×→~7.5×；QVG 用对称合理（残差零中心）、QuaRot 用对称是失误 | 本 handoff 记录 |
| 8 | 基础设施 | `quarot_quant.py` 加 clip 旋钮（QUAROT_CLIP_RATIO / QUAROT_CLIP_PCT）；`pod_run_qclip.sh` / `pod_run_var.sh`；修复 `recreate_pod.sh` 点号 pod 命名 bug | repro/ 各文件 |

## 关键数字速查（INT2 首帧 PSNR，mean±std, n=3）

QVG-Pro **31.04±0.005** ｜ QuaRot+clip(r0.99) 30.68 ｜ QuaRot 非对称 B16 30.38±0.003 ｜ QVG 28.88±0.18 ｜ QuaRot B128 24.54 ｜ RTN 21.85 ｜ QuaRot 对称 21.42

## 悬而未决 / 需要警惕

1. **INT4 pct99.7 悖论**：张量 MSE +19.7% 但视频 PSNR +1.88 dB——未复验，单 prompt、且张量数据（SF）与视频（LongCat）不同模型。多 prompt 复验前不要引用这个 +1.88。
2. 所有首帧结论基于**单 prompt（滑板手）/seed 0**；多 prompt 稳健性未验证。
3. QVG INT4 的 ±1.28 长尾波动使 INT4 组的一切方法排序不可信。

## 0714 候选方向（按讨论中的优先级）

1. **Proxy 排行榜研究**（0713 深夜讨论定型，方法论见对话）：cache-MSE 已被证明是坏 proxy；用现成的 **60+ 个已标注配置**（各类扫描的首帧 PSNR）+ KV dump 离线重放，对 5 个候选 proxy（cache-MSE / 注意力输出 MSE / 注意力图 KL / 注意力加权误差 / 单步 ε 散度）做 Spearman 秩相关排行。**缺的唯一数据：带 Q 张量的 dump**（现 dump 只有 KV，需一条 pod 补采）。这是 workshop-paper 级的产出。
2. **稀疏离群旁路**：clip 研究的正向续集——top 0.1~1% 离群提取 + 稀疏存储（~0.06B/值），目标"同压缩率打败 QVG"。
3. **trit 打包免费压缩**（今日 #7 的推论）：三元残差按 3⁵/字节打包，6.89×→~7.5×，纯工程零质量代价。
4. 多 prompt 稳健性验证（所有主结论 ×3 prompt，纯算力）。
5. 给作者的 issue（`../backup/ISSUE_DRAFT.md` 需按对称 QuaRot 指纹、方差发现更新后发出，等用户点头）。

## 基础设施状态

- 集群 pod 流水线全绿：占卡检测 + 节点黑名单 + workflow 编排均验证可靠（今日 32 条 run 零人工干预）；bad_nodes.txt 累积 ~14 个坏节点。
- KV dump（49GB，SF layer-all）在 `results/ropestudy/kv_cache_frames180.pt`，可复用。
- 本地 GPU 仍被同节点租户占用，一切生成实验走集群 pod（见 memory：k8s-self-serve-pods）。
