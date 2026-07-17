# 0717 工作目录

昨日总况读 [../0716/report-0716.md](../0716/report-0716.md)；接手须知读
[../0716/HANDOFF.md](../0716/HANDOFF.md)。本页 = 从 0716 挪过来的未完成队列。
（收尾三件套 report-0717/HANDOFF/REPRODUCE 按惯例日末补。）

## 待办队列（从 0716 结转，按优先级）

1. **发上游 issue 问 HY 帧范围**：草稿定稿在
   [../0716/issue-draft-hy-eval.md](../0716/issue-draft-hy-eval.md)，可直接发；
   作者答"帧范围"一项即可裁决 HY INT2 列复现性（现状：全程口径未复现 −10.5 dB，
   两段结构与跨崖平均可解释其数字形状）。
2. **HY −3.1 dB 归因**（调参已证伪：r=8 / split128 均无差异）：
   下一刀 = 层×头×K/V 误差分解定位、K/V 不对称预算、残差位宽/块尺寸扫描。
   工具与数组齐备（`pca_quant.py`、`sf_hy_n4*.npz`）。
3. **六格矩阵补空**：①N4 的 INT4 档定义（候选：coef 4-bit + 残差 asym 4-bit B128，
   BPE≈4.5 vs QVG 4.30——研究决策待讨论）；②SF×INT4（QVG 需按 195-latent 配置新生成）；
   ③QVG-Pro 的 paper 口径 LPIPS 补测（旧视频已清理，需重生成）。
4. **多 prompt 复验**（可信度卡）：MovieGen 套件（`assets/t2v.txt` 入口，SF 官方设定）；
   QVG 非确定性每 prompt n≥3、N4 确定性 n=1；顺带裁决 VBench 长尾分歧
   （我们 60.5 vs paper 67.3 @1400f）。战役设计草案见 0716 讨论（10 prompts × 方法阵）。
5. **N4 真 kernel 化 M0-M4**（[../0716/n4-int2-impl-plan.md](../0716/n4-int2-impl-plan.md)）：
   M0 FP8-scale 质量风险未测；M1 位打包（解锁长度/显存故事，现 fake-quant 上限 777 帧）
   → M2 融合 decode → M3 基准 → M4 流式热启动（追平对流式 QVG 的 23× 差距）。
6. **W8A8+KV2 重启**（[../0715/w8a8-kv2-plan.md](../0715/w8a8-kv2-plan.md)）：
   Phase 0 三个决策点待讨论；重启时 KV 侧换 N4。
7. **上游 issue 池**（除 HY 协议外还有 6 条：B=128 autotune bug、SF fake 路径布局矛盾、
   SF 滑窗缺失、LPIPS 怪癖、BF16@1400 不可复现、Table 5 账目滑差 + §5.2 chunk 笔误）
   ——`ISSUE_DRAFT` 汇总后待批准发送。
8. **集群善后**：Weka 恢复确认（canary 清单 `repro/backup/pods/*.yaml` 已入库，
   ENG-91011）；恢复后赦免被冤枉的黑名单节点 099/079/118/052。

## 今日实验记录

（待添加）
