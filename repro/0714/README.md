# 0714 工作目录

收尾交接（完成事项/关键数字/悬而未决/0715 方向）见 [HANDOFF.md](HANDOFF.md)；
全部实验的复现指令见 [REPRODUCE.md](REPRODUCE.md)。

## 文档地图

| 文档 | 内容 |
|---|---|
| [report-0714.md](report-0714.md) | **演示用汇总**（self-contained 五节：BPE / B 扫描 / 速度 / TNI 检验 / DeltaQuant 与 timestep）——对外 present 用这份 |
| [details-0714.md](details-0714.md) | 日报正文：量化代码考古（六阶段地图、无 Hadamard）、真实 BPE 核算与对账、QVG/Pro 参数全表、B=64 对照与九宫格汇总、paper 测速口径与 §3.5 实测 |
| [b-sweep.md](b-sweep.md) | B 扫描九宫格台账（矩阵、出处、执行方式、结果读数） |
| [kv-distributions.md](kv-distributions.md) | 三模型 K/V 分布 3D 曲面 + token norm 统计（OScaR Fig.2/3 口径）；HY 双分支结构注记 |
| [qkv-anatomy.md](qkv-anatomy.md) | SF 的 KV 值与 QKV norm 三维度解剖（视频首尾/chunk 内部/层深；L29 H9 整头离群） |
| [qk-norm.md](qk-norm.md) | QK-Norm 机制（g 的来历、Wan 整维 vs Qwen3 逐头、流程图与 g 可视化、H9 成因分解、TNI 勘误） |
| [rope-dispersion.md](rope-dispersion.md) | RoPE 分散性推导 + 真实数据验证（自 backup 迁入） |
| [reference/](reference/) | vendor 的 modeling_qwen3.py（QK-Norm 证据）等参考代码 |
| [figs/](figs/) | 全部图表（kv3d / token_norms / sf_kv / qkv / qk_norm / timestep dynamics） |

## 未完成待办（详见 HANDOFF）

- RoPE distribution 直观图：`ropestudy/step3_visualize_before_after.py` 已备好未跑（用户暂缓）
- 多 prompt 复验、LC/HY 的 QKV 解剖、proxy 排行榜启动、trit 打包、ISSUE_DRAFT 发送
