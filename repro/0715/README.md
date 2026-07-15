# 0715 工作目录

收尾三件套：`report-0715.md`（present 用，self-contained）+ `HANDOFF.md` + `REPRODUCE.md`（见 repro/README.md 约定）。

## 从 0714 接手的候选方向（详见 [../0714/HANDOFF.md](../0714/HANDOFF.md)）

1. **Proxy 排行榜研究**——cache-MSE 已被证明是坏 proxy；60+ 已标注配置在手，`results/kvplot/sf_qkv.pt` 已含 Q（部分解锁），可算注意力输出 MSE / 注意力图 KL 等候选 proxy 的秩相关
2. **多 prompt 稳健性复验**——全部首帧结论目前单 prompt/seed；INT4 p99.7 悖论也待复验
3. **LC/HY 的 QKV 三维度解剖**——采集器已通用化，各一条 pod
4. **按深度重分配量化预算**——L29 K absmax 7× 的直接推论，现成可跑
5. **trit 打包**工程验证 / **稀疏离群旁路** / **RoPE 直观图**（step3 脚本已备）/ ISSUE_DRAFT 更新发送（等确认）
