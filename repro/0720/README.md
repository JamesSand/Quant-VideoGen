# 0720 工作目录

> **新人从这里进**:[REPRODUCE-0720.md](REPRODUCE-0720.md) —— 全链路指令级
> 复现指南(环境→生成→评分→kernel 三重门→why 判决,含预期数字核对点与全部已知坑)。

前情:0718-0719 MP100 战役收敛,见 [mp100-table.md](mp100-table.md)(终表副本,
源=../0718/mp100-table.md)、[../0718/report-0719.md](../0718/report-0719.md)、
[../0718/HANDOFF-0719.md](../0718/HANDOFF-0719.md)。

状态:18 列 = 12 最优 + 4 统计平局 + 2 结构性小差距(lc:sc −0.02、hy:aq −0.96);
终版方法 = 通道轴 Budget-PCA。

## 本日议题(与用户讨论定)

- **已定:kernel 化 + 同输入速度对决**,kernel 化已完成,结果见 [kernel-results.md](kernel-results.md);**为什么赢的判决报告**见 [why-budget-pca-wins.md](why-budget-pca-wins.md)(plan 已按约定删除,判据引用于报告内)。其余候选:候选(承接 HANDOFF):剩余两列的 (a)收案/(b)解禁机制/(c)换口径;
  kernel 化 M0-M4(通道轴的转置访存入设计);0717/0718 头条文档按 MP100 重写;
  QVG-Pro 臂;paper 写作。
