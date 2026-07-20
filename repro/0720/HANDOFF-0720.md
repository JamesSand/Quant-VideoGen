# HANDOFF 0720

## 状态:两项 goal 收官 + 二审勘误完成 + paper 差异全结案

1. **Kernel 三重门全过**:[bpe-audit.md](bpe-audit.md)(逐字节合规)、
   [kernel-results.md](kernel-results.md)(encode 32.5×/1.4×/1.1×)、
   [mp100-table.md](mp100-table.md)(vs QVG 11胜7平0负,诚实重赛口径);
2. **Why 判决完稿并经二审**:[why-budget-pca-wins.md](why-budget-pca-wins.md)
   (只留成立假说,导读框 + 六图内嵌)+
   [why-refuted-and-errata.md](why-refuted-and-errata.md)(负结果台账);
   外部核查三处勘误全部核实修正(聚类口径/引用错误/单 chunk 概括),全部
   数字 8 chunk 真口径重立,效率差定格 ~2.2×、H1 证伪 24/24;
3. **KIVI-paper 复现支线结案**:[paper-diff-plan.md](paper-diff-plan.md)——
   四类 paper 差异全部定位到 paper 侧(实现弱/窗口口径/AQ 打分器/其 baseline
   视频真坏),QVG 本尊行 −0.5dB 反向背书我们管线;官方 VBench 对打完成
   (BC/IQ/SC ±0.4 一致);
4. **文档化收尾**:[REPRODUCE-0720.md](REPRODUCE-0720.md)(全链路复现指南)、
   [README.md](README.md)(总索引)、report-0720(当日叙事)。

## 结转

- 0717/0718 头条文档按 0720 终版数字重写(演化史保留);
- kernel 管线深度集成(真 cache 路径接入三管线;解锁 SF 1400f/Q3);
- 论文写作:素材齐(方法 + 终表 + 速度 + why 判决正/负两册 + f93 可视化 +
  HY 视频 + paper 差异横比口径)。

## 纪律备忘(本轮验证有效的)

单队列跑卡;kill 前独立读证据;先有账后有码,kernel 逐字节审计是终审;
外部批评逐条核实、修正双向留痕;plan 实现完即删;能量≠价值,端到端闸门
是唯一裁判;正/负结果分册,引用成对。
