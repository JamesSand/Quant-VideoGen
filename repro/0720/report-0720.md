# Report 0720 — 双 goal 收官、二审勘误、paper 差异全结案

> 三件套之一(另两件:[HANDOFF-0720.md](HANDOFF-0720.md)、
> [REPRODUCE-0720.md](REPRODUCE-0720.md));目录总索引见 [README.md](README.md)。

## 一、Goal 1:kernel 化(完,三重门全过)

**① BPE 逐字节审计合规**(LC 2.3192 / SF 2.3183 / HY cache 2.3250,fake↔kernel
偏差 ≤1.1%,[bpe-audit.md](bpe-audit.md));**② 同输入 encode 快过 QVG kmeans**
(LC 32.5× / HY 1.4× / SF 1.1×,Triton 融合 decode 与参考逐位一致,
[kernel-results.md](kernel-results.md));**③ 评测 vs QVG:18 列 = 11 显著胜 +
7 统计平局 + 0 负**([mp100-table.md](mp100-table.md))。

代价与收获:审计连环抓出 **4 个记账 bug**(块长回退×2、fp8 饱和、fp8 下溢),
受污染数字全删,LC/SF 双方(含 KIVI baseline)按真实口径重赛 ~700 次生成,
KIVI 的 LC 从虚标 33.47 回落到诚实的 30.55——我们的领先在更严格的口径下成立。

## 二、Goal 2:why-analysis(完,经二审后以真口径重立)

预注册四假说终局(正报告 [why-budget-pca-wins.md](why-budget-pca-wins.md),
每节带"大白话 + 公式 + 例子"导读框、六图内嵌):

- **H1 原命题证伪 → 修正命题成立**:同等比特下 kmeans 反而减得略多(真口径
  8 chunk × 3 模型 = 24/24 全向);真机制 = **残差结构分工**——kmeans 残差
  ≈白噪声,2-bit 格回收率在 LC/SF 全部 16 测点钉死 46-49%(教科书白噪声水平),
  我们的结构化残差回收 ~76%,格效率差 **~2.2×**;chunk 级最终误差我们 20/24
  格更小(4 个例外格如实记录,端到端不受影响);
- **H2 成立**:通道劫持,LC 判决性 1.8-2.4×(8/8 跨层稳),HY 1.4-2.4×,
  SF 效应最弱(1.1-1.8×,与其通道极差仅 7× 一致);
- **H3 成立**:质心撒不满连续 token 云(per-head 128 维真口径),K×16 码本
  只买 +0.3~+8.6pp;
- **H4 成立**:iters 2→100(计算 ×50)换 ±0.5dB 噪声——"没调好"的质疑封死。

附:HY 半区"能量≠价值"反转(fig6)、KIVI 三角定位(通道 +2.35dB、减法框架
+1.13dB)、四条可复用设计判据。

## 三、二审勘误(外部核查触发,全部核实、修正、留痕)

外部核查(GPT 复核)指出三处问题,**逐条核实后全部属实**:①kmeans 聚类口径
错误(旧版全局 64 维块 → QVG 真口径 per-head 全 D 维 token,fig2/3/5 重算);
②"0.7%"引用错误;③单 chunk 过度概括(全部改 8 chunk 复核,fig1 排序论断
撤回、SF"无红利"降级)。**修正是双向的**:效率差幅度 3-16× 回落为 ~2.2×
(旧数系口径混用),但 H1 证伪升级为 24/24 全向、"回收率钉死白噪声理论值"
的机制签名反而更硬;两处交叉验证(8 层谱均值、H2 比值)与外部核查一致。

结构动作:正报告只留成立假说,证伪/撤回/勘误/历史证伪索引全部移入
**[why-refuted-and-errata.md](why-refuted-and-errata.md)(负结果台账)**,
两文互链、引用须成对。

## 四、paper 差异全结案(fork 支线,[paper-diff-plan.md](paper-diff-plan.md))

与 QVG paper Table 1 的四类数字差异全部定位,**无一指向我们管线**:

| 差异 | 结论 |
|---|---|
| D1:LC baseline 我们高 ~10dB | paper 实现弱(三点位对称格 + post-RoPE);E1 复刻后落到 paper ±1-3dB |
| D2:HY 整块 paper 高 7-12dB | PSNR 窗口口径(paper≈发散前 ~24 帧窗,我们全程均值) |
| D3a:VBench 水位差 | 我们 AQ 重实现比官方包低 ~4.7;BC/IQ/SC 与官方 ±0.4 一致(IQ 逐位),表内结论不受影响 |
| D3b:paper baseline 的 VBench 崩坏 | 官方打分器也复现不出——其 baseline 视频真实地坏,与 D1 互证 |

反向背书:QVG 本尊行与 paper 仅差 −0.5dB。

## 五、文档化收尾

- **[REPRODUCE-0720.md](REPRODUCE-0720.md)**:全链路指令级复现指南(环境→
  素材→生成→评分→三重门→why,每阶段预期数字核对点 + 8 条已付学费的坑 +
  与 paper 横比的三条口径注意);
- README 重写为总索引;why 正报告加导读框(GitHub 可渲染的 ```math 公式)。

## 主题句(全天的线索)

**"能量不是价值,账单必须逐字节,批评必须逐条核"**——H1 修正、prope 9:0
反转、伪影偏好判据是同一枚硬币;4 个记账 bug 由真 kernel 数字节抓获,3 处
分析口径错误由外部核查抓获:两轮都是"先诚实、后胜利",最终结论反而更硬。
