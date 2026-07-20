# Report 0720 — kernel 落地三重门 + why 判决:两项 goal 全收官

> 三件套之一。终表 [mp100-table.md](mp100-table.md),速度 [kernel-results.md](kernel-results.md),
> BPE [bpe-audit.md](bpe-audit.md),why 判决 [why-budget-pca-wins.md](why-budget-pca-wins.md)。

## Goal 1:kernel 化(完)

三重门全过:**① BPE 逐字节审计合规**(LC 2.3192/SF 2.3183/HY 2.3250,fake↔kernel
偏差 ≤1.1%);**② 同输入 encode 快过 QVG kmeans**(LC 32.5×/HY 1.4×/SF 1.1×,
Triton 融合 decode 与参考逐位一致);**③ 评测 vs QVG:18 列 = 11 显著胜 + 7 统计
平局 + 0 负**。代价与收获:审计连环抓出 4 个记账 bug(块长回退×2、fp8 饱和、
fp8 下溢),LC/SF 双方(含 KIVI baseline)全部按真实口径重赛(~700 次重生成),
KIVI 的 LC 从虚标 33.47 回落到诚实的 30.55。

## Goal 2:why-analysis(完)

预注册四假说判决:**H1 证伪→修正**(kmeans 减掉的能量其实更多;真机制 = 残差
结构分工,2-bit 格能量回收效率差 3-16×)、**H2 成立**(通道劫持,LC 判决性 +
SF 均质反向验证)、**H3 成立**(质心撒不满连续薄饼,角色修正)、**H4 成立**
(iters 2→100 平坦,不是没调好)。附 HY 半区"能量≠价值"反转(fig6)与 KIVI
三角定位(通道机制 +2.35dB、减法框架 +1.13dB)。六图 + 文献锚点 + 四条可复用
设计判据,全在判决报告。

## 主题句(全天的线索)

**"能量不是价值,账单必须逐字节"**——H1 修正、prope 9:0 反转、伪影偏好判据
是同一枚硬币;四个记账 bug 全部由"真 kernel 数字节"抓获。
