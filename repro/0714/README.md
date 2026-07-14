# 0714 工作目录

候选方向见 [../0713/HANDOFF.md](../0713/HANDOFF.md)：

1. Proxy 排行榜研究（首选）——5 个候选 proxy vs 60+ 已标注配置的秩相关；缺口：带 Q 张量的 KV dump
2. 稀疏离群旁路
3. trit 打包免费压缩
4. 多 prompt 稳健性验证
5. 作者 issue（等确认）

## 待办：RoPE distribution 直观解释（用户 0714 提出，暂缓执行）

**用户的问题**：不要用 k-means 做度量（原 rope-dispersion.md 用 k-means 残差衡量可量化性，
不直观）。想直接**看**分布层面的证据：
1. pre-RoPE 的 K distribution 是有规律的——先画出来确认；
2. post-RoPE 的 K distribution 为什么就没规律了——给出原因；
3. **用户明确要求**：如果他的假设（pre 有规律 / post 全没规律）本身是错的，要直接指出。

**已知的预答案线索**（待用图呈现，勿直接下结论）：破坏是**频率局域化**的，不是全盘——
θ·T≪1 的低频通道对规律原样保留，θ·T≳π 的高频对被打散；且被打散的恰是 pre-RoPE 时
结构最好的那一半（rope-dispersion.md §B 的数据）。即用户假设一半对、一半需修正。

**已备好的工具**：`ropestudy/step3_visualize_before_after.py`（已写未跑）——六联图：
(a)(b) pre/post 热图（竖条纹 vs 高频区被搅碎）、(c)(d) 固定空间位置跨帧的单通道轨迹
（低频≈恒等 vs 高频扫圆）、(e) std 比值 vs θ·T 频率局域化散点、(f) 逐通道时间 std。
数据用 SF dump + 模型自身 RoPE（step1 同款调用），全程无 k-means。执行只需：
`.venv/bin/python repro/0714/ropestudy/step3_visualize_before_after.py`
