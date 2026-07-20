# 复现 QVG paper 的 KIVI@LongCat 指标:过程记录

> Goal(用户 0720):paper Table 1 里 KIVI 在 LC INT2 的 PSNR = **20.317**,
> 比 RTN(20.872)还低;而我们诚实重赛的正品 KIVI = **30.55**。找出 paper 的
> 数字是怎么来的,视频级复现之。本页滚动记录全过程。

## 假说(chunk 级鉴别矩阵已定,0720 侧线)

paper 的 KIVI 数字 = **两个实现选择的叠加**,缺一不可:

1. **三点位对称格**(而非 KIVI 原文的非对称 min/max 格):chunk relL2 0.067→0.221;
2. **逐通道施加在 post-RoPE K 上**(LLM 官方实现直接移植;LC 管线 cache 是
   pre-RoPE,paper 若按 LLM 习惯在 rope 后量化,通道 outlier 结构被打散):
   0.221→0.375,≈RTN(0.682)量级。

佐证:paper 的 QVG(28.716)与我们复现(28.20)一致 → 管线/协议无恙;
KVQuant 文献:RoPE 打散 pre-RoPE 的通道 outlier 结构。

## 实验设计(视频级,10 prompts,f93 PSNR,全部走同一 campaign 管线)

| 臂 | 实现 | 预测落点 |
|---|---|---|
| `kivi3pt` | pre-RoPE 逐通道 + **三点位对称格** | ~25-27(单因子) |
| `kivipaper` | **post-RoPE 施加**(劫持点作 R→量化→R⁻¹,读取时管线再施 R = paper 行为)+ 逐通道三点位 | **~20.3(命中 paper)** |
| `rtn3pt` | token 轴 + 三点位对称格 | ~20.9(顺带复现其 RTN 20.872) |
| (已有)正品 KIVI | pre-RoPE 逐通道非对称 | 30.55(诚实重赛值) |
| (已有)我们的 RTN | token 轴非对称 | 23.56 |

post-RoPE 仿真的正当性:LC cache 存 pre-RoPE、读取时施 R;在劫持点做
`R⁻¹(quant(R(k)))`,读取后等价于"量化作用于 post-RoPE 坐标"= paper 的存储行为。
R 用 LC 原装 rope_3d 频率表(grid 19×30×52,窗口相对坐标,与管线读取一致)。

## 过程日志

- [x] chunk 级鉴别矩阵(见上,侧线 0720);
- [ ] 实现三臂(pca_quant: PCA_KIVI_GRID / PCA_KIVI_PAPER / PCA_RTN_GRID);
- [ ] 冒烟:R⁻¹R 恒等性 + 各臂 hijack 生效;
- [ ] 30 jobs 生成 + f93 评分;
- [ ] 判决:kivipaper 是否落在 20.3±1。
