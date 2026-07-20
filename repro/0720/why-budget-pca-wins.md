# 为什么 Budget-PCA 赢、QVG 的 k-means 输:判决报告

> 预注册计划:why-analysis-plan(判据先于实验写死;本页如实记录**H1 原命题被证伪**
> 及其修正)。数据:三管线真实 dump chunk(`chunks/`)、MP100 诚实终表、
> `why/h1_h2_data.npz`;图 `why/fig1-5`。所有对比用 QVG 原装 kmeans 实现,
> 收敛臂给足 iters=100。

## 〇、一页结论

两个方法是同一骨架的两种减法(QVG 减最近质心,我们减 μ+子空间),胜负不在
"谁减掉的能量多"——**k-means 减得更多**(H1 原命题证伪)——而在**残差的命运**:

> **k-means 把结构吃得太干净,残差近白噪声,2-bit 均匀格在白噪声上效率极差;
> PCA 减法只拿走"方向性"结构,把通道结构和时间平滑性留给通道轴残差格,
> 2-bit 格的能量回收效率差 3-16×。减法阶段的胜利被残差阶段的溃败抵消——
> 结构分工与格子能力的匹配,才是胜负手。**

## 一、H1【表示假说】:原命题证伪 → 修正为"结构分工"命题

**原预测**:同等元数据比特下,rank-r 子空间消掉的能量 ≫ 256 质心字典。
**判决:证伪**([fig2](why/fig2_subtraction_efficiency.png))。K=256 字典只花
~0.13 bits/elem 就消掉 LC 98.5% / SF 95.1% / HY 79.2% 的能量;rank-4 花 0.25
bits 只消掉 82.1% / 68.8% / 45.6%。字典的每比特减法效率**更高**。

**修正命题(数据导出)**:最终质量由"残差能量 × 残差格效率"决定。实测
残差格效率(最终误差能量 / 残差能量,[fig3](why/fig3_residual_efficiency.png)):

| 模型 | QVG(kmeans 残差→token 轴格) | Budget-PCA(结构化残差→通道轴格) | 差距 |
|---|---|---|---|
| LC | 0.589(只回收 41% 残差能量) | **0.036**(回收 96%) | **16×** |
| SF | 0.160 | **0.022** | 7× |
| HY | 0.210 | **0.063** | 3× |

机制:kmeans 残差是"质心解释不了的部分"≈ 各向同性白噪声,minmax 均匀格在
白噪声上就是教科书的 ~40-60% 能量损失;我们的残差保留了逐通道幅度差和时间
平滑性,通道轴分块的 scale 逐块贴合,步长处处极小。

**低秩薄饼本身仍然成立**([fig1](why/fig1_spectra.png)):LC top-4 特征值占
82% 能量(λ1/λ32 = 954×)、SF 69%、HY 46%——它决定的是减法的**性价比上限**,
且与三模型的通道轴收益排序一致,但它不是对 kmeans 的胜因。

## 二、H2【几何假说】:成立(LC 判决性,SF 边界情形反向验证)

**预测**:k-means 欧氏度量被高方差通道劫持,小方差通道结构不被编码。
**判决:成立**([fig4](why/fig4_channel_error.png)):

- LC(通道方差极差 146×):QVG 在小方差通道的误差/信号比 = **我们的 2.3×**
  (0.148 vs 0.065),大方差通道两家相同(0.0036)——损伤精确集中在被劫持端;
- SF(极差仅 7×,近均质):QVG ≈ 我们,KIVI 反而最差——**通道异质性小时,
  通道机制无红利**,恰好反向验证;
- HY(极差 154×,但 post-transform):我们全通道占优;KIVI 的通道优势被
  rope/prope 混合削弱。

文献闭环:KVQuant/KIVI 原文正是此发现(pre-RoPE key 有位置一致的 outlier
通道;**RoPE 施加后通道结构被打散**)——解释了通道轴收益 LC(+3.5dB,
pre-RoPE)≫ HY(+1.3dB,post-transform)的梯度。

## 三、H3【预算假说】:成立(但角色修正)

[fig5](why/fig5_pc_plane.png):LC 的 64 维块云在主成分平面上**连续铺开**,
256 个质心只能撒离散点。率失真理论(高斯源 VQ 需要 2^(nR) 级码本追平变换
编码)预言字典的边际收益随 K 指数衰减——fig2 中 K:64→1024(16×码本)只多
消掉 LC 0.7% 能量,与理论一致。但注意角色修正:维度诅咒限制的是字典的
**上限增速**,而在 0.13 bits 的低预算点上字典已经很强;真正的分水岭仍是 §一。

## 三点五、Case Study:HY 半区秩 9:0 的"能量 ≠ 价值"反转

[fig6](why/fig6_hy_halves.png):**谱学不支持 9:0**——prope 半区反而更低秩
(top-9 能量 81.8% vs rope 半区 76.2%)且携带全部能量的 66%。按"消掉能量"
逻辑,秩应该全给 prope;但实验梯度(0717:4:4→9:0 单调变好)说明恰恰相反。
解释:prope 半区的能量是相机变换缠绕的,下游 attention 并不读取其精细结构
(其残差甚至可以换三值粗格,KP 三值反而全面变好)。**这是"能量不是价值"的
第三个实例**(与 H1 修正、伪影偏好判据同族):谱/能量类指标衡量的是数据自身,
不是它对生成的贡献——最终裁判只能是端到端闸门。

## 四、H4【工程假说】:kmeans 不是"没调好"(排除项)

LC 官配 iters=100 已充分收敛(kmeans 内部 tol 提前停:实测 K=256 时 47-100
轮收敛)。iters ∈ {2, 10, 100} 的端到端质量扫描(10 prompts):

| kmeans iters | LC f93 PSNR(同 10 prompts) |
|---|---|
| 2 | 27.13 |
| 10 | 27.72 |
| 100(官配) | 27.61 |
| **Budget-PCA** | **31.68**(全量 100;同子集 ~31.7) |

**判决:成立**——iters 从 2 到 100 曲线平坦(±0.5dB 噪声级),QVG 的差距不是
优化不足,是表示层面的(§一/§二)。"你们没把 kmeans 调好"的质疑就此封死。

## 五、KIVI 三角定位(两个假说的贡献分解)

诚实重赛后 LC PSNR:QVG 28.20 → KIVI 30.55 → 我们 31.68。
KIVI ≈ 只修 H2(通道)不减方向;我们 = H2 + 减法都占。分解:
**通道机制贡献 ≈ +2.35dB(QVG→KIVI),减法框架再 +1.13dB(KIVI→我们)**;
且 KIVI 无法用于打包/后变换数据(HY 上 17.13 < QVG 17.45),减法框架可以
(HY 我们 18.77)。

## 六、可复用的设计判据(方法论输出)

1. **给残差格留结构,别追求减法吃干抹净**——减法阶段的每比特效率不是目标
   函数,"残差结构 × 格子能力"的乘积才是;
2. **通道异质性 >10× 时通道轴必开**(LC 146×/HY 154× 受益,SF 7× 无感);
   pre-RoPE 数据红利最大,post-transform 减半;
3. **各向同性边界不许越过**(0717-0718 反复证伪的旋转/整形全家):通道级
   幅度自适应 = 安全,方向级误差整形 = 闭环毒药;
4. 无参考 IQ 类指标若出现"量化方法得分 ≥ 无损参考",即进入伪影偏好区,
   忠实型方法结构性吃亏,勿盲目优化(0718 hy:aq 教训)。

## 文献锚点

- 变换编码 vs VQ 率失真:[Nonlinear Transform Coding (Ballé et al.)](https://arxiv.org/pdf/2007.03034)、
  [Finite Block-Length Quantization Distortion](https://arxiv.org/pdf/1306.4754);
- KV 低秩结构:[Palu](https://arxiv.org/pdf/2407.21118)、
  [Eigen Attention](https://arxiv.org/html/2408.05646v1)(另注:其发现"SVD 潜空间拉伸
  分布反而伤量化"与我们"PCA 只做减法不改坐标系"互证);
- 通道 outlier 与 pre-RoPE:[KVQuant](https://proceedings.neurips.cc/paper_files/paper/2024/file/028fcbcf85435d39a40c4d61b42c99a4-Paper-Conference.pdf)、KIVI。

## 复现方法(指令级)

前提:repo 根目录、`.venv`、`source repro/backup/scripts/env_fix.sh`、单卡即可。
素材 = 三管线真实 dump chunk(`repro/0720/chunks/{lc,sf,hy}/chunk_*.pt`;若缺,
用带 `dump` 后缀的 campaign 臂重新采集,见 REPRODUCE-0718 §2)。

```bash
# ① H1/H2 主数据(奇异谱、rank 减法曲线、逐通道误差三方对比 → h1_h2_data.npz)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_h2_compute.py

# ② H1 的 kmeans 侧(质心减法单独消掉的能量,QVG 原装 kmeans,iters=100)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_kmeans_sub.py

# ③ 图 1-5(谱 / H1 判决 / 残差格效率 / H2 逐通道 / H3 质心散点)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/make_figs.py

# ④ 图 6(HY 半区分谱反转)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/fig6_hy_halves.py

# ⑤ H4(kmeans iters 扫描,20 次 LC 生成,~1.5h/8 卡)
#    生成:campaign 臂 qvgi2 / qvgi10(qvgi<N> 会把 --kmeans_max_iters 设为 N)
printf 'lc:%d:qvgi2:0\nlc:%d:qvgi10:0\n' $(seq 1 10 | sed 'p') > /tmp/h4.txt  # 或手写 20 行
CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt \
  bash repro/0718/scripts/gpu_queue.sh /tmp/h4.txt 8
#    打分:对每个臂算 f93 PSNR vs 同 prompt BF16(帮助函数同 repro/0718/scripts/stats.py)
```

关键数字核对点(复现应落在 ±噪声内):
- fig2:LC kmeans K=256 消掉 ~98.5% @0.127bits;rank-4 ~82.1% @0.25bits;
- fig3 的效率比:最终误差能量/残差能量 = QVG 0.589 vs 我们 0.036(LC);
  该比值的分子来自 kernel/bench_report.json 的 relL2²,分母 = 1 − 减法消掉比例;
- fig4:LC 小方差后 16 通道误差比 QVG 0.148 / 我们 0.065;
- fig6:prope top-9 = 81.8%、能量占 66.3%;
- H4:iters 2/10/100 → 27.13 / 27.72 / 27.61(f93,p1-10)。

逐通道误差三方对比里 KIVI/我们的臂 = `pca_quant.py` 的 PCA_KIVI=1 / 终版配置
(kernel/bp_quant.py 同数学);kmeans 一律 QVG 原装(`quant_videogen/`)。

## 诚实条款执行记录

- H1 原命题按预注册判据**证伪**并如实修正(判据 0720 实验前写死于 plan);
- kmeans 全部用 QVG 原装实现,收敛臂 iters=100;
- 所有数据来自真实管线 dump chunk 与诚实重赛后的 MP100 终表;
- 我们输掉/平局的列(hy:aq 等)的解释框架见判据 4,与本报告同一理论。
