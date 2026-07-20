# 为什么 Budget-PCA 赢、QVG 的 k-means 输:判决报告

> 预注册计划:why-analysis-plan(判据先于实验写死)。数据:三管线真实 dump
> chunk(`chunks/`,全部 8 个 = 8 个不同层)、MP100 诚实终表、
> `why/h1_h2_data.npz`;图 `why/fig1-6`。所有对比用 QVG 原装 kmeans 实现
> (per-head 全 D 维聚类真口径,与 eval 同配置),收敛臂给足 iters=100。
>
> 本页只保留**成立的假说**。判决过程中被证伪的原命题(H1 原版)、被撤回/
> 降级的论断、以及 0720 二审的三处勘误,全部在
> **[why-refuted-and-errata.md](why-refuted-and-errata.md)**(负结果台账)
> ——引用本报告时请连同它一起看,机制结论正是从那次证伪里逼出来的。

## 〇、一页结论

两个方法是同一骨架的两种减法(QVG 减最近质心,我们减 μ+子空间),胜负不在
"谁减掉的能量多"——同等比特下 k-means 反而消得略多(原 H1 证伪,
[负结果台账 §一](why-refuted-and-errata.md))——而在**残差的命运**:

> **k-means 把结构吃得太干净,残差近白噪声:其 2-bit 均匀格的能量回收率在
> 全部 16 个 LC/SF 测点上钉死在 46-49%——教科书白噪声水平;我们的残差保留
> 通道结构,同规格格子回收 ~76%(LC/SF),效率差 ~2.2×。减法阶段的小胜被
> 残差阶段的稳定溃败抵消(chunk 级最终误差我们 20/24 格更小;例外只在 HY
> 深层 3 格,见 §一)——结构分工与格子能力的匹配,才是胜负手。**

## 一、H1【表示假说,修正版】:"残差结构分工"命题成立

前置事实:减法阶段谁消掉的能量多**不是胜因**——同等元数据比特下 kmeans
反而略多(原 H1 命题按预注册判据证伪,8 chunk × 3 模型 24/24 全向;证据与
fig2 见[负结果台账 §一](why-refuted-and-errata.md))。胜负在残差阶段:

**命题(数据导出)**:最终质量由"残差能量 × 残差格效率"决定,我们赢在
**残差格效率**。实测(**同 chunk 内自洽口径**:最终误差能量 / 减法后残差
能量,8 chunk 均值[极差],[fig3](why/fig3_residual_efficiency.png)):

| 模型 | QVG(kmeans 残差→int2 B64) | Budget-PCA(结构化残差→通道轴格) | 差距 |
|---|---|---|---|
| LC | 0.52 [0.51-0.54](回收 ~48%) | **0.24 [0.21-0.33]**(回收 ~76%) | **2.2×** |
| SF | 0.52 [0.51-0.54] | **0.24 [0.21-0.38]** | 2.2× |
| HY | 0.46 [0.36-0.55] | **0.30 [0.19-0.39]** | 1.5× |

![fig3:残差格效率——QVG 回收 ~48%(白噪声水平)vs 我们 ~75%](why/fig3_residual_efficiency.png)

机制签名比幅度更硬:QVG 的回收率在全部 16 个 LC/SF 测点钉死在 **46-49%**——
这正是 minmax 均匀 2-bit 格在近白噪声上的教科书损失,层深、模型、数据都不能
动它;我们的残差保留逐通道幅度差,通道轴 scale 逐块贴合,回收 61-82%。净账:
chunk 级最终误差我们 **20/24 格更小**,且 LC 层越深差距越大(chunk_007:
6.1 vs 10.4%)。4 个例外格如实记录:SF chunk_000(0.177 vs 0.158%,两家近
无损)与 **HY 深层 3 格**(chunk 004/005/007,我们的 9:0 半区减法在深层只消
掉 49-61% 能量,减法劣势盖过格子优势)——但端到端 HY 仍是我们赢(18.77 vs
17.45),与 §三点五"能量≠价值"一致:HY 的胜负由下游 attention 读取什么决定,
不由 chunk 级 relL2 单独决定。(此表的历史勘误——旧版 16×/7×/3× 的口径
混用——见[负结果台账 §三](why-refuted-and-errata.md)。)

**低秩薄饼成立但层依赖**([fig1](why/fig1_spectra.png)):top-4 特征值能量
8 层均值 LC 55%[极差 36-82]、SF 56%[40-86]、HY 42%[36-52]——早层最扁
(chunk_001:82/70/47%),深层三家趋同。它决定的是减法性价比的上限,不是对
kmeans 的胜因(跨模型排序论断已撤回,见[负结果台账 §二](why-refuted-and-errata.md))。

![fig1:K 协方差谱——低秩薄饼层依赖,跨模型排序不稳](why/fig1_spectra.png)

## 二、H2【几何假说】:成立(LC 判决性,跨层稳)

**预测**:k-means 欧氏度量被高方差通道劫持,小方差通道结构不被编码。
**判决:成立**([fig4](why/fig4_channel_error.png);比值 = QVG/我们的小方差
16 通道误差/信号比,全部 8 chunk 复核):

- LC(通道方差极差 146×):**1.8-2.4×,8/8 全部成立**(chunk_001:0.148
  vs 0.063)——损伤精确集中在被劫持端,这是 H2 的判决性证据,跨层稳;
- HY(极差 154×,post-transform):1.4-2.4×,8/8 方向成立但波动大
  (深层 4/5/7 收窄到 1.4-1.5×,与 §一 HY 深层例外同源);KIVI 的通道优势被
  rope/prope 混合削弱;
- SF:三模型中劫持效应最弱(1.1-1.8×,8 层均值 1.5×),与其通道极差仅 7×
  一致——异质性小则通道红利小,方向上支持 H2 的几何机制。(初版"SF 无红利
  反向验证"表述过强,降级记录见[负结果台账 §二](why-refuted-and-errata.md)。)

![fig4:逐通道误差——QVG 的损伤集中在小方差通道](why/fig4_channel_error.png)

文献闭环:KVQuant/KIVI 原文正是此发现(pre-RoPE key 有位置一致的 outlier
通道;**RoPE 施加后通道结构被打散**)——解释了通道轴收益 LC(+3.5dB,
pre-RoPE)≫ HY(+1.3dB,post-transform)的梯度。

## 三、H3【预算假说】:成立(角色修正)

[fig5](why/fig5_pc_plane.png)(**单 head 的全 128 维 token 云**,即 QVG 真实
聚类对象):LC head-0 的 29640 个 token 在主成分平面上**连续铺开**,该 head
的 256 个质心只能撒离散点。率失真理论(高斯源 VQ 需 2^(nR) 级码本追平变换
编码)预言字典边际收益随 K 指数衰减——真口径实测 K:64→1024(16× 码本)多
消掉的能量:chunk_001 仅 +0.3pp(99.46→99.74%),8 层中最大的 chunk_007 也只
+8.6pp(75.7→84.3%),而质心元数据涨 16×,与理论一致。角色修正:维度诅咒限制
的是字典的**上限增速**,低预算点上字典已经很强;真正的分水岭仍是 §一。
(本节历史上的两处勘误——旧图误画全局 64 维块、"0.7%"引用错误——见
[负结果台账 §三](why-refuted-and-errata.md)。)

![fig5:LC 单 head 的 token 云连续铺开,256 质心只能撒点](why/fig5_pc_plane.png)

## 三点五、Case Study:HY 半区秩 9:0 的"能量 ≠ 价值"反转

[fig6](why/fig6_hy_halves.png):**谱学不支持 9:0**——prope 半区反而更低秩
(top-9 能量 81.8% vs rope 半区 76.2%)且携带全部能量的 66%。按"消掉能量"
逻辑,秩应该全给 prope;但实验梯度(0717:4:4→9:0 单调变好)说明恰恰相反。
解释:prope 半区的能量是相机变换缠绕的,下游 attention 并不读取其精细结构
(其残差甚至可以换三值粗格,KP 三值反而全面变好)。**这是"能量不是价值"的
第三个实例**(与 H1 修正、伪影偏好判据同族):谱/能量类指标衡量的是数据自身,
不是它对生成的贡献——最终裁判只能是端到端闸门。

![fig6:HY K 半区分谱——prope 更低秩、能量更大,但秩 9:0 给 rope 反而赢](why/fig6_hy_halves.png)

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
# ① H1/H2 chunk_001 基础数据(奇异谱、逐通道误差三方对比 → h1_h2_data.npz)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_h2_compute.py

# ② H1/H3 真口径主判决(QVG 原装 prq per-head 全 D 维聚类,eval 同配置
#    num_stages=1/K∈{64,256,1024}/int2 B64,全部 8 chunk:
#    消掉能量 + 最终 relL2² + 残差回收率)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_real_path.py
#    我方同协议对照(μ+PCA 终版配置,同 8 chunk)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_ours_path.py

# ③ H2 跨层稳健性(小方差 16 通道误差比,QVG vs 我们,同 8 chunk)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h2_multichunk.py

# ④ 图 1-5(fig2 属负结果台账;谱 / H1 真口径 / 残差格效率 / H2 逐通道 / H3 质心散点)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/make_figs.py
#    图 6(HY 半区分谱反转)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/fig6_hy_halves.py

# ⑤ H4(kmeans iters 扫描,20 次 LC 生成,~1.5h/8 卡)
#    生成:campaign 臂 qvgi2 / qvgi10(qvgi<N> 会把 --kmeans_max_iters 设为 N)
printf 'lc:%d:qvgi2:0\nlc:%d:qvgi10:0\n' $(seq 1 10 | sed 'p') > /tmp/h4.txt  # 或手写 20 行
CAMPAIGN_NS=mp100 CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt \
  bash repro/0718/scripts/gpu_queue.sh /tmp/h4.txt 8
#    打分:对每个臂算 f93 PSNR vs 同 prompt BF16(帮助函数同 repro/0718/scripts/stats.py)
```

(①-④ 同一批脚本同时产出本报告与[负结果台账](why-refuted-and-errata.md)
的全部数字;旧脚本 `h1_kmeans_sub.py` 因聚类口径错误已被 ② 取代,仅留痕。)

关键数字核对点(复现应落在 ±噪声内;kmeans 有随机性,±1pp 级):
- ②:LC chunk_001 kmeans K=256 消掉 99.6%(chunk_007 80.0%);我们 99.4%
  (72.4%);QVG 残差回收率 46-49%(LC/SF 16 测点全部)、我们 61-82%;
  chunk 级最终误差我们 20/24 格更小(例外:SF chunk_000 + HY 004/005/007);
- ③:小方差 16 通道误差比 LC 1.8-2.4× / SF 1.1-1.8× / HY 1.4-2.4×
  (chunk_001 上 LC 0.148 vs 0.063);
- fig6:prope top-9 = 81.8%、能量占 66.3%;
- H4:iters 2/10/100 → 27.13 / 27.72 / 27.61(f93,p1-10)。

逐通道误差三方对比里 KIVI/我们的臂 = `pca_quant.py` 的 PCA_KIVI=1 / 终版配置
(kernel/bp_quant.py 同数学);kmeans 一律 QVG 原装(`quant_videogen/`)。

## 诚实条款执行记录

- H1 原命题按预注册判据**证伪**并如实修正(判据 0720 实验前写死于 plan);
  证伪证据、撤回/降级的论断、外部核查触发的三处勘误,全部留痕于
  [why-refuted-and-errata.md](why-refuted-and-errata.md)(负结果台账);
- kmeans 全部用 QVG 原装实现(per-head 全 D 维真口径),收敛臂 iters=100;
- 所有数据来自真实管线 dump chunk(全部 8 个)与诚实重赛后的 MP100 终表;
- 我们输掉/平局的列(hy:aq 等)的解释框架见判据 4,与本报告同一理论。
