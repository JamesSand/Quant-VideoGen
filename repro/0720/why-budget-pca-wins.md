# 为什么 Budget-PCA 赢、QVG 的 k-means 输:判决报告

> 预注册计划:why-analysis-plan(判据先于实验写死;本页如实记录**H1 原命题被证伪**
> 及其修正)。数据:三管线真实 dump chunk(`chunks/`)、MP100 诚实终表、
> `why/h1_h2_data.npz`;图 `why/fig1-6`。所有对比用 QVG 原装 kmeans 实现,
> 收敛臂给足 iters=100。
>
> **0720 二审勘误**(外部核查后重算,三处修正,详见 §勘误):①旧版 kmeans 侧
> 用了"全局 64 维块"聚类,与 QVG 真实口径(**per-head、全 D 维 token**,
> `prq_quantize_tensor`,centroids (B,H,K,D))不符,fig2/3/5 全部按真口径重算;
> ②"K:64→1024 只多消掉 0.7%"是引用错误;③单 chunk 数字改为 4 chunk
> (=4 个不同层)复核,层依赖的论断如实降级。**核心机制结论在真口径下幸存
> 且更干净;效率差的幅度由 3-16× 修正为 ~2×(12 格全稳)。**

## 〇、一页结论

两个方法是同一骨架的两种减法(QVG 减最近质心,我们减 μ+子空间),胜负不在
"谁减掉的能量多"——**k-means 减得更多**(H1 原命题证伪,真口径 4 chunk 全部
复核)——而在**残差的命运**:

> **k-means 把结构吃得太干净,残差近白噪声:其 2-bit 均匀格的能量回收率在
> 全部 8 个 LC/SF 测点上钉死在 46-49%——教科书白噪声水平;我们的残差保留
> 通道结构,同规格格子回收 ~75%,效率差 ~2×,12 个(模型×层)格无一例外。
> 减法阶段的小胜被残差阶段的稳定溃败抵消(chunk 级最终误差我们 11/12 格
> 更小)——结构分工与格子能力的匹配,才是胜负手。**

## 一、H1【表示假说】:原命题证伪 → 修正为"结构分工"命题

**原预测**:同等元数据比特下,rank-r 子空间消掉的能量 ≫ 256 质心字典。
**判决:证伪**([fig2](why/fig2_subtraction_efficiency.png),真口径:per-head
全 D 维聚类,与 eval 完全同配置)。同等元数据(LC/SF:~0.2 bits/elem 对
~0.19)下,kmeans K=256 在 **4 个 chunk 上全部**比 μ+PCA r=4 消得多:LC
99.6/97.8/91.8/82.6% vs 99.4/97.2/87.6/74.8%(chunk 001/000/003/006);SF 同向
4/4。HY 例外说明:字典消得仍多(89.2 vs 85.8%)但其质心元数据是我们的 ~4×
(0.61 vs 0.15 bits/elem——短 chunk S=7040 摊不薄 256×256 质心表),per-bit
反而我们高;不改变"字典减法不弱"的证伪结论。

![fig2:H1 判决(真实口径)——同等比特下字典减得仍更多](why/fig2_subtraction_efficiency.png)

**修正命题(数据导出)**:最终质量由"残差能量 × 残差格效率"决定。实测
残差格效率(**同 chunk 内自洽口径**:最终误差能量 / 减法后残差能量,4 chunk
均值[极差],[fig3](why/fig3_residual_efficiency.png)):

| 模型 | QVG(kmeans 残差→int2 B64) | Budget-PCA(结构化残差→通道轴格) | 差距 |
|---|---|---|---|
| LC | 0.52 [0.51-0.53](回收 ~48%) | **0.25 [0.22-0.33]**(回收 ~75%) | **2.1×** |
| SF | 0.52 [0.51-0.54] | **0.26 [0.21-0.38]** | 2.0× |
| HY | 0.42 [0.36-0.48] | **0.25 [0.19-0.31]** | 1.7× |

![fig3:残差格效率——QVG 回收 ~48%(白噪声水平)vs 我们 ~75%](why/fig3_residual_efficiency.png)

机制签名比幅度更硬:QVG 的回收率在全部 8 个 LC/SF 测点钉死在 **46-49%**——
这正是 minmax 均匀 2-bit 格在近白噪声上的教科书损失,层深、模型、数据都不能
动它;我们的残差保留逐通道幅度差,通道轴 scale 逐块贴合,回收 62-82%。净账:
chunk 级最终误差我们 **11/12 格更小**(唯一例外 SF chunk_000:0.177 vs
0.155%,该层两家都近无损),且层越深差距越大(LC chunk_006:5.6 vs 9.1%)。

> 勘误注:旧版此表为 16×/7×/3×,系分子用视频级 relL2²、分母用错误聚类口径
> 的 chunk 级能量,两头口径不一致所致;修正后幅度回落到 ~2× 但跨 12 格全稳,
> 机制结论不变、证据反而更干净。

**低秩薄饼成立但层依赖**([fig1](why/fig1_spectra.png)):top-4 特征值能量
LC 82/77/51/41%、SF 69/86/46/40%、HY 46/52/39/42%(chunk 001/000/003/006)——
早层最扁,深层三家趋同,跨模型排序**不稳**(chunk_000 上 SF>LC)。它决定的是
减法性价比的上限,不是对 kmeans 的胜因;旧版"LC ≫ SF ≫ HY"的单 chunk 排序
论断撤回。

![fig1:K 协方差谱——低秩薄饼层依赖,跨模型排序不稳](why/fig1_spectra.png)

## 二、H2【几何假说】:LC 核心成立(跨层稳);SF"无红利"降级为部分成立

**预测**:k-means 欧氏度量被高方差通道劫持,小方差通道结构不被编码。
**判决**([fig4](why/fig4_channel_error.png);比值 = QVG/我们的小方差 16 通道
误差/信号比,4 chunk 复核):

- LC(通道方差极差 146×):**2.0-2.4×,4 chunk 全部成立**(chunk_001:0.148
  vs 0.063)——损伤精确集中在被劫持端,这是 H2 的判决性证据,跨层稳;
- SF:**1.2-1.8×,层依赖**。旧版"QVG ≈ 我们、通道机制无红利"只在 chunk_001
  (1.16×)成立,深层拉开到 1.8×——"SF 均质反向验证"**降级为部分成立**:
  方向对(SF 的劫持效应确实最弱,与其极差仅 7× 一致),但"无红利"过强;
- HY(极差 154×,post-transform):1.8-2.1×,跨层稳;KIVI 的通道优势被
  rope/prope 混合削弱。

![fig4:逐通道误差——QVG 的损伤集中在小方差通道](why/fig4_channel_error.png)

文献闭环:KVQuant/KIVI 原文正是此发现(pre-RoPE key 有位置一致的 outlier
通道;**RoPE 施加后通道结构被打散**)——解释了通道轴收益 LC(+3.5dB,
pre-RoPE)≫ HY(+1.3dB,post-transform)的梯度。

## 三、H3【预算假说】:成立(角色修正 + 口径勘误)

[fig5](why/fig5_pc_plane.png)(勘误后:**单 head 的全 128 维 token 云**,即
QVG 真实聚类对象;旧版误画全局 64 维块):LC head-0 的 29640 个 token 在主
成分平面上**连续铺开**,该 head 的 256 个质心只能撒离散点。率失真理论(高斯
源 VQ 需 2^(nR) 级码本追平变换编码)预言字典边际收益随 K 指数衰减——真口径
实测 K:64→1024(16× 码本)多消掉的能量:chunk_001 仅 +0.3pp(99.46→99.74%),
最深的 chunk_006 也只 +8.5pp(78.3→86.7%),而质心元数据涨 16×,与理论一致
(旧版"0.7%"是把 256→1024 的增量错标到 64→1024 头上,已改)。角色修正不变:
维度诅咒限制的是字典的**上限增速**,低预算点上字典已经很强;分水岭仍是 §一。

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
#    num_stages=1/K∈{64,256,1024}/int2 B64,chunk 001/000/003/006:
#    消掉能量 + 最终 relL2² + 残差回收率)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_real_path.py
#    我方同协议对照(μ+PCA 终版配置,同 4 chunk)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_ours_path.py

# ③ H2 跨层稳健性(小方差 16 通道误差比,QVG vs 我们,同 4 chunk)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h2_multichunk.py

# ④ 图 1-5(谱+跨层范围 / H1 真口径判决 / 残差格效率 / H2 逐通道 / H3 per-head 质心散点)
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

(旧脚本 `h1_kmeans_sub.py` 因聚类口径错误已被 ② 取代,保留仅为留痕。)

关键数字核对点(复现应落在 ±噪声内;kmeans 有随机性,±1pp 级):
- ②:LC chunk_001 kmeans K=256 消掉 99.6%(chunk_006 82.6%);我们 99.4%
  (74.8%);QVG 残差回收率 46-49%(LC/SF 8 测点全部)、我们 62-82%;
  chunk 级最终误差我们 11/12 格更小;
- ③:小方差 16 通道误差比 LC 2.0-2.4× / SF 1.2-1.8× / HY 1.8-2.1×
  (chunk_001 上 LC 0.148 vs 0.063);
- fig6:prope top-9 = 81.8%、能量占 66.3%;
- H4:iters 2/10/100 → 27.13 / 27.72 / 27.61(f93,p1-10)。

逐通道误差三方对比里 KIVI/我们的臂 = `pca_quant.py` 的 PCA_KIVI=1 / 终版配置
(kernel/bp_quant.py 同数学);kmeans 一律 QVG 原装(`quant_videogen/`)。

## 勘误记录(0720 二审,外部核查触发)

外部核查指出三处问题,逐条核实后全部属实,已修正:

1. **聚类口径错误(最重要)**:旧版 kmeans 侧脚本把所有 head 的 64 维块拼成
   一池做全局聚类;QVG 真实实现(`quant_videogen/functions.py::prq_quantize_tensor`)
   是 **per-head、全 D 维 token 聚类**(centroids (B,H,K,D)、ids (B,H,S)),
   `block_size=64` 只作用于残差量化。fig2/3/5 与 §一/§三全部数字按真口径重算
   (`h1_real_path.py`/`h1_ours_path.py`)。修正后 H1 证伪结论不变且更稳
   (4 chunk 全向),效率差幅度由 3-16× 回落为 ~2×(机制签名 46-49% 回收率
   反而更硬);
2. **"0.7%" 引用错误**:64→1024 的真实增量是 +0.3pp(chunk_001)到 +8.5pp
   (chunk_006);旧句把 256→1024 的增量错标到 64→1024;
3. **单 chunk 过度概括**:fig1 的"LC ≫ SF ≫ HY"排序、H2 的"SF 无红利
   反向验证"在跨 chunk(=跨层)复核下不稳,分别撤回/降级为部分成立;
   LC 的 H2 核心(2.0-2.4×)与 H4 经复核幸存。

## 诚实条款执行记录

- H1 原命题按预注册判据**证伪**并如实修正(判据 0720 实验前写死于 plan);
- 外部核查的三处批评全部核实、修正并留痕(上节);
- kmeans 全部用 QVG 原装实现(含真实聚类口径),收敛臂 iters=100;
- 所有数据来自真实管线 dump chunk 与诚实重赛后的 MP100 终表;
- 我们输掉/平局的列(hy:aq 等)的解释框架见判据 4,与本报告同一理论。
