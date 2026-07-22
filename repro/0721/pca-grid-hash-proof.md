# PCA-Grid Hash：条件最优性、复杂度与论文边界

## 1. 优化目标与编码器

对一个 head/chunk 的 \(X\in\mathbb R^{S\times D}\)，编码器在线计算四个
PCA 坐标。每个坐标用固定 Gaussian quartile 阈值
\((-\infty,-0.674\sigma),[-0.674\sigma,0),[0,0.674\sigma),
[0.674\sigma,\infty)\) 分成四区间，四个 2-bit bin 拼成
\(\ell_s\in\{0,\ldots,255\}\)。PCA 基和阈值只用于编码，不存储；解码器只读取
uint8 label、FP8 table 和 packed INT2 residual。

记 \(A\in\{0,1\}^{S\times256}\) 为 label 的 one-hot 矩阵，\(T\) 为 table，
\(Z\) 为实际反量化 residual payload。重建为

\[
\widehat X=AT+Z,\qquad
\mathcal D(T,Z)=\lVert X-AT-Z\rVert_F^2.
\]

这里的 \(\mathcal D\) 定义在实际 FP8/INT2 payload 反量化后的实值上、最终
BF16 输出 cast 之前。实现还会对 BF16 decoder 输出重新计算 SSE；refit 只有在
该 SSE 严格下降时才逐 head 接受。因此下面的最优性定理不被错误扩展到 BF16
取整后的非凸目标，而实际输出仍有单调不增保护。

## 2. 固定 code 下的 table 全局最优

**定理 1（固定 labels 与 residual payload 的条件全局最优）。**
固定 \(A,Z\)，对所有实值 table，

\[
T^\star_j={1\over n_j}\sum_{s:\ell_s=j}(X_s-Z_s),
\quad n_j=\#\{s:\ell_s=j\}
\]

是 \(\min_T\mathcal D(T,Z)\) 的全局最优解；空 cell 任意。并且

\[
\mathcal D(T,Z)=\mathcal D(T^\star,Z)
+\sum_{j:n_j>0}n_j\lVert T_j-T^\star_j\rVert_2^2 .
\]

**证明。** 令 \(Y=X-Z\)。按 label 将
\(\lVert Y-AT\rVert_F^2\) 分成 256 个互不耦合的组。每组平方和在组均值处
达到最小；将 \(Y_s-T_j=(Y_s-T^\star_j)+(T^\star_j-T_j)\) 展开，交叉项因
\(\sum_{\ell_s=j}(Y_s-T^\star_j)=0\) 消失，即得正交分解。证毕。

因此，用实际反量化后的 \(Z\) 做 grouped mean refit，而不是对未量化 residual
做 refit，才对应 post-INT2 payload MSE。FP8 factor 重选和最终 BF16 cast 不属于
该闭式子问题；研究实现会比较 refit 前后实际 BF16 重建误差，只保留严格下降的
结果。

## 3. 固定存储因子与 grid 下的离散最优

**定理 2（固定 FP8 normalization factor）。** 固定每个通道的 factor
\(f_d\)、\(A\) 与 \(Z\)。若 table 元素限制为
\(T_{jd}=f_d c_{jd}\)，其中 \(c_{jd}\) 属于 FP8-E4M3 可表示集合，则

\[
c^\star_{jd}=\operatorname{nearest}_{c\in\mathrm{FP8}}
\left(T^\star_{jd}/f_d\right)
\]

逐元素给出该离散集合上的全局最优 table。原因是定理 1 的附加项按
\((j,d)\) 完全可分，权重 \(n_j>0\) 不改变最近点。

**定理 3（固定四电平 residual grid）。** 固定 \(A,T\) 以及每个 block 的
stored FP8 scale/zero-point \((a_b,z_b)\)。每个 residual code

\[
q_i^\star=\operatorname{clip}_{\{0,1,2,3\}}
\operatorname{round}\left({X_i-(AT)_i-z_b\over a_b}\right)
\]

是该固定 grid 上的全局最优 code。目标对元素可分，故为最近四个重建值之一。

若 FP8 factor 与 residual grid 全程固定，交替执行“定理 1/2 的 table 更新”和
“定理 3 的 code 更新”时，每步 payload MSE 单调不增；table/code 状态均有限，
采用确定性 tie-break 且只接受严格下降时最终停在 coordinate-wise fixed point。
研究实现允许重新选择 FP8 factor，因此不直接援引这一收敛结论，而是在每轮后以
实际 BF16 SSE 做 accept/reject。上述结论都不是 assignments、factor exponent
或 grid 的联合全局最优。

## 4. PCA 并非 post-INT2 全局最优：显式反例

下面 8 个已中心化三维样本，rank-1 PCA 的连续 residual energy 更低，但对每个
通道使用 min-max asymmetric 四电平量化后，最终量化误差更高：

```text
[-0.262,-0.135, 0.752] [ 0.306, 0.996, 2.549]
[ 0.374, 0.324,-6.399] [-0.184,-0.100,-2.618]
[ 0.205,-0.922,-4.406] [ 0.404, 0.330,-0.634]
[ 0.211, 0.813,13.616] [-1.054,-1.307,-2.860]
```

| predictor direction | residual energy | post-INT2 error |
|---|---:|---:|
| PCA \(v=(0.013,0.073,0.997)\) | **4.701** | 0.369 |
| alternate \(v=(0.004,-0.124,-0.992)\) | 5.507 | **0.232** |

alternate residual energy 高 17%，post-INT2 MSE 却低 37%。因此
Eckart–Young 只能证明量化前低秩 residual energy 最优，不能证明最终 INT2
重建误差最优；本文的严格证明必须限定在固定 labels/grid/codes 的子问题。

## 5. 真实 BPE 闭式账本

令 \(N=\lceil S/G\rceil\)，table size \(K=256\)。单个 K 或 V tensor：

\[
\mathrm{BPE}
= {2GN\over S}                 \quad\text{(padded INT2)}
+ {16N\over S}                 \quad\text{(FP8 scale+zero)}
+ {8\over D}                   \quad\text{(uint8 label)}
+ {8K\over S}                  \quad\text{(FP8 table)}
+ {16\over S}                  \quad\text{(table/residual int8 exponents)}.
\]

SF/HY 在 K/V 间共享 labels，因此 cache 平均口径的 label 项由 \(8/D\) 变为
\(4/D\)。实际 tensor 逐字节审计：

| model | \(S,D,G\) | label policy | hash BPE | QVG actual BPE |
|---|---|---|---:|---:|
| LC | 29640,128,64 | K/V separate | **2.3864** | 2.4639 |
| SF | 37440,128,64 | shared V-derived | **2.3364** | 2.4063 |
| HY | 7040,256,64 | shared V-rope-derived | **2.5588** | 3.3199 |

所有 padding、table、labels、scale/zero-point 和 normalization exponent 都已入账。

## 6. 复杂度与 kernel 形态

设 label source 维度为 \(D_\ell\)、固定 subspace iteration 次数为 \(I\)：

- label：\(O(SD_\ell^2+ID_\ell^2r+SD_\ell r)\)；
- uint8 sort + grouped reduction：当前实现 \(O(S\log S+SD)\)；
- residual quant/pack：\(O(SD)\)；
- fused decode：\(O(SD)\)，一次完成 label gather、INT2 unpack、FP8
  scale/zero 和输出；
- QVG assignment：每次迭代 \(O(SKD)\)，另有 grouped reduction。

关键差异不是声称 PCA 无成本，而是不存在 \(S\times K\times D\) nearest-centroid
search。SF 使用一轮 V-derived label 并在 K/V 间共享；HY 只在 V 的 128-d
rope half 上计算 label。

8 个真实 chunk 的 total encode+decode 几何平均加速及 bootstrap 95% 下界：

| model | speedup | 95% lower |
|---|---:|---:|
| LC | 7.326× | 7.092× |
| SF | 1.108× | 1.105× |
| HY | 1.281× | 1.271× |

## 7. 预注册实验结论

PCA-Grid Hash 的 G1（all-in BPE）、G2（8 chunk K/V MSE）和 G3
（encode+decode 总延迟）通过。三模型单样本 paired canary 在 G4 失败：

- LC 通过早停门槛；
- SF 的 Image Quality 相对 QVG 为 \(-2.84\) point；
- HY 的 LPIPS、Background Consistency、Subject Consistency、Aesthetic 与
  Image Quality 越过 margin。

因此按预注册规则没有为 hash 启动 MP100，也不能宣称三项硬要求达成。有限回退
factor-only product grid 在 chunk 级 G1–G3 通过，但 paired canary 同样失败
（SF IQ \(-2.14\) point；HY PSNR \(-0.188\) dB、LPIPS \(+0.0066\)、
AQ \(-2.07\)、IQ \(-1.53\)）。既有完整 MP100/10-seed factor-grid 数据的
90% CI TOST 中，LC、SF 与 HY 三个 primary reconstruction 指标通过，但 HY
guardrail 失败：BC 未证明等价或优效，IQ 的下界
\(-0.619<-0.30\)，AQ 下界 \(-1.152<-0.30\)。这同样不能替代 G4。

## 8. 可宣称与不可宣称

可以严格宣称：

1. 固定 analytic labels、固定 decoded residual payload 时，实值 table refit
   为全局最优；
2. 再固定 FP8 factor 时，nearest FP8 table code 为离散全局最优；
3. 固定 table 与四电平 grid 时，round/clamp residual codes 为离散全局最优；
4. 实际 BPE、chunk MSE 和 kernel total latency 已分别通过预注册 G1–G3；
5. G4 已被实验否决，当前候选没有同时满足三项硬要求。

不可宣称：

- PCA、四分位阈值、rank、labels 或 normalization exponent 的联合全局最优；
- 对所有数据分布的 post-INT2 全局最优；
- attention error 或视频质量的解析最优；
- 用较低 tensor MSE 或 primary 指标通过掩盖 G4 guardrail 失败。

论文中的“optimal”必须写成 **fixed-code conditional optimum**；端到端质量只能由
paired generation experiment 验证。
