# Budget-PCA(通道轴版)方法讲解:三个模型各自怎么做,代码在哪

> ⚠️ 0720 勘误:本文写作时的 LC/SF 战绩与 BPE 引用了记账修正前的数字。以
> [mp100-table.md](mp100-table.md)(终表)与 [bpe-audit.md](bpe-audit.md)
> (逐字节审计:LC 2.3192 / SF 2.3183 / HY cache 2.3250)为准;终版战绩:
> LC 31.68/0.9370/0.0547、SF 四维 93.21/66.65/88.69/53.27、HY 18.77/0.5016/0.3286。
> 方法机制、代码链接、env 配置部分仍然有效(kernel 实现另见 kernel/bp_quant.py)。

> 0720 版,对应 [mp100-table.md](mp100-table.md) 的终版配置。引擎
> [`pca_quant.py`](../backup/scripts/pca_quant.py),注入器
> [`pca_launcher.py`](../backup/scripts/pca_launcher.py)。所有行号可点击跳转。

## 〇、一句话

每当一段 KV 要被量化时:**减去它自己的均值和 top-r 主成分(精确可恢复的"减法"),
剩下的残差用 2-bit 均匀格量化——格子沿"通道轴"分块**(本轮吸收 KIVI 洞见的升级,
零 BPE 代价)。无字典、无 k-means、无校准、无迭代,全部组件是 GEMM/逐元素运算。

## 一、引擎:一次量化事件里发生什么(`pca_quant.py`)

入口是 [`pca_fake_quant(x, r)`](../backup/scripts/pca_quant.py#L173),`x` 形状
`[B, H, S, D]`(S 个 token、每头 D 维),五步:

1. **减均值**:`mu = X.mean(dim=1)` —— 每(头,chunk)一个 D 维均值,顺带把
   KIVI 说的"通道偏移"全部吸掉(μ 就是逐通道均值);
2. **算基**([#L190-L191](../backup/scripts/pca_quant.py#L190-L191)):
   `cov = Xc^T Xc / S` → `torch.linalg.eigh` 取 top-r 特征向量 —— **只用本
   chunk 自己的数据,一次特征分解,没有校准集**;
3. **量化系数**([#L194](../backup/scripts/pca_quant.py#L194)):每 token 的 r 个
   投影系数做 2-bit 非对称量化(每 token 一对 fp8 scale/zp)。注意顺序:
4. **残差在系数量化之后算**([#L198](../backup/scripts/pca_quant.py#L198)):
   `res = Xc − q(c)·V_rᵀ` —— 系数的量化误差被残差吸收,编解码两端用同一个
   q(c)·V_rᵀ,所以低秩通道自身零重建误差,**全部误差只存在于残差这一处**;
5. **残差过均匀格**([`_quant_residual`](../backup/scripts/pca_quant.py#L122)),
   这里就是本轮的关键升级——**量化轴**([#L145](../backup/scripts/pca_quant.py#L145)):

```python
if RES_AXIS == "channel":                       # PCA_RES_AXIS[_K/_V]=channel
    xt = x.transpose(-1, -2).contiguous()       # [.., S, D] -> [.., D, S]
    g = <128/96/64/... 中能整除 S 的最大块>
    return _asym_quant_lastdim_grouped(xt, 2, g).transpose(-1, -2)
```

- 默认 **token 轴**:每 token 沿 D 维分块,一块一对 scale/zp——适合"token 间
  差异大"的数据;
- **channel 轴**:转置后每通道沿 token 维分块——适合"通道间方差差几个量级"的
  数据(KIVI 对 K 的洞见;LC/SF 的 pre-RoPE K 正是这种)。**元数据量不变,BPE
  完全一样**,只是 scale 的"方向"换了。
- 每张量/每半区可独立选格子、块长、轴:
  [`_res_grid_override`](../backup/scripts/pca_quant.py#L34) 用环境变量
  `PCA_RES_{GRID,BLOCK,AXIS}_{K,V,KP,VP}` 逐层覆盖。

解码([#L206](../backup/scripts/pca_quant.py#L206)):`x̂ = μ + q(c)V_rᵀ + r̂es`,
一次瘦 GEMM + 逐元素加。

## 二、注入器:怎么接到三个模型上(`pca_launcher.py`)

QVG fork 的三条管线都通过统一接口 `quant_videogen.compress.compress_kv_cache`
做量化。launcher 在目标脚本 import 之前把它替换掉
([#L21-L36](../backup/scripts/pca_launcher.py#L21-L36)):命令行传
`--quant_type naive-int2` 时劫持为我们的 `pca_fake_quant_kv`,其余 quant_type
(如 QVG 自己的 kmeans)原样放行。所以**我们和 QVG/baseline 走同一个钩子、同一个
量化时机,比较天然公平**。日志里的 hijack 计数就是劫持次数(必须 >0,否则该
run 作废)。

## 三、三个模型各自怎么做

### LongCat(条件窗一次性量化)

- **量化时机**:每段续写开始,73 帧条件窗 prefill 完成后整窗量化一次
  ([`longcat_video_dit.py#L541`](../../experiments/LongCat/longcat_video/modules/longcat_video_dit.py#L541));
  cache 存的是 **pre-RoPE K**(读取时按段内固定 grid 现转);
- **配置**:`PCA_R=4, asym B128, K/V 都用 channel 轴`
  ```bash
  PCA_R=4 PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_V_MODE=pca \
  PCA_RES_AXIS_K=channel PCA_RES_AXIS_V=channel
  ```
- **为什么**:pre-RoPE K 的通道方差异质性最强(KIVI 在 LC 上 33.5dB 碾压即证据),
  channel 轴一举 +5.2dB;V 同样受益(+1.1dB)。**BPE 2.3125**
  (2 + 残差 scale/zp 0.125 + 系数 0.0625 + 系数 scale/zp 0.125);
- **战绩**:35.23/0.9635/0.0385(次优 KIVI 33.47/0.9548/0.0453)。

### HY-WorldPlay(唯一的 post-transform、256 维打包模型)

- **量化时机**:chunk 老化出近期窗后按 token 段量化
  ([`pipeline…relative_rope.py#L377`](../../experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py#L377));
  cache 每 token 是 **[k_rope‖k_prope] 256 维打包**(写入时已转好,读取不重转);
- **配置**:半区秩 + 半区格子 + K 通道轴三件套
  ```bash
  PCA_R=4 PCA_HALF_R_K=9,0 PCA_HALF_R_V=9,0 PCA_RES_GRID=asym PCA_RES_BLOCK=128 \
  PCA_RES_AXIS_K=channel PCA_RES_BLOCK_K=64 PCA_RES_GRID_KP=ternary PCA_RES_BLOCK_KP=64
  ```
  逐项读:两半区分开做 PCA([`_hq`](../backup/scripts/pca_quant.py#L682),
  秩 **9:0** = prope 半区数据价值低,秩全给 rope 半区);K 残差 **channel 轴 +
  B64 细块**(保真+一致性双赢的甜点);K-prope 半区残差**三值格**(该半区连
  噪声都不值 4 电平,还省 BPE)。**BPE 2.29**;
- **战绩**:18.50/0.4822/0.3406 + SC 94.67(全部最优),BC/IQ 统计平局,AQ −0.96
  (结构性,见表终注)。

### Self-Forcing(滑窗流式,走 pre-RoPE 存储路径)

- **量化时机**:chunk(24 latent)老化后量化;QVG fork 的 SF 存储是 BSHD/BHSD
  不一致的,须 `PCA_SF_STORE_FIX=1`
  ([launcher#L41](../backup/scripts/pca_launcher.py#L41))在 store 时转置修复;
- **配置**:**与 LC 完全相同**(方法统一性的直接体现)
  ```bash
  PCA_R=4 PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_V_MODE=pca \
  PCA_RES_AXIS_K=channel PCA_RES_AXIS_V=channel PCA_SF_STORE_FIX=1
  ```
  另注意命令必须带 `--quant_type naive-int2 --quant_block_size 64 --use_ema
  --local_attn_size 195`,帧数参数是 latent 单位(180=717px);**BPE 2.3125**;
- **战绩**:VBench 四维全部第一(93.18/66.69/88.66/53.22)。

## 四、三模型对照速览

| | LC | HY | SF |
|---|---|---|---|
| 量化对象 | pre-RoPE K + V | **post-rope‖prope 打包 256 维** | pre-RoPE K + V |
| 量化时机 | 段首条件窗一次 | chunk 老化逐段 | chunk 老化逐段 |
| 秩 | K=V=4 | **半区 9:0 / 9:0** | K=V=4 |
| 残差格 | asym B128 | K:asym **B64**;KP:**三值** B64;V:asym B128 | asym B128 |
| 残差轴 | K/V 都 channel | K channel(V token) | K/V 都 channel |
| BPE | 2.3125 | 2.29 | 2.3125 |
| 特殊件 | — | 半区机制(唯一 packed 模型) | store 转置修复 |

**共同骨架一个字不变**:μ 减法 → 本 chunk eigh → 系数 2bit → 残差均匀格。
三个模型的差异全部是"格子/轴/秩往哪放"的静态配置,与 QVG 逐模型调
iters/K/块参数同性质,但我们没有任何迭代或校准组件。
