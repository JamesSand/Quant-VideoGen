# qvg4pt(QVG 四点位 int2)实现全链路 walkthrough

> 目的:自查/审计用。qvg4pt 是 0721 的反事实臂——**QVG 原装 kmeans 减法一个字
> 不动,只把它的残差格从三电平对称换成四电平非对称 B64**,回答"QVG 若把 2-bit
> 的四个码字用满会怎样"。终值:**PSNR 32.29 / SSIM 0.9478 / LPIPS 0.0405**
> (LC 10 prompts,f93),BPE **2.589**(实存口径,超预算 11%)。
> 所有行号以当前 HEAD 为准,代码全部逐字摘抄。

## 〇、一页链路图

```
jobs_aux_q4.txt: "lc:1:qvg4pt:0" … "lc:10:qvg4pt:0"
  → repro/0718/scripts/gpu_queue.sh(逐 GPU 领 job)
  → repro/0718/scripts/campaign.sh:95 解析 → :128 命中 qvg4*) 分支
      export PCA_QVG4=1 PCA_FP8SIM=1
      torchrun repro/backup/scripts/pca_launcher.py $LCC --quant_type naive-int2
  → pca_launcher.py:25-42 monkey-patch:凡 quant_type 匹配 naive-int\d+,
      把 quant_videogen.compress.compress_kv_cache 劫持为 pca_fake_quant_kv
  → pca_launcher.py:393-394 runpy 执行 experiments/LongCat/run_long_t2v.py
  → 每 segment 一次:pipeline_longcat_video.py:1195 dit.quantize_kv_cache
      (条件窗 prefill 之后、50 步去噪之前,只此一次)
  → longcat_video_dit.py:573 逐层调 compress_kv_cache(k, v, …)
      k/v 形状 [1, 32, S, 128],K 是 pre-RoPE(attention.py:150-159 先 clone 再 rope)
  → pca_quant.py:775 dispatch(PCA_QVG4=1)
  → pca_quant.py:646 _qvg4_fake_quant_kv ← 核心实现
      = QVG 原装 batch_kmeans_Euclid(K=256, iters=100)减法
      + 四电平 asym B64 残差格 + fp8 元数据模拟
  → 评分:f93 vs bf16_rep0(score_fp8.py 同款协议)→ 32.29
```

## 一、这条臂和邻居们的差异(先看这张表)

| 臂 | 减法 | 质心存储 | 残差格 | 残差元数据 | BPE(LC 实存) |
|---|---|---|---|---|---|
| QVG 原版 | kmeans K=256 | fp32 | **三电平对称** absmax B64(token 轴) | scale(fp8)=0.125 | 2.464 |
| **qvg4pt(本文)** | 同上,不动 | fp32(同上) | **四电平非对称** min-max B64(token 轴) | scale+zp(fp8)=**0.25** | **2.589** |
| qvg-nom(qvgprot) | 同上 | **bf16** | 四电平非对称 B128(token 轴) | scale+zp(fp8)=0.125 | 2.3257(名义,压线) |
| Budget-PCA(我们) | μ+top-r PCA | — | 四电平非对称 B128(**通道轴**) | scale+zp(fp8)=0.125 | 2.3195 |

qvg4pt 与 QVG 原版**唯一**的差别是残差格那一列;与 qvg-nom 的差别是质心 dtype
和 B64/B128。

## 二、入口:job spec → campaign 分支

Job 文件 `repro/0721/mpfull/jobs_aux_q4.txt`(fp8 合法口径重跑,10 行):

```
lc:1:qvg4pt:0
…
lc:10:qvg4pt:0
```

解析(`repro/0718/scripts/campaign.sh:95,106-107`):`IFS=: read -r KIND A B C D`
→ `lc)` 分支绑定 `P=$A; ARM=$B; REP=${C:-0}`。ARM=qvg4pt 在 `case $ARM in`
(:118)里首先命中 `qvg4*)`:

```bash
# campaign.sh:128-131,逐字
qvg4*) export PCA_QVG4=1 PCA_FP8SIM=1
      PCA_TARGET=experiments/LongCat/run_long_t2v.py PYTHONPATH=experiments/LongCat \
        torchrun --nproc_per_node=1 --standalone repro/backup/scripts/pca_launcher.py \
        $LCC --quant_type naive-int2 --quant_block_size 64 > $LOG 2>&1; RC=$? ;;
```

要点:
- **只导出两个环境变量**:`PCA_QVG4=1`(选臂)、`PCA_FP8SIM=1`(元数据 fp8 模拟,
  0721 勘误后补上的——首轮漏设导致 32.85 作废,见 §七);
- 跑的不是 run_long_t2v.py 本体,而是 **pca_launcher.py**,靠 `PCA_TARGET` 转发;
- `--quant_type naive-int2` 只是**劫持暗号**(见 §三),不是真的跑 naive-int2;
- `$LCC`(campaign.sh:114-117)= LongCat 单段续写的公共参数:
  `--init_video_path $BASE/lc/base/$P-0.mp4 --num_segments 1 --num_cond_frames 73 --seed 0 …`;
- 对照:QVG 原版臂走的是原生路径 `--quant_type triton-nstages-kmeans-int2`
  (campaign.sh:24),**不经过** launcher。

## 三、挂载机制:launcher monkey-patch

`repro/backup/scripts/pca_launcher.py:25-42`(节选逐字):

```python
def _patched(k, v, quant_type, quant_config, quantize_fn):
    if re.fullmatch(r"naive-int(\d+)", quant_type) is None:
        return _orig(k, v, quant_type, quant_config, quantize_fn)
    ...
    return pca_fake_quant_kv(k, v)

_compress.compress_kv_cache = _patched
```

```python
# pca_launcher.py:393-394
_target = os.environ.get("PCA_TARGET", "experiments/LongCat/run_long_t2v.py")
runpy.run_path(_target, run_name="__main__")
```

即:launcher 在目标脚本 import 之前,把 fork 公共库
`quant_videogen/compress.py:171` 的 `compress_kv_cache` 换成 `_patched`;
`--quant_type naive-int2` 命中正则后,真正执行的是 `pca_quant.pca_fake_quant_kv`。
run_long_t2v.py 本身对 pca_quant **零感知**——它只把 quant_type 装进
`pipe.dit.quant_config`(run_long_t2v.py:245-252)。

## 四、触发点:什么时候、对什么张量量化

- **每 segment 恰好一次**(不是逐 chunk-age-out):
  `pipeline_longcat_video.py:1185-1197`——条件窗 latents 先经 DiT prefill
  (`_cache_clean_latents`,`return_kv=True`),随后、在 50 步去噪**之前**:

  ```python
  kv_cache_dict = self.dit.quantize_kv_cache(
      kv_cache_dict, offload_kv_cache=offload_kv_cache
  )
  ```

- 逐层调用(`longcat_video_dit.py:541` 定义,`:573-575` 调用):

  ```python
  k_quant, v_quant = compress_kv_cache(
      k, v, self.quant_config.quant_type, self.quant_config, quantize_fn
  )
  ```

- **张量形状 [B, H, S, D] = [1, 32, S_cond, 128]**(num_heads=32,head_dim=128,
  longcat_video_dit.py:203-205);
- **K 是 pre-RoPE**:`attention.py:150-159`,cache 在 q_norm/k_norm 之后、
  `rope_3d` 之前 `k.clone()`;解码侧(attention.py:271-286)每步对
  `cat([k_cache, k])` 重新施加 rope——与 QVG paper 的 pre-RoPE key caching
  协议一致。

## 五、核心实现:`_qvg4_fake_quant_kv`

`repro/backup/scripts/pca_quant.py:646-662`,全文逐字:

```python
def _qvg4_fake_quant_kv(k, v):
    """反事实臂(0721):QVG 原装 kmeans 减法(per-head 全 D 维,K=256,LC 官配
    iters=100)不动,残差格由其三电平对称换成【四电平非对称 B64 + fp8 s/z】
    ——回答"QVG 若把 2-bit 的四个码字用满会怎样"。BPE 代价:残差 s/z 从
    0.125 涨到 0.25(B64 asym 双参数),质心/索引账不变。"""
    from quant_videogen.kmeans.kmeans_euclid import batch_kmeans_Euclid
    iters = int(os.environ.get("PCA_QVG4_ITERS", "100"))
    outs = []
    for x in (k, v):
        B, H, S, D = x.shape
        X = x.float().view(B * H, S, D).contiguous()
        lab, cent, _, _ = batch_kmeans_Euclid(X, n_clusters=256, max_iters=iters)
        g = torch.gather(cent, 1, lab.long().unsqueeze(-1).expand(-1, -1, D))
        res = (X - g).view(B, H, S, D)
        rq = _asym_quant_lastdim_grouped(res, 2, 64, mse_opt=False, fp8_per_row=True)
        outs.append((g.view(B, H, S, D) + rq).to(x.dtype))
    return outs[0], outs[1]
```

dispatch 入口(`pca_quant.py:775-779`):`PCA_QVG4=1` 时打印
`[pca_quant] QVG-4pt counterfactual: kmeans sub + ASYM 4-level B64 residual`
(该行可在 `repro/0718/logs/lc_1_qvg4pt_0.log:20` 实证)并进入上述函数。
dispatch 优先级:FACTOR_GRID → GRID_HASH → QVGPRO → **QVG4** → KIVI_POST →
KIVI_PAPER → KIVI → RTN → 我们的 pca 主路径。

### 5.1 减法:QVG 原装 kmeans,一个字没动

`quant_videogen/kmeans/kmeans_euclid.py:29-86` 的 `batch_kmeans_Euclid`——
这是 **QVG 自己发布的库代码**(Triton assign + sorted centroid update):

- 输入 reshape 成 `[B*H, S, D]` → **per-(batch,head) 聚类,全 128 维**,与其
  eval 真口径一致;
- `n_clusters=256`,`max_iters` 由 `PCA_QVG4_ITERS` 控制,默认 100 = LC 官配;
  `tol=1e-4`(label 变化率早停,`:80`);随机初始化取自数据点(`torch.randint`,`:54`);
- 空簇保旧质心(`_euclid_iter`,`:13-25`);
- 重建 = `gather(cent, lab)`,残差 = X − 最近质心。

### 5.2 残差格:四电平非对称 min-max,token 轴 B64

`pca_quant.py:89-103`(`_asym_quant_lastdim_grouped`,mse_opt=False 路径逐字):

```python
xg = x.reshape(*S[:-1], S[-1] // group, group)     # [B,H,S, D/64=2, 64]
mn = xg.amin(dim=-1, keepdim=True)
mx = xg.amax(dim=-1, keepdim=True)
scale = ((mx - mn) / (2 ** bits - 1)).clamp_min(1e-8)   # bits=2 → 除以 3
if PCA_FP8SIM:
    scale = _fp8(scale, per_row=fp8_per_row).clamp_min(1e-8)
    mn = _fp8(mn, per_row=fp8_per_row)
q = torch.clamp(torch.round((xg - mn) / scale), 0, 2 ** bits - 1)   # q ∈ {0,1,2,3}
return (q * scale + mn).reshape(S)
```

- 分组沿**最后一维 D**:每 token 的 128 通道切成 2 块 × 64——即 **token 轴
  B64**,与 QVG 原装的分块方向相同(对照可比,只换格子);
- `q ∈ {0,1,2,3}` **四个码字全用满**,min-max 非对称覆盖整个区间;
- 每块存 (scale, zero-point=块 min) 两个 fp8 参数 → 16/64 = **0.25 bits/elem**。

### 5.3 fp8 元数据模拟(`PCA_FP8SIM=1`)

`pca_quant.py:70-79` 的 `_fp8`:scale 和 mn 各自除以归一因子后过
`float8_e4m3fn` 往返。qvg4pt 传 `fp8_per_row=True` → 因子 =
`amax(dim=(-2,-1))`,对 [B,H,S,2,1] 的 scale 张量而言是**每 token 一个因子**
(scale、mn 各一个)。⚠ 记账口径见 §六。

### 5.4 对照:QVG 原装的三电平格长什么样

QVG 发布代码 `quant_videogen/sim/quant/lowbit_quantize.py:675-677`:

```python
def get_intx_max_value(num_bits: int) -> int:
    return (1 << (num_bits - 1)) - 1  # num_bits=2 → 1
```

随后 `_blockwise_intx_quantize_triton`(:680-727)做
`clamp(round(y), -1, +1)`——**码字只有 {-1, 0, +1} 三个**,对称 absmax,
每块只存一个 scale。2-bit 四个码字浪费一个,这就是其 46-49% 回收率天花板的
代码级来源(交叉实验证据见 `repro/0720/why/grid_cross.py` 与 why 报告 §一)。
pca_quant.py 里与之对齐的复刻是 `_ternary_quant_blocked`(:123-131),
pcatern 臂用的就是它。

## 六、记账:BPE 2.589 的逐项分解

LC:S=29640(token/chunk),D=128,K=256,块 G=64。全部按**实存逐字节**:

| 项 | 计算 | bits/elem |
|---|---|---|
| INT2 残差码 | 2(128 整除 64,无 padding) | 2.0000 |
| 残差 scale+zp(fp8,asym B64) | 2×8/64 | 0.2500 |
| 簇索引(uint8/token) | 8/128 | 0.0625 |
| 质心表(**fp32**,K×D 摊到 S×D) | 256×32/29640 | 0.2764 |
| **合计** | | **2.5889 ≈ 2.589** |

对照 QVG 原版(唯一差别是 s/z 项 0.25→0.125):2 + 0.125 + 0.0625 + 0.2764 =
**2.4639**,与 `repro/0721/grid_hash_screen.py:21` 的常数
`QVG_BPE = {"lc": 2.4639, "sf": 2.4063, "hy": 3.3199}` 一致。

⚠ **诚实脚注(本文首次明确)**:§5.3 的 fp8 归一因子在 qvg4pt 臂是每 token
两个(scale、mn 各一),**未计入 2.589**。若严格实存:int8 指数存法 +2×8/128
= +0.125 → 2.714;bf16 存法 +0.25 → 2.839。即 **2.589 是对 qvg4pt 有利的
下界**,严格记账它超预算更多(11% → 17-22%)。方向上只强化"四点位 QVG 装不进
合同"的结论,不影响任何已发布判决;qvg-nom(qvgprot)不受影响——它的 token
轴走 `fp8_per_row=False`,单个全局因子摊销 ≈ 0。

## 七、评分:32.29 怎么来的

- **生成**:auxq pod(`repro/0721/mpfull/pods/pod-auxq.yaml`,CAMPAIGN_NS=mp100)
  跑 jobs_aux_q4.txt,产出覆盖写入
  `results/multiprompt/mp100/lc/qvg4pt_rep0/p{1..10}/{P}-0/segment_1.mp4`;
  台账 `repro/0718/logs/ledger_zhizhousha-qvg-mpfull-auxq.txt:1-10`(全 OK)。
- **协议**(与 mp100 表同款,实现 `repro/0720/score_fp8.py`):
  参考 = 同 prompt 的 **bf16_rep0**(BF16-KV 续写,同 seed 同 init);
  取第 93 帧(首个生成帧);帧 /255;PSNR = 10·log10(1/mse);
  SSIM = paper metric.py 口径(11×11 avg_pool,C1=0.01²,C2=0.03²);
  LPIPS = paper 口径([0,1] 直喂 vgg,不归一)。
- **逐 prompt 终值**(0722 补落盘,`repro/0721/qvg4pt-fp8-score.json`):

| p | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 1 | 25.42 | 0.8597 | 0.0712 |
| 2 | 29.86 | 0.9559 | 0.0603 |
| 3 | 35.54 | 0.9855 | 0.0088 |
| 4 | 36.02 | 0.9636 | 0.0270 |
| 5 | 26.59 | 0.8645 | 0.0777 |
| 6 | 35.40 | 0.9767 | 0.0275 |
| 7 | 34.73 | 0.9553 | 0.0544 |
| 8 | 33.61 | 0.9754 | 0.0255 |
| 9 | 33.65 | 0.9689 | 0.0219 |
| 10 | 32.06 | 0.9722 | 0.0306 |
| **均值** | **32.29** | **0.9478** | **0.0405** |

- **一键复现**(单 GPU,~4h 生成 + 分钟级评分):

```bash
# 生成(10 个 job)
CAMPAIGN_NS=mp100 bash repro/0718/scripts/gpu_queue.sh repro/0721/mpfull/jobs_aux_q4.txt 1
# 评分(CPU 亦可)
.venv/bin/python repro/0721/score_qvg4pt_f93.py   # 落盘 repro/0721/qvg4pt-fp8-score.json
```

## 八、历史勘误与注意事项

1. **首轮 32.85 作废**:qvg4*/qvgpro* 分支首版漏设 `PCA_FP8SIM=1`,scale/zp
   以 fp32 参与模拟(等效 BPE 更高)。修复重跑后终值 32.29(−0.56)。记录见
   `repro/0720/why-refuted-and-errata.md` §七。
2. **campaign.sh:261 的 hijack 校验模式**(`*:pca*|*pca:*|*:rtn*|*:kivi*`)
   字面上不含 qvg4pt;auxq 台账仍记 hijack=1,但依赖的是日志 announce 行而非
   该 pattern——若复用此臂建议顺手把 pattern 补上 `*:qvg4*|*:qvgpro*`。
3. kmeans 用 `torch.randint` 随机初始化且 Triton centroid update 含 atomic
   语义 → **同 seed 不保证逐位复现**(与 QVG 原版同性质);我们方法无此问题。
4. 相关文档:机制归因 `repro/0720/why-budget-pca-wins.md` §一(grid_cross
   交叉矩阵);卖点与攻防 `repro/0721/sell-budget-pca.md`;证据阶梯
   `repro/0720/report-0720.md` §4.1。
