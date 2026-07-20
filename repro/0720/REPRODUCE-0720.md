# REPRODUCE 0720 — Budget-PCA 全链路指令级复现指南

> 目标读者:没跟过本项目的新人。跟着本文从零环境走到三个交付物:
> ① **MP100 定案表**([mp100-table.md](mp100-table.md),vs QVG 11胜7平0负);
> ② **kernel 三重门**(BPE 逐字节合规 + encode 快过 kmeans + 数学等价);
> ③ **why 判决报告**([why-budget-pca-wins.md](why-budget-pca-wins.md))。
> 每一步都给命令和预期数字核对点;对不上先查 §9 已知坑。

## 0. 一页地图:什么结论在哪个文件、由哪个脚本产生

| 交付物 | 文档 | 生成脚本 | 预期核心数字 |
|---|---|---|---|
| 终表(质量) | [mp100-table.md](mp100-table.md) | `repro/0718/scripts/{campaign,gpu_queue}.sh` → `repro/0720/score_fp8.py` → `aggregate_fp8.py` | LC 31.68/0.9370/0.0547;SF 四维全第一;HY 18.77 |
| BPE 审计 | [bpe-audit.md](bpe-audit.md) | `repro/0720/kernel/bpe_audit.py` | LC 2.3192 / SF 2.3183 / HY cache 2.3250(全 ≤2.326) |
| 速度对决 | [kernel-results.md](kernel-results.md) | `repro/0720/kernel/bench_speed.py` | encode LC 32.5× / HY 1.4× / SF 1.1× 快过 kmeans |
| why 判决 | [why-budget-pca-wins.md](why-budget-pca-wins.md)(成立假说)+ [why-refuted-and-errata.md](why-refuted-and-errata.md)(负结果台账) | `repro/0720/why/*.py`(§7) | 格效率差 ~2.2×;H1 原命题证伪 24/24;H4 平坦 |
| 方法讲解 | [method-explainer.md](method-explainer.md) | — | (读物,行号可点击) |
| paper 差异盘点 | [paper-diff-plan.md](paper-diff-plan.md) | `repro/0720/{score_e1,vbench_official}.py` | 四类差异全部定位到 paper 侧 |

## 1. 环境

```bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen   # repo 根(QVG paper fork)
. .venv/bin/activate                                          # python 3.12 venv(已建好)
source repro/backup/scripts/env_fix.sh                        # 必须:TRITON_PTXAS 指向 CUDA-13 ptxas 等
```

- 硬件:8×H100 dev pod(单卡也能跑,时间×8);模型权重在 `ckpts/`
  (LongCat-Video-13B / Self-Forcing-Wan / HY-WorldPlay-8B,fork 原始 README 的下载方式)。
- **每个 shell 都要 source env_fix.sh**,忘了会撞 triton/ptxas 版本错。
- 作图需中文字体:`~/.local/share/fonts/NotoSansSC.ttf`(缺了 `make_figs.py` 会豆腐块)。
- 需要更多卡:用 k8s 自开 pod(kubeconfig 在 `low-precision-project/k8s-from-h100-pod/`,
  **凭证,永不入 git**);现成清单 `repro/0718/scripts/pod-mp100-w{1,2,3}.yaml`、
  `repro/0720/kernel/pod-{audit,bench}.yaml`。

## 2. 素材(prompt 集与产物目录)

- **100 条随机 MovieGen prompt(seed=42)已固化**,直接用,不要重抽:
  - `repro/0718/prompts100/selected.txt` — 100 行,LC/评分用(第 i 行 = prompt i,1-based);
  - `repro/0718/prompts100/p{1..100}.txt` — 单行文件,SF 用;
  - `repro/0718/prompts100/source_lines.txt` — 它们在源文件
    `repro/0717/MovieGenVideoBench_extended.txt` 中的行号(留痕,重抽=换考卷,禁止)。
- HY 不用 prompt,用固定场景 × seed 0-9。
- 产物目录:`results/multiprompt/mp100/{lc,sf,hy}/...`(由 `CAMPAIGN_NS=mp100` 决定);
  日志与台账 `repro/0718/logs/`;评分缓存 `repro/0718/npz/`。

## 3. 终版配置速查(方法本体只是一组环境变量)

引擎 `repro/backup/scripts/pca_quant.py` + 注入器 `pca_launcher.py`(劫持
`compress_kv_cache`,与 QVG/baseline 同钩子同时机;详见 method-explainer §二)。

| 模型 | 终版臂名 | 关键 env | BPE(审计值) |
|---|---|---|---|
| LC | `pcakaxvaxfp8` | `PCA_R=4 PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_V_MODE=pca PCA_RES_AXIS_K=channel PCA_RES_AXIS_V=channel PCA_FP8SIM=1` | 2.3192 |
| SF | `pcaa128kaxvaxfp8` | 同 LC + `PCA_SF_STORE_FIX=1` | 2.3183 |
| HY | `pcav90kpternkaxkb64fp8` | LC 基础 + `PCA_HALF_R_K=9,0 PCA_HALF_R_V=9,0 PCA_RES_BLOCK_K=64 PCA_RES_GRID_KP=ternary PCA_RES_BLOCK_KP=64` | 2.3250(cache 级) |

臂名后缀 → env 的映射逻辑在 `campaign.sh::apply_variant`(子串匹配,新增变体注意
子串碰撞,见 §9)。baseline 臂:`rtnfp8` / `kivifp8` / `quarot` / `qvg`
(QVG = 原装 triton-nstages-kmeans-int2,K=256,LC iters=100、SF/HY iters=2,
见 campaign.sh 顶部 QVG_LC/QVG_SFHY)。

## 4. 生成(≈2000 个视频,8 卡一到两天)

任务语法(`campaign.sh <job>`,一行一个任务):

```
lc_base:<P>                    # LC 条件窗基底(每个 prompt 先要有它)
lc:<P>:<arm>:<rep>             # LC 续写;P=1..100;rep 固定 0
sf:<P>:<arm>:<rep>[:latents]   # SF;默认 700 帧窗(180 latents)
hy:<S>:<arm>                   # HY;S=seed 0..9
```

执行(**永远只开一个队列实例**,见 §9):

```bash
export CAMPAIGN_NS=mp100 \
       CAMPAIGN_PROMPTS=repro/0718/prompts100/selected.txt \
       CAMPAIGN_SF_DIR=repro/0718/prompts100
# jobs 文件一行一任务。先 LC 基底:
bash repro/0718/scripts/gpu_queue.sh repro/0718/jobs100_bases.txt 8
# 再六臂全量(自己写 jobs 文件,臂名以 mp100-table 各表的"覆盖数"行为准):
#   LC/SF: bf16 / rtnfp8 / kivifp8 / quarot / qvg / pca 终版臂(§3 表)
#   HY:    bf16 / rtn / kivi / quarot / qvg / pcav90kpternkaxkb64fp8
# 历史参考:jobs100_{bases,main}.txt = 初版战役;repro/0720/jobs_m1regen.txt =
# 记账修正后的诚实重赛(fp8 终版臂 210 行)。注意 jobs100_main.txt 里的旧臂名
# (rtn/kivi 无 fp8 后缀)是修正前口径,LC/SF 勿照抄。
bash repro/0718/scripts/gpu_queue.sh <你的 jobs.txt> 8
```

**每个 job 完成后核对台账** `repro/0718/logs/ledger_$(hostname -s).txt`:
`OK` 才算数;`NOHIJACK` = pca 臂没劫持到量化调用,**该 run 作废必须重跑**
(hijack 计数是"我们的量化真的生效了"的唯一证据)。

## 5. 评分 → 终表

```bash
# 终版入口(0720 记账修正口径)= score_fp8.py + aggregate_fp8.py:
for i in $(seq 0 7); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python repro/0720/score_fp8.py $i 8 \
    > repro/0718/logs/score_shard$i.log 2>&1 &
done; wait
.venv/bin/python repro/0720/aggregate_fp8.py    # 重写 repro/0718/mp100-table.md
cp repro/0718/mp100-table.md repro/0720/mp100-table.md   # 0720 副本同步
# (repro/0718/scripts/score100_all.sh 是初版战役的同构入口,协议相同)
```

协议(全部硬编码在 score 脚本,改动=换口径,先对齐再动):

- **PSNR/SSIM/LPIPS 都是 vs 同 seed BF16 参考**:LC = f93 单帧;HY = frames[13:]
  均值;SF 无条件前缀(onset=帧1)近无损区无判别力 → **不进三指标矩阵**,只走
  VBench(记忆:SF no-prefix eval exclusion);
- SSIM = paper 的 `metric.py` 口径(11×11 avg_pool);LPIPS = paper 口径
  (**[0,1] 直喂 vgg 不归一**,绝对值勿与其他论文横比);
- VBench 四维 = CLIP-B/32(BC) / MUSIQ(IQ) / DINO-B/16(SC) / CLIP-L/14+LAION(AQ),
  实现 `repro/0718/scripts/vbench4.py`(与官方包 BC/IQ/SC ±0.4 一致,AQ 系统性
  低 ~4.7 → 只做表内横比,勿与 paper 绝对值比,见 §10);
- 胜负判定 = 逐列**配对双侧符号检验**(aggregate_fp8.py 内置,p<0.05 才叫显著)。

预期落点(±噪声):LC 31.68/0.9370/0.0547;HY 18.77/0.5016/0.3286 + SC 94.68;
SF 93.21/66.65/88.69/53.27;vs QVG = 11 显著胜 + 7 平 + 0 负。

## 6. Kernel 三重门(真实现的三项验收)

真 kernel:`repro/0720/kernel/bp_quant.py`(encode,whole-graph torch.compile)+
`bp_triton.py`(Triton fused decode)。

```bash
# 门 1:BPE 逐字节审计(数 bytes,不是公式)——预期 LC 2.3192/SF 2.3183/HY 2.3250
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/kernel/bpe_audit.py
# 门 2+3:速度对决(vs QVG triton kmeans,同输入 CUDA-event 计时)+ fake↔kernel 等价
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/kernel/bench_speed.py
```

预期:encode LC 32.5×/HY 1.4×/SF 1.1× 快;combined LC 27.3×/HY 1.16×/SF ~平
(0.95-1.10 波动);fake↔kernel relL2 差 ≤1.1%;Triton decode 与 reference 逐位一致。
结果落 `repro/0720/kernel/bench_report.json`,汇总解读在 [kernel-results.md](kernel-results.md)。

## 7. Why 判决(机制分析,需要 §2 的 dump chunks)

素材 = 三管线真实 dump chunk(`repro/0720/chunks/{lc,sf,hy}/chunk_00{0..7}.pt`,
每模型 8 个 = 前 8 次量化事件)。缺了就用带 `dump` 后缀的臂重采:
`lc:1:pcakaxvaxfp8dump:0`(SF/HY 同理,PCA_DUMP_DIR 自动指向 chunks/$KIND)。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_h2_compute.py    # 谱+逐通道误差
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_real_path.py    # kmeans 真口径(8 chunk)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h1_ours_path.py    # 我方同协议(8 chunk)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/h2_multichunk.py   # H2 跨层
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/make_figs.py       # fig1-5
CUDA_VISIBLE_DEVICES=0 .venv/bin/python repro/0720/why/fig6_hy_halves.py  # fig6
```

预期核对点(kmeans 有随机性 ±1pp):QVG 残差回收率 46-49%(LC/SF 16 测点全部)
vs 我们 61-82%;chunk 级误差我们 20/24 格更小;H2 LC 1.8-2.4×(8/8)。
H4(iters 扫描)要真生成:campaign 臂 `qvgi2`/`qvgi10` × 10 prompts,预期
27.13/27.72/27.61 vs 我们 31.68。
**注意**:kmeans 侧一律用 QVG 原装实现、**per-head 全 D 维 token 聚类**口径
(centroids (B,H,K,D);`block_size=64` 只管残差格)——旧的全局 64 维块口径是
已勘误的错误(`h1_kmeans_sub.py` 仅留痕,勿用)。

## 8. 逐字节记账规则(改配置前必读)

**记账对称原则:任何一边都不允许有未计费的精度/比特。** 具体:

- 残差 2-bit + 每块 fp8 scale/zp;块长**必须整除或补零**——"找不到整除块长就
  缩小到 g=8"的 fallback 会让元数据翻倍还测不出来(0720 最大教训,LC 曾因此
  虚标 +3.5dB,KIVI baseline 同样中招);
- fp8 走 E4M3 就要处理饱和(>448)与跨通道 2^18 动态范围:channel 轴用
  **每通道 bf16 归一因子**,token 轴用全局因子(bp_quant.py 已实现,别退化);
- 系数打包补齐到 4 的倍数(HY r=9 曾漏,BPE 2.51→2.325);
- 每秩 +2bit/token = +0.0156 BPE;r4 asym B128 = 2.3125 帐面,审计值含全部
  元数据零头。变更任何格子/块长/轴,先跑门 1 审计再跑质量。

## 9. 已知坑(每条都付过学费)

1. **单队列纪律**:gpu_queue 显存预检是"检查后启动"(TOCTOU),两个并发实例
   会双订同一张卡 → OOM 连环(一次损失 60 job)。多批任务合并成一个 jobs 文件;
2. **kill 前独立读证据**:先 `nvidia-smi --query-compute-apps` + `ps` 完整命令行
   (只读),确认 pid 归属后再单独 kill——按 pid 邻近猜测曾误杀健康任务;
3. **SF 帧数是 latent 单位**且 %3==0(180=700 帧窗);SF 存储 BSHD/BHSD 不一致,
   我们臂须 `PCA_SF_STORE_FIX=1`,QuaRot 臂须 `QUAROT_SF_STORE_FIX=1`;
4. triton cache:按 job 复制会爆配额;campaign.sh 已改 pod-local per-GPU
   (`/tmp/qvg_triton/gpu$ID`),别改回来;
5. 臂名子串碰撞:`apply_variant` 是子串匹配(`*kptern*` 会吃掉 `kptern128`
   之类),新变体先 grep 现有 case;两臂分数逐位相同 = 十有八九撞名;
6. LC 续写前必须有同 prompt 的 `lc_base`(campaign.sh 有 base-wait 守卫,
   但 base 失败会让下游全卡住,先跑通 bases 再发 main);
7. pca 臂 `hijack==0` 即作废(见 §4);QVG 臂的不确定性来自 kmeans atomic_add
   而非 seed,重复跑直接同 seed 重复即可;
8. HY 快照:`results/.../hy/` 下 mp4 很大,**视频/npz 不入 git**(用户点名要的
   除外);kubeconfig 永不入 git。

## 10. 与 paper 原文横比的口径注意(引用终表时带上)

- **paper 的 LC baseline 低 ~10dB 是它的实现弱**(三点位对称格 + post-RoPE,
  E1 复现至 ±1-3dB);我们的 RTN/KIVI 是诚实强实现,表=更严格考场;
- **HY 比 paper 低 7-12dB 是 PSNR 窗口口径**(paper≈发散前 ~24 帧窗,我们全程均值);
- **AQ 绝对值不可与 paper 比**(打分器 −4.7 + prompt 子集 −2.5;表内同尺同刻,
  方法间结论不受影响);BC/IQ/SC 与官方 VBench ±0.4 一致(IQ 逐位一致,
  验证脚本 `repro/0720/vbench_official.py`,原始结果 `repro/0720/e2b/`);
- 全部证据链:[paper-diff-plan.md](paper-diff-plan.md)。

---
*配套阅读顺序建议:method-explainer(方法)→ 本文(复现)→ mp100-table(结果)
→ why-budget-pca-wins(机制)→ paper-diff-plan(与原文横比)。*
