# Kernel 化 Plan:通道轴 Budget-PCA 真实现 + 与 QVG k-means 的同输入速度对决

> 目标(用户 0720 定):把 Budget-PCA 用 kernel 实现;**对同样输入的 KV cache,
> 对比我们与 QVG(triton-nstages-kmeans)的量化速度,要求我们更快**(结构性
> 优势:无 k-means 迭代)。继承 [../0716/n4-int2-impl-plan.md](../0716/n4-int2-impl-plan.md)
> 的 M0-M4 框架,按通道轴终版配置更新。

## 一、为什么我们理应更快(先把账算清,再动手)

同一次量化事件(一个 chunk,S 个 token × H 头 × D 维),两边的计算构成:

| | QVG(kmeans-int2,K=256 质心) | Budget-PCA(通道轴版) |
|---|---|---|
| 迭代? | **k-means++ 初始化 + iters 轮**(LC 官配 **iters=100**,SF/HY=2);每轮 assignment 是 [S,K] 距离阵 + argmin + atomic 质心更新 | **零迭代** |
| 主要 FLOPs | iters × (S·K·B 距离计算) | 3 个 GEMM:协方差 S·D²、投影 S·D·r、重建 S·r·D(r=4,后两项可忽略)+ 一次 eigh(D³=128³,批 H 个,微不足道) |
| 逐元素 | 残差 PRQ 量化 + 打包 | 残差 blockwise 量化 + 打包(同量级) |
| 粗账(LC,S=37440,D=128,K=256,B=64) | ~100 × 37440×256×64 ≈ **6.1 TFLOP** | 37440×128² ≈ **0.6 GFLOP** + eigh ≈0 |

理论差 3-4 个数量级(LC);SF/HY(iters=2)也有 ~40× 的 FLOPs 差。QVG 自己的
paper 承认 k-means 是延迟大头(其 §4.3 为此做了 centroid caching,称省 3×)。
**我们的风险不在 FLOPs,在实现质量**:launch 开销、转置访存、eigh 的 CPU 回退。
所以 plan 的核心是"少写 kernel,多复用":QVG 的 int2 打包存储、fp8 scale 布局、
分页 cache(`quant_videogen/real/`、`kernels/`)全部照用,只替换"怎么算出这些位"。

## 二、要实现什么(存储格式与 kernel 清单)

### 存储(每头每 chunk;沿用 QVG 的 packed uint8 基建)

| 组件 | 格式 | 体积(bits/elem 摊) |
|---|---|---|
| 残差 2-bit 码 | uint8 打包,4 值/字节;**K 按通道轴分块**(块=64/128 token),V 同(LC/SF)或 token 轴(HY V) | 2.0 |
| 残差 scale/zp | fp8 E4M3,每块一对 | 0.125-0.25 |
| 系数 2-bit 码 | uint8 打包,每 token r=4 个 | 0.0625 |
| 系数 scale/zp | fp8,每 token 一对 | 0.125 |
| μ、V_r 基 | bf16,每 chunk 一份(D + D×r) | 摊销 ≈0 |

与 fake-quant 记账一致:LC/SF 2.3125、HY 2.29——**先有账后有码,不许实现时偷精度**。

### Kernel 清单(能用 torch/cuBLAS 的绝不手写)

1. **encode 统计段(torch 现成)**:`mean` → 协方差 GEMM(`baddbmm`)→
   批量 `torch.linalg.eigh`(cuSOLVER,[B·H,128,128],确认走 GPU 不回 CPU)
   → 投影/重建 GEMM;
2. **K1:系数量化打包 kernel(Triton)**:每 token r 个系数 → minmax → fp8
   scale/zp → 2-bit 码打包。逐元素,一小时级工作量;
3. **K2:残差通道轴量化打包 kernel(Triton,本 plan 唯一有含量的 kernel)**:
   输入 [S,D] 残差,**按通道读、沿 token 分块**求 minmax→量化→打包。访存是
   转置模式——两个方案 M1 阶段对测选优:
   (a) 先显式 `transpose().contiguous()` 再复用 QVG 的 lastdim 量化 kernel(零新
   kernel,多一次拷贝);(b) 直写 strided Triton kernel(省拷贝,块内 token 连续
   读天然 coalesced——通道轴其实**更适合** [S,D] 行主序:同一块的 64 个 token
   在同一列,跨行 stride=D 固定);
4. **K3:融合 decode kernel(Triton)**:拆位 → 残差 dequant → + q(c)·V_rᵀ(r=4
   的窄 GEMM 在 kernel 内做寄存器级累加)→ +μ → 写出 bf16。attention 前一次
   完成,替换 QVG 的 `triton_prq_dequantize_tensor` 调用点;
5. **HY 特例**:256 维打包按半区各跑一遍(秩 9/0、KP 三值格 = K2 的 ternary
   分支),打包布局不变。

## 三、速度对决协议(同输入、同 GPU、同计时法)

- **输入**:从三条管线各 dump 真实待量化 chunk(LC 条件窗 37440 tok、HY 7040
  tok×256 维、SF 37440 tok),存 npz——**双方吃同一份张量**;
- **对手**:`triton-nstages-kmeans-int2` 原装路径(LC iters=100 / SF·HY iters=2,
  官配不动;另报 QVG 的 centroid-caching 热启动臂,不给它吃亏的机会);
- **计时**:CUDA events + `torch.cuda.synchronize`,预热 10 次、计 100 次中位数
  (协议沿用 [../0716/kernel-speed.md](../0716/kernel-speed.md) 同 chunk 口径);
  encode / decode / 端到端逐视频三个表分开报;
- **同时报质量**:kernel 版对 fake-quant 版逐张量 max|Δ|(应为 fp8/打包舍入级),
  并抽 10 prompt 重跑 f93/VBench 确认不掉点(M3 验收门);
- **成功判据**:三个模型的 encode 中位延迟均 < QVG;端到端生成 walltime ≤ QVG;
  显存占用打出 2-bit 真压缩(顺带解锁 SF>777 帧,回收 Q3)。

## 四、里程碑(总 ~4-5 天)

- **M0(0.5 天)数值预检**:fake 路径加 fp8-scale/打包舍入模拟臂,确认 2.3125
  账本下质量不掉(0716 M0 原样,加通道轴);跑通批量 eigh 的 GPU 路径微基准;
- **M1(1.5 天)encode+存储**:K1/K2 + QVG packed cache 对接,LC 先行(单事件最
  简单),对拍 fake-quant;通道轴两方案(转置拷贝 vs strided kernel)对测定版;
- **M2(1 天)融合 decode**:K3 + 三管线读取点替换;SF store-fix 在真路径下的
  等价物(BSHD 布局直接按正确轴打包,顺带消灭那个上游 bug);
- **M3(1 天)速度对决 + 质量回归**:§三协议全表,写 `kernel-results.md`;
  含 SF 长度极限重测(真 2-bit 显存 → 1400 帧解锁?);
- **M4(0.5 天)HY 半区特例收尾** + 文档/REPRODUCE;
- (可选 M5)流式 encode 基复用——对齐 QVG centroid caching 的工程对等物,
  只在 M3 发现 encode 意外落后时才启动。

## 五、风险与对策

| 风险 | 对策 |
|---|---|
| `torch.linalg.eigh` 批量小矩阵在某些版本走 CPU/慢核 | M0 先微基准;备选:协方差幂迭代取 top-4(r 固定且小,10 次幂迭代 ≪ eigh)或 `torch.lobpcg` |
| 通道轴打包的转置访存拖慢 K2 | 两方案对测(§二.3);实在不行 K 残差落盘就存转置布局,decode 侧一并转回 |
| fp8 scale 精度掉点 | M0 预检门,掉点则 scale 升 fp16(BPE +0.06,仍 <2.326,记账同步改) |
| QVG 的 nstages/clip 变体计时口径争议 | 全部变体各报一行,配置原文引用 compress.py#L107 |
| LC iters=100 对比"太容易赢"被质疑 | 加 iters=2 的 LC 非官配臂作参考行,注明官配出处 |

## 六、不做什么

- 不写 attention kernel(量化/解量化边界为止,attention 用各管线原装);
- 不做 INT4/QVG-Pro 对决(另案);
- 不为速度牺牲任何已定案的质量配置——配置冻结,kernel 只是等价实现。
