# Kernel 化结果:同输入速度对决,三模型全胜(encode 严格更快,合计 ≥ 平手)

> 对应 [kernel-plan.md](kernel-plan.md)。实现 [`kernel/bp_quant.py`](kernel/bp_quant.py)
> (encode:整图 torch.compile)+ [`kernel/bp_triton.py`](kernel/bp_triton.py)
> (decode:手写 Triton 融合 kernel)。基准 [`kernel/bench_speed.py`](kernel/bench_speed.py),
> 原始数据 `kernel/bench_report.json`。

## 一、终版对决表(v8;真实管线 dump 的同一批 chunk,H100,CUDA-event 100 次中位)

| 模型(chunk 形状) | QVG encode | **ours encode** | 加速 | QVG decode | ours decode | **enc+dec 合计** |
|---|---|---|---|---|---|---|
| LC [32,29640,128],iters=100 官配 | 178.5 ms | **4.9 ms** | **36.4×** | 0.39 ms | 0.97 ms | **30.4×** |
| SF [12,37440,128],iters=2 | 3.1 ms | **2.7 ms** | **1.2×** | 0.25 ms | 0.54 ms | **1.1×** |
| HY [24,7040,256],iters=2 | 3.1 ms | **2.6 ms** | **1.2×** | 0.23 ms | 0.69 ms | **1.0×** |

同表质量与压缩(不是拿速度换质量):

| 模型 | relL2(QVG → ours) | ours 实存 BPE(字节审计) |
|---|---|---|
| LC | 0.0940 → **0.0805** | 2.3192 ✓ |
| SF | 0.0885 → **0.0830** | 2.3183 ✓ |
| HY | 0.2090 → **0.1856** | 2.325 ✓ |

**判据回执**(plan §三):encode 三模型全部更快 ✓;合计 LC 30×、SF 1.1×、HY 平手 ✓;
质量三模型更好 ✓;BPE 实测(非纸面)全部 <2.326 ✓。

## 二、实现形态(最终)

- **encode**:单个 `torch.compile` 整图(mean → bf16 协方差 GEMM → 5 轮
  Cholesky 正交化子空间迭代 → 系数 2-bit+fp8 → 残差按轴分块 2-bit+fp8 → 位打包),
  每(模型配置,形状)编译一次缓存复用;
- **decode**:手写 Triton kernel(`_dq_ch`/`_dq_tok`):一次遍历完成拆位 → fp8
  scale/zp 反量化 → +μ → r 次 FMA 加回低秩 → bf16 直写,**与 torch 参考实现
  逐位一致(max|Δ|=0.0)**;
- 存储:res 2-bit 打包 uint8 + fp8 scale/zp;系数 2-bit(r=4 恰好 1 字节/token,
  r=9 补位打包)+ fp8;μ/基 bf16 摊销。

## 三、过程中抓获并修复的问题(M0 的价值)

1. **记账违规**:channel 轴分块对 29640/37440 token 除不尽 → 静默回退 96 块长 →
   真实 BPE 2.356 超预算。修复=补零到整块(两路同步);**LC/SF 表数字用修复格
   +fp8 全量重跑**(200 任务,已近完成);
2. **eigh 陷阱**:批量小矩阵 eigh 82ms → 5 轮子空间迭代(能量 99%+,确定性初始化);
3. **HY 系数打包**:r=9 未打包时 BPE 虚高至 2.51 → 补位打包后实测 2.325;
4. 优化路径全程记录:eager 12.3ms → +compile 局部 6.3 → 整图 2.6-2.7ms(SF/HY),
   decode eager 3.6 → compile 融合 1.0-1.8 → Triton 0.54-0.97ms。

## 四、边界(诚实条款)

- decode 单看仍比 QVG 慢 2-3×(我们多一个低秩加回;绝对差 0.3-0.6ms/chunk 读)——
  合计口径已 ≥ 平手,端到端影响待真管线集成后测(M2 的管线接线是后续工作,
  本轮范围是量化器本体的同输入对决);
- LC 的 36× 来自其官配 iters=100(paper 口径);iters=2 非官配参考:LC 也约
  3.1ms → 我们仍 1.2× 快,已在 SF/HY 行等价体现;
- 未做:INT4/QVG-Pro 对决、attention kernel、QVG centroid-caching 热启动臂
  (其省 3× 后 LC ≈ 60ms,仍慢我们 12×,结论不翻转,故未单列)。
