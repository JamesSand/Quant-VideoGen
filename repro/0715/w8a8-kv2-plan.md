# Plan：W8A8 模型量化 + 2-bit KV cache（0715）

目标：给视频 DiT 上 **W8A8**（权重+激活 8-bit，线性层走低比特 GEMM）并叠加 **2-bit KV cache**，
一次拿到三块收益：权重显存 ~2×、线性层计算加速（H100 FP8/INT8 tensor core）、KV 显存 ~7×。
主实验模型 Self-Forcing（最便宜），结论再推 LongCat/HY。

## 0. 为什么这个组合是合理的（基于我们已有的发现）

| 已有发现 | 对本计划的指导 |
|---|---|
| 视频激活随 timestep **模式级**漂移（V corr 0.33、中层 Q/K 0.64，0714 实测） | 激活量化必须用**运行时逐 token 动态 scale**，不依赖静态 per-channel smoothing（DeltaQuant 教训）；8-bit 的 256 级对这种漂移的容忍度远高于 4-bit——这正是选 W8A8 而不是 W4A4 的理由 |
| 量化难度深度不均（L29 K absmax 7×，H9 整头） | 需要先做**逐层敏感度扫描**，预留混合精度出口（个别层保 bf16） |
| KV 2-bit 已充分验证（QVG 28.88 / 我们的非对称 QuaRot B64 28.85） | KV 侧直接复用，不重新发明；两条路线都测（见 §2.3） |
| INT2 的 KV 已带来访存加速（LC −20%） | W8A8 的计算加速与之正交，端到端收益应叠加——用已建立的 paper 口径测速协议验证 |

## 1. 范围与格式决策

- **量化范围**：DiT block 内全部线性层——QKV 投影、attention 输出投影、FFN 两层、cross-attn
  的投影。**不量化**：LayerNorm/RMSNorm、adaLN 调制 MLP（首轮保守，敏感度扫描后再定）、
  patchify/unpatchify、text encoder、VAE。
- **数值格式**：主选 **FP8 E4M3**（权重 per-channel scale + 激活 per-token 动态 scale；
  H100 原生 `torch._scaled_mm`，工程路径最短）；对照 INT8（对称 per-channel W + 动态
  per-token A）。fake-quant 阶段两种都测，真 kernel 阶段只做胜者。
- **KV 2-bit**：QVG 发布路径（`triton-nstages-kmeans-int2`, B=64）为主；
  我们的非对称 QuaRot B=64（BPE 更低、确定性）为对照。

## 2. 分阶段计划

### Phase 0 —— fake-quant 可行性矩阵（1 天，~10 条 pod）

launcher 式 monkey-patch `nn.Linear`（quarot_launcher 同款手法，不改 repo 源码），
SF 单段短跑，frame-93 PSNR 口径：

| 臂 | W/A | KV | 目的 |
|---|---|---|---|
| A0 | bf16 | bf16 | 参考 |
| A1 | bf16 | int2 (QVG) | 已有（28.88），复用 |
| A2 | **W8A8-FP8** | bf16 | 单独看 W8A8 伤多少 |
| A3 | W8A8-INT8 | bf16 | 格式对照 |
| A4 | **W8A8-FP8** | **int2** | 组合——本计划的主角 |
| A5 | W8A8-FP8 | int2 (QuaRot asym B64) | KV 路线对照 |

**判定**：`PSNR(A4) ≥ PSNR(A1) − 0.5 dB` → 误差近似可叠加，直进 Phase 2；
掉 0.5~2 dB → Phase 1 修复；掉 >2 dB → 逐层归因。
（QVG 臂 n=3，W8A8 确定性臂 n=1。）

### Phase 1 —— 校准细化（按需，1 天）

- **逐层敏感度扫描**：一次只量化一个深度段（如 0-9/10-19/20-29），定位掉点层；
  重点怀疑对象：首末 block（0714 深度发现）、FFN 第二层。
- 修复手段优先级：①敏感层保 bf16（混合精度，成本最低）②激活 per-token → per-token-per-128ch
  细粒度 ③SmoothQuant λ（必须用 0714 的全步采集数据做**逐 timestep 验证**后才准上）。
- 顺手补一个数据缺口：抓 FFN 激活的 per-channel 分布（现有 qkv 钩子扩一行），
  确认激活侧离群与 timestep 漂移在 8-bit 网格下的余量。

### Phase 2 —— 真 kernel + 端到端收益（1-2 天）

- FP8 路径：权重离线转 FP8 + `torch._scaled_mm` 替换 Linear forward（激活 scale 运行时算）；
  不追求 fused epilogue，首版能跑就行。
- 用 paper 口径测速协议（同 GPU、日志内计时、热身后计量——0714 的教训全部适用）测四臂：
  bf16 / W8A8 / bf16+KVint2 / **W8A8+KVint2**，报端到端时间、峰值显存、线性层耗时占比
  （TIME_BENCH 分解）。
- 附加实验：**LC 长度极限重测**——权重省下的 ~13.6GB（LC 27GB→13.5GB）能换多少帧
  （0712 极限协议复用）。

### Phase 3 —— 报告（半天）

`report-0715.md`（present 版）：组合收益总表（显存/速度/质量三轴 × 四臂）、
误差可叠加性结论、与 SVDQuant/DeltaQuant 的定位对比（他们 W4A4 难而我们 W8A8+KV2
换取近无损）。

## 3. 评测协议

- 质量：frame-93 首帧 PSNR（既有参考视频/口径），QVG 臂 n=3；可选加测全视频 PSNR 曲线。
- 速度/显存：paper 口径端到端 + TIME_BENCH 分解 + `torch.cuda.max_memory_allocated`。
- 全部 pod 走既有基础设施（NODE_BUSY/黑名单/结果文件），REPRODUCE.md 记录每条命令。

## 4. 成功标准

1. 质量：W8A8+KVint2 相对 KVint2-only 掉 ≤0.5 dB（理想 ≤0.2）
2. 速度：线性层 ≥1.5×，端到端（叠加 KV 访存收益）SF ≥1.3×
3. 显存：权重 ~2× + KV ~7×，LC 峰值显存给出实测数字

## 5. 风险与对策

| 风险 | 对策 |
|---|---|
| 激活 timestep 漂移伤 8-bit | 动态 per-token scale 天然免疫幅度漂移；模式漂移在 256 级下余量大，Phase 1 有逐步验证兜底 |
| adaLN 调制敏感 | 首轮不量化；若量化则单独敏感度测试 |
| fake-quant 与真 kernel 有出入 | Phase 2 用 A4 同配置回测 PSNR，出入 >0.1 dB 则查 epilogue 精度 |
| `torch._scaled_mm` 对非 2^n 形状/小 batch 低效 | 先 profile 单层，必要时退 cutlass/TE；速度目标打折也先拿质量结论 |
| 组合误差非线性放大 | Phase 0 的 A2/A4 差分直接量化这一项，掉点则逐层归因 |

## 6. 待确认后启动

Phase 0 的 6 臂矩阵（~10 pod，半天出数）可以立即开跑。需要你拍板的：
①格式先测 FP8+INT8 双轨还是只 FP8；②KV 对照臂 A5 要不要（+1 pod）；③评测除 frame-93
外要不要加全视频 PSNR。
