# 0714 Handoff — 代码考古/BPE/速度三问 · B 扫描 · KV/QKV 分布解剖 · 三篇论文精读

> 演示用汇总：[report-0714.md](report-0714.md)（五节，self-contained，可直接 present）
> 日报正文：[details-0714.md](details-0714.md)；实验台账：[b-sweep.md](b-sweep.md)
> 复现指令：[REPRODUCE.md](REPRODUCE.md)；文档地图：[README.md](README.md)

## 今日完成（按时间序）

| # | 工作 | 关键结果 | 产物 |
|---|---|---|---|
| 1 | **三问调查**（代码考古 / BPE / 速度口径） | ①QVG pipeline 六阶段 file:line 地图，**全 repo 无 Hadamard**，"INT2"实为三元；②BPE 公式 `r + 8/B + S·8/128 + S·4096/N` 与 README 显存**逐字节吻合**（67.318 vs 67.32 MB/层），paper 6.94× vs 发布配置 6.88×（chunk 假设差 17%），附录 Table 5 漏记 scale 项；③paper 只报端到端开销百分比，无任何 kernel 级 benchmark | details-0714.md §1-3 |
| 2 | repro/ 目录重构 | `0713/ 0714/ backup/` 三层结构；脚本内路径全部改写；first_frames/limit_videos 归入 backup | repro/README.md |
| 3 | **papers/ 建立**，三篇论文入库精读 | 2602.02958（QVG 本体）、2605.26266（Jensen bias 校正）、**2605.19660 OScaR**（TNI 诊断，美团 LongCat 团队）、DeltaQuant CVPR26（timestep 动态性）——各附与本项目的关联 | papers/README.md |
| 4 | QVG vs QVG-Pro 参数全表 | dataclass 默认值就是 Pro（S=4/B=16）但**无任何官方 Pro 发布脚本**；iters 官方脚本内部差 50 倍（LC 100 vs SF/HY 2） | report §1.3 |
| 5 | B=64 对齐的 QuaRot 对照 | **非对称 QuaRot B64（BPE 2.25）与 QVG（2.326）统计打平**：28.85 vs 28.88±0.18；对称 B64 崩至 19.14 | report §2.5 |
| 6 | **B 扫描九宫格**（3 方法 × B∈{16,64,128}，QVG 格 n=3） | QVG 30.96/28.88/28.41、QuaRot 30.38/28.85/24.54、+clip(r0.99) 30.68/29.07/25.35；**QVG 粗块端韧性 9 倍于 QuaRot**（64→128：−0.47 vs −4.31）；**QVG-Pro 优势解构：S=4 仅 +0.08 dB，B=16 贡献 +2.08**；clip 增益随 B 增大（+0.30/+0.22/+0.81） | b-sweep.md、report §2.6 |
| 7 | **发布 kernel bug 修复** | `quant_pack` autotune 含 `BLOCK_D<Q_BLOCK_SIZE` 非法配置 → `tl.arange(0,0)` 崩溃，B=128 在官方真打包路径根本跑不了；修复 = autotune 剪枝（提交 8b81883） | quant_videogen/real/quant_pack.py |
| 8 | **paper 口径三模型测速**（同 GPU 顺序跑，日志内生成计时） | paper 声称"慢 1.5~4.3%"，实测**三模型全负开销**：LC **−19.9%**、SF **−5.0%**、HY **−30.2%**（稳态 chunk 口径；墙钟 −68% 是假象——bf16 臂独付 336s 冷 PVC 加载）；SF"43s"之谜 = 发布脚本实际 717 帧（4 倍工作量） | report §3.5、summary §3 |
| 9 | **KV 分布 3D 曲面图**（三模型，OScaR Fig.2 风格） | K 全有通道墙（SF 5.4× / HY 12.8× / LC 9.0×）、V 全平坦——视频 DiT 与 LLM 同构；HY 缓存实为 [rotary 分支‖**PRoPE 相机投影分支**] 256 维拼接（原生双分支，QVG 只做打包）；LC 单 chunk=29,640 token 与 BPE 对账互证 | kv-distributions.md、figs/kv3d_*.png |
| 10 | **Token norm 统计**（OScaR Fig.3 口径，三模型） | **视频 K 侧无 TNI**：极值比 LC 1.03× / HY 1.27× / SF 1.41×，无低 norm sink token（机制：文本/sink 载体全在 cross-attn，self-attn cache 纯视频 patch）；norm 参差在 V 侧（3.5-5.6×）但 per-token 轴免疫 | kv-distributions.md、figs/token_norms_*.png |
| 11 | **QKV 三维度解剖**（SF：时间窗 × chunk 内部 × 层深） | ①视频首尾分布**无漂移**（<5%）；②chunk 内无 sink；③**L29 的 H9 整头离群**（norm≈105 vs 其他头 15，absmax 93.5 vs 13.1）——归因：**W_k 能量集中 16.6×（ch95）× g 增益 5.4 对齐相乘**（corr 0.83），本质是两根巨型通道（ch95/ch49） | qkv-anatomy.md、qk-norm.md 附录 |
| 12 | QK-Norm 机制文档 + g 可视化 | g = 训练学出的共享参数（非逐 token）；**Wan 整维归一 vs Qwen3 逐头归一**——前者允许整头倾斜（H9 的存在条件）；勘误：QK-Norm 挡不住 TNI（Qwen3 有 QK-Norm 仍被 OScaR 观察到 sink）；vendor 了 modeling_qwen3.py 作证据 | qk-norm.md、figs/qk_norm_*.png、reference/ |
| 13 | **timestep 轴动态实测**（同一 block 全部去噪 forward） | **V 模式级剧变 corr≈0.33**、中层 Q/K 0.64-0.66、**L29 巨型通道跨步静态 0.95-0.99**；SF cache 存的是去噪后干净重编码的 K/V → 步间漂移不进 cache；补齐 image-vs-video 辨析（幅度级 vs 模式级，DeltaQuant 未做的对照） | summary §5、figs/qkv_timestep_dynamics.png |
| 14 | report-0714.md 演示文档 | 五节 self-contained：BPE 表（QVG/Pro/QuaRot 对称+非对称）、B 扫描、速度、TNI 检验、DeltaQuant/timestep | report-0714.md |
| 15 | 双远端 | 新增 remote `together` → **github.com/togethercomputer/quant-video-gen**，全史已推 | git remote |

## 关键数字速查

- 九宫格（INT2 frame-93）：QVG **30.96/28.88/28.41**（B=16/64/128）｜QuaRot 非对称 30.38/28.85/**24.54**｜+clip 30.68/**29.07**/25.35（B64 处超 QVG）
- BPE：QVG **2.326**(6.88×)｜Pro 3.30(4.85×)｜QuaRot 非对称 B64 **2.25**(7.11×)｜对称 B64 2.125(7.53×)
- 速度（paper 口径）：LC **−19.9%**｜SF −5.0%｜HY **−30.2%**（稳态）
- L29 H9：整头 norm 104.8 vs 其他头 ~15；absmax 93.5 vs 中层 13.1；ch95 的 K rms 89.9 = x̂ 16.6 × g 5.4
- timestep 漂移：V corr 0.33｜中层 Q/K 0.64｜L29 巨型通道 0.97±

## 悬而未决 / 需要警惕

1. **一切主结论仍是单 prompt（滑板手）/seed 0**——多 prompt 复验没做，尤其"asym B64 打平 QVG"这种要对外讲的结论。
2. **LC/HY 的 QKV 解剖未做**（采集器已通用化）；H9 式整头离群是否只在 Wan 系整维 QK-Norm 下出现——Qwen3 式逐头归一的模型可作反证测试。
3. **Proxy 排行榜**部分解锁：`results/kvplot/sf_qkv.pt` 已含 Q（3 层 × 3 窗，末去噪步）+ `sf_qkv_steps.pt` 含全步——注意力输出 MSE/注意力图 KL 两个 proxy 已可算；全量仍需更多层。
4. RoPE distribution 直观图（用户暂缓的待办）：`ropestudy/step3_visualize_before_after.py` 已写好未跑。
5. trit 打包（+16-22% 免费压缩）仍是纯工程待办。
6. `backup/ISSUE_DRAFT.md` 待更新后发送（新增候选：对称 QuaRot 指纹、6.94 vs 6.88 对账、Table 5 笔误、B=128 kernel bug、asym-B64 打平）——**发出前须用户点头**。
7. HY 测速的墙钟教训已成规范：**pod 内多臂顺序跑，第一臂吃冷缓存——对比必须用日志内生成计时**。

## 0715 候选方向

1. **多 prompt 稳健性批**（九宫格关键格 + asym B64 对照 × 3 prompt，纯算力）。
2. **Proxy 排行榜启动**（sf_qkv.pt 的 Q 已在手，先算注意力输出 MSE / 注意力图 KL 与 60+ 已标注配置的秩相关）。
3. **按深度重分配量化预算**实验（L29 K 细块/高位宽、浅层更狠——qkv-anatomy 推论 #1，直接可跑）。
4. LC/HY 的 QKV 解剖补齐（各一条 pod）。
5. 作者 issue 更新与发送（等确认）。
