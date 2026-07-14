# 生成长度极限实测视频（80GB H100，单卡）

每条 run 从提议尺寸开始，OOM 自动降档，**能完整跑通的最大尺寸即实测极限**。
bf16 = 不量化 KV-cache；qvg-int2 = 论文主方法（triton-nstages-kmeans-int2, S=1/B=64/K=256）。

| 文件 | 模型 | 精度 | 长度 | 实测极限的含义 | 峰值显存 |
|---|---|---|---|---|---:|
| `longcat_bf16_1493frames_99.5s.mp4` | LongCat-Video | bf16 | 1493 帧 / 99.5s | 滑窗设计显存恒定，**无显存上限**（70 段为本次设定，超 paper 附录的 1400 帧） | 59.5 GB |
| `longcat_qvg-int2_1493frames_99.5s.mp4` | LongCat-Video | INT2 | 同上 | 同上（INT2 更快：KV 载入流量 ~7× 更小） | 41.0 GB |
| `selfforcing_bf16_777frames_48.6s.mp4` | Self-Forcing | bf16 | 777 帧 / 48.6s | **bf16 显存极限**（195 latent；228/210 OOM） | 66.7 GB |
| `selfforcing_qvg-int2_2397frames_149.8s.mp4` | Self-Forcing | INT2 | **2397 帧 / 2 分 30 秒** | INT2 极限（600 latent；720 OOM）——**解锁 3.1×**；bf16 同长需 ~154GB | 37.0 GB |
| `hyworldplay_bf16_317frames_19.8s.mp4` | HY-WorldPlay | bf16 | 317 帧 / 19.8s | **bf16 显存极限**（20 chunks；22/24/26 OOM） | 59.3 GB |
| `hyworldplay_qvg-int2_829frames_51.8s.mp4` | HY-WorldPlay | INT2 | **829 帧 / 51.8 秒** | INT2 极限（52 chunks；60 OOM）——**解锁 2.6×**；bf16 同长需 ~110GB | 36.2 GB |

关键工程发现：INT2 的实际长度解锁是 **~3×**，不是存储压缩比的 7×——发布实现的注意力读取路径把整段历史反量化回 bf16 并全量重应用 RoPE，该瞬态仍按全长 bf16 计价，成为新瓶颈（SF 720-latent 与 HY 60-chunk 的 OOM 都发生在这条路径上）。

原始输出与日志：`results/limits/`、`repro/backup/logs/limit_*.log`、`repro/backup/race/result_limit_*.txt`。
