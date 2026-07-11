# 实验细节全记录（QuantVideoGen 复现）

> 本文档是 [REPORT.md](REPORT.md) 的附录：全部实验过程、命令、失败与修复。
> 时间：2026-07-10 ~ 07-11。机器：`zhizhousha-dev-8gpu-ssh`（node research-common-h100-073，本身是 k8s pod）+ 集群 pod。

## 1. 环境搭建与修复

```bash
uv venv .venv --python /usr/bin/python3.12
uv pip install -e ".[all]" torchaudio==2.8.0
uv pip install ".../flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-...whl"
```

对 README 的三处修正：
1. **flash-attn wheel ABI**：README 给的 `cxx11abiFALSE` 与 PyPI torch 2.8.0（manylinux_2_28 = CXX11-ABI TRUE）不兼容，import 报 undefined symbol → 用 `cxx11abiTRUE`。
2. **torchaudio 需钉 2.8.0**：lingbot extra 的裸 `torchaudio` 解析到 2.11.0（ABI 目标 torch 2.11，运行时崩）。
3. **`.[all]` 无依赖冲突**（`uv pip compile` 验证：numpy 1.26.4 / transformers 4.51.3 联合可满足）。

运行时环境修复：
- **NGC 容器的 `TRITON_PTXAS_PATH` 等指向 CUDA 13.0 工具链**，triton 3.4 不认 ptxas 13.0（"Triton only support CUDA 10.0 or higher, but got CUDA version: 13.0"）→ `env_fix.sh` unset 全部 `TRITON_*_PATH`。
- **并发 JIT 撞共享 `~/.triton/cache`**：5 个进程同时编译 `_triton_rope` 时读到半写文件（FileNotFoundError *.cubin）→ 每 run 独立 `TRITON_CACHE_DIR`。
- HY 的 `LIBRARY_PATH` stubs 导出按原脚本保留；`libcuda.so` 已在 ldconfig，非必需。

模型准备（省 133GB 重复下载）：LongCat-Video (78G)、Wan2.1-T2V-1.3B (17G)、self_forcing_dmd.pt (5.7G) 直接 symlink `/shared/huggingface/hub` 的 snapshot；HY-WorldPlay (52.3G) wget；Wan2.2-TI2V-5B-Diffusers (~32G) 由 HY 首次运行自动下载（README 未提及）。

## 2. 评测工具链

- 唯一评测器：`experiments/LongCat/longcat_video/utils/metric.py`（逐帧 PSNR/SSIM/LPIPS 取平均，帧 0 恒被跳过；LPIPS-VGG 在 import 时实例化并下载 ~530MB 权重）。
- LongCat：`base.sh` 产 93 帧 init；bf16/qvg 共享同一 init；输出为累计式 `segment_N.mp4`；**必须 `--skip_frames 93`**（init 前缀两边同内容，MSE≈0 会产生 inf 毒化平均——实测因 x264 lookahead 前缀为有限 ~41.7 dB，但混入平均同样失真）。
- HY：仓库自带 bf16 (14 chunks/221 帧) 与 qvg (12 chunks/189 帧) 脚本**帧数不对齐**，无法直接比对 → `run_hy_bf16_matched.sh` 用 qvg 几何重跑基线。`generate.py` 吞异常且 exit 0 → 每 run 检查 `err.txt`。
- Self-Forcing：只有 prompt 0 的对是同 seed 有效比较（QVG run 的 k-means 消耗全局 RNG，使 prompt 1 初始噪声错位）。

## 3. QVG 主实验（文档协议）

| 实验 | 配置 | 结果 (PSNR dB) |
|---|---|---|
| SF INT2 全程 (prompt 0, skip 93) | 官方脚本 | 16.33；分叉点精确在 frame 93（首个量化事件 QUANT_FACTOR=8）；分叉前数值噪声底 ~37-44（两 run 的 cache 代码路径不同） |
| LC bf16/INT2/INT4 × 发布版/RNG 隔离版 | seed 0, prompt_idx 1（滑板手，= paper teaser） | 全程 (seg10, skip93)：INT2 12.70/12.63，INT4 12.51/12.02；seg1-only：17.35/17.91、20.91/21.51 |
| HY matched bf16 + INT2/INT4 | 12 chunks 几何 | 18.67 / 22.51 |

关键中间数据：
- 逐帧曲线：LC INT2 首个生成帧 29.4 → 24.8 → 22.4 → … → 15.5；HY 分叉起点 (frame 29) INT2 25.4 / INT4 33.5。
- KV 内存日志：LC bf16 464.00/22272.00 MB、INT2 67.32/3231.28、INT4 125.43/6020.53；HY INT2 141.18/4235.45、bf16(48帧几何) 825.00/24750.00、INT4 244.31/7329.20。
- SF bf16 KV：1645.31 MB/层、49359.40 MB 总。

## 4. 假设检验记录（时间序）

| # | 假设 | 实验 | 结果 | 判定 |
|---|---|---|---|---|
| 1 | kernel 数值错误 | `kernel_ab_test.py`：triton 实路径 vs 纯 torch 模拟，同数据 | INT2 rel_err 0.1492 vs 0.1497；INT4 0.0225 vs 0.0225 | ❌ 排除 |
| 2 | RNG 污染（k-means randint 消耗全局 CUDA RNG） | `longcat_rngiso_launcher.py` 把 randint 重定向到独立 Generator | seg1 17.91 vs 17.35（种子方差内）；全程 12.63 vs 12.70 | ❌ 非主因（污染真实存在但不解释差距） |
| 3 | 混沌放大上界 | 两个只差 kmeans 种子的 INT2 run 互比 | 22.41 dB → quant-vs-bf16 上限 ≈ 25.4 | ✅ 证明 28.716 不可达 |
| 4 | 配置错误 | 对照 KV 内存日志 | 与 README 逐位一致 | ❌ 排除 |
| 5 | 真实数据量化误差异常 | sim 假量化 run 打印逐层 rel-L2 | INT2 K 0.267/V 0.453（paper Fig 7 范围内）；INT4 0.042/0.071 | ❌ 误差正常 |
| 6 | QVG-Pro 配置 (S=4,B=16) | 1 segment | 22.05（paper Pro 30.376） | ❌ |
| 7 | 未 skip 前缀（codec 噪声抬高） | seg10 no-skip | 21.86 | ❌ |
| 8 | 统一评测窗口存在 | 全窗口扫描（单方法） | INT2 需 [1..179)、INT4 需 [1..121)，互斥 | ❌（单方法层面） |
| 9 | fullkv workload（增长上下文） | bf16+INT2+INT4 各 6 段 | 逐段递减 19.9→13.5→14.6；bf16 在 seg4 (153帧上下文) OOM | ❌ |
| 10 | HY 14-chunk 几何 | 补跑 shipped bf16 + 同几何 INT2/INT4 | 18.14/21.59 ≈ matched 几何 | ❌ |
| 11 | 外部资源（eval 脚本/视频/issue/版本差异） | GitHub、git 全历史、项目主页、arXiv v1-v5 diff | 全空；v1 与 v5 数字/协议一字不差 | ❌ |
| 12 | **起点窗口协议族（联合多方法约束）** | `protocol_search.py` | 见 §6 | ✅ **成立** |

## 5. QuaRot/RTN 基线移植与矩阵

- QuaRot 官方 repo（spcl/QuaRot fork）只支持 LLaMA；QVG 仓库无任何 QuaRot/KIVI 代码（工作区+git 历史零命中）。
- 移植（`quarot_quant.py`）：head_dim=128 的 Sylvester 正交 Hadamard；fake-quant 旋转→分块 RTN→反旋转在数学上等价于在旋转基中做注意力；K、V 都旋转（QuaRot 通过 v_proj/o_proj 融合旋转 V）；对称/非对称、B16/B128 可配；TF32 需关闭（否则旋转往返引入 1e-4 误差）。注入方式：launcher 在目标脚本 import 前 patch `compress_kv_cache`，劫持 `naive-int*`。
- 单元测试：H@Hᵀ 偏差 6e-8；重离群数据 INT4 增益 1.22×（旋转设计场景 ✓）；平滑结构数据旋转反而略差（RTN 0.053 vs QuaRot 0.065 @INT4）。
- 矩阵（12 配置 × 独占集群 GPU）结果：

| 配置 | LC | HY |
|---|---:|---:|
| QuaRot INT2 非对称 B16 | 17.72 | 19.21 |
| QuaRot INT2 对称 B16 | 18.93 | — |
| QuaRot INT2 非对称 B128 | 17.14 | — |
| QuaRot INT4 非对称 B16 | 20.45 | 21.87 |
| QuaRot INT4 对称 B16 | 21.09 | — |
| QuaRot INT4 非对称 B128 | 18.81 | — |
| RTN INT2 B16 | 16.47 | 17.95 |
| RTN INT4 B16 | 26.26 | 21.49 |

- 排序分析：INT2 下 QuaRot>RTN（两模型，+1~2 dB，与 paper 同向）；HY INT4 下 QuaRot−RTN = +0.38（paper +0.36）；**LC INT4 排序反转**（RTN 高 6 dB）。

## 6. 联合协议搜索

- 数据基座：23 个视频对的逐帧 PSNR/SSIM/LPIPS 数组（`precompute_arrays.py` → `protosearch/*.npz`），含 4 个补跑的 5-segment 基线长视频（193 帧，扩窗用）。
- 搜索空间：窗口 [a,b) 全组合 × {逐帧均值, 窗口 MSE} 聚合 × 百分位协议 × QuaRot 三变体替换 × QVG 发布版/rngiso 替换；约束 = 同 block 所有方法共用同一规则。
- 结果（详见 REPORT.md §一.3）：LC INT2 单帧窗口 [93,94) 四方法同时命中（最坏 0.98 dB，QVG 0.002）；HY INT4 [0,32) 1.39；LC INT4 [78,102) 2.45；HY INT2 [23,36) 2.55。扩窗至 193 帧不改善。
- 六方首帧检验（引出搜索的中间结果）：frame94 上 QVG INT2 +0.64 / RTN INT2 +0.98 / QuaRot(非对称) INT2 **+8.8** / QuaRot INT4 −0.70 / RTN INT4 +2.24 / QVG INT4 −3.55 → 非对称 QuaRot 与 paper 不符，对称变体 −0.16 相符。

## 6-bis. 生成长度极限实验

- 方法：每模型 bf16 + INT2 各一条，从提议尺寸起 OOM 自动降档（`repro/pod_run_limit.sh`），能跑通的最大尺寸即实测极限；全部在集群 1-GPU pod 上执行。
- 阶梯记录：SF bf16 228→210→**195✓**；SF INT2 900→720→**600✓**；HY bf16 26→24→22→（细档）20✓；HY INT2 60→**52✓**；LongCat 70 段直接跑通（bf16/INT2 双双 1493 帧）。
- OOM 根因取证：SF INT2 死于 `causal_rope_apply_long_input`（71.9GB 实分配，碎片仅 4.25GB——真实墙）；HY INT2 初次死于碎片（15.9GB reserved-unallocated，`expandable_segments:True` 缓解后仍在 60 chunks 撞读取瞬态墙）。证明瓶颈在"整段反量化 + 全量 RoPE"的读取瞬态，非 cache 存储。
- Self-Forcing 限定：>180 latent 需 `local_attn_size = num_output_frames`（代码不支持窗口淘汰，prerope 路径越界即 ValueError）；输出帧数=4L−3。
- HY 限定：`num_chunk ≤ memory_frames/4`（越界触发非连续记忆断言）；噪声缓冲默认 241 latent（`--num_frames 961`）为设计上限；动作序列每条目 8 latent。
- 视频收纳：`limit_videos/`（含 README 索引）；原始输出 `results/limits/`。

## 7. 基础设施：k8s 实验流水线

- 本机 8 卡长期被同节点其他租户的 privileged pod（`charlie/glm52bo-worker`，sglang 服务，绕过调度器不申报 `nvidia.com/gpu`）物理占用；本地抢跑失败两轮（外部调度在 ~3 分钟内回填空卡，我们模型加载要 5-13 分钟）。
- 方案：用户 kubeconfig（`low-precision-project/k8s-from-h100-pod/`，context `zhizhousha`）在集群建 1-GPU pod，挂 `home-zhizhousha` + `shared-data` PVC，镜像与 dev pod 相同（venv 直接复用）。
- 工具：`gen_pod_manifests.sh`（12 配置 manifest）、`pod_run_one.sh`（含占卡检测快速失败：free<72GB → exit 42）、`recreate_pod.sh`（坏节点 nodeAffinity 黑名单）、`pod_run_long.sh`（N-segment 长视频）。
- 事故记录：1 个 pod 落在 055 节点（也有 squatter，51GB 时 OOM）→ 黑名单重建成功；8-GPU 单 pod 因全集群无整空节点 Pending（59 节点全忙），删除。
- 弃用的本地方案（留档）：`gpu_sentinel.py`（显存哨兵占位-衰减交接）、`race_run_one.sh`——集群 pod 方案出现后不再需要。

## 8. 失败与修复流水账

| 事故 | 根因 | 修复 |
|---|---|---|
| SF/LC 首轮全崩 | NGC TRITON_PTXAS_PATH → CUDA13 ptxas | env_fix.sh |
| LC INT2/INT4 二轮崩 | 并发 JIT 共享 triton cache 竞争 | 每 run 独立 TRITON_CACHE_DIR |
| HY INT2/INT4 用了系统 python | 组合命令 bash `&&`/`&` 优先级 | 独立后台命令重发 |
| HY quarot pod ModuleNotFoundError: models | runpy 不把 wan/ 加入 sys.path | PYTHONPATH 加 experiments/HY-WorldPlay/wan |
| 集群首波 12 pod 全 OOM | 启动后外部任务回收本地 GPU（该波误在本地跑）+ 装载慢 | 全部迁移集群 + 占卡检测 |
| fullkv bf16 OOM@seg4 | 153 帧上下文 cache 超 80GB | 记录为发现（QVG 内存价值主张的佐证） |

## 9. Git 提交索引（JamesSand/Quant-VideoGen fork）

- `88e8f7c` 首次复现研究（QVG 主实验 + 证据链）
- `39f6be5` QuaRot/RTN 基线矩阵（12 配置 + k8s 流水线）
- `e0c3440` 六方首帧检验
- `558d58a` 联合协议搜索（起点窗口协议定位）
