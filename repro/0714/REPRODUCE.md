# 0714 实验复现指令

> 工作目录一律为 repo 根 `Quant-VideoGen/`。公共前置（venv、env_fix、kubeconfig、pod 模板、
> NODE_BUSY/recreate 流程、共享输入与 bf16 参考视频）与 0713 完全一致，见
> [../0713/REPRODUCE.md](../0713/REPRODUCE.md) §0，此处不重复。
> 评测口径不变：**frame-93 首帧 PSNR** vs `results/longcat/bf16/1-0/segment_1.mp4`。

## 1. B 扫描九宫格补跑（8 pod）

前置修复（B=128 会触发发布 kernel 崩溃，先打补丁）：`quant_videogen/real/quant_pack.py`
的 autotune 剪掉 `BLOCK_D < Q_BLOCK_SIZE` 的配置（提交 8b81883）。验证方法：修复前
`quant_block_size=128` 必现 `tl.arange(0,0)` 崩溃，修复后金丝雀 run 正常出片。

```bash
# QVG B=16 / B=128，各 n=3（QVG 非确定性）：
bash repro/backup/scripts/pod_run_var.sh qvg_b16  <1|2|3>   # pod 内执行
bash repro/backup/scripts/pod_run_var.sh qvg_b128 <1|2|3>
#   两臂参数（摘自 runner）：--quant_type triton-nstages-kmeans-int2
#   --quant_block_size {16|128} --cache_num_{k,v}_centroids 256
#   --kmeans_max_iters 100 --num_prq_stages 1
# QuaRot+clip 的 B=64/128 两格（qclip runner 第 5 参 = 块大小，缺省 16）：
bash repro/backup/scripts/pod_run_qclip.sh int2 0.99 100 qclip_int2_r0.99_b64  64
bash repro/backup/scripts/pod_run_qclip.sh int2 0.99 100 qclip_int2_r0.99_b128 128
#   runner 内固定：QUAROT_SYM=0、K/V 双旋转、QUAROT_CLIP_RATIO=$2、QUAROT_CLIP_PCT=$3
```

manifests：`repro/backup/pods/var_qvg_b{16,128}_run{1..3}.yaml`、`qclip_int2_r0.99_b{64,128}.yaml`。
产物 `results/varstudy/…`、`results/qclip/…`；状态 `repro/backup/race/result_<tag>.txt`。

## 2. B=64 对齐的 QuaRot 两臂

```bash
bash repro/backup/scripts/pod_run_var.sh quarot_asym64 1   # QUAROT_BLOCK=64 QUAROT_SYM=0
bash repro/backup/scripts/pod_run_var.sh quarot_sym64  1   # QUAROT_BLOCK=64 QUAROT_SYM=1
#   都经 quarot_launcher.py 劫持 --quant_type naive-int2 --quant_block_size 64
```

## 3. 速度实验

### 3a. speed_sf / speed_lc（热身+计时两遍协议；结果后被 §3b 的纯 paper 口径取代）

```bash
bash repro/backup/scripts/pod_run_speed.sh sf   # SF 180-latent：bf16 warm/meas + int2 warm/meas
bash repro/backup/scripts/pod_run_speed.sh lc   # LC 单段续写：TIME_BENCH=5 算子分解，bf16/int2 各两遍
```

### 3b. paper 口径（发布工作量，bf16 → qvg 顺序跑）

```bash
bash repro/backup/scripts/pod_run_paperspeed.sh sf   # bf16 ×1 + qvg ×2（pass1 含 triton 编译）
bash repro/backup/scripts/pod_run_paperspeed.sh lc   # 10 段全量，bf16/qvg 各一遍
bash repro/backup/scripts/pod_run_paperspeed.sh hy   # 12 chunk 匹配几何，bf16 ×1 + qvg ×2
```

**⚠️ 墙钟陷阱（HY 教训）**：pod 内第一臂独自支付冷 PVC 权重加载（HY 实测 ~336s），
`result_paperspeed_*.txt` 里的 wall 只能看趋势；**正式数字必须用日志内生成计时**——
HY 用 `Generate time for chunk N` 的稳态段（chunk 4-11）、SF 用去噪 tqdm、LC 用分段
tqdm 均值。修正后：LC −19.9%、SF −5.0%、HY −30.2%（details-0714.md §3.5）。

## 4. KV 分布采集与 3D 图（三模型）

```bash
# SF：直接用既有 49GB dump（results/ropestudy/kv_cache_frames180.pt，layer 15）
# LC / HY：现场采集（双钩子 launcher：ChunkedKVCache.write + compress_kv_cache 兜底）
bash repro/backup/scripts/pod_run_kvplot.sh lc   # 产出 results/kvplot/lc_kv.pt（compress 钩，48 层调用取中段）
bash repro/backup/scripts/pod_run_kvplot.sh hy   # 产出 hy_kv.pt（cache-write 钩，callers 行号区分 K/V）

# 出图（KV 值一律 3D surface —— 见 memory kv-plot-style）：
.venv/bin/python repro/backup/scripts/plot_kv3d.py sf repro/0714/figs/kv3d_sf.png
.venv/bin/python repro/backup/scripts/plot_kv3d.py lc repro/0714/figs/kv3d_lc.png
.venv/bin/python repro/backup/scripts/plot_kv3d.py hy repro/0714/figs/kv3d_hy.png
# token norm（OScaR Fig.3 口径：逐位置跨 head 箱线 + 中位 head 热图）：
for M in sf lc hy; do .venv/bin/python repro/backup/scripts/plot_token_norms.py $M repro/0714/figs/token_norms_$M.png; done
```

## 5. QKV 三维度解剖采集（SF）

```bash
bash repro/backup/scripts/pod_run_qkv.sh          # 末去噪步版 → results/kvplot/sf_qkv.pt
bash repro/backup/scripts/pod_run_qkv_steps.sh    # 全步版   → results/kvplot/sf_qkv_steps.pt
```

采集器 `qkv_capture_launcher.py` 钩在 `CausalWanSelfAttention.attn_kv_cache_prerope`
（q/k/v 原始入参，K 未加 RoPE）。环境变量语义：
- `QKV_LAYERS=0,15,29`——按创建序号定位层
- `QKV_WINDOWS=0-5,87-92,174-179`——按 block 起始帧过滤（steps 版用 `87-89` 单 block）
- `QKV_ALL_STEPS=1`——每次 forward 追加保存（=每去噪步一份）；缺省 0 为覆盖写（留最后一步）
- SF 每 block = 4 去噪步 [t=1000/750/500/250] + 1 次干净上下文重编码，双生成器共 10 次 forward

```bash
# 出图：
.venv/bin/python repro/backup/scripts/plot_sf_kv_summary.py    # KV 值三图（3D surface，固定 H4 对比 + H9 展品图）
.venv/bin/python repro/backup/scripts/plot_qkv_anatomy.py      # QKV norm 三图（OScaR 逐位置箱线，zoom=连续 32 token 跨帧界）+ 统计表
.venv/bin/python repro/backup/scripts/plot_qkv_steps.py        # timestep 动态六图 + 每步统计
.venv/bin/python repro/backup/scripts/plot_qk_norm_flow.py     # QK-Norm 流程图
```

## 6. QK-Norm 的 g 提取（checkpoint 直读）

```python
import torch
sd = torch.load('ckpts/Self-Forcing/self_forcing_dmd.pt', map_location='cpu',
                mmap=True, weights_only=False)['generator_ema']
g = sd['model.blocks.29.self_attn.norm_k.weight'].float().view(12, 128)  # [head, ch]
# 每头 rms：g.pow(2).mean(1).sqrt()；H9 大通道：g[9].abs().topk(8)
# 与实测 K 对齐：ch_rms = K[:,9,:].pow(2).mean(0).sqrt()；corr(|g[9]|, ch_rms) ≈ 0.83
```

可视化脚本内嵌在会话命令中（figs/qk_norm_g_l29.png 的生成代码见 git 历史 `4e88cac`）。

## 7. 文档类产物

- BPE 全表/对账：公式核算 + `pdftotext` 读 paper 附录（无需 GPU）。
- HY 双分支（rotary‖PRoPE）结构：读
  `experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py:121-233`。
- Qwen3 QK-Norm 证据：`repro/0714/reference/modeling_qwen3.py:195-218`（vendor 自本地 transformers）。
