# 0713 实验复现指令

> 路径说明：0714 做过目录重构，本文所有路径按**重构后**位置书写（脚本在 `repro/backup/scripts/`，当日实际执行时它们位于 `repro/` 根）。所有命令的工作目录均为 repo 根 `Quant-VideoGen/`。

## 0. 公共前置

```bash
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
source .venv/bin/activate
source repro/backup/scripts/env_fix.sh        # NGC 镜像必须：unset TRITON_*_PATH（CUDA-13 ptxas 会毁掉 triton 3.4）
export KUBECONFIG=/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100
```

- 生成一律走 **1 卡 H100 集群 pod**（本地卡被占）。pod manifest 模板见 `repro/backup/pods/var_qvg_run2.yaml`；每个 runner 自带 NODE_BUSY 检查（空闲 <72GB → exit 42），此时用 `repro/backup/scripts/recreate_pod.sh <tag> <坏节点名>` 拉黑重建。
- 共享输入（已在盘上，重生成方法见 `repro/backup/EXPERIMENTS.md`）：
  - LongCat 权重 `ckpts/LongCat-Video`
  - 续写起点（bf16 首段）`results/longcat/base/1-0.mp4`
  - **bf16 参考视频** `results/longcat/bf16/1-0/segment_1.mp4`（113 帧；重生成：`bash scripts/LongCat/run_bf16.sh` 的单段续写等价命令，参数同下 COMMON，quant_type=none）
- 所有 run 的公共生成参数（写死在各 runner 内）：
  `--workload 480p_long_gen --num_segments 1 --num_cond_frames 73 --seed 0 --prompt_source text_to_video_from_file --prompt assets/t2v.txt --prompt_idx 1`

## 1. QuaRot + 旋转后裁剪扫描（20 run）

Runner：`repro/backup/scripts/pod_run_qclip.sh <int2|int4> <clip_ratio> <clip_pct> <tag>`
量化配置（runner 内固定）：QuaRot 非对称、B=16、K/V 双旋转，即
`QUAROT_BLOCK=16 QUAROT_SYM=0 QUAROT_ROTATE_K=1 QUAROT_ROTATE_V=1`，经 `quarot_launcher.py` 劫持 `naive-int{2,4}`。
**Clip 旋钮**（实现在 `repro/backup/scripts/quarot_quant.py`，作用于 Hadamard 旋转之后、分块量化之前）：
- `QUAROT_CLIP_RATIO=r`：逐块收缩，`mn=amin(block)*r, mx=amax(block)*r`（r=1.0 关闭）
- `QUAROT_CLIP_PCT=p`：全局绝对值 p 分位截断 `clamp(±percentile(|x|,p))`（p=100 关闭；分位用 repo 的 `compute_percentile_by_sorting`，torch.quantile 超 2^24 元素会炸）

当日 20 个扫描点（b 臂 ratio 为主，a 臂 pct 对照）：

```bash
# 每位宽 6 个 ratio 点（pct=100 关）+ 4 个 pct 点（ratio=1.0 关）
for BITS in int2 int4; do
  for R in 0.85 0.90 0.925 0.95 0.975 0.99; do
    bash repro/backup/scripts/pod_run_qclip.sh $BITS $R 100 qclip_${BITS}_r${R}   # pod 内执行
  done
  for P in 99.0 99.5 99.7 99.9; do
    bash repro/backup/scripts/pod_run_qclip.sh $BITS 1.0 $P qclip_${BITS}_p${P}
  done
done
```

实际以 pod 方式并行：manifests 在 `repro/backup/pods/qclip_*.yaml`（20 个），`kubectl apply -f` 即可；产物 `results/qclip/<tag>/1-0/segment_1.mp4`，状态 `repro/backup/race/result_<tag>.txt`，日志 `repro/backup/logs/<tag>.log`。

## 2. INT2 方差研究（6 方法 × 3 次）

Runner：`repro/backup/scripts/pod_run_var.sh <method> <runidx>`，method ∈ `qvgpro | qvg | quarot_asym16 | quarot_sym16 | quarot_asym128 | rtn16`。
每方法 run2/run3 为当日新跑（manifests `repro/backup/pods/var_*_run{2,3}.yaml`），run1 复用此前单次结果。各方法确切参数（摘自 runner）：

| method | 命令要点 |
|---|---|
| qvgpro | `--quant_type triton-nstages-kmeans-int2 --quant_block_size 16 --num_prq_stages 4 --cache_num_{k,v}_centroids 256 --kmeans_max_iters 100`（经 `longcat_rngiso_launcher.py`，kmeans randint 隔离到种子 20260710 的独立 generator）|
| qvg | 同上但 `--quant_block_size 64 --num_prq_stages 1`，直跑 `experiments/LongCat/run_long_t2v.py`（发布版 RNG 行为）|
| quarot_asym16 | `QUAROT_BLOCK=16 QUAROT_SYM=0` + `quarot_launcher.py --quant_type naive-int2 --quant_block_size 16` |
| quarot_sym16 | 同上但 `QUAROT_SYM=1` |
| quarot_asym128 | `QUAROT_BLOCK=128 QUAROT_SYM=0`，`--quant_block_size 128` |
| rtn16 | 无 launcher（不旋转），直跑 `--quant_type naive-int2 --quant_block_size 16` |

```bash
for M in qvgpro qvg quarot_asym16 quarot_sym16 quarot_asym128 rtn16; do
  for I in 2 3; do kubectl apply -f repro/backup/pods/var_${M}_run${I}.yaml; done
done
```

产物 `results/varstudy/var_<method>_run<idx>/1-0/segment_1.mp4`。

## 3. 评测口径：frame-93 首帧 PSNR

一切 0713 数字都是 **全视频第 93 帧**（= 第一张生成帧；前 93 帧为共享 init 前缀）对 bf16 参考的 PSNR：

```bash
# 批量：把新 (参考, 待测) 对加进 repro/backup/scripts/precompute_arrays.py 的 PAIRS 字典
#   参考一律是 results/longcat/bf16/1-0/segment_1.mp4
python repro/backup/scripts/precompute_arrays.py     # 产出 repro/backup/protosearch/<name>.npz
python - <<'EOF'
import numpy as np
d = np.load('repro/backup/protosearch/<name>.npz'); print(round(float(d['psnr'][93]), 3))
EOF
```

单对快速版（当日 qclip/var 用的就是这个逻辑，CPU 即可）：

```python
import imageio.v3 as iio, numpy as np
ref = iio.imread('results/longcat/bf16/1-0/segment_1.mp4', index=93, plugin='pyav')
gen = iio.imread('results/qclip/qclip_int2_r0.99/1-0/segment_1.mp4', index=93, plugin='pyav')
mse = np.mean((ref.astype(np.float64)/255 - gen.astype(np.float64)/255)**2)
print(10*np.log10(1/mse))
```

## 4. 首帧图与 contact sheet

`repro/0713/first_frames/` 的 25 张图：对每个 run 抽 frame 93 存 PNG（文件名 = 配置），再拼 4×3 网格并标注 PSNR：

```python
import imageio.v3 as iio
from PIL import Image, ImageDraw
for tag in TAGS:   # 每个扫描点
    img = iio.imread(f'results/qclip/{tag}/1-0/segment_1.mp4', index=93, plugin='pyav')
    Image.fromarray(img).save(f'repro/0713/first_frames/{tag}.png')
# contact sheet：PIL 网格粘贴 + draw.text 写 "tag  PSNR=xx.xx"（无 CJK 字体，标注用英文）
```

## 5. Clip 张量级刨析（纯 CPU，不需要 pod）

数据：真实 SF KV dump `results/ropestudy/kv_cache_frames180.pt`（49GB）。

```python
import sys, torch
sys.path.insert(0, 'repro/backup/scripts')
from quarot_quant import hadamard, quarot_fake_quant   # TF32 已在模块内关闭
d = torch.load('results/ropestudy/kv_cache_frames180.pt', weights_only=False, mmap=True)
K = torch.cat([c for c in d['kv_cache'][15]['k'].chunks], dim=1).float()  # BSHD [1,S,12,128]
H = hadamard(128, K.device)
Kr = K @ H                                   # 旋转后
# 三个量：max 缩减、含被裁值的净 MSE、幸存值/被裁值分解
#   ratio 臂: 逐 16-ch 块 amin/amax×r 后 clamp；pct 臂: 全局 |x| 分位 clamp
#   MSE 一律在"旋转回原基后 vs 原始 K"上计算; 分解用 clip 掩码分别对幸存/被裁元素求和
```

具体数字与结论见 `report-0713.md` "Clip 的张量级刨析" 一节。

## 6. 总表 3 次均值

纯文档操作：`report-0713.md` 总表的 INT2 行 = §2 方差研究三次 frame-93 PSNR 的 mean±std；INT4 的 QVG 行 = 三次异构测量（released/rngiso/clip-p100 对照臂）均值。原始三次数值保留在 report 的"INT2 方差实验"表里。
