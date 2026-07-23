# qvg4pt 最小 demo(0722)

用 qvg4pt(QVG kmeans 减法 + 四电平非对称 B64 残差格 + fp8 s/z)量化 KV cache,
跑一个单 prompt、单 segment 的 LongCat 视频续写。全链路解读见
[../qvg4pt-code-walkthrough.md](../qvg4pt-code-walkthrough.md)。

## 文件

| 文件 | 内容 |
|---|---|
| `qvg4pt_quant.py` | 量化实现,数值与 `repro/backup/scripts/pca_quant.py` 逐行一致(fp8 合法口径);kmeans 直接 import QVG 发布库(减法一个字不动) |
| `qvg4pt_launcher.py` | 最小 monkey-patch launcher:naive-int2 → qvg4pt,其余放行 |
| `run_demo.sh` | 一键 demo(1 GPU,~25 分钟):`bash run_demo.sh [prompt_idx]` |
| `minimal_one_frame.py` + `run_one_frame.sh` | **一帧最小版**(1 GPU,~5 分钟):单文件内联 affine int2 实现(存 mn+步长,格子与原版逐位一致),去噪 50→8 步提速,出 `demo_out/oneframe_f93.png` |
| `test_equivalence.py` | 拷贝版 vs 原版等价测试(CPU 验格子逐位一致;GPU 加验全臂) |
| `demo_out/` | 产出视频(不入 git) |

## 跑法

```bash
# 等价性自检(CPU 即可,GPU 更全)
.venv/bin/python repro/0722/qvg4pt-demo/test_equivalence.py
# demo(需 1 张 GPU + mp100 的 p1 base 视频)
bash repro/0722/qvg4pt-demo/run_demo.sh 1
# 成功标志:日志出现
#   [qvg4pt_launcher] hijacking naive-int2 -> qvg4pt (kmeans K=256 sub + ...)
# 且产出 demo_out/1-0/segment_1.mp4
```

## 与正式跑批的关系

campaign 正式路径(`campaign.sh:128` qvg4* 分支)走
`repro/backup/scripts/pca_launcher.py` + `PCA_QVG4=1`,分支多、依赖全量
pca_quant;本 demo 把 qvg4pt 一条路抽成自包含三件套,数值等价
(`test_equivalence.py` 为证),便于单独阅读/复现。终值(10 prompts):
PSNR 32.29 / SSIM 0.9478 / LPIPS 0.0405 @ BPE 2.589(实存,超预算 11%),
逐 prompt 见 `repro/0721/qvg4pt-fp8-score.json`。

## 实测记录(0722)

1 卡 H100 pod 实跑通过,证据链:

```
PASS grid: bit-exact on (2, 4, 96, 128)          ← 拷贝版 vs 原版,格子逐位一致
PASS full arm: bit-exact (same seed)              ← 全臂(含 kmeans)逐位一致
[qvg4pt_launcher] hijacking naive-int2 -> qvg4pt (kmeans K=256 sub +
  ASYM 4-level B64 residual, fp8 s/z) k(1, 32, 29640, 128)   ← 劫持生效,
  形状 = [B=1, H=32, S=29640 条件 token, D=128],pre-RoPE
DEMO OK: repro/0722/qvg4pt-demo/demo_out/1-0/segment_1.mp4   ← 3.7MB,moov 完整
```

## 一帧最小版实测(0722)

1 卡 pod 实跑通过(全程 ~10 分钟):qvg4pt 在全部层触发
(`[one_frame] qvg4pt on k(1, 32, 29640, 128)` 逐层打印),8 步去噪出片,
首个生成帧见 [oneframe_f93.png](oneframe_f93.png)(577KB,画面完好)。
