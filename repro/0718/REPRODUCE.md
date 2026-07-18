# REPRODUCE 0718 — multi-prompt campaign 指令级复现

前提：dev pod 8×H100，repo 根目录，`.venv` + `repro/backup/scripts/env_fix.sh`。

## 1. 素材

```bash
# 10+10 条 prompt(MovieGen 行 1,101,...,901 + 51,151,...,951)已在
repro/0718/prompts/selected.txt      # LC 用(prompt_idx 1-20,1-based)
repro/0718/prompts/p{1..20}.txt      # SF 用(单行文件)
```

## 2. 生成（单任务入口 + 8 卡队列）

```bash
# 任务语法: lc_base:<P> | lc:<P>:<arm>:<rep> | sf:<P>:<arm>:<rep>[:latents] | hy:<S>:<arm>
# arm: bf16 | qvg | pca<变体>；变体后缀: r6/r8/a128/tern/vtern/ktern/kptern/kptern128/kp0/v90/v00/vmean
bash repro/0718/scripts/gpu_queue.sh <jobs.txt> 8       # GPUS="3 4 5" 可指定卡
# 队列文件: jobs_phase{0a,0b,1a,1b,2a,3}.txt(基线) jobs_sweep{1..7}.txt jobs_validate.txt
```

终版三臂（写新 jobs 文件即可）：LC `lc:P:pca:0`；SF `sf:P:pcaa128:0`；
HY `hy:S:pcav90kptern`。QVG 臂 = `qvg`(LC iters=100,SF/HY iters=2,同 seed 0 重复)。

每 job 落一行台账 `repro/0718/logs/ledger.txt`：`<job> OK|FAIL|NOFILE|NOHIJACK rc wall hijack out`
——**pca 臂必须 hijack≥1**,NOHIJACK 即作废。

## 3. 评分

```bash
.venv/bin/python repro/0718/scripts/stats.py          # 基线全表 -> stats-output.md
.venv/bin/python repro/0718/scripts/sweep_stats.py    # 变体对比(共享 npz/vbench 缓存)
# 单视频 VBench: .venv/bin/python repro/backup/scripts/vbench_iq.py <mp4...>
```

协议窗口：LC = f93 单帧(+[93:] 均值辅助)；HY = frames[13:] 均值；SF = VBench 700
前缀窗(180 latents=717px)。QVG 对照 = 3 重复取均值。

## 4. 已知坑

- SF `--num_output_frames` 是 **latent** 帧,须 %3==0(180=VBench700;360 会触发
  上游 raise,双边都不能跑);
- torchrun 崩溃留 NCCL 僵尸占卡:先 `nvidia-smi --query-compute-apps` + `ps` 独立
  读证据再 kill;gpu_queue.sh 已有开跑前显存守卫;
- triton cache 勿按 job 复制(配额);HY 需要 `--seed`(0718 新增参数)。

## 5. BPE 合法性速查

每秩 +2bit/token(=+0.0156);r4 asym B128 = 2.3125,r5 起超线(2.328)。
HY v90+kptern = 2.258。变更配置先对账,历史记录 research-log #35a。
