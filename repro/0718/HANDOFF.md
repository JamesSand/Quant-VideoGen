# HANDOFF 0718

## 状态

Multi-prompt campaign 定稿（[multi-prompt-results.md](multi-prompt-results.md)）。
终版配置见 [report-0718.md](report-0718.md)（LC r4 asym / SF r4 asym / HY
v90+KP-tern），LC/SF 全胜含 held-out，HY 参考指标胜、VBench 结构性负 0.78。

## 唯一待仲裁（阻塞 goal 完全达成的最后一项）

**HY VBench 语义**：MUSIQ-IQ 奖励 QVG 字典伪影（其分数常高于 BF16 无损参考），
忠实型方法无解（8 配置 × 10 seeds 前沿在案）。选项：
(a) 接受参考指标为 HY 质量准绳（现终版即全胜）；
(b) 坚持 VBench → 换全三值配置（VB +0.11 但参考三指标全负）；
(c) 换 VBench 其他维度/完整套件重测（IQ 单维的伪影偏好可能被平均掉）。

## 下一步队列

1. 用户仲裁 HY VBench → 更新 eval-protocol.md 与 0717 final-method-results.md
   头条（当前头条仍是 0717 n=1 版,需要以 campaign 版为准重写);
2. 老三样结转:kernel 化 M0-M4(解锁 Q3 的 1400f)、rope 半区 KLT 裁决实验
   (0717 README 待办五)、QVG-Pro LPIPS 复测;
3. LC p9 型坏 basis 瞬态:秩增大有效但预算非法——想免预算的稳健化
   (e.g. shrinkage 基、μ-only 回退判据)可作新 idea。

## 关键操作事实

- 全部生成/评分基建在 `repro/0718/scripts/`(campaign.sh + gpu_queue.sh + 三个
  stats 脚本),视频在 `results/multiprompt/`(2.6G,不进 git),npz 缓存
  `repro/0718/npz/`,台账 `repro/0718/logs/ledger.txt`;
- **勿再给每 job 建 triton cache 副本**(20G 撑爆配额;campaign.sh 已如此,
  重用前先改 TRITON_CACHE_DIR 策略);
- torchrun 崩溃会留 NCCL 僵尸占几十 GB;kill 前先独立一步读 `ps` 证据
  (0718 误杀过健康的 LC base,教训在 memory);
- HY generate.py 现在有 `--seed`(本日加,default 0 行为不变)。
