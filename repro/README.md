# repro/ — 复现与实验工作区

## 目录结构

```
repro/
├── 0713/        ← 按日期的工作目录：当日 report + handoff + 当日可视化
│   ├── report-0713.md    当日日报（总表 / clip 扫描 / 方差研究 / 张量刨析）
│   ├── HANDOFF.md        当日收尾交接：完成事项、关键数字、开放问题、次日候选方向
│   └── first_frames/     当日首帧对比图（contact sheet + 单帧）
├── 0714/        ← 三问调查+B扫描+KV/QKV分布解剖+论文精读（summary/report/HANDOFF/REPRODUCE 齐备）
└── backup/      ← 日期目录之外的全部积累
    ├── REPORT.md             复现总报告（结论层，入口文档）
    ├── EXPERIMENTS.md        全部实验明细（≥10 次 QuaRot 尝试等）
    ├── CLIP_STUDY.md         原始 KV 纯丢弃 clip 研究（scoped variant）
    ├── ARCHITECTURES.md      三个模型的生成流程解析
    ├── ISSUE_DRAFT.md        给作者的 issue 草稿（未发送）
    ├── scripts/              全部脚本：launcher / pod runner / manifest 生成 / 环境修复
    ├── pods/                 历史 pod manifest（内嵌路径为迁移前旧路径，复用前需重新生成）
    ├── race/                 每条 run 的 result_*.txt 记录 + bad_nodes.txt
    ├── metrics/ figs/ protosearch/   数据与图表
    ├── logs/ triton_cache/   运行日志与 JIT 缓存（不入库）
    ├── first_frames_rawclip/ 原始 KV clip 研究的首帧图（本地，不入库）
    └── limit_videos/         生成长度极限实测视频（mp4 本地，不入库；README 入库）
```

## 约定

- **每个日期工作目录（`MMDD/`）必须包含三个 md 文件**：
  | 文件 | 定位 |
  |---|---|
  | `report-MMDD.md` | 当日日报，**拿去 present 的文件**——self-contained（背景 + 证据图表 + 结论），外人不读其他文档也能懂；技术深挖放独立专题 md（如 `details-MMDD.md`） |
  | `HANDOFF.md` | 收尾交接：当日完成表、关键数字速查、悬而未决、次日候选方向 |
  | `REPRODUCE.md` | 当日**全部实验**的具体复现指令与参数——runner 用法、确切扫描点、pod 命令、评测口径与代码片段，一律从实际脚本抄录 |
- 跨天复用的脚本/数据一律放 `backup/`，日期目录只放当日产物与文档。
- 大文件（视频、日志、JIT 缓存、批量帧图）不入库，md 中注明本地路径。
- 每个 deliverable 完成即 commit + push。
