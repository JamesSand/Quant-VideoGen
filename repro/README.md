# repro/ — 复现与实验工作区

## 目录结构

```
repro/
├── 0713/        ← 按日期的工作目录：当日 report + handoff + 当日可视化
│   ├── report-0713.md    当日日报（总表 / clip 扫描 / 方差研究 / 张量刨析）
│   ├── HANDOFF.md        当日收尾交接：完成事项、关键数字、开放问题、次日候选方向
│   └── first_frames/     当日首帧对比图（contact sheet + 单帧）
├── 0714/        ← 当前工作日
└── backup/      ← 日期目录之外的全部积累
    ├── REPORT.md             复现总报告（结论层，入口文档）
    ├── EXPERIMENTS.md        全部实验明细（≥10 次 QuaRot 尝试等）
    ├── CLIP_STUDY.md         原始 KV 纯丢弃 clip 研究（scoped variant）
    ├── ROPE_DISPERSION.md    RoPE 分散性推导
    ├── ARCHITECTURES.md      三个模型的生成流程解析
    ├── ISSUE_DRAFT.md        给作者的 issue 草稿（未发送）
    ├── scripts/              全部脚本：launcher / pod runner / manifest 生成 / 环境修复
    ├── pods/                 历史 pod manifest（内嵌路径为迁移前旧路径，复用前需重新生成）
    ├── race/                 每条 run 的 result_*.txt 记录 + bad_nodes.txt
    ├── metrics/ figs/ protosearch/ ropestudy/   数据与图表
    ├── logs/ triton_cache/   运行日志与 JIT 缓存（不入库）
    ├── first_frames_rawclip/ 原始 KV clip 研究的首帧图（本地，不入库）
    └── limit_videos/         生成长度极限实测视频（mp4 本地，不入库；README 入库）
```

## 约定

- 每个工作日在 `repro/` 下开一个 `MMDD/` 目录，日报命名 `report-MMDD.md`，收尾写 `HANDOFF.md`，并附 `REPRODUCE.md`（当日全部实验的具体指令与参数）。
- 跨天复用的脚本/数据一律放 `backup/`，日期目录只放当日产物与文档。
- 大文件（视频、日志、JIT 缓存、批量帧图）不入库，md 中注明本地路径。
- 每个 deliverable 完成即 commit + push。
