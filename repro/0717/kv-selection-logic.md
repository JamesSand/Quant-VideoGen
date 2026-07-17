# 三个模型的 KV 选取逻辑：当前生成帧到底读哪些缓存

> 逐模型梳理"生成新内容时注意力从 KV cache 里取什么"，含代码锚点。
> 这决定了量化误差**何时何处**被激活（配合 [rope-coordinate-finding.md](rope-coordinate-finding.md)
> 与断崖机制），也是评测协议差异的根源。

## 一、LongCat（分段续写，固定长条件窗）

**结构**：视频续写任务——固定 73 帧条件窗 → 每段生成 20 新帧 → 窗口滑动进入下一段。

**选取逻辑**（[pipeline_longcat_video.py#L1180-L1200](../../experiments/LongCat/longcat_video/pipeline_longcat_video.py#L1180-L1200)）：
1. 段开始：对 73 帧条件窗做 prefill 得到其 KV（[`_cache_clean_latents`](../../experiments/LongCat/longcat_video/pipeline_longcat_video.py#L1184-L1192)），
   **随即整窗量化**（[`quantize_kv_cache`](../../experiments/LongCat/longcat_video/modules/longcat_video_dit.py#L541-L580)，唯一一次量化事件）；
2. 段内 denoising 的每一步：新 20 帧的 query 读取 **整个量化后的条件窗 KV +
   本段自身 token**（段内全注意力，无因果掩码筛选、无检索——条件窗全量常驻）；
3. 下一段：窗口滑动重新锚定，重复 1-2（[`rope_3d`](../../experiments/LongCat/longcat_video/modules/rope_3d.py#L115-L143) 按当前 grid 现算位置，读取点在 [attention.py#L187](../../experiments/LongCat/longcat_video/modules/attention.py#L187)）。

**含义**：条件窗 = 100% 被读的"全热"内容 → 量化误差从第一个生成帧就全额激活
→ **frame-93（首个生成帧）协议**恰好测到纯净的误差注入点；不存在检索选择性，
时间轴预算分配在 LC 上无意义（单事件、全热）。

## 二、HY-WorldPlay（近期窗 + 阈值切换的 FOV 记忆检索）

**结构**：I2V 世界模型，逐 chunk（4 latent = 16 帧）自回归；KV cache 按帧对齐
写入，老化出近期窗后量化（8-latent 跨度事件）。

**选取逻辑**（[pipeline_wan_w_mem_relative_rope.py#L1055-L1075](../../experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py#L1055-L1075)）——两个阶段：

1. **早期（current_frame_idx < context_window_length，[默认 16 latent](../../experiments/HY-WorldPlay/wan/generate.py#L201)）**：
   [`selected_frame_indices = range(0, current_frame_idx)`](../../experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py#L1073-L1075) —— **全历史直读**：
   新 chunk 的 query 读取此前所有帧的 KV（近期帧 BF16 + 老化帧已量化）。
   我们 189 帧 run 的断崖（f29 ≈ latent 7-8，chunk 2-3）**发生在这个阶段**：
   回访时刻模型直读全部量化历史，无检索过滤；
2. **后期（≥16 latent）**：[`select_mem_frames_wan`](../../experiments/HY-WorldPlay/wan/models/utils.py#L88-L145)
   —— **近期窗 + FOV 检索**：
   - 近期窗：最近 temporal_context_size（44 像素帧）恒在；
   - 记忆配额：memory_frames − temporal_context = **4 帧**（1 个 4 帧块）；
   - 检索：历史每 4 帧一个候选块，用相机 **FOV 重叠度**（[`calculate_fov_overlap_similarity`](../../experiments/HY-WorldPlay/hyvideo/utils/retrieval_context.py#L139-L160)，60°×35° 视锥 Monte-Carlo）对当前 chunk 的 query 帧打分，
     **贪心取重叠最大的块**直至配额满——赢者通吃，选中的老帧按相对位置重新
     排位（相对 RoPE 现算 + prope 按当前相机变换）。

**⚠️ 0717 修正——阶段 2 在我们的评测中从未执行**：paper 协议 run 传
`--memory_frames 48` = 全部 48 latent（pod_run_paperspeed.sh#L65），触发条件
`current_frame_idx ≥ context_window_length(=48)` 永不满足；且 QVG fork 的连续性
assert（[pipeline#L1076-L1081](../../experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py#L1076-L1081)）
禁止非前缀选择——检索若真执行会当场崩溃。as-evaluated 的 HY = 全历史前缀直读，
cache 存 post-rope‖post-prope、追加写入、读取不重转（[processor#L165-L185](../../experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py#L165-L185)），
**位置写入时定死** = 固定读取坐标系。阶段 2 描述保留为上游设计意图（真部署
检索配置下才生效）。详见 [rope-coordinate-finding.md](rope-coordinate-finding.md) §二/§三。

**含义**：量化误差的激活在评测配置下是持续直读式的（断崖 f29 发生于全历史
直读阶段）；"事件性检索激活"属于未评测的设计意图配置。

## 三、Self-Forcing（纯近期滚动窗 + 可选 sink）

**结构**：T2V，逐 block（3 latent = 12 帧）因果自回归；KV cache 追加式写入，
chunk（24 latent = 37440 token）老化后量化。

**选取逻辑**（[causal_model.py#L113-L262](../../experiments/Self-Forcing/wan/modules/causal_model.py#L113-L262)）：
1. 注意力尺寸：[`max_attention_size = 32760 token（=21 latent） if local_attn_size==-1 else local_attn_size×1560`](../../experiments/Self-Forcing/wan/modules/causal_model.py#L131)——新 block 的 query 读取
   [`kv_cache[k][max(0, end − max_attention_size) : end]`](../../experiments/Self-Forcing/wan/modules/causal_model.py#L262)，**纯近期滑窗、无检索**；
2. 可选 [`sink_size`](../../experiments/Self-Forcing/wan/modules/causal_model.py#L229)：保留开头 sink_size 帧恒在窗内（默认 0；这是 KVSink 式
   sink 机制的原生接口）；
3. 我们的评测 run 用 `--local_attn_size 195` = 全历史注意力（发布代码的滑窗
   路径在 pre-RoPE 分支不可用，见 0716 上游问题清单）。

**含义**：全注意力配置下每个新 block 读全部量化历史——误差持续均匀注入、无
事件性激活；且 SF 无条件前缀（首 block 即被量化影响）——这就是 SF 被移出参考
三指标矩阵、只测 VBench 的结构原因。

## 四、三模型对照表

| | LC | HY | SF |
|---|---|---|---|
| 读取模式 | 条件窗全量常驻 | 全历史（<16 latent）→ 近期窗+FOV 检索 4 帧（≥16） | 近期滑窗（21 latent）或全历史 |
| 检索/选择性 | 无 | **有（FOV 赢者通吃）** | 无（可选 sink） |
| 量化误差激活 | 首生成帧即全额 | 事件性（回访/检索） | 持续均匀 |
| 位置语义 | 段内固定（cache 段内生死） | **固定（0717 修正）**：post-rope/prope 写入时定死、追加读取不重转；检索重排仅为未评测的设计意图 | **绝对位置写入时施加，滑窗只截断**（固定） |
| 对应评测协议 | frame-93 | 全程均值（断崖支配） | VBench（无参考） |
