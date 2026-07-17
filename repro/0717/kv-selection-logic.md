# 三个模型的 KV 选取逻辑：当前生成帧到底读哪些缓存

> 逐模型梳理"生成新内容时注意力从 KV cache 里取什么"，含代码锚点。
> 这决定了量化误差**何时何处**被激活（配合 [rope-coordinate-finding.md](rope-coordinate-finding.md)
> 与断崖机制），也是评测协议差异的根源。

## 一、LongCat（分段续写，固定长条件窗）

**结构**：视频续写任务——固定 73 帧条件窗 → 每段生成 20 新帧 → 窗口滑动进入下一段。

**选取逻辑**（`pipeline_longcat_video.py:1180-1200`）：
1. 段开始：对 73 帧条件窗做 prefill 得到其 KV（`_cache_clean_latents`），
   **随即整窗量化**（`quantize_kv_cache`，唯一一次量化事件）；
2. 段内 denoising 的每一步：新 20 帧的 query 读取 **整个量化后的条件窗 KV +
   本段自身 token**（段内全注意力，无因果掩码筛选、无检索——条件窗全量常驻）；
3. 下一段：窗口滑动重新锚定，重复 1-2（`rope_3d` 按当前 grid 现算位置）。

**含义**：条件窗 = 100% 被读的"全热"内容 → 量化误差从第一个生成帧就全额激活
→ **frame-93（首个生成帧）协议**恰好测到纯净的误差注入点；不存在检索选择性，
时间轴预算分配在 LC 上无意义（单事件、全热）。

## 二、HY-WorldPlay（近期窗 + 阈值切换的 FOV 记忆检索）

**结构**：I2V 世界模型，逐 chunk（4 latent = 16 帧）自回归；KV cache 按帧对齐
写入，老化出近期窗后量化（8-latent 跨度事件）。

**选取逻辑**（`pipeline_wan_w_mem_relative_rope.py:1055-1075`）——两个阶段：

1. **早期（current_frame_idx < context_window_length，默认 16 latent）**：
   `selected_frame_indices = range(0, current_frame_idx)` —— **全历史直读**：
   新 chunk 的 query 读取此前所有帧的 KV（近期帧 BF16 + 老化帧已量化）。
   我们 189 帧 run 的断崖（f29 ≈ latent 7-8，chunk 2-3）**发生在这个阶段**：
   回访时刻模型直读全部量化历史，无检索过滤；
2. **后期（≥16 latent）**：`select_mem_frames_wan`（`wan/models/utils.py:90-143`）
   —— **近期窗 + FOV 检索**：
   - 近期窗：最近 temporal_context_size（44 像素帧）恒在；
   - 记忆配额：memory_frames − temporal_context = **4 帧**（1 个 4 帧块）；
   - 检索：历史每 4 帧一个候选块，用相机 **FOV 重叠度**（`calculate_fov_overlap
     _similarity`，60°×35° 视锥 Monte-Carlo）对当前 chunk 的 query 帧打分，
     **贪心取重叠最大的块**直至配额满——赢者通吃，选中的老帧按相对位置重新
     排位（相对 RoPE 现算 + prope 按当前相机变换）。

**含义**：量化误差的激活是**事件性**的（回访/检索命中时），这是断崖机制与
"锚点内容最热"结论的结构根源；同一 key 多次被以不同旋转读取 = 无固定读取坐标系。

## 三、Self-Forcing（纯近期滚动窗 + 可选 sink）

**结构**：T2V，逐 block（3 latent = 12 帧）因果自回归；KV cache 追加式写入，
chunk（24 latent = 37440 token）老化后量化。

**选取逻辑**（`wan/modules/causal_model.py:118-262`）：
1. 注意力尺寸：`max_attention_size = 32760 token（=21 latent） if local_attn_size==-1
   else local_attn_size×1560`——新 block 的 query 读取
   `kv_cache[k][max(0, end − max_attention_size) : end]`，**纯近期滑窗、无检索**；
2. 可选 `sink_size`：保留开头 sink_size 帧恒在窗内（默认 0；这是 KVSink 式
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
| 位置语义 | 每段重锚定 | 相对 RoPE + 检索重排 + prope 相机变换 | 窗口滚动重排 |
| 对应评测协议 | frame-93 | 全程均值（断崖支配） | VBench（无参考） |
