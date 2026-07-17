# 坐标系审计（三次修正定稿）：三个模型的评测管线全是固定读取坐标系；各向异性失败的机制重新开放

> 本页最初回答"为什么不能拿 post-RoPE 的 KV 做 PCA 再量化"，先后经用户三次追问
> 修正（LC → SF → HY），最终结论与初版相反：**在 QVG 评测所用的三条管线里，
> 量化 KV 的读取坐标系全部固定**。各向异性方法的经验失败规律不变，但其机制
> 归因（"读取旋转动态"）被证伪，重新成为开放问题。修正史保留在 §五，作为
> 反面教材与诚实记录。

## 一、LLM 里 post-RoPE 量化的地基（不变）

标准 LLM 的位置绝对且一次性：token s 的 key 写入时被 R_s 旋转一次，之后永远以
R_s·k 的形态被读取——每个 key 有唯一的 post-RoPE 形态，可在该固定坐标系里做
PCA/各向异性整形（KIVI、OSCAR，其校准脚本自注 "Dump post-RoPE Q/K/V"）。

## 二、三管线坐标系审计（as-evaluated，全部固定）

判断标准 = **量化 KV 生命周期内读取旋转是否唯一**。逐一验代码：

- **LC：固定**——条件窗 KV 段首 prefill+量化，段内所有步用同一套 grid 位置
  读取；下一段旧 cache 丢弃重建。生到死一种旋转；
- **SF：固定**——原生路径 rope 写入时按**绝对帧索引**施加
  （[causal_model.py#L222-L226](../../experiments/Self-Forcing/wan/modules/causal_model.py#L222-L226)，缓存 post-RoPE key），
  滑窗只截断读取范围、不重锚定——**滑窗注意力不影响 RoPE 编码**。QVG fork 的
  pre-RoPE 路径（[#L277-L304](../../experiments/Self-Forcing/wan/modules/causal_model.py#L277-L304)）读取时现转但位置仍绝对
  （且未实现滑窗，超窗 raise），语义不变；
- **HY：也是固定（0717 三次修正，本页原判"最重灾区"被证伪）**——证据链四环：
  1. rope 与 prope 都在**写入前**施加（[arwan…relative_rope.py#L121-L122](../../experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py#L121-L122) 的
     `key_rope = apply_rotary_emb(key, *rotary_emb)`、[#L151](../../experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py#L151) 的 `prope_qkv`），
     cache 存的是 **post-rope‖post-prope** 的 256 维打包；
  2. 读取直接 `chunk(2)` 拆半拼接（[#L165-L176](../../experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py#L165-L176)），**从不重转**；
  3. 写入只追加尾部 `write(old_end, total, new_k[:,:,old_end:])`（[#L182-L185](../../experiments/HY-WorldPlay/wan/models/dits/arwan_w_action_w_mem_relative_rope.py#L182-L185)），
     旧 token（含已量化的）永不被覆盖——每 chunk 重跑的 cache forward 里旧帧
     的重算结果直接丢弃；
  4. 位置 = 窗口槽位（`current_start=0, current_end=len(selected)*880`），但 QVG
     fork 的连续性 assert（[pipeline…relative_rope.py#L1076-L1081](../../experiments/HY-WorldPlay/wan/inference/pipeline_wan_w_mem_relative_rope.py#L1076-L1081)，注释自证
     "so that old quantized tokens remain valid"）强制 selected 必须是 0..N-1
     前缀 → **槽位 ≡ 全局帧号，位置永不变**。且 paper 协议 run 传
     `--memory_frames 48` = 全部 48 latent（pod_run_paperspeed.sh#L65），FOV 检索
     分支触发条件 `current_frame_idx ≥ 48` 永不满足——**检索重排在我们全部评测
     中从未执行过一次**（若执行，assert 会当场崩溃）。

**讽刺反转**：HY 恰恰是三模型中唯一量化 **post-transform** K 的（LC/SF 量化
pre-RoPE、读取时确定性重转）。"pre-RoPE caching 是为覆盖动态坐标系"的说法
不成立——三条管线的量化 K 语义全部等价于 LLM 情形。

## 三、"relative rope" 是设计意图，不是评测现实

文件名里的 relative 指上游 HY 的 reconstituted context memory 设计（paper
§3.3）：检索选中的老帧应按**当次窗口内相对序号**重新编码——那才是真动态坐标系。
但 QVG fork 的"量化一次 + cache 只追加"结构上要求前缀选择，用 assert 把这条路
挡死了。**若未来在真检索配置下评测/部署（memory_frames < n_latent），本节的
动态坐标系问题会真实出现**——届时各向异性整形的旋转搅拌论证（写入存
R_s·(k+ε)，读取需 R_s'·k，误差方向被 R_s'R_s⁻¹ 混合）重新生效。这是配置边界，
必须随协议声明。

## 四、证伪之后：什么还站着，什么倒了

**站着的（纯经验规律，与机制归因解耦）**：
1. 固定基上的各向异性误差整形（KLT/NF 格/白化/按维比特）在生成端失败，HY 最惨
   （16.6）、LC 轻伤（30.9）——5 次张量-生成脱钩实录不变；
2. 各向同性方案存活（QuaRot LC 30.38、均匀残差格）；
3. PCA 减法安全（Budget-PCA 三关全胜，结果不受本次修正影响——它本来就没依赖
   坐标系假设，HY 上它实际作用于 post-rope‖post-prope 数据）；
4. 时间/token 轴预算分配有效（N19）。

**倒了的（机制归因）**：上述失败**不能**再归因于"读取旋转动态"——as-evaluated
坐标系全固定，OSCAR 的前提在这三条管线里其实并未被违反。真实机制重新开放，
现存候选（均未证实）：
- **闭环反馈累积**：各向异性整形把残差误差集中到低方差方向，其承载的细节丢失
  随自回归逐步复利（N20 仅在 SF 700f 长时程掉线、V-KLT 闭环毒性支持此说）；
- **度量错位**：K 自协方差 ≠ attention 读取度量（q 加权 + softmax 非线性）；
  注意 attn_metric 验证失败说明简单的 q 加权修正也不够；
- **跨 token 位置混合**：单一固定基拟合的是不同位置旋转混合后的平均结构，
  逐 token 的重要方向随位置旋转而变——HY 的 prope 半区相机变换逐帧剧烈变化，
  或可解释其最惨（这也与 prope 半区秩归零有效自洽）；
- **prope 半区的时变读取度量**（复核代理补充的精化）：key 固定 ≠ 度量固定——
  query 侧 prope 变换跟**当前相机**走，同一存储 key 被不同读取以不同方向敏感度
  探测。坐标系固定但读取度量逐读取变化，这在 prope 半区上部分恢复了各向异性
  整形的失败论证（rope 半区不适用：q 侧 rope 只依赖 q 自己的位置，对固定 k 的
  探测方向族是确定的）。**靶向验证实验（待做）**：只对 rope 半区做 KLT、prope
  半区留均匀格——若仍崩，说明死因在闭环/度量错位而非 prope；若不崩，时变度量
  说得到支持。

## 五、修正史（三问三改，全部来自用户追问）

1. 初版：三模型皆动态，HY 最重、LC 每段重锚定、SF 随窗重排；
2. 一次修正（LC）：LC 量化 cache 段内生死、位置固定 → LC ≈ LLM；
3. 二次修正（SF）：滑窗只截断不重锚定、rope 绝对位置写入时施加 → SF 固定；
4. 三次修正（HY，本次）：post-rope 缓存 + 追加写入 + 前缀 assert + 检索分支
   从未执行 → HY 固定。理论作为机制解释整体证伪，降级为"设计意图 vs 评测
   现实"的配置边界分析（§三）。

**教训**：机制理论必须逐行验证"实际执行的代码路径 + 实际使用的配置"，
文件名/设计文档/paper 描述都不算数。
