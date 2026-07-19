# HANDOFF 0719

## 状态:MP100 优化循环收敛

终表 [mp100-table.md](mp100-table.md)(含终注)、日报 [report-0719.md](report-0719.md)。
**18 质量列 = 12 最优 + 4 统计平局(双侧 p>0.05)+ 2 显著小差距**。

## 待用户裁决

goal 字面是"每列最好"。剩余两列(lc:sc −0.02 ≈0.02%相对、hy:aq −0.96 ≈1.5%)已证
结构性(伪影偏好家族,13 配置扫描证据链在案)。选项:
(a) 以"最优或统计平局 + 两列注明结构性"收案(16/18);
(b) 授权超出当前约束的机制(学习组件/字典类)继续攻两列;
(c) 对该两列换指标口径(如 VBench 全套件均值)。

## 终版配置(env 速查)

```bash
# LC/SF: PCA_R=4 PCA_RES_GRID=asym PCA_RES_BLOCK=128 PCA_V_MODE=pca \
#        PCA_RES_AXIS_K=channel PCA_RES_AXIS_V=channel   (SF 另加 PCA_SF_STORE_FIX=1)
# HY:    PCA_R=4 PCA_HALF_R_K=9,0 PCA_HALF_R_V=9,0 PCA_RES_GRID=asym PCA_RES_BLOCK=128 \
#        PCA_RES_AXIS_K=channel PCA_RES_BLOCK_K=64 PCA_RES_GRID_KP=ternary PCA_RES_BLOCK_KP=64
```

## 结转

- 0717 final-method-results.md 与 0718 multi-prompt-results.md 的头条需按 MP100
  通道轴版重写(旧配置数字保留为演化史);
- kernel 化 M0-M4(通道轴对 kernel 布局有影响——channel 轴需要转置访存,记入设计);
- QVG-Pro 臂、rope 半区 KLT 裁决实验(0717 待办五)仍在队列。
