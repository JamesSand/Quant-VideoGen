# k8s/ — research-common H100 集群操作知识库

在共享的 research-common H100 (Kubernetes) 集群上跑 QVG 实验时的**操作层**记录:
怎么接入、怎么诊断"卡被占满却看不到进程"、怎么找真正空闲的节点、以及规范开 pod 的姿势。

> 与 `../backup/pods/`(逐实验的 pod manifest)、`../backup/scripts/pod_run_*.sh`(逐实验的
> 提交脚本)互补:那边是"跑某个具体实验",这里是"集群怎么用、出问题怎么查"。

## 什么时候读哪份

| 文件 | 定位 |
|---|---|
| [`cluster-access.md`](cluster-access.md) | 接入:kubeconfig 在哪、身份/权限、kubectl 为何会消失+已做的持久化修复、PVC。**开 pod 前先确认这份的前提都满足。** |
| [`gpu-occupancy-diagnosis.md`](gpu-occupancy-diagnosis.md) | 诊断:`nvidia-smi` 显示卡满、却 "No running processes found" 的排查手册(附 charlie/GLM-5.2 实例) |
| [`pod-config-comparison.md`](pod-config-comparison.md) | 规范 vs 绕过:合法申请(走 device plugin)与账外占卡(privileged+不申请 GPU)的逐字段对比,以及为什么别学后者 |
| [`find-free-nodes.md`](find-free-nodes.md) | 选节点:扫全集群找"记账+账外皆空"的整机的方法 + 脚本 + 快照结果 |
| [`dev-8gpu-vetted.yaml`](dev-8gpu-vetted.yaml) | 现成的 8 卡 dev pod manifest,`nodeAffinity` 只落到已核验的空闲节点 |

## 快速开始(环境已配好,新 shell 直接可用)

```bash
# kubectl 已装在 ~/bin(home PVC,扛 pod 重建);KUBECONFIG 已在 ~/.bashenv/.bashrc 导出
kubectl config view --minify -o jsonpath='ns={.contexts[0].context.namespace}\n'  # -> zhizhousha
kubectl auth can-i create pods                                                     # -> yes

# 开一台已核验空闲的 8 卡整机
kubectl apply -f dev-8gpu-vetted.yaml

# 查自己的 pod / 进去
kubectl get pods -n zhizhousha -o wide
kubectl exec -it <pod> -n zhizhousha -- bash
```

若 `kubectl` 找不到(极端情况),退路二进制:`/shared/syanamandra/bin/kubectl`;
凭证:`/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100`。

## 一句话背景

2026-07 排查一次"8 卡 H100 全满但 pod 内 `nvidia-smi` 看不到进程"时整理:根因是**另一个用户的
特权 pod 绕过了 GPU 调度器**(privileged + `runtimeClassName: nvidia` + 不申请 `nvidia.com/gpu`),
物理占满了调度器记账给我的整机。详见 `gpu-occupancy-diagnosis.md`。
