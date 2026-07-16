# pod-config-comparison — 规范申请 vs 账外占卡

对比两类 8 卡 pod:**规范写法**(走 device plugin,受调度器保护,= 我们该用的)
与 **账外占卡**(privileged + 不申请 GPU,= charlie 那种,别学)。理解差异能避免重蹈
"被 double-booking"的覆辙。背景见 [`gpu-occupancy-diagnosis.md`](gpu-occupancy-diagnosis.md)。

## TL;DR

- 两类 pod 常常**大部分字段相同**:`privileged: true`、`runtimeClassName: nvidia`、
  `hostNetwork/hostIPC`、nvcr pytorch 镜像。
- **唯一的决定性差异**:规范写法在 `resources` 里写了 **`nvidia.com/gpu: 8`**;账外写法**没写**,
  改用手动 `CUDA_VISIBLE_DEVICES=0-7` + 靠 privileged/nvidia-runtime 直接抓物理卡。
- 后果:账外 pod 不被调度器记账 → 同一批卡又被派给别人 → double-booking。**这就是坑人的根因。**

## 逐字段对比

| 维度 | 规范(我们的 dev pod) | 账外(charlie glm52bo-worker) | 说明 |
|---|---|---|---|
| **GPU 申请** | **`nvidia.com/gpu: 8`(requests+limits)** | **无** | ★ 决定性差异 |
| 拿卡机制 | device plugin 经 CDI 注入分配到的卡;`NVIDIA_VISIBLE_DEVICES=/var/run/nvidia-container-devices` | privileged 直接开 `/dev/nvidia*` + 镜像默认 `NVIDIA_VISIBLE_DEVICES=all` | 前者被记账,后者不被 |
| `CUDA_VISIBLE_DEVICES` | 不设(由分配决定) | 手动 `0-7` | |
| `privileged` | true(仅用于 useradd/sudo,对拿卡其实多余) | true | 相同 |
| `runtimeClassName` | nvidia | nvidia | 相同 |
| `hostNetwork` / `hostIPC` | true / true | true / true | 相同 |
| `hostPID` | 未设 | true | 多机/调试需要;也是其进程在 host 可见、pod 内不可见的原因 |
| `nodeSelector` | `node-group: default`(+ 本仓库加了 affinity 限定空闲节点) | 点名绑具体 hostname | |
| `tolerations` | 无 | `operator: Exists`(容忍所有 taint) | 账外写法可能落到隔离节点 |
| 资源 | cpu 96 / mem 700Gi | cpu 16–32 / mem 100–200Gi | |
| Secret | 走 `secretKeyRef` / 环境注入 | ⚠ `HF_TOKEN` **明文硬编码**(泄露风险,值已脱敏) | 见下 |

## "改成账外那样"要动什么(⚠ 反模式,别做)

在规范 YAML 上:删掉 `resources` 里的 `nvidia.com/gpu`(requests+limits 两处)、加
`env: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`、`nodeSelector` 绑到目标节点(privileged 已有)。
**后果**:调度器以为该节点空 → 别人被派上来 / 你自己落到"看似空实则被账外占满"的节点,
原地重演本次事故;同节点负载互相踩显存;将来准入策略上线会被直接拒。

## 正确做法

- **单机满整机(本仓库场景)**:保持 `nvidia.com/gpu: 8`,让调度器保证独占;再用 `nodeAffinity`
  只落到已核验空闲的节点(见 [`dev-8gpu-vetted.yaml`](dev-8gpu-vetted.yaml) 与 [`find-free-nodes.md`](find-free-nodes.md))。
- **多机训练**:可借账外写法的 `hostNetwork/hostPID/NCCL_*` + master/worker 结构,
  但**每个节点的 pod 仍各自申请 `nvidia.com/gpu: 8`**——既能跨机通信,又不破坏记账。
- **缩权限**:单机场景可去掉 `privileged`(只为 useradd/sudo 用),GPU 照走 device plugin。

## 安全提醒

charlie 的 worker pod 把 `HF_TOKEN` 明文写在 env(本文已脱敏为 `hf_****`)。任何能
`get pod -n charlie` 的人都能读到明文 = 泄露。规范做法用 `secretKeyRef` 引用 K8s Secret。
建议连同"GPU 记账绕过"一起反馈给 charlie 或集群管理员。
