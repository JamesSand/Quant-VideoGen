# find-free-nodes — 扫出"真正空闲"的整机

调度器的记账**只认走 device plugin 申请的卡**,看不到账外特权 pod(见
[`pod-config-comparison.md`](pod-config-comparison.md))。所以"真正空闲"要同时满足两条:
**(a) 记账为 0** 且 **(b) 没有账外占卡嫌疑 pod**;再加上 Ready / 可调度 / 未隔离 / 有满 8 卡。

## 方法

抓两份 JSON,离线聚合:

```bash
export KUBECONFIG=/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100
kubectl get nodes -o json > /tmp/nodes.json
kubectl get pods -A -o json > /tmp/pods.json
python3 find_free_nodes.py    # 见下
```

判据:
- 只看 `node-group=default` 且名字含 `h100` 的节点;`allocatable nvidia.com/gpu >= 8`(<8 的剔除,装不下 8 卡 pod)。
- **记账占用** = 该节点上所有非 Succeeded/Failed pod 的 `nvidia.com/gpu` 请求之和,要 == 0。
- **账外嫌疑** = 非基础设施 namespace 的 pod,`privileged` 或 `runtimeClassName: nvidia`
  或带 `CUDA_VISIBLE_DEVICES`/`NVIDIA_VISIBLE_DEVICES`,但请求 0 GPU —— 要 == 0。
- Ready=True、`spec.unschedulable != true`、无 quarantine taint/label。

> 账外检测**偏保守(宁可多报)**:CSI canary、gpu-zombie-kill 之类会被误标为嫌疑,
> 但这只会让你多跳过几台,不会把脏节点当干净。它**不**验证物理 `nvidia-smi`(跨节点跑不了),
> 真要 100% 确认得调度上去后本地 `nvidia-smi`,或查 DCGM/Prometheus。

## 脚本 `find_free_nodes.py`

```python
import json
nodes=json.load(open('/tmp/nodes.json'))['items']
pods=json.load(open('/tmp/pods.json'))['items']
INFRA={'kube-system','gpu-operator','nvsentinel','csi-wekafs','network-operator','node-problem-detector',
 'observability','kube-prometheus-stack','metallb-system','cert-manager','kuberay','kubeflow','gpud','kyverno',
 'registry','default','local-path-storage','monitoring','calico-system','tigera-operator'}
def gpu_req(p):
    t=0
    for c in p['spec'].get('containers',[]):
        r=c.get('resources',{})
        v=r.get('requests',{}).get('nvidia.com/gpu') or r.get('limits',{}).get('nvidia.com/gpu')
        if v: t+=int(v)
    return t
def is_bypass(p):
    if p['metadata']['namespace'] in INFRA: return False
    if gpu_req(p)>0: return False
    rc=p['spec'].get('runtimeClassName')
    for c in p['spec'].get('containers',[]):
        sc=c.get('securityContext',{}) or {}
        envs={e.get('name') for e in c.get('env',[])}
        if sc.get('privileged') or rc=='nvidia' or 'NVIDIA_VISIBLE_DEVICES' in envs or 'CUDA_VISIBLE_DEVICES' in envs:
            return True
    return False
acct={}; byp={}
for p in pods:
    if p['status'].get('phase') in ('Succeeded','Failed'): continue
    n=p['spec'].get('nodeName')
    if not n: continue
    acct[n]=acct.get(n,0)+gpu_req(p)
    if is_bypass(p): byp.setdefault(n,[]).append(f"{p['metadata']['namespace']}/{p['metadata']['name']}")
free=[]
for nd in nodes:
    name=nd['metadata']['name']; labels=nd['metadata'].get('labels',{})
    if 'h100' not in name or labels.get('node-group')!='default': continue
    alloc=int(nd['status'].get('allocatable',{}).get('nvidia.com/gpu',0) or 0)
    if alloc<8: continue
    ready=any(c['type']=='Ready' and c['status']=='True' for c in nd['status'].get('conditions',[]))
    sched=not nd['spec'].get('unschedulable',False)
    taints=nd['spec'].get('taints',[]) or []
    quar=any('quarantine' in t.get('key','') or 'nvsentinel' in t.get('key','') for t in taints) or any('quarantine' in k for k in labels)
    if acct.get(name,0)==0 and not byp.get(name) and ready and sched and not quar:
        free.append(name)
print(f"{len(free)} free 8-GPU nodes:")
for n in sorted(free): print(" ", n)
```

## 快照 @ 2026-07-16 ~21:5x

- 共 **51** 台 `node-group=default` 的 H100;**12** 台判定空闲。
- 满 8 卡且空闲(可直接用):
  `005 050 051 064 068 073 079 097 099 110 114`(节点名 `research-common-h100-0XX.cloud.together.ai`)。
- 特例:`058` 只有 7 张 allocatable(疑似坏一张),已从 8 卡候选剔除。
- 账外嫌疑命中:`086`(charlie/glm52bo-worker,真占卡)、`040`(zhizhousha/csi-canary,误报)、
  `096`(apanda/gpu-zombie-kill,误报)。
- 我当时的 dev pod 在 `059`,8 卡合法记账、`nvidia-smi` 实测全 0% —— 已是干净整机。

> ⚠️ 这是**时间点快照**,集群天天变。隔一阵要用就重扫。好在
> [`dev-8gpu-vetted.yaml`](dev-8gpu-vetted.yaml) 里的 `nvidia.com/gpu: 8` 会让调度器兜底,
> 不会误落到记账已满的节点(只是可能仍撞上账外占用者,所以才叠加 `nodeAffinity` 白名单)。
