# gpu-occupancy-diagnosis — "卡满却看不到进程" 排查手册

## 症状

pod 内 `nvidia-smi`:显存被占到 ~90%、有算力/功耗,但进程表是 **`No running processes found`**。

## 根因(两层)

1. **PID namespace 隔离**:`nvidia-smi` 的显存/利用率是从物理 GPU 全局读的(整机口径),
   但进程列表要把 GPU 上的 PID 映射到**当前 PID namespace**里。占卡进程若在**别的 pod**
   的 namespace 里,你就只看得到"卡被占"、看不到是谁。这本身是正常容器隔离,不是 bug。
2. **有 pod 绕过了 GPU 调度器**:某个 `privileged: true` + `runtimeClassName: nvidia` +
   **不申请 `nvidia.com/gpu`** 的 pod,物理上吃了整机 8 张卡,但调度器不知道 → 把同一批卡
   记账分配给了你的 pod → **double-booking**。详见 [`pod-config-comparison.md`](pod-config-comparison.md)。

## 排查步骤

```bash
export KUBECONFIG=/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100

# 0) 先看物理占用(整机口径),确认是"活跃占用"还是"僵尸显存"(util=0 才是泄漏)
nvidia-smi

# 1) 我在哪台物理节点?
kubectl get pod <mypod> -n zhizhousha -o custom-columns=NODE:.spec.nodeName
#   或在 pod 内: cat /etc/hostname  (dev pod 的 name 常与 nodeName 无关,以 kubectl 为准)

# 2) 这台节点上所有 namespace 的 pod + 各自 GPU 申请
NODE=research-common-h100-0XX.cloud.together.ai
kubectl get pods -A --field-selector spec.nodeName=$NODE \
  -o custom-columns=NS:.metadata.namespace,NAME:.metadata.name,STATUS:.status.phase,GPU:'.spec.containers[*].resources.limits.nvidia\.com/gpu'

# 3) 找"账外占卡"嫌疑:privileged / nvidia runtime / 带 CUDA_VISIBLE_DEVICES 但不申请 GPU 的 pod
kubectl get pod <suspect> -n <ns> -o json | python3 -c "
import json,sys; p=json.load(sys.stdin); s=p['spec']
print('runtimeClassName:', s.get('runtimeClassName'), 'hostPID:', s.get('hostPID'))
for c in s['containers']:
    print(c['name'],'privileged=',(c.get('securityContext') or {}).get('privileged'),
          'gpu_req=', (c.get('resources',{}).get('limits',{}) or {}).get('nvidia.com/gpu'))
    print('  cmd:', ' '.join((c.get('command') or [])+(c.get('args') or []))[:200])
"
```

> 只能 `list pods -A` 拿到 spec,**读不了别人 ns 的 `pods/log`**(RBAC)。定位到"命令行 + 目录"
> 一般够了。要看对方进程,得在**宿主机**上 `nvidia-smi`(host PID namespace 能看到全部)。

## 实例:charlie 的 GLM-5.2 sglang(2026-07)

- 占卡者:`charlie/glm52bo-worker-*`,命令是
  `python -m sglang.launch_server --model-path zai-org/GLM-5.2-FP8 --tensor-parallel-size 16
  --nnodes 2 --node-rank 1 --mem-fraction-static 0.85 ...`(2 节点 TP16 推理 worker)。
- `--mem-fraction-static 0.85` 正好对上观测到的每卡 ~90% 显存;sglang 推理对上中等 util + ~160W。
- 配置:`privileged: true`、`runtimeClassName: nvidia`、`CUDA_VISIBLE_DEVICES=0-7`、`hostPID/hostIPC/hostNetwork: true`、
  **`resources` 里无 `nvidia.com/gpu`**、`nodeSelector` 点名绑到具体节点。
- 代码目录:`/home/charlie/CoQuant/.RUD/hybridmodel-testing/work/CoQuant`(conda env `oscar`)。
- 时间线:07-10 在节点 073;到 07-16 已迁到 086(pod 后缀 `-jx6pz`→`-wcff5`)。**集群状态天天变,
  每次都要现查,别记死某台。**

## 结论口径

- `nvidia-smi` 有占用、`util>0` → 真在跑,不是僵尸显存。
- pod 内看不到进程 = 占卡者在别的 namespace(正常隔离)。
- 若该节点"记账为 0 却物理被占" = 有账外特权 pod(反模式);"记账已满" = 正常被别人合法占用。
- 想要干净整机:用 [`find-free-nodes.md`](find-free-nodes.md) 的口径选"记账+账外皆空"的节点。
