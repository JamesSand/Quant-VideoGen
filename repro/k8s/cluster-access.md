# cluster-access — 接入 research-common H100 集群

开 pod 前先确认下面几个前提都满足。踩过的坑:kubectl 会随 pod 重建消失、pod 自带的
default SA 权限几乎为零。

## 1. 凭证(kubeconfig)

- **路径**:`/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100`
  (在 `home-zhizhousha` PVC 上,持久)。
- **身份**:context `zhizhousha`,cluster `research-common`,user `zhizhousha`,默认 namespace `zhizhousha`。
- **权限**(`kubectl auth can-i`):
  | 操作 | 结果 |
  |---|---|
  | `create/delete pods`、`create jobs`(自己的 `zhizhousha` ns) | ✅ yes |
  | `list pods -A`(跨 namespace 只读) | ✅ yes |
  | 在别人的 ns(如 `charlie`)`create`/读 `pods/log` | ❌ no |

> ⚠️ **pod 自带的 in-cluster default SA(`serviceaccount:zhizhousha:default`)权限极小**——
> 连 `create pods` 都是 `no`,只能对自己做 SelfSubjectAccessReview。所以**任何 kubectl 操作都必须
> 显式用上面那把 kubeconfig**,不能靠容器默认凭证。

## 2. kubectl(易失!已做持久化)

**坑**:dev pod 的 rootfs 是 ephemeral 的。手动装在 `/usr/local/bin/kubectl` 的二进制会在
**pod 重建时被抹掉**(本 pod `zhizhousha-dev-8gpu-ssh*` 于 2026-07-16 07:28 重建后 kubectl 就没了)。

**已做的修复(扛重建)**:
- kubectl 拷到 **`~/bin/kubectl`**(在 `home-zhizhousha` RWX PVC 上 → 重建后仍在)。
- `~/.bashenv`(非交互 shell,pod 的 `BASH_ENV` 指向它)和 `~/.bashrc`(交互 shell)里加了一段:
  ```bash
  # >>> k8s dev env (added by claude) >>>
  export PATH="$HOME/bin:$PATH"
  export KUBECONFIG=/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100
  # <<< k8s dev env (added by claude) <<<
  ```
- **退路二进制**:`/shared/syanamandra/bin/kubectl`(别人 PVC 里,v1.29.6)。

**验证(模拟 skill 的非交互调用)**:
```bash
env BASH_ENV=$HOME/.bashenv bash -c 'which kubectl; kubectl auth can-i create pods'
# 期望: /home/zhizhousha/bin/kubectl  +  yes
```

## 3. 存储(PVC)

dev pod 的两个卷,均已 `Bound`:

| PVC | 挂载点 | 容量 / 模式 / StorageClass |
|---|---|---|
| `home-zhizhousha` | `/home/zhizhousha` | 10Ti / RWX / `weka-data` |
| `shared-data` | `/shared` | 100Ti / RWX / `shared-weka` |

> `/home/zhizhousha` 和 `/shared` 跨 pod 重建持久;容器根文件系统里的其他改动(装的包、
> `/usr/local/bin` 等)**不持久**。要留东西就放这两个卷。

## 4. 准入(admission)

集群跑着 Kyverno。实测:`privileged: true` + `nvidia.com/gpu: 8` 的 dev pod
`kubectl apply --dry-run=server` **能通过**,当前没有策略拦截规范写法的 pod。
(将来若上策略封"特权+不申请 GPU"的账外 pod,受影响的是 `pod-config-comparison.md` §绕过那种写法,不是你。)
