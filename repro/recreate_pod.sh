#!/bin/bash
# Recreate a failed config pod, excluding known-bad nodes via nodeAffinity.
# Usage: recreate_pod.sh <tag> [bad_node_to_add]
set -eu
TAG=${1:?tag}
REPRO=/home/zhizhousha/workspace/video-project/Quant-VideoGen/repro
export KUBECONFIG=/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/kubeconfig/research-common-h100
export PATH=/home/zhizhousha/workspace/low-precision-project/k8s-from-h100-pod/bin:$PATH

[ $# -ge 2 ] && echo "$2" >> $REPRO/race/bad_nodes.txt
touch $REPRO/race/bad_nodes.txt
BAD=$(sort -u $REPRO/race/bad_nodes.txt | sed 's/^/                - /')

PODNAME=zhizhousha-qvg-$(echo $TAG | tr '_' '-')
kubectl delete pod $PODNAME --ignore-not-found --wait=true

# regenerate manifest with NODE_NAME env + anti-affinity for bad nodes
python3 - "$TAG" <<'EOF'
import sys, yaml, pathlib
tag = sys.argv[1]
repro = pathlib.Path('/home/zhizhousha/workspace/video-project/Quant-VideoGen/repro')
mf = yaml.safe_load((repro / 'pods' / f'{tag}.yaml').read_text())
bad = [l.strip() for l in (repro / 'race' / 'bad_nodes.txt').read_text().splitlines() if l.strip()]
spec = mf['spec']
c = spec['containers'][0]
envs = [e for e in c.get('env', []) if e.get('name') != 'NODE_NAME']
envs.append({'name': 'NODE_NAME', 'valueFrom': {'fieldRef': {'fieldPath': 'spec.nodeName'}}})
c['env'] = envs
if bad:
    spec['affinity'] = {'nodeAffinity': {'requiredDuringSchedulingIgnoredDuringExecution': {
        'nodeSelectorTerms': [{'matchExpressions': [
            {'key': 'kubernetes.io/hostname', 'operator': 'NotIn', 'values': bad}]}]}}}
out = repro / 'pods' / f'{tag}.yaml'
out.write_text(yaml.safe_dump(mf, sort_keys=False))
print(f'rewrote {out} excluding {len(bad)} bad node(s)')
EOF

# clear previous result so the watcher waits for the fresh one
rm -f $REPRO/race/result_$TAG.txt
kubectl apply -f $REPRO/pods/$TAG.yaml
