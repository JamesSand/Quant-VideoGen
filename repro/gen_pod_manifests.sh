#!/bin/bash
# Generate one 1-GPU pod manifest per baseline config into repro/pods/.
set -eu
OUTDIR=/home/zhizhousha/workspace/video-project/Quant-VideoGen/repro/pods
mkdir -p $OUTDIR

CONFIGS=(
  "lc int2 16 0 1 lc_quarot_int2_asym_b16"
  "lc int4 16 0 1 lc_quarot_int4_asym_b16"
  "lc int2 16 1 1 lc_quarot_int2_sym_b16"
  "lc int4 16 1 1 lc_quarot_int4_sym_b16"
  "lc int2 128 0 1 lc_quarot_int2_asym_b128"
  "lc int4 128 0 1 lc_quarot_int4_asym_b128"
  "lc int2 16 0 0 lc_rtn_int2_b16"
  "lc int4 16 0 0 lc_rtn_int4_b16"
  "hy int2 16 0 1 hy_quarot_int2_asym_b16"
  "hy int4 16 0 1 hy_quarot_int4_asym_b16"
  "hy int2 16 0 0 hy_rtn_int2_b16"
  "hy int4 16 0 0 hy_rtn_int4_b16"
)

for cfg in "${CONFIGS[@]}"; do
  set -- $cfg
  KIND=$1; BITS=$2; BLOCK=$3; SYM=$4; ROT=$5; TAG=$6
  PODNAME=zhizhousha-qvg-$(echo $TAG | tr '_' '-')
  cat > $OUTDIR/$TAG.yaml <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $PODNAME
  labels:
    owner: zhizhousha
    purpose: qvg-quarot-baseline
spec:
  restartPolicy: Never
  nodeSelector:
    node-group: default
  runtimeClassName: nvidia
  containers:
    - name: main
      image: nvcr.io/nvidia/pytorch:25.11-py3
      command: ["bash", "-lc", "bash /home/zhizhousha/workspace/video-project/Quant-VideoGen/repro/pod_run_one.sh $KIND $BITS $BLOCK $SYM $ROT $TAG"]
      workingDir: /home/zhizhousha
      env:
        - name: HOME
          value: /home/zhizhousha
        - name: HF_HOME
          value: /shared/huggingface
      resources:
        requests: { cpu: "12", memory: 96Gi, nvidia.com/gpu: "1" }
        limits:   { cpu: "12", memory: 96Gi, nvidia.com/gpu: "1" }
      volumeMounts:
        - { name: home, mountPath: /home/zhizhousha }
        - { name: shared, mountPath: /shared }
        - { name: shm, mountPath: /dev/shm }
  volumes:
    - name: home
      persistentVolumeClaim: { claimName: home-zhizhousha }
    - name: shared
      persistentVolumeClaim: { claimName: shared-data }
    - name: shm
      emptyDir: { medium: Memory, sizeLimit: 64Gi }
EOF
  echo "wrote $OUTDIR/$TAG.yaml ($PODNAME)"
done
