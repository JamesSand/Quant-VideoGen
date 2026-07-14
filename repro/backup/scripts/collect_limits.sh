#!/bin/bash
# Collect limit-push videos into a self-describing folder: limit_videos/
set -u
cd /home/zhizhousha/workspace/video-project/Quant-VideoGen
mkdir -p limit_videos

copy_one() {  # $1 src, $2 dst-name
  if [ -f "$1" ]; then
    cp -f "$1" "limit_videos/$2"
    echo "collected: $2  ($(du -h "limit_videos/$2" | cut -f1))"
  else
    echo "MISSING: $1"
  fi
}

frames() { python - "$1" <<'PY' 2>/dev/null
import sys, imageio
print(sum(1 for _ in imageio.get_reader(sys.argv[1])))
PY
}

LCB=results/limits/lc_bf16/1-0/segment_70.mp4
LCI=results/limits/lc_int2/1-0/segment_70.mp4
SFB=results/limits/sf_bf16/0-0_ema.mp4
SFI=results/limits/sf_int2/0-0_ema.mp4
HYB=results/limits/hy_bf16/0-0.mp4
HYI=results/limits/hy_int2/0-0.mp4

copy_one $LCB longcat_bf16_$(frames $LCB 2>/dev/null || echo x)frames.mp4
copy_one $LCI longcat_qvg-int2_$(frames $LCI 2>/dev/null || echo x)frames.mp4
copy_one $SFB selfforcing_bf16_$(frames $SFB 2>/dev/null || echo x)frames.mp4
copy_one $SFI selfforcing_qvg-int2_$(frames $SFI 2>/dev/null || echo x)frames.mp4
copy_one $HYB hyworldplay_bf16_$(frames $HYB 2>/dev/null || echo x)frames.mp4
copy_one $HYI hyworldplay_qvg-int2_$(frames $HYI 2>/dev/null || echo x)frames.mp4

ls -la limit_videos/
