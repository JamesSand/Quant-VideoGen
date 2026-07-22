#!/usr/bin/env python3
"""SF qvg-nom 判决评分:qvgprot/qvgproc ×100 的 vbench4(700 帧窗,与 mp100 同协议)。
Usage: score_sf_qvgpro.py <shard> <nshards>   (one per GPU, CUDA_VISIBLE_DEVICES 已设)
Cache: repro/0718/npz/vbench4_sfq_shard<i>.json
"""
import os, sys, glob, subprocess

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
SHARD, NSH = int(sys.argv[1]), int(sys.argv[2])
PY = ".venv/bin/python"
os.environ["VBENCH4_CACHE"] = f"repro/0718/npz/vbench4_sfq_shard{SHARD}.json"

vids = []
for arm in ["qvgprot", "qvgproc"]:
    for p in range(1, 101):
        g = glob.glob(f"results/multiprompt/mp100/sf/{arm}_rep0_f180/p{p}/*.mp4")
        if g:
            vids.append(g[0])
mine = [v for i, v in enumerate(vids) if i % NSH == SHARD]
print(f"shard {SHARD}/{NSH}: {len(mine)} videos", flush=True)
if mine:
    subprocess.run([PY, "repro/0718/scripts/vbench4.py", *mine, "--max-frames", "700"],
                   env=os.environ, check=False)
print("SHARD DONE", flush=True)
