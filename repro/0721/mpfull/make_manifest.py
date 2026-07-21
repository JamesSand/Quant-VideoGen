#!/usr/bin/env python3
"""MPFULL 可复核清单:所有生成视频的 sha256 + 尺寸 + 台账快照。
产出:repro/0721/mpfull/manifest.tsv(入 git)+ ledger-snapshot.txt。
"""
import hashlib, os

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
rows = []
for root, _, files in os.walk("results/multiprompt/mpfull"):
    for fn in sorted(files):
        if not fn.endswith(".mp4"):
            continue
        p = os.path.join(root, fn)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        rows.append((p, os.path.getsize(p), h.hexdigest()))
rows.sort()
with open("repro/0721/mpfull/manifest.tsv", "w") as f:
    f.write("path\tbytes\tsha256\n")
    for p, sz, hx in rows:
        f.write(f"{p}\t{sz}\t{hx}\n")
import glob
with open("repro/0721/mpfull/ledger-snapshot.txt", "w") as f:
    for lf in sorted(glob.glob("repro/0718/logs/ledger_*.txt")):
        for line in open(lf):
            if "mpfull" in line:
                f.write(f"{os.path.basename(lf)}: {line}")
print(f"manifest: {len(rows)} videos")
