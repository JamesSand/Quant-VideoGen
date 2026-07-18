#!/usr/bin/env python3
"""Phase 0 smoke check: metrics + VBench end-to-end on prompt#1 / seed0 videos."""
import os, subprocess, sys, shutil, glob

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
PY = ".venv/bin/python"
NPZD = "repro/0718/npz"; os.makedirs(NPZD, exist_ok=True)
os.makedirs("repro/backup/protosearch", exist_ok=True)

def count_frames(p):
    import imageio.v3 as iio
    return sum(1 for _ in iio.imiter(p, plugin="pyav"))

def ref_metrics(model, ref, tests, n):
    r = subprocess.run([PY, "repro/backup/scripts/sf_ref_metrics.py", ref, str(n)] + [t for t, _ in tests],
                       capture_output=True, text=True)
    print(r.stdout); print(r.stderr[-800:] if r.returncode else "", file=sys.stderr)
    for t, name in tests:  # rescue npz from tag collision
        tag = t.split("/")[-2]
        src = f"repro/backup/protosearch/sf_{tag}.npz"
        if os.path.exists(src):
            shutil.move(src, f"{NPZD}/{model}_{name}.npz")
    return r.returncode

def first(pat):
    g = sorted(glob.glob(pat))
    return g[0] if g else None

lc_ref = first("results/multiprompt/lc/bf16_rep0/p1/*/segment_1.mp4")
lc_qvg = first("results/multiprompt/lc/qvg_rep0/p1/*/segment_1.mp4")
lc_pca = first("results/multiprompt/lc/pca_rep0/p1/*/segment_1.mp4")
sf_ref = first("results/multiprompt/sf/bf16_rep0_f180/p1/*.mp4")
sf_qvg = first("results/multiprompt/sf/qvg_rep0_f180/p1/*.mp4")
sf_pca = first("results/multiprompt/sf/pca_rep0_f180/p1/*.mp4")
hy_ref = "results/multiprompt/hy/bf16_s0/0-0.mp4"
hy_qvg = "results/multiprompt/hy/qvg_s0/0-0.mp4"
hy_pca = "results/multiprompt/hy/pca_s0/0-0.mp4"

fails = []
print("== LC f93 metrics ==")
if lc_ref and lc_qvg and lc_pca:
    n = count_frames(lc_ref); print(f"LC segment frames: {n}")
    if ref_metrics("lc_p1", lc_ref, [(lc_qvg, "qvg_p1"), (lc_pca, "pca_p1")], n): fails.append("lc_metrics")
else:
    fails.append("lc_missing"); print("MISSING LC videos", lc_ref, lc_qvg, lc_pca)

print("== HY full-video metrics ==")
if all(os.path.exists(p) for p in (hy_ref, hy_qvg, hy_pca)):
    n = count_frames(hy_ref); print(f"HY frames: {n}")
    if ref_metrics("hy_s0", hy_ref, [(hy_qvg, "qvg_s0"), (hy_pca, "pca_s0")], n): fails.append("hy_metrics")
else:
    fails.append("hy_missing"); print("MISSING HY videos")

print("== VBench (all three models, arm-wise) ==")
vids = [v for v in (sf_ref, sf_qvg, sf_pca, lc_ref, lc_pca, hy_ref, hy_pca) if v and os.path.exists(v)]
if len(vids) < 7: fails.append(f"vbench_missing_{7-len(vids)}")
r = subprocess.run([PY, "repro/backup/scripts/vbench_iq.py"] + vids, capture_output=True, text=True)
print(r.stdout)
if r.returncode: fails.append("vbench"); print(r.stderr[-800:], file=sys.stderr)

print("PHASE0CHECK", "FAIL:" + ",".join(fails) if fails else "ALL-OK")
