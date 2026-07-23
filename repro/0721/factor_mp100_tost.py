#!/usr/bin/env python3
"""Apply the preregistered paired non-inferiority gates to the bounded fallback.

The videos are the completed MP100/10-seed Budget-PCA factor-grid campaign.
Each row is paired by prompt (LC/SF) or seed (HY).  A 90% paired Student-t
interval is the TOST-compatible alpha=0.05 decision rule.
"""
import glob
import json
import math
import os

import numpy as np
from scipy.stats import t

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")
NPZ = "repro/0718/npz"

VB = {}
for path in sorted(glob.glob(f"{NPZ}/vbench4_shard*.json")) + [
    f"{NPZ}/vbench4.json"
]:
    if os.path.exists(path):
        with open(path) as f:
            VB.update(json.load(f))

ARMS = {
    "lc": "pcakaxvaxfp8",
    "sf": "pcaa128kaxvaxfp8",
    "hy": "pcav90kpternkaxkb64fp8",
}
MARGINS = {
    "PSNR": 0.10,
    "SSIM": 0.002,
    "LPIPS": 0.003,
    "BC": 0.20,
    "IQ": 0.30,
    "SC": 0.30,
    "AQ": 0.30,
}


def one(pattern):
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def vb(path, max_frames):
    return VB.get(f"{path}::{max_frames}") if path else None


def ref_value(model, index, arm):
    if model == "lc":
        path = f"{NPZ}/mp100_lc_p{index}_{arm}.npz"
        if not os.path.exists(path):
            return None
        z = np.load(path)
        if len(z["psnr"]) <= 93:
            return None
        return np.asarray([z["psnr"][93], z["ssim"][93], z["lpips"][93]])
    path = f"{NPZ}/hy_s{index}_{arm}.npz"
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return np.asarray(
        [z["psnr"][13:].mean(), z["ssim"][13:].mean(), z["lpips"][13:].mean()]
    )


def video(model, index, arm):
    if model == "lc":
        return one(
            f"results/multiprompt/mp100/lc/{arm}_rep0/p{index}/*/segment_1.mp4"
        )
    if model == "sf":
        return one(
            f"results/multiprompt/mp100/sf/{arm}_rep0_f180/p{index}/*.mp4"
        )
    return f"results/multiprompt/hy/{arm}_s{index}/0-{index}.mp4"


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    if n < 2:
        return mean, -math.inf, math.inf
    sem = float(values.std(ddof=1) / math.sqrt(n))
    radius = float(t.ppf(0.95, n - 1) * sem)
    return mean, mean - radius, mean + radius


paired = {model: {metric: [] for metric in MARGINS} for model in ARMS}
for model, candidate in ARMS.items():
    indices = range(1, 101) if model in ("lc", "sf") else range(10)
    for index in indices:
        if model != "sf":
            qref = ref_value(model, index, "qvg")
            cref = ref_value(model, index, candidate)
            if qref is not None and cref is not None:
                paired[model]["PSNR"].append(float(cref[0] - qref[0]))
                paired[model]["SSIM"].append(float(cref[1] - qref[1]))
                # Orient every effect so positive means the candidate is better.
                paired[model]["LPIPS"].append(float(qref[2] - cref[2]))
        qvideo = video(model, index, "qvg")
        cvideo = video(model, index, candidate)
        max_frames = 700 if model == "sf" else 0
        qvb, cvb = vb(qvideo, max_frames), vb(cvideo, max_frames)
        if qvb is not None and cvb is not None:
            for metric in ("BC", "IQ", "SC", "AQ"):
                paired[model][metric].append(float(cvb[metric] - qvb[metric]))

primary = {
    "lc": {"PSNR", "SSIM", "LPIPS"},
    "sf": {"BC", "IQ", "SC", "AQ"},
    "hy": {"PSNR", "SSIM", "LPIPS"},
}
results = {}
overall = True
for model, metrics in paired.items():
    results[model] = {}
    for metric, values in metrics.items():
        if not values:
            continue
        mean, lower, upper = interval(values)
        margin = MARGINS[metric]
        equivalent = lower > -margin and upper < margin
        # One-sided superiority uses the same alpha and paired t interval and
        # must not be rejected merely because it exceeds the upper TOST bound.
        superior = lower > 0
        passed = equivalent or superior
        role = "primary" if metric in primary[model] else "guardrail"
        results[model][metric] = {
            "n": len(values),
            "oriented_mean": mean,
            "ci90": [lower, upper],
            "noninferiority_margin": margin,
            "role": role,
            "equivalent": equivalent,
            "superior": superior,
            "pass": passed,
        }
        overall &= passed

payload = {
    "candidate": "factor-only product-grid (completed Budget-PCA campaign)",
    "arms": ARMS,
    "orientation": "positive is better; LPIPS is QVG minus candidate",
    "models": results,
    "pass": overall,
}
with open("repro/0721/factor_mp100_tost.json", "w") as f:
    json.dump(payload, f, indent=2)

lines = [
    "# Factor-only fallback: paired MP100 TOST",
    "",
    "90% paired t interval; every effect is oriented so positive is better.",
    "",
    "| model | role | metric | n | mean | 90% CI | TOST margin | result |",
    "|---|---|---|---:|---:|---:|---:|---|",
]
for model, metrics in results.items():
    for metric, row in metrics.items():
        lines.append(
            f"| {model} | {row['role']} | {metric} | {row['n']} | "
            f"{row['oriented_mean']:+.4f} | "
            f"[{row['ci90'][0]:+.4f}, {row['ci90'][1]:+.4f}] | "
            f"±{row['noninferiority_margin']:.4f} | "
            f"{'SUPERIOR' if row['superior'] else ('EQUIV' if row['equivalent'] else 'FAIL')} |"
        )
lines += ["", f"**G4 fallback: {'PASS' if overall else 'FAIL'}**", ""]
with open("repro/0721/factor-mp100-tost.md", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
raise SystemExit(0 if overall else 1)
