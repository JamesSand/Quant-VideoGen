#!/usr/bin/env python3
"""Score the bounded factor-grid fallback before any new large campaign."""
import glob
import json
import os
import subprocess

import imageio.v3 as iio
import lpips
import numpy as np
import torch
import torch.nn.functional as F

os.chdir("/home/zhizhousha/workspace/video-project/Quant-VideoGen")


def one(pattern):
    values = glob.glob(pattern)
    if not values:
        raise FileNotFoundError(pattern)
    return values[0]


PATHS = {
    "sf": {
        "qvg": one("results/multiprompt/sf/qvg_rep0_f180/p1/*.mp4"),
        "factor": one("results/multiprompt/sf/pcafactor_rep0_f180/p1/*.mp4"),
    },
    "hy": {
        "ref": "results/multiprompt/hy/bf16_s0/0-0.mp4",
        "qvg": "results/multiprompt/hy/qvg_s0/0-0.mp4",
        "factor": "results/multiprompt/hy/pcafactor_s0/0-0.mp4",
    },
}


def ssim_batch(a, b):
    c1, c2 = 0.01**2, 0.03**2
    m1, m2 = F.avg_pool2d(a, 11, 1, 5), F.avg_pool2d(b, 11, 1, 5)
    v1 = F.avg_pool2d(a * a, 11, 1, 5) - m1**2
    v2 = F.avg_pool2d(b * b, 11, 1, 5) - m2**2
    v12 = F.avg_pool2d(a * b, 11, 1, 5) - m1 * m2
    return (
        ((2 * m1 * m2 + c1) * (2 * v12 + c2))
        / ((m1**2 + m2**2 + c1) * (v1 + v2 + c2))
    ).mean((1, 2, 3))


perceptual = lpips.LPIPS(net="vgg").cuda().eval()


def ref_metrics(ref_path, test_path):
    values, batch_a, batch_b = [], [], []

    def flush():
        if not batch_a:
            return
        a = torch.stack(batch_a).cuda(non_blocking=True)
        b = torch.stack(batch_b).cuda(non_blocking=True)
        with torch.no_grad():
            mse = (a - b).square().mean((1, 2, 3)).clamp_min(1e-30)
            psnr = 10 * torch.log10(1 / mse)
            ssim = ssim_batch(a, b)
            lpips_value = perceptual(a, b).reshape(-1)
        values.extend(
            zip(
                psnr.cpu().tolist(),
                ssim.cpu().tolist(),
                lpips_value.cpu().tolist(),
            )
        )
        batch_a.clear()
        batch_b.clear()

    for ref_frame, test_frame in zip(
        iio.imiter(ref_path, plugin="pyav"),
        iio.imiter(test_path, plugin="pyav"),
    ):
        batch_a.append(
            torch.from_numpy(np.asarray(ref_frame)).float().permute(2, 0, 1) / 255
        )
        batch_b.append(
            torch.from_numpy(np.asarray(test_frame)).float().permute(2, 0, 1) / 255
        )
        if len(batch_a) == 8:
            flush()
    flush()
    return np.asarray(values)


result = {"paths": PATHS, "metrics": {"sf": {}, "hy": {}}}
for arm in ("qvg", "factor"):
    selected = ref_metrics(PATHS["hy"]["ref"], PATHS["hy"][arm])[13:].mean(0)
    result["metrics"]["hy"][arm] = {
        "PSNR": float(selected[0]),
        "SSIM": float(selected[1]),
        "LPIPS": float(selected[2]),
    }

cache_path = "repro/0721/factor_canary_vbench.json"
env = dict(os.environ, VBENCH4_CACHE=cache_path)
subprocess.run(
    [".venv/bin/python", "repro/0718/scripts/vbench4.py"]
    + [PATHS["hy"][arm] for arm in ("qvg", "factor")],
    check=True,
    env=env,
)
subprocess.run(
    [".venv/bin/python", "repro/0718/scripts/vbench4.py"]
    + [PATHS["sf"][arm] for arm in ("qvg", "factor")]
    + ["--max-frames", "700"],
    check=True,
    env=env,
)
with open(cache_path) as f:
    vbench = json.load(f)
for model in ("sf", "hy"):
    max_frames = 700 if model == "sf" else 0
    for arm in ("qvg", "factor"):
        result["metrics"][model].setdefault(arm, {})
        result["metrics"][model][arm].update(
            vbench[f"{PATHS[model][arm]}::{max_frames}"]
        )

margins = {
    "PSNR": -0.10,
    "SSIM": -0.002,
    "LPIPS": 0.003,
    "BC": -0.20,
    "IQ": -0.30,
    "SC": -0.30,
    "AQ": -0.30,
}
deltas, passed = {}, True
for model, arms in result["metrics"].items():
    deltas[model] = {}
    for metric, factor_value in arms["factor"].items():
        if metric == "n":
            continue
        delta = factor_value - arms["qvg"][metric]
        deltas[model][metric] = delta
        if metric == "LPIPS":
            passed &= delta <= margins[metric]
        else:
            passed &= delta >= margins[metric]
result["deltas_factor_minus_qvg"] = deltas
result["pass"] = bool(passed)
with open("repro/0721/factor_canary_score.json", "w") as f:
    json.dump(result, f, indent=2)

lines = [
    "# Factor-only product-grid paired canary",
    "",
    "| model | metric | QVG | factor | delta |",
    "|---|---|---:|---:|---:|",
]
for model, arms in result["metrics"].items():
    for metric, factor_value in arms["factor"].items():
        if metric == "n":
            continue
        lines.append(
            f"| {model} | {metric} | {arms['qvg'][metric]:.4f} | "
            f"{factor_value:.4f} | {factor_value-arms['qvg'][metric]:+.4f} |"
        )
lines += ["", f"**Fallback canary: {'PASS' if passed else 'FAIL'}**", ""]
with open("repro/0721/factor-canary-report.md", "w") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
raise SystemExit(0 if passed else 1)
